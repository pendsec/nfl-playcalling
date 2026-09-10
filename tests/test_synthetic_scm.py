"""
Synthetic SCM regression tests — V2.

These simulate data from a KNOWN structural causal model (src.data.load.
generate_synthetic) and assert the pipeline recovers the right answers. Per
CLAUDE.md they run on every change to OPE / Q-model / propensity code.

What the known SCM guarantees (see generate_synthetic docstring):
  * 12 actions (6 coverage shells x pressure) across all downs;
  * the causal-best coverage is an `ideal_shell` that rises with distance, and
    the causal-best pressure is blitz on long downs;
  * the behavior policy is confounded by a hidden `coach_read` (blitzes / plays
    tighter under a favorable read, which also lowers EPA), so the observational
    comparison is biased — the sensitivity analysis is what bounds it;
  * Cover-0 is only ever called on 3rd/4th-and-short (a seeded positivity hole).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load import generate_synthetic, true_policy_value
from src.data.features import build_dataset, time_aware_split
from src.schemas.dataset import Dataset
from src.scm import graph as scm
from src.scm import identify
from src.models.behavior.propensity import fit_behavior_model
from src.models.outcome.q_model import fit_q_model, crossfit_q
from src.ope.direct_method import behavior_recovery_check, dm_policy_value
from src.ope.doubly_robust import dr_policy_value
from src.ope.aipw import aipw_contrasts
from src.models.behavior.propensity import _cv_splits
from src.models.outcome.q_model import _shrunk_cell_mean
from src.ope.sensitivity import sensitivity_bounds, attach as attach_sensitivity
from src.policy.conservative import learn_conservative_policy
from src.policy.greedy import learn_greedy_policy
from src.evaluation import compare_policies, compare_v1_to_v2


# Light config so the calibrated-GBM fits stay fast in CI.
CFG = {
    "data": {"team": "SYN", "down": "all", "holdout_season": 2023},
    "action": {"blitz_rusher_threshold": 5, "drop_uncharted_coverage": True},
    "reward": {"negate_epa": True},
    "behavior_model": {"type": "lightgbm", "n_estimators": 120, "learning_rate": 0.05,
                       "max_depth": 5, "min_child_samples": 30,
                       "calibration_folds": 3, "cross_fit_folds": 2},
    "outcome_model": {"type": "lightgbm", "n_estimators": 150, "learning_rate": 0.05,
                      "max_depth": 5, "min_child_samples": 20, "min_cell": 20,
                      "cross_fit_folds": 5},
    "policy": {"alpha": 1.0, "support_threshold": 0.05},
    "ope": {"clip_propensity": [0.01, 0.99], "weight_clip": 20.0},
    "evaluation": {"baseline_name": "v1_greedy",
                   "candidate_name": "v2_conservative",
                   "regression_margin": 0.02},
    "seed": 42,
}
WEIGHT_CLIP = CFG["ope"]["weight_clip"]


@pytest.fixture(scope="module")
def fitted():
    pbp = generate_synthetic(n_plays=9000, seed=7)
    ds = build_dataset(pbp, CFG)
    train, test = time_aware_split(ds, CFG["data"]["holdout_season"])
    beh, diag = fit_behavior_model(train, CFG)
    q = fit_q_model(train, CFG)
    mu_train = crossfit_q(train, CFG)
    policy = learn_conservative_policy(train, q, beh, CFG)
    return dict(ds=ds, train=train, test=test, beh=beh, q=q, diag=diag,
                mu_train=mu_train, policy=policy)


# ── Data layer ───────────────────────────────────────────────────────────────
def test_dataset_12_actions(fitted):
    ds = fitted["ds"]
    assert ds.n_actions == 12
    assert ds.df[ds.action_col].nunique() == 12
    assert set(ds.df["down"].unique()) == {1, 2, 3, 4}
    assert len(ds.df) > 4000


def test_no_postsnap_leakage_in_state(fitted):
    ds = fitted["ds"]
    forbidden = {"epa", "reward", "action", "off_playcall",
                 "_coach_read", "_true_blitz_effect"}
    assert forbidden.isdisjoint(set(ds.state_cols))


def test_scm_adjustment_matches_state(fitted):
    """The SCM adjustment set and the engineered state must be identical."""
    ds = fitted["ds"]
    assert set(scm.adjustment_columns()) == set(ds.state_cols)
    identify.assert_adjustment_consistency(ds)  # raises on drift / mediator leak


# ── Behavior model ───────────────────────────────────────────────────────────
def test_pi_b_beats_majority_and_calibrated(fitted):
    diag = fitted["diag"]
    assert diag["oof_accuracy"] > diag["majority_baseline"]
    assert diag["ece"] < 0.30           # calibrated within a loose tolerance
    assert 0.0 < diag["ess_ratio"] <= 1.0


# ── Outcome model ────────────────────────────────────────────────────────────
def test_q_recovers_known_best_response(fitted):
    """On long downs the deep-zone blitz should out-Q a tight no-blitz call.

    True SCM: ideal_shell rises with distance and blitz helps on long downs, so
    e.g. cover4_blitz (a=9) should beat cover1_no_blitz (a=2) when ydstogo is big.
    """
    test, q = fitted["test"], fitted["q"]
    long_df = test.df[test.df["ydstogo"] >= 13]
    assert len(long_df) > 20
    q_all = q.predict_all_actions(long_df)
    assert q_all[:, 9].mean() > q_all[:, 2].mean()


# ── OPE ──────────────────────────────────────────────────────────────────────
def test_dr_recovers_behavior_value(fitted):
    """Core smoke test: DR under pi_b recovers the empirical mean reward."""
    rec = behavior_recovery_check(fitted["train"], fitted["q"], fitted["beh"],
                                  q_all=fitted["mu_train"], weight_clip=WEIGHT_CLIP)
    assert rec["passes"], rec


def test_dr_robust_to_outcome_misspecification(fitted):
    """Double robustness: with a badly-misspecified Q but correct propensities,
    DR still recovers pi_b's value while the Direct Method does not.

    Evaluating pi_b on its own logged data makes the importance weights 1, so
    with Q == 0 the DM value collapses to 0 but the DR correction restores the
    empirical mean reward.
    """
    train, beh = fitted["train"], fitted["beh"]
    a = train.df[train.action_col].to_numpy()
    r = train.df[train.reward_col].to_numpy()
    pi_b = beh.propensity(train.df)
    bad_q = np.zeros((len(r), train.n_actions))

    emp = r.mean()
    dm = dm_policy_value(bad_q, pi_b).value
    dr = dr_policy_value(bad_q, pi_b, pi_b, a, r, WEIGHT_CLIP).value
    assert abs(dr - emp) < abs(dm - emp)          # DR strictly better
    assert abs(dr - emp) < 0.01                   # and essentially exact


# ── Policy ───────────────────────────────────────────────────────────────────
def test_conservative_not_worse_than_behavior(fitted):
    """DR should rate the conservative policy at least as good as pi_b."""
    test, q, beh, policy = (fitted[k] for k in ("test", "q", "beh", "policy"))
    a = test.df[test.action_col].to_numpy()
    r = test.df[test.reward_col].to_numpy()
    q_all = q.predict_all_actions(test.df)
    pi_b = beh.propensity(test.df)
    out = policy.act(test.df)
    v_pi = dr_policy_value(q_all, pi_b, out["policy_probs"], a, r, WEIGHT_CLIP).value
    v_b = dr_policy_value(q_all, pi_b, pi_b, a, r, WEIGHT_CLIP).value
    assert v_pi >= v_b - 1e-6


def test_conservative_improves_true_value(fitted):
    """Against ground truth, the conservative policy beats the behavior policy."""
    test, policy = fitted["test"], fitted["policy"]
    a = test.df[test.action_col].to_numpy()
    out = policy.act(test.df)
    assert true_policy_value(test.df, out["chosen"]) > true_policy_value(test.df, a)


# ── Positivity ───────────────────────────────────────────────────────────────
def test_positivity_flags_seeded_hole(fitted):
    """Cover-0 (actions 0,1) is seeded to never occur on 1st down."""
    ds = fitted["ds"]
    assert not identify.is_action_observed(ds, 0, down=1)
    assert not identify.is_action_observed(ds, 1, down=1)
    assert identify.is_action_observed(ds, 0, down=3)   # but present on 3rd-short
    pos = identify.check_positivity(fitted["train"], min_count=10)
    assert pos["n_absent_cells"] > 0
    assert pos["pct_cells_supported"] < 1.0


# ── Sensitivity ──────────────────────────────────────────────────────────────
def test_sensitivity_widens_and_brackets_truth(fitted):
    """At gamma=1 the bound is a point; it widens with gamma and, once wide
    enough, brackets the policy's true value that gamma=1 misses (the signature
    of the SCM's real unobserved confounding)."""
    test, q, beh, policy = (fitted[k] for k in ("test", "q", "beh", "policy"))
    a = test.df[test.action_col].to_numpy()
    r = test.df[test.reward_col].to_numpy()
    q_all = q.predict_all_actions(test.df)
    pi_b = beh.propensity(test.df)
    out = policy.act(test.df)
    true_v = true_policy_value(test.df, out["chosen"])

    sb = sensitivity_bounds(q_all, pi_b, out["policy_probs"], a, r,
                            [1.0, 1.5, 2.0, 3.0], WEIGHT_CLIP).set_index("gamma")
    # Point at gamma=1, monotonically widening.
    assert sb.loc[1.0, "width"] == pytest.approx(0.0, abs=1e-9)
    widths = sb["width"].to_numpy()
    assert np.all(np.diff(widths) > 0)
    # gamma=1 point misses the truth (confounding bias); a wider gamma brackets it.
    assert abs(sb.loc[1.0, "v_lower"] - true_v) > 1e-3
    assert sb.loc[3.0, "v_lower"] <= true_v <= sb.loc[3.0, "v_upper"]


# ── Identification (DoWhy) ───────────────────────────────────────────────────
def test_dowhy_backdoor_identifies(fitted):
    model = identify.build_causal_model(fitted["train"])
    est = identify.identify_estimand(model)
    bd = est.get_backdoor_variables()
    assert len(bd) == len(scm.adjustment_columns())


# ── OPE provenance (CLAUDE.md causal discipline) ─────────────────────────────
def test_ope_estimates_carry_graph_version_and_estimator(fitted):
    """Every OPE estimate must declare the graph it assumed and how it was computed.

    CLAUDE.md: "Every OPE estimate ships with: point estimate, confidence
    interval, sensitivity bound for unobserved confounders, and the assumed
    causal graph version." An unlabelled number is not a causal claim.
    """
    test, q, beh = fitted["test"], fitted["q"], fitted["beh"]
    df = test.df
    q_all = q.predict_all_actions(df)
    pi_b = beh.propensity(df)
    a = df[test.action_col].to_numpy()
    r = df[test.reward_col].to_numpy()

    res = dr_policy_value(q_all, pi_b, pi_b, a, r,
                          weight_clip=WEIGHT_CLIP, policy_name="behavior")
    assert res.estimator == "DR-SN"
    assert res.policy == "behavior"
    assert res.graph_version == scm.version_tag()
    assert res.graph_version.startswith(scm.GRAPH_VERSION)

    rec = res.to_record()
    for key in ("value", "ci_lo", "ci_hi", "n", "estimator", "graph_version",
                "sensitivity_computed", "robust_to_gamma"):
        assert key in rec
    # No sensitivity attached yet — the record must say so rather than imply one.
    assert rec["sensitivity_computed"] is False
    assert rec["robust_to_gamma"] is None


def test_graph_fingerprint_tracks_structural_edits():
    """The fingerprint must change when the DAG's structural claims change.

    GRAPH_VERSION is hand-maintained and can go stale; the fingerprint is the
    guard that makes a silent edit visible on every stored estimate.
    """
    before = scm.fingerprint()
    assert before == scm.fingerprint()          # deterministic

    scm.EDGES.append((scm.PLAYER_PROXIES, scm.OFF_FORMATION))
    try:
        assert scm.fingerprint() != before
    finally:
        scm.EDGES.pop()
    assert scm.fingerprint() == before


def test_sensitivity_attaches_to_its_own_estimate(fitted):
    """A sensitivity bound must bind to the estimate it qualifies, not float free."""
    test, q, beh, policy = fitted["test"], fitted["q"], fitted["beh"], fitted["policy"]
    df = test.df
    q_all = q.predict_all_actions(df)
    pi_b = beh.propensity(df)
    a = df[test.action_col].to_numpy()
    r = df[test.reward_col].to_numpy()
    probs = policy.act(df)["policy_probs"]

    v_b = dr_policy_value(q_all, pi_b, pi_b, a, r, WEIGHT_CLIP, policy_name="behavior")
    v_p = dr_policy_value(q_all, pi_b, probs, a, r, WEIGHT_CLIP,
                          policy_name="conservative")
    sens = sensitivity_bounds(q_all, pi_b, probs, a, r, [1.0, 1.5, 3.0],
                              WEIGHT_CLIP, compare_value=v_b.value,
                              policy_name="conservative")

    attach_sensitivity(v_p, sens)
    assert v_p.to_record()["sensitivity_computed"] is True
    assert v_p.robust_to_gamma is not None

    # Attaching the conservative policy's bound to the behavior estimate is the
    # exact mix-up this guards against.
    with pytest.raises(ValueError):
        attach_sensitivity(v_b, sens)


# ── Version-over-version regression gate ─────────────────────────────────────
def test_regression_gate_passes_on_v1_to_v2(fitted):
    """V2's conservative policy must not be significantly worse than V1's greedy."""
    cmp = compare_v1_to_v2(fitted["test"], fitted["q"], fitted["beh"], CFG)
    assert cmp.n == len(fitted["test"].df)
    assert not cmp.regressed, f"V2 regressed against V1: {cmp.report()}"
    assert cmp.to_record()["blocks_merge"] is False
    assert cmp.candidate.graph_version == scm.version_tag()


def test_regression_gate_catches_a_deliberately_bad_policy(fitted):
    """The gate must actually fail — a worst-Q policy has to trip it.

    A gate that never fires is not a gate. We hand it a policy that picks the
    *lowest*-Q supported action on every play and require a REGRESSION verdict.
    """
    test, q, beh = fitted["test"], fitted["q"], fitted["beh"]
    df = test.df
    good = learn_greedy_policy(test, q, beh, CFG).act(df)

    # Deliberately terrible candidate: argmin Q among supported calls.
    q_all = good["q_all"]
    masked = np.where(good["support"], q_all, np.inf)
    worst = masked.argmin(axis=1)
    no_support = ~good["support"].any(axis=1)
    worst[no_support] = q_all[no_support].argmin(axis=1)
    bad_probs = np.zeros_like(q_all)
    bad_probs[np.arange(len(df)), worst] = 1.0

    cmp = compare_policies(test, q, beh, good["policy_probs"], bad_probs,
                           weight_clip=WEIGHT_CLIP, margin=0.02,
                           baseline_name="v1_greedy", candidate_name="worst_q")
    assert cmp.regressed
    assert cmp.verdict == "REGRESSION"
    assert cmp.to_record()["blocks_merge"] is True

# ── AIPW variance controls ───────────────────────────────────────────────────
def test_aipw_weight_controls_recover_a_known_null_contrast():
    """Raw 1/pi_b weights let a mis-scaled propensity invent an effect.

    Ground truth here is a NULL contrast: every play earns reward 1 regardless of
    the call, so ATE(rare vs common) is exactly 0. But pi_b claims action 0 is
    taken 2% of the time when it is really taken 30%, so the raw inverse weights
    sum to ~15x n and manufacture a large positive effect. This is the shape of
    the defect that shipped a +7.58 EPA/play "significant" contrast on real data.

    Clipping alone is not enough — it bounds any single play's leverage but not
    the systematic mis-scaling. Self-normalization is what forces the correction
    to be an average, and the two together recover the truth.
    """
    n = 600
    q_all = np.zeros((n, 2))
    pi_b = np.tile([0.02, 0.98], (n, 1))
    actions = np.array([0] * 180 + [1] * 420)
    rewards = np.ones(n)
    labels = {0: "rare", 1: "common"}

    def ate(**kw):
        d = aipw_contrasts(q_all, pi_b, actions, rewards, labels, baseline=1, **kw)
        return float(d["ate_vs_baseline"].iloc[0])

    raw = ate(weight_clip=np.inf, self_normalize=False)
    clip_only = ate(weight_clip=20.0, self_normalize=False)
    fixed = ate(weight_clip=20.0, self_normalize=True)

    assert raw > 10.0                          # the defect: an effect from nothing
    assert abs(clip_only) < abs(raw)           # clipping helps but does not fix it
    assert fixed == pytest.approx(0.0, abs=1e-6)   # both controls recover the truth


def test_aipw_reports_support_behind_each_contrast(fitted):
    """A contrast estimated off a handful of plays must say so in the output."""
    test, q, beh = fitted["test"], fitted["q"], fitted["beh"]
    df = test.df
    a = df[test.action_col].to_numpy()
    contrasts = aipw_contrasts(
        q.predict_all_actions(df), beh.propensity(df), a,
        df[test.reward_col].to_numpy(), test.action_labels,
        baseline=0, weight_clip=WEIGHT_CLIP,
    )
    assert {"n_taken", "n_taken_baseline"} <= set(contrasts.columns)
    assert (contrasts["n_taken_baseline"] == int((a == 0).sum())).all()
    for _, row in contrasts.iterrows():
        label_to_id = {v: k for k, v in test.action_labels.items()}
        assert row["n_taken"] == int((a == label_to_id[row["action"]]).sum())


# ── Rare-class cross-fitting ─────────────────────────────────────────────────
def test_cv_splits_keeps_unsplittable_class_on_the_training_side():
    """One singleton class must not disable cross-fitting for every other action.

    This is what made the V2 propensity diagnostics dishonest: a lone play in one
    of twelve cells sent accuracy/ECE/ESS down an in-sample fallback path, and an
    in-sample ESS ratio measures memorization, not overlap.
    """
    y = np.array([0] * 50 + [1] * 50 + [2])       # class 2 has a single member
    splits, rare = _cv_splits(y, requested=3, seed=0)

    assert splits, "a single un-splittable class must not veto cross-fitting"
    assert rare.sum() == 1 and rare[-1]

    singleton = len(y) - 1
    for train_idx, test_idx in splits:
        assert singleton in train_idx      # still informs every fit
        assert singleton not in test_idx   # but never scored out-of-fold

    # Every splittable row is scored exactly once across the folds.
    scored = np.concatenate([te for _, te in splits])
    assert sorted(scored.tolist()) == list(range(len(y) - 1))


def test_behavior_diagnostics_are_out_of_fold(fitted):
    """pi_b diagnostics must be cross-fitted and account for every training row."""
    diag, train = fitted["diag"], fitted["train"]
    assert diag["diagnostics_source"] == "oof"
    assert diag["n_splits"] >= 2
    assert diag["n_diagnostic"] + diag["n_excluded_rare"] == len(train.df)
    # An in-sample ESS ratio on a memorized fit sits near 1; an honest one does not.
    assert diag["ess_ratio"] < 0.95


# ── Q-model shrinkage ────────────────────────────────────────────────────────
def test_thin_q_cells_shrink_toward_the_global_mean():
    """A cell too thin for a regressor must not contribute an extreme constant.

    On real data a 9-play cell produced a -2.30 fallback against a global mean
    near -0.09, and because a fallback is constant across every state it then
    dominated the policy's argmax everywhere.
    """
    global_mean = 0.0
    one_play = _shrunk_cell_mean(np.array([-5.0]), global_mean, 20)
    near_full = _shrunk_cell_mean(np.full(19, -5.0), global_mean, 20)

    assert _shrunk_cell_mean(np.array([]), global_mean, 20) == global_mean
    assert one_play == pytest.approx(-5.0 / 21.0)      # a lone play barely moves it
    assert abs(one_play) < abs(near_full) < 5.0        # graceful with cell size
    assert near_full > -5.0                            # never the raw cell mean


# ── Conservatism scaling ─────────────────────────────────────────────────────
def test_alpha_actually_moves_the_policy_toward_supported_calls(fitted):
    """The conservatism penalty must have authority over the argmax.

    With the penalty in raw reward units it spanned ~0.23 against a Q spread of
    ~3.1 and could not change a single decision. Scaling it by the Q model's own
    within-state spread (QModel.q_scale) is what makes alpha a real knob.
    """
    test, q, beh = fitted["test"], fitted["q"], fitted["beh"]
    assert q.q_scale > 0

    pi_b = beh.propensity(test.df)
    idx = np.arange(len(pi_b))
    support = []
    for alpha in (0.0, 1.0, 4.0):
        cfg = {**CFG, "policy": {**CFG["policy"], "alpha": alpha}}
        out = learn_conservative_policy(test, q, beh, cfg).act(test.df)
        support.append(float(pi_b[idx, out["chosen"]].mean()))

    assert support[0] < support[1] < support[2], support

# ── DAG structural claims (v2.1) ─────────────────────────────────────────────
def test_declared_mediator_is_actually_a_mediator():
    """A variable in MEDIATORS must be a descendant of the treatment.

    Through v2.0 `off_playcall` was declared a mediator but had no incoming edge
    from `def_playcall`, so nothing in the graph made it one — the label and the
    structure disagreed, and only the label was ever read.
    """
    children_of_treatment = {t for s_, t in scm.EDGES if s_ == scm.TREATMENT}
    for m in scm.MEDIATORS:
        assert m in children_of_treatment, (
            f"{m} is declared a mediator but the treatment has no edge into it"
        )


def test_selection_node_is_a_collider_on_a_treatment_outcome_path():
    """The selection the dataset performs must be visible in the graph.

    Every modelled row has coverage_charted = 1. That node takes both
    `off_playcall` (runs are ~3% charted) and `epa` (a sack leaves no shell) as
    parents, which is what makes selecting on it both block part of the total
    effect and open a non-causal path.
    """
    assert scm.SELECTION, "the dataset is selected on something; say what"
    for sel in scm.SELECTION:
        parents = {s_ for s_, t in scm.EDGES if t == sel}
        assert len(parents) >= 2, f"{sel} is not a collider: parents={parents}"
        assert scm.OUTCOME in parents
        # ...and reachable from the treatment, or selecting on it would be benign.
        assert parents & ({scm.TREATMENT} | scm.MEDIATORS)


def test_selection_variables_cannot_be_used_as_features(fitted):
    """Adjusting for a variable the sample is conditioned on is a contradiction."""
    ds = fitted["ds"]
    assert not (set(scm.SELECTION) & set(ds.state_cols))

    polluted = Dataset(
        df=ds.df.assign(**{c: 1 for c in scm.SELECTION}),
        numeric_state=ds.numeric_state + sorted(scm.SELECTION),
        categorical_state=ds.categorical_state, action_col=ds.action_col,
        reward_col=ds.reward_col, n_actions=ds.n_actions,
        action_labels=ds.action_labels,
    )
    with pytest.raises(ValueError, match="[Ss]election"):
        identify.assert_adjustment_consistency(polluted)


def test_graph_version_was_bumped_for_the_structural_change():
    """Structural edits must move GRAPH_VERSION, not just the fingerprint.

    The fingerprint catches an unversioned edit after the fact; the version is
    what a human reads on a stored estimate. v2.0 predates the mediator edge and
    the selection node, so an estimate stamped v2.0 assumed a different graph.
    """
    assert scm.GRAPH_VERSION >= "v2.1"
    assert scm.version_tag().startswith(scm.GRAPH_VERSION + "+")
