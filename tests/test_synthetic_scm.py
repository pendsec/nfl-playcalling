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
from src.scm import graph as scm
from src.scm import identify
from src.models.behavior.propensity import fit_behavior_model
from src.models.outcome.q_model import fit_q_model, crossfit_q
from src.ope.direct_method import behavior_recovery_check, dm_policy_value
from src.ope.doubly_robust import dr_policy_value
from src.ope.sensitivity import sensitivity_bounds
from src.policy.conservative import learn_conservative_policy


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
    "policy": {"alpha": 0.05, "support_threshold": 0.05},
    "ope": {"clip_propensity": [0.01, 0.99], "weight_clip": 20.0},
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
