"""
Every-snap pipeline: personnel package x front, all defenses, runs included.

Anchored on `generate_synthetic_alignment`, a known SCM with deterrence built in
— a heavier alignment lowers the chance the offense runs at all — so the value of
a call is a mixture over the offense's response, and `true_alignment_value`
computes it exactly. That makes three claims checkable against truth rather than
against another estimate: the learned policy beats the coordinator, DR recovers
the behavior policy's value, and the sensitivity band brackets the true value.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _config import CFG, CFG_L1, WEIGHT_CLIP
from src.data.features import (LEVEL1_LABELS, TEAM_STATE, _qb_on_field,
                               build_dataset, personnel_db_count)
from src.data.load import true_alignment_value
from src.evaluation import compare_to_baseline
from src.models.outcome.q_model import fit_q_model
from src.ope.doubly_robust import dr_policy_value
from src.ope.ipw import ipw_policy_value, weight_diagnostics
from src.ope.ladder import ope_ladder
from src.ope.sensitivity import sensitivity_bounds
from src.schemas.dataset import Dataset
from src.scm import graph as scm
from src.scm import identify


# ── Data layer ───────────────────────────────────────────────────────────────
def test_personnel_parses_to_defensive_back_count():
    s = pd.Series(["3 CB, 2 DE, 2 DT, 1 FS, 2 ILB, 1 SS",   # nickel: 5
                   "2 CB, 2 DE, 2 DT, 1 FS, 3 ILB, 1 SS",   # base: 4
                   "4 CB, 1 DB, 2 DE, 1 DT, 1 FS, 1 ILB, 1 SS",  # dime+: 7
                   "", None])
    assert personnel_db_count(s).tolist() == [5, 4, 7, 0, 0]


def test_level1_labels_land_on_the_right_bin_edges():
    """Package by DB count, front by box count — checked at every edge."""
    pers = {4: "2 CB, 1 FS, 1 SS", 5: "3 CB, 1 FS, 1 SS", 6: "4 CB, 1 FS, 1 SS"}
    rows, want = [], []
    for n_db, pkg in ((4, 0), (5, 1), (6, 2)):
        for box, front in ((4, 0), (5, 0), (6, 1), (7, 2), (8, 3), (10, 3)):
            rows.append({"defense_personnel": pers[n_db], "n_defense_box": float(box)})
            want.append(pkg * 4 + front)
    from src.data.features import _label_level1
    out = _label_level1(pd.DataFrame(rows), CFG_L1)
    assert out["action"].tolist() == want
    assert set(out["action_label"]) <= set(LEVEL1_LABELS.values())


def test_unlabeled_alignments_are_dropped_not_guessed():
    from src.data.features import _label_level1
    df = pd.DataFrame({"defense_personnel": ["3 CB, 1 FS, 1 SS", "", "3 CB, 1 FS, 1 SS"],
                       "n_defense_box": [6.0, 6.0, np.nan]})
    assert len(_label_level1(df, CFG_L1)) == 1


def test_every_snap_dataset_keeps_runs_and_every_defense(fitted_l1):
    df = fitted_l1["ds"].df
    assert set(df["play_type"]) == {"pass", "run"}
    assert 0.2 < (df["play_type"] == "run").mean() < 0.8
    assert df["action"].nunique() == 12
    assert df["defteam"].nunique() == 8
    assert set(TEAM_STATE) <= set(fitted_l1["ds"].state_cols)


def test_qb_feature_does_not_reveal_play_type(fitted_l1):
    """nflfastR records the passer only on passes. Grouping a QB feature by it
    directly defaults the feature on every run, and its value alone then says
    whether the play was a run — a post-snap fact in the pre-snap state."""
    df = fitted_l1["ds"].df
    zero = df.groupby("play_type")["qb_avg_epa"].apply(lambda x: (x == 0).mean())
    assert abs(zero["run"] - zero["pass"]) < 0.05, zero.to_dict()


def test_qb_on_field_fills_within_the_offenses_game():
    df = pd.DataFrame({
        "game_id": ["g1"] * 4 + ["g2"] * 2, "posteam": ["A"] * 4 + ["B"] * 2,
        "passer_player_id": [None, "QB1", None, None, None, "QB9"],
    })
    assert _qb_on_field(df).tolist() == ["QB1"] * 4 + ["QB9"] * 2


# ── Causal graph ─────────────────────────────────────────────────────────────
def test_shell_pressure_graph_is_unchanged_by_the_refactor():
    """Pinned: the dropback graph's structure — and so every estimate stamped
    with it — must not move because a second graph was added beside it."""
    assert scm.version_tag() == "v2.2+14cc1d02d555"
    assert scm.SHELL_PRESSURE_GRAPH.version_tag() == scm.version_tag()


def test_graph_is_chosen_by_what_the_action_is():
    assert scm.graph_for(CFG_L1) is scm.ALIGNMENT_GRAPH
    assert scm.graph_for(CFG) is scm.SHELL_PRESSURE_GRAPH
    with pytest.raises(ValueError):
        scm.graph_for({"action": {"scheme": "disguise"}})


def test_alignment_graph_routes_the_effect_through_both_mediators():
    """The offense's run/pass choice (deterrence) and the paired coverage call
    are both downstream of the alignment — so both are mediators, and neither
    may be conditioned on when estimating the alignment's total effect."""
    g = scm.ALIGNMENT_GRAPH
    children = {t for s, t in g.edges if s == g.treatment}
    assert {scm.OFF_PLAYCALL, scm.DEF_COVERAGE} <= g.mediators <= children
    assert not g.selection, "no dropback selection on the every-snap sample"
    assert scm.TEAMS in g.adjustment_set()


def test_every_snap_state_matches_the_alignment_graph(fitted_l1):
    ds = fitted_l1["ds"]
    identify.assert_adjustment_consistency(ds, scm.ALIGNMENT_GRAPH)
    assert set(ds.state_cols) == set(scm.ALIGNMENT_GRAPH.adjustment_columns())


def test_treatment_columns_are_rejected_as_state(fitted_l1):
    """Conditioning on the action being evaluated makes its effect unlearnable."""
    ds = fitted_l1["ds"]
    polluted = Dataset(df=ds.df, numeric_state=ds.numeric_state + ["n_defense_box"],
                       categorical_state=ds.categorical_state, action_col=ds.action_col,
                       reward_col=ds.reward_col, n_actions=ds.n_actions,
                       action_labels=ds.action_labels)
    with pytest.raises(ValueError, match="Treatment column"):
        identify.assert_adjustment_consistency(polluted, scm.ALIGNMENT_GRAPH)


# ── IPW and the ladder ───────────────────────────────────────────────────────
def test_ipw_is_unbiased_with_known_propensities():
    """No outcome model at all: reweighting by the true propensities recovers a
    deterministic policy's value when rewards depend on the action."""
    rng = np.random.default_rng(0)
    n, k = 200_000, 3
    pi_b = rng.dirichlet(np.full(k, 3.0), n)
    a = (rng.random(n)[:, None] > np.cumsum(pi_b, 1)).sum(1).clip(0, k - 1)
    effect = np.array([0.0, 0.3, -0.2])
    r = effect[a] + rng.normal(0, 1, n)
    target = np.zeros((n, k)); target[:, 1] = 1.0          # always action 1
    est = ipw_policy_value(pi_b, target, a, r, weight_clip=1e9)
    assert est.value == pytest.approx(0.3, abs=4 * est.se)
    assert est.estimator == "IPW"


def test_snipw_is_undefined_rather_than_zero_without_support():
    pi_b = np.full((4, 2), 0.5); a = np.zeros(4, int); r = np.ones(4)
    never = np.zeros((4, 2)); never[:, 1] = 1.0            # never the logged call
    assert np.isnan(ipw_policy_value(pi_b, never, a, r, self_normalize=True).value)


def test_weight_diagnostics_report_what_the_estimate_rests_on():
    pi_b = np.full((10, 2), 0.5); a = np.array([0] * 3 + [1] * 7)
    target = np.zeros((10, 2)); target[:, 0] = 1.0
    d = weight_diagnostics(pi_b, target, a)
    assert d["support_share"] == pytest.approx(0.3)
    assert 0 < d["ess_ratio"] <= 1


def test_ladder_has_every_rung_stamped_with_the_right_graph(fitted_l1):
    t, q, beh = fitted_l1["test"], fitted_l1["q"], fitted_l1["beh"]
    d = t.df
    tag = scm.ALIGNMENT_GRAPH.version_tag()
    lad = ope_ladder(q.predict_all_actions(d), beh.propensity(d),
                     fitted_l1["policy"].act(d)["policy_probs"],
                     d[t.action_col].to_numpy(), d[t.reward_col].to_numpy(),
                     WEIGHT_CLIP, policy_name="conservative", graph_version=tag)
    assert lad["estimator"].tolist() == ["DM", "IPW", "SNIPW", "DR-SN"]
    assert (lad["graph_version"] == tag).all()


def test_regression_gate_stamps_the_alignment_graph(fitted_l1):
    cmp = compare_to_baseline(fitted_l1["test"], fitted_l1["q"], fitted_l1["beh"], CFG_L1)
    assert cmp.candidate.graph_version == scm.ALIGNMENT_GRAPH.version_tag()
    assert not cmp.regressed, cmp.report()


# ── Outcome model ────────────────────────────────────────────────────────────
def _true_v(df):
    from src.data.load import _alignment_components, alignment_heaviness
    h = alignment_heaviness(np.arange(12))[None, :]
    p, rr, rp = _alignment_components(h, df["_p_run_logit"].to_numpy()[:, None])
    return p * rr + (1 - p) * rp + (df["_qb_term"] + df["_base"]).to_numpy()[:, None]


def test_pooled_q_does_not_inflate_the_spread_between_actions(fitted_l1):
    """One regressor per action estimates each cell's effect with independent
    noise, inflating the spread across actions — and a policy's argmax then
    chases whichever cell is most overestimated. Pooling over (package, front)
    has to keep that spread near the truth's."""
    tr, te = fitted_l1["train"], fitted_l1["test"]
    true_sd = _true_v(te.df).std(1).mean()
    pooled_sd = fitted_l1["q"].predict_all_actions(te.df).std(1).mean()
    per_action = fit_q_model(tr, {**CFG_L1, "outcome_model": {**CFG_L1["outcome_model"],
                                                              "structure": "per_action"}})
    per_sd = per_action.predict_all_actions(te.df).std(1).mean()
    assert pooled_sd < per_sd
    assert pooled_sd / true_sd < 2.0


# ── Against the truth ────────────────────────────────────────────────────────
def test_conservative_policy_beats_the_coordinator_in_truth(fitted_l1):
    d = fitted_l1["test"].df
    chosen = fitted_l1["policy"].act(d)["chosen"]
    assert true_alignment_value(d, chosen) > true_alignment_value(d, d["action"].to_numpy())


def test_dr_recovers_the_behavior_policys_true_value(fitted_l1):
    t, q, beh = fitted_l1["test"], fitted_l1["q"], fitted_l1["beh"]
    d = t.df; a = d["action"].to_numpy(); pib = beh.propensity(d)
    est = dr_policy_value(q.predict_all_actions(d), pib, pib, a, d["reward"].to_numpy())
    assert est.value == pytest.approx(true_alignment_value(d, a), abs=3 * est.se)


def test_sensitivity_band_brackets_the_true_policy_value(fitted_l1):
    """With the hidden read confounding the outcome, the point estimate can miss;
    a wide enough band must not."""
    t, q, beh = fitted_l1["test"], fitted_l1["q"], fitted_l1["beh"]
    d = t.df; a = d["action"].to_numpy(); pib = beh.propensity(d)
    out = fitted_l1["policy"].act(d)
    band = sensitivity_bounds(q.predict_all_actions(d), pib, out["policy_probs"], a,
                              d["reward"].to_numpy(), [1.0, 3.0], WEIGHT_CLIP).set_index("gamma")
    truth = true_alignment_value(d, out["chosen"])
    assert band.loc[3.0, "v_lower"] <= truth <= band.loc[3.0, "v_upper"]


def test_dr_recovers_the_true_policy_value_without_confounding(fitted_l1_unconfounded):
    """The one setting where OPE should hit the TRUE value of a new policy."""
    f = fitted_l1_unconfounded
    d = f["test"].df; a = d["action"].to_numpy(); pib = f["beh"].propensity(d)
    out = f["policy"].act(d)
    est = dr_policy_value(f["q"].predict_all_actions(d), pib, out["policy_probs"], a,
                          d["reward"].to_numpy(), WEIGHT_CLIP)
    assert est.value == pytest.approx(true_alignment_value(d, out["chosen"]), abs=3 * est.se)
