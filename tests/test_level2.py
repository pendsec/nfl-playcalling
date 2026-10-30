"""
Level 2 — coverage shell x pressure given the alignment — and the factored policy.

Anchored on `generate_synthetic_alignment(coverage_effects=True)`: the same
alignment SCM with a known coverage structure on passes (the ideal safety
structure deepens with distance; a blitz pays more from a heavy alignment), a
coordinator whose coverage call follows the state, the alignment and his
hidden read, and `true_dropback_value` for the exact per-dropback value of any
factored policy.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _config import CFG_L1
from src.data.features import (build_level2_dataset, level2_config, level2_labels,
                               shell_index)
from src.data.load import generate_synthetic_alignment, true_dropback_value
from src.policy.hierarchical import alignment_frames, evaluate_factored, factor
from src.scm import graph as scm
from src.scm import identify


def _cfg(shells="family", axes=("package", "front")):
    lvl1 = {**CFG_L1["action"]["level1"], "axes": list(axes)}
    return {**CFG_L1, "action": {**CFG_L1["action"], "level1": lvl1,
                                 "level2": {"shells": shells, "blitz_rusher_threshold": 5}}}


# ── Taxonomy ─────────────────────────────────────────────────────────────────
def test_family_taxonomy_groups_by_safety_structure():
    cov = pd.Series(["COVER_0", "COVER_1", "COVER_3", "COVER_2", "COVER_4",
                     "COVER_6", "2_MAN", "COMBO", "PREVENT", None])
    got = shell_index(cov, _cfg()).tolist()
    assert got[:7] == [0, 1, 1, 2, 2, 2, 2]
    assert all(np.isnan(got[7:]))


def test_full_taxonomy_keeps_six_shells_and_drops_two_man():
    cov = pd.Series(["COVER_0", "COVER_1", "COVER_3", "COVER_6", "2_MAN"])
    got = shell_index(cov, _cfg("full")).tolist()
    assert got[:4] == [0, 1, 3, 5] and np.isnan(got[4])
    assert len(level2_labels(_cfg("full"))) == 12
    assert level2_labels(_cfg())[3] == "single_high_blitz"


def test_unknown_shell_mode_is_rejected():
    with pytest.raises(ValueError, match="shells"):
        level2_labels(_cfg("halves"))


# ── Dataset and graph ────────────────────────────────────────────────────────
def test_level2_rows_are_charted_dropbacks_with_the_alignment_in_the_state(l2):
    d1, d2 = l2["ds1"].df, l2["ds2"].df
    assert set(d2["play_type"]) == {"pass"}
    assert len(d2) == (d1["play_type"] == "pass").sum()     # every synthetic pass is charted
    assert {"l1_package", "l1_front"} <= set(l2["ds2"].state_cols)
    np.testing.assert_array_equal(d2["l1_package"], d2["l1_action"] // 4)
    np.testing.assert_array_equal(d2["l1_front"], d2["l1_action"] % 4)
    identify.assert_adjustment_consistency(l2["ds2"], scm.graph_for(l2["cfg2"]))


def test_package_only_level1_keeps_the_box_out_of_level2(l2):
    """Without the box in the call, the box is a sibling of the coverage — the
    level-2 state carries the package only, and the graph says so."""
    cfg = _cfg(axes=["package"])
    from src.data.features import build_dataset
    ds1 = build_dataset(l2["pbp"], cfg)
    ds2 = build_level2_dataset(ds1, level2_config(cfg))
    assert "l1_package" in ds2.state_cols and "l1_front" not in ds2.state_cols
    g = scm.graph_for(level2_config(cfg))
    assert "l1_front" not in g.adjustment_columns()
    assert g.fingerprint() != scm.graph_for(l2["cfg2"]).fingerprint()
    identify.assert_adjustment_consistency(ds2, g)


def test_coverage_graph_selects_on_charting_and_adjusts_for_the_alignment(l2):
    g = scm.graph_for(l2["cfg2"])
    assert g.treatment == scm.DEF_COVERAGE
    assert scm.DEF_ALIGNMENT in g.adjustment_set()
    assert g.selection == {scm.CHARTED}
    assert (scm.DEF_COVERAGE, scm.OFF_PLAYCALL) not in g.edges   # the stated assumption
    assert "n_defense_box" in g.def_choice_columns[scm.DEF_FRONT]


def test_level2_config_drops_the_read_allowance():
    cfg = {**_cfg(), "policy": {"type": "safe", "lcb_kappa": 2.0,
                                "read_allowance": {"axes": ["front"], "scale": 2.0},
                                "level2": {"lcb_kappa": 1.0}}}
    c2 = level2_config(cfg)
    assert c2["action"]["scheme"] == "level2"
    assert c2["policy"]["lcb_kappa"] == 1.0 and "read_allowance" not in c2["policy"]
    assert cfg["action"].get("scheme") != "level2"          # the original is untouched


def test_synthetic_level1_frame_is_unchanged_by_coverage_effects():
    a = generate_synthetic_alignment(2000, seed=3)
    b = generate_synthetic_alignment(2000, seed=3, coverage_effects=True)
    differ = {c for c in a.columns if not a[c].equals(b[c])}
    assert differ == {"defense_coverage_type", "number_of_pass_rushers", "epa"}
    runs = a["play_type"] == "run"
    np.testing.assert_allclose(a.loc[runs, "epa"], b.loc[runs, "epa"])


# ── The factored policy ──────────────────────────────────────────────────────
def test_alignment_frames_set_only_the_alignment_columns(l2):
    df = l2["test2"].df.head(5)
    frames = alignment_frames(df, l2["cfg"])
    assert len(frames) == 12
    assert frames[7]["l1_package"].eq(1).all() and frames[7]["l1_front"].eq(3).all()
    np.testing.assert_array_equal(frames[7]["ydstogo"], df["ydstogo"])


def test_factor_is_a_joint_distribution():
    rng = np.random.default_rng(0)
    p1 = rng.dirichlet(np.ones(12), 4)
    p2 = rng.dirichlet(np.ones(6), (4, 12))
    j = factor(p1, p2)
    assert j.shape == (4, 72)
    np.testing.assert_allclose(j.sum(1), 1.0)
    np.testing.assert_allclose(j[:, 7 * 6 + 3], p1[:, 7] * p2[:, 7, 3])


def test_coordinator_row_is_exactly_zero_lift(l2_eval):
    tab = l2_eval["table"].set_index("policy")
    assert tab.loc["coordinator", "lift"] == 0.0
    np.testing.assert_allclose(l2_eval["policies"]["coordinator"].sum(1), 1.0)
    np.testing.assert_allclose(l2_eval["policies"]["both levels"].sum(1), 1.0)


def test_level2_policy_beats_the_coordinator_in_truth(l2_unconf):
    """Where the effects are not confounded, the coverage policy at the
    coordinator's own alignment is worth more in truth than his coverage."""
    fam = np.arange(6) // 2
    df = l2_unconf["test2"].df
    true = {k: true_dropback_value(df, p, fam) for k, p in l2_unconf["eval"]["policies"].items()}
    assert true["coverage only"] > true["coordinator"]


def _lift_at_logged_alignment(f, kappa):
    """(true lift, DR lift) of the level-2 policy at the logged alignment."""
    import copy
    from src.ope.doubly_robust import dr_per_play
    from src.policy.safe import learn_safe_policy
    df, fam = f["test2"].df, np.arange(6) // 2
    cfg = copy.deepcopy(f["cfg2"]); cfg["policy"]["lcb_kappa"] = kappa
    out = learn_safe_policy(f["train2"], f["q2"], f["beh2"], cfg, f["pol2"].ensemble).act(df)
    onehot = np.eye(12)[df["l1_action"].to_numpy()][:, :, None]
    joint = lambda p: (onehot * p[:, None, :]).reshape(len(df), -1)
    true = (true_dropback_value(df, joint(out["policy_probs"]), fam)
            - true_dropback_value(df, joint(out["pi_b"]), fam))
    a, r = df["action"].to_numpy(), df["reward"].to_numpy()
    dr = (dr_per_play(out["q_all"], out["pi_b"], out["policy_probs"], a, r)
          - dr_per_play(out["q_all"], out["pi_b"], out["pi_b"], a, r)).mean()
    return true, dr


def test_confounding_inflates_dr_and_caution_limits_the_damage(l2, l2_unconf):
    """The failure mode level 2's bound is set against. When the coordinator's
    read picks the coverage AND improves the result, DR credits the policy with
    the read's outcomes: far above the truth, and far more than without the
    read. Raising the bound cannot make the policy worse in truth."""
    true0, dr0 = _lift_at_logged_alignment(l2, 0.0)
    true3, _ = _lift_at_logged_alignment(l2, 3.0)
    u_true0, u_dr0 = _lift_at_logged_alignment(l2_unconf, 0.0)
    assert dr0 - true0 > (u_dr0 - u_true0) + 0.03
    assert true3 >= true0


def test_dr_recovers_the_level2_gain_without_confounding(l2_unconf):
    """With the read's effect switched off, DR's estimate of the coverage
    policy's gain matches the true gain within its own error."""
    res, df = l2_unconf["eval"], l2_unconf["test2"].df
    fam = np.arange(6) // 2
    true = {k: true_dropback_value(df, p, fam) for k, p in res["policies"].items()}
    row = res["table"].set_index("policy").loc["coverage only"]
    assert abs(row["lift"] - (true["coverage only"] - true["coordinator"])) <= 3 * row["lift_se"]


# ── V2 -> V3 projection ──────────────────────────────────────────────────────
def test_v2_calls_project_onto_level2_calls():
    from src.evaluation.projection_gate import v2_to_level2
    m = v2_to_level2(_cfg())
    assert m.shape == (12, 6)
    np.testing.assert_array_equal(m.sum(1), 1.0)
    assert m[2 * 1 + 1, 3] == 1 and m[2 * 3 + 0, 2] == 1      # C1/C3 -> single-high
    assert m[2 * 5 + 1, 5] == 1                               # C6 blitz -> two-high blitz
    np.testing.assert_array_equal(v2_to_level2(_cfg("full")), np.eye(12))
