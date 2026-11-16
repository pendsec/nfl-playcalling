"""`src.pipeline`: the stages run.py and the notebook share return the numbers the
estimators give when called directly."""

from __future__ import annotations

import numpy as np
import pytest

from _config import CFG_L1, WEIGHT_CLIP
from src.ope.doubly_robust import dr_per_play, dr_policy_value
from src.pipeline import holdout_ope, judge_report
from src.policy.safe import PolicyFit, defer_to, learn_policy
from src.scm import graph as scm
from src.stats import mean_se

GV = scm.ALIGNMENT_GRAPH.version_tag()


def _cfg(policy_type, judge="train"):
    return {**CFG_L1,
            "policy": {**CFG_L1["policy"], "type": policy_type, "lcb_kappa": 1.0},
            "ope": {**CFG_L1["ope"], "judge_propensity": judge, "judge_folds": 3},
            "sensitivity": {"gamma_range": [1.0, 2.0]}}


def test_holdout_ope_matches_the_estimators_called_directly(fitted_l1):
    f = fitted_l1
    cfg = _cfg("conservative")
    fit = PolicyFit(beh=f["beh"], q=f["q"], ensemble=None, policy=f["policy"])
    h = holdout_ope(f["train"], f["test"], cfg, fit, GV)
    d = f["test"].df
    a, r = d["action"].to_numpy(), d["reward"].to_numpy()
    q, pib = f["q"].predict_all_actions(d), f["beh"].propensity(d)
    assert h.judge is f["beh"]                                  # judge_propensity: train
    np.testing.assert_array_equal(h.pi_b, pib)
    want = dr_policy_value(q, pib, f["policy"].act(d)["policy_probs"], a, r, WEIGHT_CLIP,
                           graph_version=GV, policy_name="conservative")
    assert h.v_policy.value == want.value and h.v_policy.se == want.se
    assert h.v_policy.sensitivity is h.sensitivity              # the bound is attached
    assert list(h.sensitivity["gamma"]) == [1.0, 2.0]
    assert h.ladder["estimator"].tolist() == ["DM", "IPW", "SNIPW", "DR-SN"] * 2
    assert h.lifts == {} and h.report["judge"] == "train" and "log_loss_diff" not in h.report


def test_safe_policy_with_a_pooled_judge_reports_both_lifts(fitted_l1, l1_ensemble):
    f = fitted_l1
    cfg = _cfg("safe", judge="pooled_crossfit")
    policy = learn_policy(f["train"], f["q"], f["beh"], cfg, ensemble=l1_ensemble)
    fit = PolicyFit(beh=f["beh"], q=f["q"], ensemble=l1_ensemble, policy=policy)
    h = holdout_ope(f["train"], f["test"], cfg, fit, GV)
    d = f["test"].df
    a, r, games = d["action"].to_numpy(), d["reward"].to_numpy(), d["game_id"].to_numpy()
    q = f["q"].predict_all_actions(d)
    assert h.judge is not f["beh"]
    base = dr_per_play(q, h.pi_b, h.pi_b, a, r, WEIGHT_CLIP)
    anchored = defer_to(h.out, h.pi_b)
    want = mean_se(dr_per_play(q, h.pi_b, anchored, a, r, WEIGHT_CLIP) - base, games)
    assert h.lifts["anchored"] == want
    assert (h.report["lift_anchored"], h.report["lift_anchored_se"]) == want
    assert set(h.lifts) == {"as built", "anchored"}
    assert h.report["log_loss_diff"] == pytest.approx(
        h.report["judge_log_loss"] - h.report["train_fit_log_loss"])


def test_defer_to_replaces_only_the_deferred_rows():
    probs = np.eye(3)[[0, 1, 2]]
    pi_b = np.full((3, 3), 1 / 3)
    out = {"policy_probs": probs, "deferred": np.array([True, False, True])}
    got = defer_to(out, pi_b)
    np.testing.assert_array_equal(got[1], probs[1])
    np.testing.assert_array_equal(got[[0, 2]], pi_b[[0, 2]])
    assert defer_to({"policy_probs": probs}, pi_b) is probs


def test_judge_report_restricts_the_worst_balance_to_supported_calls():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(3), 1000)
    p[:, 2] = 0.001                                   # never supported
    p /= p.sum(1, keepdims=True)
    a = np.array([rng.choice(3, p=row) for row in p])
    row = judge_report(_cfg("safe"), a, p, p, {0: "x", 1: "y", 2: "z"}, np.arange(1000) // 10)
    assert row["train_fit_balance_call"] in ("x", "y")
    assert row["judge_log_loss"] == row["train_fit_log_loss"]
