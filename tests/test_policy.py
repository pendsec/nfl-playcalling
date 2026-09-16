"""
Policy learning and the regression gate.

The gate is deliberately asymmetric: it blocks on a proven deterioration past a
non-inferiority margin, not on a failure to prove improvement. A gate that never
fires is not a gate, so one test hands it a deliberately terrible policy and
requires a REGRESSION verdict.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scm import graph as scm
from _config import CFG, WEIGHT_CLIP
from src.data.load import true_policy_value
from src.evaluation import compare_policies, compare_to_baseline
from src.ope.doubly_robust import dr_policy_value
from src.policy.conservative import learn_conservative_policy
from src.policy.greedy import learn_greedy_policy


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


def test_regression_gate_passes_for_the_conservative_policy(fitted):
    """The conservative policy must not be significantly worse than the baseline."""
    cmp = compare_to_baseline(fitted["test"], fitted["q"], fitted["beh"], CFG)
    assert cmp.n == len(fitted["test"].df)
    assert not cmp.regressed, f"candidate regressed against baseline: {cmp.report()}"
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
                           baseline_name="greedy_baseline", candidate_name="worst_q")
    assert cmp.regressed
    assert cmp.verdict == "REGRESSION"
    assert cmp.to_record()["blocks_merge"] is True
