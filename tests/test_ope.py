"""
Off-policy evaluation: recovery, double robustness, variance controls, provenance.

The behavior-recovery smoke test is the gate CLAUDE.md requires before any
off-policy number is believed — OPE has to reproduce the behavior policy's own
mean reward first.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scm import graph as scm
from _config import WEIGHT_CLIP
from src.data.load import true_policy_value
from src.ope.aipw import aipw_contrasts
from src.ope.direct_method import behavior_recovery_check, dm_policy_value
from src.ope.doubly_robust import dr_policy_value
from src.ope.sensitivity import attach as attach_sensitivity, sensitivity_bounds


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
