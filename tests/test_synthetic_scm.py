"""
Synthetic SCM regression tests.

These simulate data from a KNOWN structural causal model (src.data.load.
generate_synthetic) and assert the pipeline recovers the right answers. Per
CLAUDE.md these run on every change to OPE / Q-model / propensity code.

What the known SCM guarantees (see generate_synthetic docstring):
  * the true causal best response is to BLITZ on 3rd-and-long;
  * the behavior policy is confounded (blitzes more when the hidden coach_read
    is favorable), so the naive observational comparison is biased.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.load import generate_synthetic
from src.data.features import build_v1_dataset, time_aware_split
from src.models.behavior.propensity import fit_behavior_model
from src.models.outcome.q_model import fit_q_model
from src.ope.direct_method import behavior_recovery_check, dm_policy_value
from src.policy.greedy import learn_greedy_policy


CFG = {
    "data": {"team": "SYN", "down": 3, "holdout_season": 2023},
    "action": {"blitz_rusher_threshold": 5, "drop_uncharted_coverage": True},
    "reward": {"negate_epa": True},
    "behavior_model": {"type": "logistic", "C": 1.0, "max_iter": 1000,
                       "cross_fit_folds": 5},
    "outcome_model": {"type": "ridge", "alpha": 1.0},
    "policy": {"support_threshold": 0.05},
    "ope": {"clip_propensity": [0.01, 0.99]},
    "seed": 42,
}


@pytest.fixture(scope="module")
def fitted():
    pbp = generate_synthetic(n_plays=6000, seed=7)
    ds = build_v1_dataset(pbp, CFG)
    train, test = time_aware_split(ds, CFG["data"]["holdout_season"])
    beh, diag = fit_behavior_model(train, CFG)
    q = fit_q_model(train, CFG)
    policy = learn_greedy_policy(train, q, beh, CFG)
    return dict(ds=ds, train=train, test=test, beh=beh, q=q,
                policy=policy, diag=diag)


def test_dataset_shape(fitted):
    ds = fitted["ds"]
    assert ds.n_actions == 4
    assert len(ds.df) > 1000
    # All four actions should appear in the simulated data.
    assert ds.df[ds.action_col].nunique() == 4


def test_behavior_model_beats_majority(fitted):
    diag = fitted["diag"]
    # pi_b should learn the ydstogo->blitz signal, beating the majority class.
    assert diag["oof_accuracy"] >= diag["majority_baseline"]


def test_ope_recovers_behavior_value(fitted):
    """The core smoke test: DM under pi_b ~= empirical mean reward."""
    rec = behavior_recovery_check(fitted["train"], fitted["q"], fitted["beh"])
    assert rec["passes"], rec


def test_q_model_learns_blitz_helps_on_long(fitted):
    """On 3rd-and-long, Q should rank a blitz above the matching no-blitz call.

    True effect: blitz lowers EPA (raises reward) and the effect grows with
    distance. We check zone_blitz (a=1) > zone_no_blitz (a=0) on long downs.
    """
    test = fitted["test"]
    q = fitted["q"]
    long_df = test.df[test.df["ydstogo"] >= 12]
    assert len(long_df) > 20
    q_all = q.predict_all_actions(long_df)
    assert q_all[:, 1].mean() > q_all[:, 0].mean()


def test_learned_policy_not_worse_than_behavior(fitted):
    """OPE should rate the greedy policy at least as good as pi_b on holdout."""
    test, q, beh, policy = (fitted[k] for k in ("test", "q", "beh", "policy"))
    out = policy.act(test.df)
    q_all = q.predict_all_actions(test.df)
    v_greedy = dm_policy_value(q_all, out["policy_probs"]).value
    v_behavior = dm_policy_value(q_all, beh.propensity(test.df)).value
    assert v_greedy >= v_behavior - 1e-9


def test_no_postsnap_leakage_in_state(fitted):
    """State columns must not include the outcome or any post-snap field."""
    ds = fitted["ds"]
    forbidden = {"epa", "reward", "action", "_coach_read", "_true_blitz_effect"}
    assert forbidden.isdisjoint(set(ds.state_cols))
