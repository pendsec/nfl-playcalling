"""
Off-Policy Evaluation — Direct Method (V1).

The Direct Method plugs the fitted Q-model into a candidate policy:

    V_DM(pi) = E_s [ sum_a  pi(a|s) * Q(s, a) ]

It is biased if Q is misspecified (V2 upgrades this to doubly-robust), but it is
the right first estimator for the skeleton and it gives us the mandatory smoke
test: recover the behavior policy's known average reward. If V_DM(pi_b) does not
land near the empirical mean reward, the Q-model is broken and no candidate
policy estimate can be trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..models.outcome.q_model import QModel
from ..models.behavior.propensity import BehaviorModel
from ..data.features import Dataset


@dataclass
class OPEResult:
    value: float          # estimated mean reward per play under the policy
    se: float             # standard error across plays
    n: int

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.value - 1.96 * self.se, self.value + 1.96 * self.se)


def dm_policy_value(q_all: np.ndarray, policy_probs: np.ndarray) -> OPEResult:
    """Direct-Method value of a (possibly stochastic) policy.

    q_all, policy_probs: shape (n, n_actions). policy_probs rows sum to 1.
    """
    per_play = (policy_probs * q_all).sum(axis=1)
    return OPEResult(
        value=float(per_play.mean()),
        se=float(per_play.std(ddof=1) / np.sqrt(len(per_play))),
        n=len(per_play),
    )


def behavior_recovery_check(
    ds: Dataset, q: QModel, beh: BehaviorModel
) -> dict:
    """Smoke test: does OPE recover the behavior policy's mean reward?

    Three numbers that should all agree:
      * empirical mean reward (ground truth on this data),
      * DM on the taken action  E[Q(s, a_taken)]  (Q's in-sample fit),
      * DM under pi_b           E_s sum_a pi_b(a|s) Q(s,a).
    """
    df = ds.df
    rewards = df[ds.reward_col].to_numpy()
    actions = df[ds.action_col].to_numpy()

    empirical = float(rewards.mean())
    empirical_se = float(rewards.std(ddof=1) / np.sqrt(len(rewards)))

    dm_taken = float(q.predict_taken(df, actions).mean())

    q_all = q.predict_all_actions(df)
    pi_b = beh.propensity(df)
    dm_pi_b = dm_policy_value(q_all, pi_b)

    return {
        "empirical_reward": empirical,
        "empirical_se": empirical_se,
        "dm_taken_action": dm_taken,
        "dm_under_pi_b": dm_pi_b.value,
        "recovery_gap": abs(dm_pi_b.value - empirical),
        # Pass if DM-under-pi_b is within ~2 empirical SEs of the truth.
        "passes": abs(dm_pi_b.value - empirical) <= max(2 * empirical_se, 0.02),
    }
