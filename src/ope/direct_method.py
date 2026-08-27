"""
Off-Policy Evaluation — Direct Method + the mandatory recovery smoke test.

The Direct Method plugs the fitted Q-model into a candidate policy:

    V_DM(pi) = E_s [ sum_a  pi(a|s) * Q(s, a) ]

It is biased if Q is misspecified — which is exactly V1's pathology — so V2 uses
it only as the regression term inside the doubly-robust estimator
(`ope.doubly_robust.dr_policy_value`) and as one leg of the smoke test.

`behavior_recovery_check` is the gate before trusting ANY candidate-policy
estimate: DM under pi_b and, crucially, **DR under pi_b** must both land near the
empirical mean reward. DR is the one that must pass — if it doesn't, the
propensities or the OPE wiring are broken.
"""

from __future__ import annotations

import numpy as np

from ..schemas.dataset import Dataset
from ..schemas.behavior import BehaviorModel
from ..schemas.outcome import QModel
from ..schemas.ope import OPEResult
from .doubly_robust import dr_policy_value


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
    ds: Dataset, q: QModel, beh: BehaviorModel,
    q_all: np.ndarray | None = None, weight_clip: float = 20.0,
) -> dict:
    """Smoke test: does OPE recover the behavior policy's mean reward?

    Numbers that should agree with the empirical mean reward:
      * DM under pi_b   E_s sum_a pi_b(a|s) Q(s,a)   (biased if Q is off),
      * DR under pi_b   the doubly-robust value      (must pass — the gate).

    Pass `q_all` (e.g. cross-fit Q on the training rows) to keep the check honest
    in-sample; otherwise the fitted Q is scored on its own training data.
    """
    df = ds.df
    rewards = df[ds.reward_col].to_numpy()
    actions = df[ds.action_col].to_numpy()

    empirical = float(rewards.mean())
    empirical_se = float(rewards.std(ddof=1) / np.sqrt(len(rewards)))

    if q_all is None:
        q_all = q.predict_all_actions(df)
    pi_b = beh.propensity(df)

    dm_pi_b = dm_policy_value(q_all, pi_b)
    dr_pi_b = dr_policy_value(q_all, pi_b, pi_b, actions, rewards,
                             weight_clip=weight_clip)

    tol = max(2 * empirical_se, 0.02)
    return {
        "empirical_reward": empirical,
        "empirical_se": empirical_se,
        "dm_taken_action": float(q_all[np.arange(len(df)), actions].mean()),
        "dm_under_pi_b": dm_pi_b.value,
        "dr_under_pi_b": dr_pi_b.value,
        "recovery_gap": abs(dr_pi_b.value - empirical),
        # The DR leg is the gate: it must recover the empirical mean.
        "passes": abs(dr_pi_b.value - empirical) <= tol,
    }
