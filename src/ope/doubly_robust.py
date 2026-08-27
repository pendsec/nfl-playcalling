"""
Off-Policy Evaluation — Doubly-Robust policy value (V2).

The DR estimator combines the Direct Method (V1's plug-in Q) with an
importance-weighted correction on the taken action:

    V_DR(pi) = (1/n) Σ_i Σ_a pi(a|s_i) q(s_i,a)          <- direct method term
             +  weighted-mean_i [ w_i (r_i - q(s_i,a_i)) ] <- IPW correction
      with   w_i = pi(a_i|s_i) / pi_b(a_i|s_i)  (clipped).

Doubly robust: consistent if EITHER q OR pi_b is correct. The correction is what
fixes V1's Direct-Method pathology — when q extrapolates badly off-support, the
residual (r - q) on logged plays pulls the estimate back toward reality instead
of trusting a Q-value no data supports.

Two variance controls:
  * weight clipping (switch-DR flavor): cap w_i so a single tiny propensity can't
    dominate — essential here because positivity is weak on rare calls.
  * self-normalization (SNDR): divide the correction by the mean weight, trading
    a little bias for much lower variance under poor overlap.
"""

from __future__ import annotations

import numpy as np

from ..schemas.ope import OPEResult


def dr_policy_value(
    q_all: np.ndarray,
    pi_b: np.ndarray,
    policy_probs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    weight_clip: float = 20.0,
    self_normalize: bool = True,
) -> OPEResult:
    """Doubly-robust value of a (possibly stochastic) policy.

    All array args are aligned by play. q_all / pi_b / policy_probs are
    (n, n_actions); actions and rewards are length n.
    """
    n = len(rewards)
    idx = np.arange(n)
    actions = np.asarray(actions)

    # Direct-method term: expected Q under the target policy, per play.
    dm = (policy_probs * q_all).sum(axis=1)

    # Importance weights on the taken action, clipped for stability.
    w = policy_probs[idx, actions] / pi_b[idx, actions]
    w = np.clip(w, 0.0, weight_clip)
    residual = rewards - q_all[idx, actions]

    if self_normalize and w.sum() > 0:
        # SNDR: per-play score with weights normalized to mean 1.
        w_norm = w * (n / w.sum())
        per_play = dm + w_norm * residual
    else:
        per_play = dm + w * residual

    return OPEResult(
        value=float(per_play.mean()),
        se=float(per_play.std(ddof=1) / np.sqrt(n)),
        n=n,
    )
