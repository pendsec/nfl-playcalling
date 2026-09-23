"""
Off-Policy Evaluation — inverse propensity weighting, the middle rung.

    V_IPW(pi)   = (1/n) sum_i  w_i r_i
    V_SNIPW(pi) = sum_i w_i r_i / sum_i w_i          (self-normalized)
      with  w_i = pi(a_i|s_i) / pi_b(a_i|s_i), clipped.

IPW uses no outcome model at all: it reweights the logged rewards by how much
more (or less) often the candidate policy would have made the call the DC
actually made. That makes it the cleanest check on the propensity model — and
the noisiest estimator of the three, because for a deterministic candidate every
play where the DC called something else contributes exactly zero, and the plays
that remain are weighted by one over a propensity that can be small.

Its place in the ladder is diagnostic. DM trusts the outcome model completely,
IPW trusts the propensities completely, DR needs only one of the two to be right.
When all three agree, neither nuisance model is doing anything strange; when IPW
and DM disagree, DR's correction term is the thing to read.
"""

from __future__ import annotations

import numpy as np

from ..schemas.ope import OPEResult
from ..scm.graph import version_tag


def _weights(pi_b: np.ndarray, policy_probs: np.ndarray, actions: np.ndarray,
             weight_clip: float) -> np.ndarray:
    idx = np.arange(len(actions))
    actions = np.asarray(actions)
    w = policy_probs[idx, actions] / pi_b[idx, actions]
    return np.clip(w, 0.0, weight_clip)


def ipw_per_play(
    pi_b: np.ndarray,
    policy_probs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    weight_clip: float = 20.0,
    self_normalize: bool = False,
) -> np.ndarray:
    """Per-play IPW scores, whose mean is the IPW (or SNIPW) value."""
    w = _weights(pi_b, policy_probs, actions, weight_clip)
    if self_normalize:
        total = w.sum()
        if total <= 0:
            # The candidate never agrees with any logged call: there is nothing
            # to reweight, and the honest answer is "undefined", not zero.
            return np.full(len(rewards), np.nan)
        w = w * (len(rewards) / total)
    return w * np.asarray(rewards)


def ipw_policy_value(
    pi_b: np.ndarray,
    policy_probs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    weight_clip: float = 20.0,
    self_normalize: bool = False,
    policy_name: str = "unspecified",
    graph_version: str | None = None,
) -> OPEResult:
    """IPW value of a (possibly stochastic) policy — no outcome model involved."""
    per_play = ipw_per_play(pi_b, policy_probs, actions, rewards,
                            weight_clip=weight_clip, self_normalize=self_normalize)
    n = len(per_play)
    return OPEResult(
        value=float(np.mean(per_play)),
        se=float(np.std(per_play, ddof=1) / np.sqrt(n)),
        n=n,
        estimator="SNIPW" if self_normalize else "IPW",
        policy=policy_name,
        graph_version=graph_version or version_tag(),
    )


def weight_diagnostics(pi_b: np.ndarray, policy_probs: np.ndarray,
                       actions: np.ndarray, weight_clip: float = 20.0) -> dict:
    """How much of the logged data an importance-weighted estimate rests on.

    `support_share` is the fraction of plays with a nonzero weight — for a
    deterministic policy, the plays where the DC made the call the policy would
    have made. `ess_ratio` is the effective sample size of those weights as a
    share of n. Read them before reading an IPW number: an estimate resting on
    5% of plays with an ESS of 2% is a statement about a handful of snaps.
    """
    idx = np.arange(len(actions))
    raw = policy_probs[idx, np.asarray(actions)] / pi_b[idx, np.asarray(actions)]
    w = np.clip(raw, 0.0, weight_clip)
    n = len(w)
    ess = (w.sum() ** 2) / (w ** 2).sum() if (w ** 2).sum() > 0 else 0.0
    return {
        "support_share": float((w > 0).mean()),
        "ess_ratio": float(ess / n),
        "max_weight": float(raw.max()) if n else 0.0,
        "share_clipped": float((raw > weight_clip).mean()),
    }
