"""
The DM -> IPW -> DR ladder: one policy, four estimators, side by side.

Each rung trusts a different thing. DM trusts the outcome model; IPW and SNIPW
trust the propensities; DR trusts either. Reading them together is the point —
agreement says neither nuisance model is misbehaving, and a gap says which one
to look at. Every row carries its own provenance, like any other OPE estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .direct_method import dm_policy_value
from .doubly_robust import dr_policy_value
from .ipw import ipw_policy_value, weight_diagnostics


def ope_ladder(
    q_all: np.ndarray,
    pi_b: np.ndarray,
    policy_probs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    weight_clip: float = 20.0,
    policy_name: str = "unspecified",
    graph_version: str | None = None,
) -> pd.DataFrame:
    """DM, IPW, SNIPW and DR estimates of one policy, one row each."""
    kw = dict(policy_name=policy_name, graph_version=graph_version)
    results = [
        dm_policy_value(q_all, policy_probs, **kw),
        ipw_policy_value(pi_b, policy_probs, actions, rewards,
                         weight_clip=weight_clip, self_normalize=False, **kw),
        ipw_policy_value(pi_b, policy_probs, actions, rewards,
                         weight_clip=weight_clip, self_normalize=True, **kw),
        dr_policy_value(q_all, pi_b, policy_probs, actions, rewards,
                        weight_clip=weight_clip, **kw),
    ]
    frame = pd.DataFrame([r.to_record() for r in results])
    diag = weight_diagnostics(pi_b, policy_probs, actions, weight_clip)
    for key, value in diag.items():
        frame[key] = value
    return frame
