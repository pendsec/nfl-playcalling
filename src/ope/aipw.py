"""
AIPW treatment-effect contrasts (interpretability, NOT the policy judge).

This estimates the Average Treatment Effect of each defensive call relative to a
baseline call — "how much does cover-2 blitz change EPA vs base, averaged over
plays." It shares the DR nuisances (q, pi_b) but answers a DIFFERENT question
than OPE: it contrasts two *fixed actions*, whereas `ope.doubly_robust` values a
whole *policy* pi(A|S). Use it for the causal story surfaced to coaches, never to
rank candidate policies.

    psi_d(i) = q(s_i, d) + 1{a_i == d}/pi_b(d|s_i) * (r_i - q(s_i, d))
    ATE(d vs base) = mean_i [ psi_d(i) - psi_base(i) ]

Doubly robust in the same sense as the policy-value estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aipw_contrasts(
    q_all: np.ndarray,
    pi_b: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    action_labels: dict[int, str],
    baseline: int = 0,
) -> pd.DataFrame:
    """AIPW ATE of every action vs `baseline` (reward units; +ve = better)."""
    n, k = q_all.shape
    idx = np.arange(n)
    actions = np.asarray(actions)

    psi = np.empty((n, k))
    for d in range(k):
        indicator = (actions == d).astype(float)
        psi[:, d] = q_all[:, d] + indicator / pi_b[:, d] * (rewards - q_all[:, d])

    psi_base = psi[:, baseline]
    rows = []
    for d in range(k):
        if d == baseline:
            continue
        diff = psi[:, d] - psi_base
        ate = float(diff.mean())
        se = float(diff.std(ddof=1) / np.sqrt(n))
        rows.append({
            "action": action_labels.get(d, d),
            "baseline": action_labels.get(baseline, baseline),
            "ate_vs_baseline": ate,
            "se": se,
            "ci_lo": ate - 1.96 * se,
            "ci_hi": ate + 1.96 * se,
            "significant": not (ate - 1.96 * se < 0 < ate + 1.96 * se),
        })
    return pd.DataFrame(rows).sort_values("ate_vs_baseline", ascending=False)
