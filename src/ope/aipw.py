"""
AIPW treatment-effect contrasts (interpretability, NOT the policy judge).

This estimates the Average Treatment Effect of each defensive call relative to a
baseline call — "how much does cover-2 blitz change EPA vs base, averaged over
plays." It shares the DR nuisances (q, pi_b) but answers a DIFFERENT question
than OPE: it contrasts two *fixed actions*, whereas `ope.doubly_robust` values a
whole *policy* pi(A|S). Use it for the causal story surfaced to coaches, never to
rank candidate policies.

    psi_d(i) = q(s_i, d) + w_d(i) * (r_i - q(s_i, d)),
        w_d(i) = 1{a_i == d} / pi_b(d | s_i)   (clipped, then self-normalized)
    ATE(d vs base) = mean_i [ psi_d(i) - psi_base(i) ]

Doubly robust in the same sense as the policy-value estimator.

Variance controls (identical to `ope.doubly_robust`, and mandatory here for the
same reason): contrasting a *fixed* action means every play that did not take
that action contributes zero weight, so the whole correction rests on the plays
that did. On a single-team 12-action slice that can be a handful of plays with
propensities sitting on the clip floor, where a raw 1/pi_b reaches ~100 and the
weights sum to many times n. Clipping caps any single play's leverage;
self-normalization (Hajek) forces the correction to be an average rather than a
sum inflated by however far the weights drift from mean 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aipw_scores(
    q_d: np.ndarray,
    pi_b_d: np.ndarray,
    took_d: np.ndarray,
    rewards: np.ndarray,
    weight_clip: float,
    self_normalize: bool,
) -> np.ndarray:
    """Per-play AIPW pseudo-outcomes for one fixed action d.

    `took_d` is the 0/1 indicator that play i actually took action d. Plays that
    took some other call contribute only the direct-method term q(s_i, d).
    """
    w = np.where(took_d, 1.0 / pi_b_d, 0.0)
    w = np.clip(w, 0.0, weight_clip)

    total = w.sum()
    if total <= 0:
        # Action never taken on these rows: no residual correction is possible,
        # so the contrast falls back to the pure direct method for this action.
        return q_d.copy()
    if self_normalize:
        w = w * (len(rewards) / total)
    return q_d + w * (rewards - q_d)


def aipw_contrasts(
    q_all: np.ndarray,
    pi_b: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    action_labels: dict[int, str],
    baseline: int = 0,
    weight_clip: float = 20.0,
    self_normalize: bool = True,
) -> pd.DataFrame:
    """AIPW ATE of every action vs `baseline` (reward units; +ve = better).

    `n_taken` / `n_taken_baseline` are reported alongside every contrast: an
    effect estimated off a handful of logged plays is not made trustworthy by a
    narrow-looking CI, and the count is what tells the reader which it is.
    """
    n, k = q_all.shape
    actions = np.asarray(actions)

    took = actions[:, None] == np.arange(k)[None, :]
    psi = np.column_stack([
        _aipw_scores(q_all[:, d], pi_b[:, d], took[:, d], rewards,
                     weight_clip, self_normalize)
        for d in range(k)
    ])

    psi_base = psi[:, baseline]
    n_base = int(took[:, baseline].sum())
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
            "n_taken": int(took[:, d].sum()),
            "n_taken_baseline": n_base,
        })
    return pd.DataFrame(rows).sort_values("ate_vs_baseline", ascending=False)
