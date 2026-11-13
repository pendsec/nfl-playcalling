"""Small statistics shared across the pipeline: means with standard errors,
heterogeneity across groups, and the non-inferiority rule the gates use.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_se(x: np.ndarray, clusters: np.ndarray | None = None) -> tuple[float, float]:
    """Mean of `x` and its standard error.

    With `clusters`, observations in the same cluster (a game) are not treated
    as independent: the SE is the cluster-robust one. Without, it is the iid
    `sd / sqrt(n)`, infinite when there is a single observation.
    """
    x = np.asarray(x, float)
    mean = float(x.mean())
    if clusters is None:
        n = len(x)
        return mean, float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    _, inv = np.unique(clusters, return_inverse=True)
    g = np.bincount(inv, weights=x - mean)
    n_g = len(g)
    if n_g < 2:
        return mean, float("nan")
    se = np.sqrt((g ** 2).sum() * n_g / (n_g - 1)) / len(x)
    return mean, float(se)


def regresses(diff: float, se: float, margin: float) -> bool:
    """Non-inferiority: the whole 95% interval sits below `-margin`."""
    return diff + 1.96 * se < -margin


def heterogeneity(est: np.ndarray, se: np.ndarray) -> dict:
    """Cochran's Q, I^2 and the DerSimonian-Laird tau across groups."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[ok], se[ok]
    k = len(est)
    if k < 2:
        return {"k": k, "pooled": float("nan"), "pooled_se": float("nan"), "q": float("nan"),
                "p": float("nan"), "i2": float("nan"), "tau": float("nan")}
    w = 1.0 / se ** 2
    pooled = float((w * est).sum() / w.sum())
    q = float((w * (est - pooled) ** 2).sum())
    df = k - 1
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - df) / c)
    return {"k": k, "pooled": pooled, "pooled_se": float(np.sqrt(1.0 / w.sum())),
            "q": q, "p": float(stats.chi2.sf(q, df)),
            "i2": max(0.0, (q - df) / q) if q > 0 else 0.0, "tau": float(np.sqrt(tau2))}
