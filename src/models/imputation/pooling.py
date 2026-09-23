"""Rubin's rules — combining estimates across multiple imputations.

Multiple imputation is only honest if the final interval reflects the label
uncertainty. Averaging m point estimates and quoting the average's standard
error would report a *narrower* interval the more draws you take, which is
backwards: extra draws reduce Monte-Carlo noise, they do not tell you what the
missing labels were.

Rubin's total variance splits the difference properly:

    T = Ubar + (1 + 1/m) * B

`Ubar` is the average within-draw variance (what you would have known with the
labels in hand) and `B` is the between-draw variance (what the imputation does
not know). The (1 + 1/m) factor corrects for having finitely many draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RubinResult:
    """A pooled estimate across m imputations, with its provenance."""
    estimate: float
    se: float
    within_var: float       # Ubar — variance you would have had with labels
    between_var: float      # B    — variance from not knowing the labels
    m: int
    df: float               # Barnard-Rubin style degrees of freedom

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.estimate - 1.96 * self.se, self.estimate + 1.96 * self.se)

    @property
    def fraction_missing_information(self) -> float:
        """Share of total variance contributed by not knowing the labels.

        The headline honesty metric. Near 0 means the imputation barely mattered;
        near 1 means the estimate is mostly a statement about the imputation
        model rather than about the data.
        """
        total = self.se ** 2
        if total <= 0:
            return 0.0
        return float(((1 + 1 / self.m) * self.between_var) / total)

    def to_record(self) -> dict:
        lo, hi = self.ci95
        return {
            "estimate": self.estimate, "se": self.se,
            "ci_lo": lo, "ci_hi": hi,
            "within_var": self.within_var, "between_var": self.between_var,
            "m_imputations": self.m,
            "fraction_missing_information": self.fraction_missing_information,
        }


def pool_rubin(estimates, variances) -> RubinResult:
    """Pool m imputation estimates and their variances into one result.

    `estimates[i]` and `variances[i]` come from running the SAME downstream
    estimator on the i-th completed dataset. Variances are squared standard
    errors, not standard errors.
    """
    q = np.asarray(estimates, dtype=float)
    u = np.asarray(variances, dtype=float)
    if q.shape != u.shape or q.ndim != 1 or len(q) < 2:
        raise ValueError(
            f"need matching 1-D estimates/variances with m >= 2, "
            f"got {q.shape} and {u.shape}"
        )
    m = len(q)
    qbar = float(q.mean())
    ubar = float(u.mean())
    b = float(q.var(ddof=1))
    total = ubar + (1 + 1 / m) * b

    # Barnard-Rubin degrees of freedom; infinite when the draws agree exactly.
    if b > 0:
        df = (m - 1) * (1 + ubar / ((1 + 1 / m) * b)) ** 2
    else:
        df = float("inf")

    return RubinResult(estimate=qbar, se=float(np.sqrt(total)),
                       within_var=ubar, between_var=b, m=m, df=df)
