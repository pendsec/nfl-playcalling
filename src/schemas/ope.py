"""`OPEResult` — one off-policy-evaluation estimate: point value, SE, and 95% CI.

Pure data container returned by the estimators in `src.ope` (the Direct Method
in V1). The `value` is the estimated mean reward per play under the evaluated
policy; `ci95` derives the normal-approximation confidence interval from `se`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OPEResult:
    value: float          # estimated mean reward per play under the policy
    se: float             # standard error across plays
    n: int

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.value - 1.96 * self.se, self.value + 1.96 * self.se)
