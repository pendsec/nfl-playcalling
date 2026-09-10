"""`OPEResult` — one off-policy-evaluation estimate, with its provenance.

Pure data container returned by the estimators in `src.ope`. The `value` is the
estimated mean reward per play under the evaluated policy; `ci95` derives the
normal-approximation confidence interval from `se`.

An OPE number is meaningless without the assumptions behind it, so this carries
the full record CLAUDE.md's causal discipline requires:

    point estimate + confidence interval + sensitivity bound + graph version

`estimator` and `policy` say *what was computed*, `graph_version` says *under
which DAG* (see `scm.graph.version_tag`), and `sensitivity` — attached after the
fact by `ope.sensitivity.attach` — says *how much unobserved confounding the
conclusion survives*. `to_record()` flattens all of it into one row, so a saved
estimate can never drift apart from the assumptions that produced it.

This module is deliberately dependency-free (no scm import) to keep `schemas` a
leaf layer; the estimators stamp the graph version in as they build the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OPEResult:
    value: float          # estimated mean reward per play under the policy
    se: float             # standard error across plays
    n: int

    # ── Provenance (CLAUDE.md: every OPE estimate declares its assumptions) ──
    estimator: str = "unspecified"       # "DM" | "DR" | ...
    policy: str = "unspecified"          # which policy was valued
    graph_version: str = "unversioned"   # scm.graph.version_tag() at estimation

    # Sensitivity bound for unobserved confounding. Populated by
    # `ope.sensitivity.attach`; None means the bound has not been computed yet,
    # which `to_record()` reports honestly rather than silently omitting.
    sensitivity: Any | None = field(default=None, repr=False)

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.value - 1.96 * self.se, self.value + 1.96 * self.se)

    @property
    def robust_to_gamma(self) -> float | None:
        """Largest Gamma at which this policy still beats its comparison.

        Reads the attached sensitivity table. Returns None when no sensitivity
        analysis was attached, or when the table has no comparison column (a
        bound was computed but nothing was being beaten). A value of 1.0 means
        the conclusion does not survive even mild unobserved confounding.
        """
        sens = self.sensitivity
        if sens is None or "beats_comparison" not in getattr(sens, "columns", []):
            return None
        robust = sens.loc[sens["beats_comparison"], "gamma"]
        return float(robust.max()) if len(robust) else None

    def to_record(self) -> dict:
        """Flatten estimate + assumptions into one row for logging / CSV.

        Deliberately includes the provenance fields even when unset, so a record
        missing its graph version is visible in the output rather than absent.
        """
        lo, hi = self.ci95
        return {
            "policy": self.policy,
            "estimator": self.estimator,
            "value": self.value,
            "se": self.se,
            "ci_lo": lo,
            "ci_hi": hi,
            "n": self.n,
            "graph_version": self.graph_version,
            "sensitivity_computed": self.sensitivity is not None,
            "robust_to_gamma": self.robust_to_gamma,
        }
