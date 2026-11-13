"""
`CausalGraph` — one hand-built DAG, with everything that makes it a causal claim.

The project carries more than one graph because the treatment differs between
pipelines: the dropback pipeline's treatment is coverage shell x pressure, the
league-wide pipeline's is personnel package x front. Each graph declares what is
adjusted for, what is a mediator, what the sample is selected on, and which
columns ARE the treatment — and each stamps its own version and fingerprint onto
the estimates computed under it.

Containers are held BY REFERENCE, never copied. `graph.py` builds its graphs
from module-level lists and sets, and an edit to one of those (including a
test's deliberate edit) has to move the fingerprint of the graph built from it.
A copy would silently freeze the fingerprint at import time.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(eq=False)
class CausalGraph:
    name: str                         # human label: what the treatment is, on what sample
    version: str                      # hand-maintained; bump on any structural change
    treatment: str
    outcome: str
    edges: list[tuple[str, str]]
    observed_confounders: list[str]   # abstract nodes forming the backdoor set
    confounder_columns: dict[str, list[str]]
    observed: set[str]
    unobserved: set[str]
    mediators: set[str]
    selection: set[str] = field(default_factory=set)
    defensive_choice: set[str] = field(default_factory=set)
    def_choice_columns: dict[str, list[str]] = field(default_factory=dict)
    # Concrete columns that ARE the treatment. Never a state feature: a model
    # that conditions on the action it is estimating the effect of learns
    # nothing about that effect.
    treatment_columns: list[str] = field(default_factory=list)
    sample_note: str = ""             # one line on what the sample is conditioned on

    def adjustment_set(self) -> set[str]:
        """Abstract confounder nodes to condition on (backdoor set given observed)."""
        return set(self.observed_confounders)

    def adjustment_columns(self) -> list[str]:
        """Concrete feature columns forming the adjustment set (mediators excluded)."""
        cols: list[str] = []
        for node in self.observed_confounders:
            cols.extend(self.confounder_columns[node])
        return cols

    def fingerprint(self) -> str:
        """Short content hash of the DAG's structural claims.

        Two estimates with the same version label but different fingerprints
        were computed under different graphs — the guard against an edit
        someone forgot to version-bump.
        """
        payload = "|".join([
            ";".join(f"{s}->{t}" for s, t in sorted(self.edges)),
            ";".join(sorted(self.observed)),
            ";".join(sorted(self.unobserved)),
            ";".join(sorted(self.mediators)),
            ";".join(sorted(self.selection)),
            ";".join(sorted(self.defensive_choice)),
            f"{self.treatment}->{self.outcome}",
            ";".join(self.adjustment_columns()),
        ])
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def version_tag(self) -> str:
        """`version+fingerprint` — the string stamped onto every OPE record."""
        return f"{self.version}+{self.fingerprint()}"

    def describe(self) -> str:
        """Human-readable summary for the run log."""
        lines = [
            f"SCM — {self.name} [{self.version_tag()}]",
            f"  treatment={self.treatment}, outcome={self.outcome}",
            f"  adjustment set (condition on): {sorted(self.adjustment_set())}",
            f"    -> {len(self.adjustment_columns())} feature columns",
            f"  mediators (do NOT condition on): {sorted(self.mediators)}",
            f"  unobserved confounders: {sorted(self.unobserved)} "
            f"(bounded by the sensitivity analysis)",
        ]
        if self.selection:
            lines.append(f"  SELECTED ON: {sorted(self.selection)} = 1")
        if self.sample_note:
            lines.append(f"    {self.sample_note}")
        if self.defensive_choice:
            lines.append(f"  defensive choices carried but NOT adjusted for: "
                         f"{sorted(self.defensive_choice)}")
        return "\n".join(lines)
