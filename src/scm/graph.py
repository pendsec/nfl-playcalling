"""
Causal Graph (SCM Skeleton) — V1.

A hand-built DAG over the V1 variables. Its job is to discipline what every
downstream model conditions on. The key V1 rulings:

  * adjustment set = pre-snap state S (the backdoor set for A -> R here).
  * off_playcall is a MEDIATOR of A -> R and must NOT be conditioned on when
    estimating the call's effect (the Q-model obeys this).
  * coach_read is the unobserved confounder (S, U) -> A and (S, U) -> R. In V1
    it is unmeasured; the sensitivity analysis that bounds its impact arrives
    in V2. We record it here so the assumption is explicit, not hidden.

No networkx dependency — the graph is a plain edge list plus helpers.
"""

from __future__ import annotations

# Nodes
STATE = "game_state"          # pre-snap S (down, distance, field, score, time, personnel)
COACH_READ = "coach_read"     # unobserved confounder U (matchup intuition / film)
OFF_PLAYCALL = "off_playcall"  # offense's call — post-snap MEDIATOR
DEF_PLAYCALL = "def_playcall"  # treatment A
EPA = "epa"                   # outcome R (reward = -EPA)

OBSERVED = {STATE, OFF_PLAYCALL, DEF_PLAYCALL, EPA}
UNOBSERVED = {COACH_READ}

# Directed edges (parent -> child).
EDGES = [
    (STATE, DEF_PLAYCALL),
    (STATE, OFF_PLAYCALL),
    (STATE, EPA),
    (COACH_READ, DEF_PLAYCALL),   # confounding into the call
    (COACH_READ, EPA),            # confounding into the outcome
    (DEF_PLAYCALL, EPA),          # the causal effect we want
    (OFF_PLAYCALL, EPA),          # mediator -> outcome
]

TREATMENT = DEF_PLAYCALL
OUTCOME = EPA
MEDIATORS = {OFF_PLAYCALL}


def adjustment_set() -> set[str]:
    """Variables to condition on to identify A -> R via the backdoor criterion.

    With coach_read unobserved we condition on the observed state S. This leaves
    residual confounding through coach_read — the explicit V1 caveat that V2's
    sensitivity analysis quantifies. Mediators are excluded by construction.
    """
    return {STATE}


def to_gml() -> str:
    """GML serialization (e.g. for DoWhy / inspection in later versions)."""
    nodes = sorted(OBSERVED | UNOBSERVED)
    lines = ["graph [", "  directed 1"]
    for n in nodes:
        lines.append(f'  node [ id "{n}" label "{n}" ]')
    for s, t in EDGES:
        lines.append(f'  edge [ source "{s}" target "{t}" ]')
    lines.append("]")
    return "\n".join(lines)


def describe() -> str:
    """Human-readable summary for the run log."""
    return (
        f"V1 SCM: treatment={TREATMENT}, outcome={OUTCOME}\n"
        f"  adjustment set (condition on): {sorted(adjustment_set())}\n"
        f"  mediators (do NOT condition on): {sorted(MEDIATORS)}\n"
        f"  unobserved confounders: {sorted(UNOBSERVED)} "
        f"(handled by V2 sensitivity analysis)"
    )
