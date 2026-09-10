"""
Causal Graph (SCM Skeleton) — V2.

A hand-built DAG over the project variables. Its job is to discipline what every
downstream model conditions on. The key rulings:

  * adjustment set = observed pre-snap confounders (game state, offensive
    personnel/formation, offensive tendencies, player/unit proxies). This is the
    backdoor set for def_playcall -> epa given the observed variables.
  * off_playcall is a post-snap MEDIATOR/competing-cause and is NEVER conditioned
    on when estimating the call's total effect (the Q-model obeys this).
  * coach_read is the UNOBSERVED confounder U: (S,U) -> def_playcall and
    (S,U) -> epa. Conditioning on the observed set leaves residual confounding
    through coach_read — the explicit caveat the V2 sensitivity analysis bounds.

The abstract structural nodes below document the mechanism; the concrete
`CONFOUNDER_COLUMNS` map ties each node to the engineered feature columns so the
adjustment set is a single source of truth shared by the Q-model, the DoWhy
identification, and the leak/consistency tests.

No networkx dependency here — the graph is a plain edge list plus helpers.
"""

from __future__ import annotations

# ── Abstract nodes ───────────────────────────────────────────────────────────
GAME_STATE = "game_state"        # down, distance, field, score, time, timeouts
OFF_PERSONNEL = "off_personnel"  # RB/TE/WR counts
OFF_FORMATION = "off_formation"  # formation, shotgun, no-huddle
OFF_TENDENCIES = "off_tendencies"  # rolling offensive pass tendency
PLAYER_PROXIES = "player_proxies"  # QB / defensive-unit quality (historical EPA)
COACH_READ = "coach_read"        # UNOBSERVED confounder U (matchup intuition / film)
OFF_PLAYCALL = "off_playcall"    # offense's call — post-snap MEDIATOR (excluded)
DEF_PLAYCALL = "def_playcall"    # treatment A
EPA = "epa"                      # outcome R (reward = -EPA)

OBSERVED_CONFOUNDERS = [
    GAME_STATE, OFF_PERSONNEL, OFF_FORMATION, OFF_TENDENCIES, PLAYER_PROXIES,
]
OBSERVED = set(OBSERVED_CONFOUNDERS) | {OFF_PLAYCALL, DEF_PLAYCALL, EPA}
UNOBSERVED = {COACH_READ}

TREATMENT = DEF_PLAYCALL
OUTCOME = EPA
MEDIATORS = {OFF_PLAYCALL}

# Directed edges (parent -> child).
EDGES = [
    (GAME_STATE, OFF_PERSONNEL), (GAME_STATE, OFF_FORMATION),
    (GAME_STATE, OFF_TENDENCIES), (GAME_STATE, DEF_PLAYCALL), (GAME_STATE, EPA),
    (OFF_PERSONNEL, OFF_FORMATION), (OFF_PERSONNEL, OFF_PLAYCALL),
    (OFF_PERSONNEL, DEF_PLAYCALL), (OFF_PERSONNEL, EPA),
    (OFF_FORMATION, OFF_PLAYCALL), (OFF_FORMATION, DEF_PLAYCALL), (OFF_FORMATION, EPA),
    (OFF_TENDENCIES, DEF_PLAYCALL), (OFF_TENDENCIES, OFF_PLAYCALL),
    (PLAYER_PROXIES, DEF_PLAYCALL), (PLAYER_PROXIES, EPA),
    (OFF_PLAYCALL, EPA),
    (DEF_PLAYCALL, EPA),                 # the causal effect we want
    (COACH_READ, DEF_PLAYCALL),          # unobserved confounding into the call
    (COACH_READ, EPA),                   # unobserved confounding into the outcome
]

# ── Abstract node -> concrete feature columns ────────────────────────────────
# The single source of truth for the adjustment set. Must stay in sync with the
# data layer's state features (asserted by tests).
CONFOUNDER_COLUMNS = {
    GAME_STATE: [
        "down", "ydstogo", "yardline_100", "score_diff",
        "game_seconds_remaining", "half_seconds_remaining", "qtr",
        "posteam_timeouts_remaining", "defteam_timeouts_remaining",
        "is_two_minute_drill", "is_red_zone", "is_third_or_fourth", "scoring_opp",
    ],
    OFF_PERSONNEL: ["num_rb", "num_te", "num_wr"],
    OFF_FORMATION: ["formation", "shotgun", "no_huddle"],
    OFF_TENDENCIES: ["off_pass_tendency"],
    PLAYER_PROXIES: ["def_team_avg_epa", "qb_avg_epa"],
}


# ── Graph version ────────────────────────────────────────────────────────────
# Every causal estimate this repo produces is only valid *relative to this DAG*,
# so each OPE record carries the version below (CLAUDE.md causal discipline:
# "Every OPE estimate ships with ... the assumed causal graph version").
#
# GRAPH_VERSION is the human-facing label — bump it whenever the structural
# claims change (an edge added/removed, a variable moved between observed,
# mediator, and unobserved). `fingerprint()` is the machine-facing guard: it
# hashes the actual structure, so an edit that someone forgets to version-bump
# still shows up as a different fingerprint on the stored estimate.
GRAPH_VERSION = "v2.0"


def fingerprint() -> str:
    """Short content hash of the DAG's structural claims.

    Covers the edge list, the observed/unobserved split, the mediator set, and
    the concrete adjustment columns — i.e. everything that changes what the
    backdoor adjustment means. Two estimates with the same GRAPH_VERSION but
    different fingerprints were computed under different graphs.
    """
    import hashlib

    payload = "|".join([
        ";".join(f"{s}->{t}" for s, t in sorted(EDGES)),
        ";".join(sorted(OBSERVED)),
        ";".join(sorted(UNOBSERVED)),
        ";".join(sorted(MEDIATORS)),
        f"{TREATMENT}->{OUTCOME}",
        ";".join(adjustment_columns()),
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def version_tag() -> str:
    """`GRAPH_VERSION+fingerprint` — the string stamped onto every OPE record."""
    return f"{GRAPH_VERSION}+{fingerprint()}"


def adjustment_set() -> set[str]:
    """Abstract confounder nodes to condition on (backdoor set given observed)."""
    return set(OBSERVED_CONFOUNDERS)


def adjustment_columns() -> list[str]:
    """Concrete feature columns forming the adjustment set (mediators excluded).

    This is what the Q-model conditions on and what DoWhy receives as common
    causes. Deliberately excludes off_playcall and anything result-derived.
    """
    cols: list[str] = []
    for node in OBSERVED_CONFOUNDERS:
        cols.extend(CONFOUNDER_COLUMNS[node])
    return cols


def to_gml() -> str:
    """GML serialization of the abstract DAG (for inspection / DoWhy)."""
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
        f"V2 SCM [{version_tag()}]: treatment={TREATMENT}, outcome={OUTCOME}\n"
        f"  adjustment set (condition on): {sorted(adjustment_set())}\n"
        f"    -> {len(adjustment_columns())} feature columns\n"
        f"  mediators (do NOT condition on): {sorted(MEDIATORS)}\n"
        f"  unobserved confounders: {sorted(UNOBSERVED)} "
        f"(bounded by the sensitivity analysis)"
    )
