"""
Causal Graph (SCM Skeleton).

A hand-built DAG over the project variables. Its job is to discipline what every
downstream model conditions on. The key rulings:

  * adjustment set = observed pre-snap confounders (game state, offensive
    personnel/formation, offensive tendencies, player/unit proxies). This is the
    backdoor set for def_playcall -> epa given the observed variables.
  * off_playcall is a MEDIATOR and is NEVER conditioned on when estimating the
    call's total effect (the Q-model obeys this). The defense calls its play
    pre-snap, and the look it shows drives the offense's audible / RPO read, so
    def_playcall -> off_playcall is a real edge: part of what a call does is
    change whether the offense runs at all.
  * coach_read is the UNOBSERVED confounder U: (S,U) -> def_playcall and
    (S,U) -> epa. Conditioning on the observed set leaves residual confounding
    through coach_read — the explicit caveat the sensitivity analysis bounds.
  * coverage_charted is a SELECTION node, and the one the dataset is built by
    conditioning on. A coverage shell only exists as a label where NGS charted
    one, so every modelled row has coverage_charted = 1. That is not innocuous:
    charting depends on off_playcall (runs are charted ~3% of the time vs ~94%
    of dropbacks) and on how the play resolved (a sack, scramble or throwaway
    leaves no shell to label), which makes it a collider on
    def_playcall -> off_playcall -> coverage_charted <- epa. Selecting on it
    therefore both BLOCKS the run/pass-deterrence portion of the total effect
    and OPENS a non-causal path. See `SELECTION_NOTES` for the measured size of
    each, and the README's limitations section for what it means for the
    estimand.

The abstract structural nodes below document the mechanism; the concrete
`CONFOUNDER_COLUMNS` map ties each node to the engineered feature columns so the
adjustment set is a single source of truth shared by the Q-model, the DoWhy
identification, and the leak/consistency tests.

No networkx dependency here — the graph is a plain edge list plus helpers.
"""

from __future__ import annotations

from .dag import CausalGraph

# ── Abstract nodes ───────────────────────────────────────────────────────────
GAME_STATE = "game_state"        # down, distance, field, score, time, timeouts
OFF_PERSONNEL = "off_personnel"  # RB/TE/WR counts
OFF_FORMATION = "off_formation"  # formation, shotgun, no-huddle
OFF_TENDENCIES = "off_tendencies"  # rolling offensive pass tendency
PLAYER_PROXIES = "player_proxies"  # QB / defensive-unit quality (historical EPA)
COACH_READ = "coach_read"        # UNOBSERVED confounder U (matchup intuition / film)
OFF_PLAYCALL = "off_playcall"    # offense's run/pass call — MEDIATOR (excluded)
DEF_PLAYCALL = "def_playcall"    # treatment A
EPA = "epa"                      # outcome R (reward = -EPA)
CHARTED = "coverage_charted"     # SELECTION: was a coverage shell labeled at all?
DEF_FRONT = "def_front"          # defensive box/front commitment — a SIBLING of
                                 # the treatment, not a cause of it

OBSERVED_CONFOUNDERS = [
    GAME_STATE, OFF_PERSONNEL, OFF_FORMATION, OFF_TENDENCIES, PLAYER_PROXIES,
]
OBSERVED = set(OBSERVED_CONFOUNDERS) | {OFF_PLAYCALL, DEF_PLAYCALL, EPA, CHARTED,
                                        DEF_FRONT}
UNOBSERVED = {COACH_READ}

TREATMENT = DEF_PLAYCALL
OUTCOME = EPA
MEDIATORS = {OFF_PLAYCALL}

# Variables the sample is selected ON, rather than adjusted for. Every modelled
# row has CHARTED = 1 by construction of the dataset, so this is an assumption
# the estimates carry whether or not it is written down — which is why it is
# written down, and hashed into the graph fingerprint.
SELECTION = {CHARTED}

# Other components of the defense's own decision. These are observed, and they
# are pre-snap, which makes them tempting to drop into the adjustment set — but
# they are SIBLINGS of the treatment, not causes of it. Conditioning on part of
# the defense's own call blocks a slice of the effect being estimated, so they
# are declared here and guarded out of the state by
# `scm.identify.assert_adjustment_consistency`.
#
# def_front is carried rather than modeled: it is the only defensive
# attribute charted on ~99% of ALL snaps (runs included, where coverage is
# charted on ~3%), which is why it becomes the primary axis of the factored
# action space rather than staying a spectator.
DEFENSIVE_CHOICE = {DEF_FRONT}
DEF_CHOICE_COLUMNS = {DEF_FRONT: ["n_defense_box"]}

# Measured on the SF 2021-2023 slice, so the strength of each selection
# path is a number rather than a worry. Kept next to the edges they qualify.
SELECTION_NOTES = {
    "charting_rate_pass": 0.936,   # dropbacks carry a shell label
    "charting_rate_run": 0.030,    # runs essentially never do -> runs are excluded
    "dropped_pass_plays": 242,     # charted-NaN dropbacks, dropped from the slice
    # The dropped dropbacks are the defense's BEST outcomes (sacks, scrambles,
    # throwaways resolve no shell), so selection shifts the reward level:
    "mean_reward_kept": -0.036,
    "mean_reward_dropped": +0.922,
    # ...but near-identically across the pressure axis (9.3% of blitzes dropped
    # vs 9.4% of non-blitzes), so it moves every policy's level together rather
    # than distorting the contrast OPE actually compares.
    "drop_rate_blitz": 0.093,
    "drop_rate_no_blitz": 0.094,
}

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
    (DEF_PLAYCALL, OFF_PLAYCALL),        # the look drives audibles / RPO reads,
                                         # so off_playcall is a genuine mediator
    (COACH_READ, DEF_PLAYCALL),          # unobserved confounding into the call
    (COACH_READ, EPA),                   # unobserved confounding into the outcome
    (GAME_STATE, DEF_FRONT), (OFF_PERSONNEL, DEF_FRONT),
    (OFF_FORMATION, DEF_FRONT),
    (COACH_READ, DEF_FRONT),             # the same hidden read drives the front
    (DEF_FRONT, EPA),                    # +0.068 reward/defender vs the run,
                                         # -0.020 vs the pass (2021-23, league)
    (DEF_FRONT, OFF_PLAYCALL),           # and it moves the offense's hand: run
                                         # rate runs 14% (<=5 box) to 65% (8+)
    (OFF_PLAYCALL, CHARTED),             # coverage is charted on dropbacks only
    (EPA, CHARTED),                      # shorthand for "the play broke down":
                                         # a sack/scramble is both a great
                                         # defensive outcome and unchartable, so
                                         # selection depends on the outcome
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
    OFF_FORMATION: ["formation", "shotgun", "no_huddle",
                    # FTN-charted pre-snap presentation (2022+).
                    "is_motion", "n_offense_backfield", "qb_location"],
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
# v2.1 — added def_playcall -> off_playcall (making the declared mediator an
# actual one) and the coverage_charted selection node. Both are structural
# claims, so every estimate stamped v2.0 was computed under a different graph.
# v2.2 — added the def_front node (FTN box count: a defensive choice, carried
# but never adjusted for) and the FTN pre-snap offensive confounders
# is_motion / n_offense_backfield / qb_location to the adjustment set.
GRAPH_VERSION = "v2.2"


# ── The graphs themselves ────────────────────────────────────────────────────
# Built from the module-level containers above BY REFERENCE (see dag.py), so the
# fingerprint tracks any edit to them.
SHELL_PRESSURE_GRAPH = CausalGraph(
    name="coverage shell x pressure, charted dropbacks",
    version=GRAPH_VERSION,
    treatment=TREATMENT, outcome=OUTCOME, edges=EDGES,
    observed_confounders=OBSERVED_CONFOUNDERS,
    confounder_columns=CONFOUNDER_COLUMNS,
    observed=OBSERVED, unobserved=UNOBSERVED, mediators=MEDIATORS,
    selection=SELECTION, defensive_choice=DEFENSIVE_CHOICE,
    def_choice_columns=DEF_CHOICE_COLUMNS,
    treatment_columns=["defense_coverage_type", "number_of_pass_rushers"],
    sample_note=("the estimand is conditional on a charted dropback, so "
                 "run/pass deterrence is outside it"),
)


# ── Every-snap graph: treatment = personnel package x front ─────────────────
# The alignment the defense shows before the snap — which personnel package is
# on the field, and how many defenders are in the box. Both are charted on ~100%
# of snaps, runs included, and both are chosen before the snap, so this is the
# first treatment in the project that is simultaneously callable and fully
# observed. Three structural differences from the dropback graph:
#
#   * No dropback selection. Runs are in the sample, so the offense's run/pass
#     choice is a MEDIATOR the effect flows through (deterrence: run rate falls
#     from 64% against base personnel to 14% against dime), not a variable the
#     sample is conditioned on.
#   * The coverage call is a MEDIATOR too. The DC pairs a shell and a pressure
#     with the alignment, and that pairing is part of what calling an alignment
#     does — so the level-1 estimand is the alignment's TOTAL effect, including
#     the coverage that typically comes with it. Coverage is also measured
#     post-snap and missing on runs, so it could not be adjusted for anyway.
#   * Team identity is a confounder. A defense's personnel and scheme drive its
#     calls AND its outcomes, and the opposing offense shapes both as well.
TEAMS = "team_identity"          # defense + offense identity (confounder)
DEF_ALIGNMENT = "def_alignment"  # treatment: personnel package x front
DEF_COVERAGE = "def_coverage"    # the paired shell x pressure — a MEDIATOR here

ALIGNMENT_OBSERVED_CONFOUNDERS = OBSERVED_CONFOUNDERS + [TEAMS]
ALIGNMENT_CONFOUNDER_COLUMNS = {**CONFOUNDER_COLUMNS, TEAMS: ["defteam", "posteam"]}
ALIGNMENT_EDGES = [
    # structure among the pre-snap state
    (GAME_STATE, OFF_PERSONNEL), (GAME_STATE, OFF_FORMATION),
    (GAME_STATE, OFF_TENDENCIES), (OFF_PERSONNEL, OFF_FORMATION),
    (TEAMS, PLAYER_PROXIES),
    # confounders -> the call, the outcome, and the offense's run/pass choice
    (GAME_STATE, DEF_ALIGNMENT), (GAME_STATE, EPA), (GAME_STATE, DEF_COVERAGE),
    (OFF_PERSONNEL, DEF_ALIGNMENT), (OFF_PERSONNEL, EPA), (OFF_PERSONNEL, OFF_PLAYCALL),
    (OFF_FORMATION, DEF_ALIGNMENT), (OFF_FORMATION, EPA), (OFF_FORMATION, OFF_PLAYCALL),
    (OFF_TENDENCIES, DEF_ALIGNMENT), (OFF_TENDENCIES, OFF_PLAYCALL),
    (PLAYER_PROXIES, DEF_ALIGNMENT), (PLAYER_PROXIES, EPA),
    (TEAMS, DEF_ALIGNMENT), (TEAMS, EPA), (TEAMS, OFF_PLAYCALL),
    # the hidden read
    (COACH_READ, DEF_ALIGNMENT), (COACH_READ, DEF_COVERAGE), (COACH_READ, EPA),
    # the effect, and the paths it travels
    (DEF_ALIGNMENT, EPA),
    (DEF_ALIGNMENT, OFF_PLAYCALL),       # deterrence
    (DEF_ALIGNMENT, DEF_COVERAGE),       # the coverage paired with the alignment
    (OFF_PLAYCALL, EPA), (DEF_COVERAGE, EPA),
]
ALIGNMENT_OBSERVED = (set(ALIGNMENT_OBSERVED_CONFOUNDERS)
                      | {OFF_PLAYCALL, DEF_ALIGNMENT, DEF_COVERAGE, EPA})
ALIGNMENT_MEDIATORS = {OFF_PLAYCALL, DEF_COVERAGE}

# v3.0 — the every-snap alignment graph (new treatment, team confounders, the
# coverage call and the offense's run/pass choice both mediators, no selection).
ALIGNMENT_GRAPH_VERSION = "v3.0"

ALIGNMENT_GRAPH = CausalGraph(
    name="personnel package x front, every snap",
    version=ALIGNMENT_GRAPH_VERSION,
    treatment=DEF_ALIGNMENT, outcome=EPA, edges=ALIGNMENT_EDGES,
    observed_confounders=ALIGNMENT_OBSERVED_CONFOUNDERS,
    confounder_columns=ALIGNMENT_CONFOUNDER_COLUMNS,
    observed=ALIGNMENT_OBSERVED, unobserved=UNOBSERVED,
    mediators=ALIGNMENT_MEDIATORS,
    treatment_columns=["defense_personnel", "n_defense_box",
                       "personnel_package", "front"],
    sample_note=("no dropback selection: every FTN- and personnel-charted "
                 "scrimmage snap, runs included"),
)


def graph_for(cfg: dict) -> CausalGraph:
    """The graph a configuration's estimates are made under.

    Keyed on the ACTION SCHEME rather than a version string, because the graph
    is a claim about the treatment: change what the action is, and the backdoor
    set, the mediators and the selection all change with it.
    """
    scheme = cfg.get("action", {}).get("scheme", "shell_pressure")
    if scheme == "level1":
        return ALIGNMENT_GRAPH
    if scheme == "shell_pressure":
        return SHELL_PRESSURE_GRAPH
    raise ValueError(f"unknown action scheme {scheme!r}")


# ── Module-level API: the shell x pressure graph ─────────────────────────────
# Kept so existing callers read naturally; each delegates to the graph object,
# so there is one implementation of every structural computation.
def fingerprint() -> str:
    return SHELL_PRESSURE_GRAPH.fingerprint()


def version_tag() -> str:
    return SHELL_PRESSURE_GRAPH.version_tag()


def adjustment_set() -> set[str]:
    return SHELL_PRESSURE_GRAPH.adjustment_set()


def adjustment_columns() -> list[str]:
    return SHELL_PRESSURE_GRAPH.adjustment_columns()


def to_gml() -> str:
    return SHELL_PRESSURE_GRAPH.to_gml()


def describe() -> str:
    return SHELL_PRESSURE_GRAPH.describe()
