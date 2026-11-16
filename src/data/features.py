"""
Data Layer — feature engineering: raw play-by-play -> (S, A, R) table.

Discipline enforced here (CLAUDE.md):
  * Pre-snap features only in S. Nothing derived from the play's own result.
    History features (rolling tendencies / EPA proxies) use strictly PAST plays
    (expanding mean, shifted one play) so no look-ahead leaks in.
  * Action labels with charted-coverage noise are the observed treatment;
    uncharted / off-taxonomy coverages (NaN, PREVENT, 2_MAN) are dropped rather
    than guessed (treated as latent).
  * Reward R = -EPA (defense minimizes offensive EPA).

Action space: 6 coverage shells {C0,C1,C2,C3,C4,C6} x pressure {no-blitz, blitz}
= 12 actions, across all downs.

Scope caveat — the slice is CHARTED DROPBACKS, not every snap. The coverage axis
comes from nflfastR's `defense_coverage_type`, which is NGS charting on pass
plays: runs are charted on roughly 3% of snaps, so `drop_uncharted_coverage`
would silently delete nearly every run anyway. `cfg['data']['play_types']` makes
that a declared restriction rather than a side effect, and `slice_report`
quantifies it for the run log. It matters for reading a recommendation: "cover-3
on 1st and 10" here means *given the offense drops back*, not unconditionally.

The `Dataset` returned bundles the frame with the column metadata every
downstream model needs, so feature lists are defined once, here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .load import (COVERAGE_SHELLS, FTN_POSTSNAP, FTN_PRESNAP_DEFENSE,
                   N_SHELLS, ftn_coverage)
from ..schemas.dataset import Dataset

# 12-action space: coverage shell (6) x pressure (blitz / no-blitz).
# Encoding action = 2*shell + blitz, so the id doubles as (shell, pressure).
_SHELL_NAMES = ["cover0", "cover1", "cover2", "cover3", "cover4", "cover6"]
ACTION_LABELS = {
    2 * c + b: f"{_SHELL_NAMES[c]}_{'blitz' if b else 'no_blitz'}"
    for c in range(N_SHELLS) for b in (0, 1)
}
N_ACTIONS = 2 * N_SHELLS  # 12

# Pre-snap state features S. `down` now varies (all downs) so it is included.
NUMERIC_STATE = [
    "down", "ydstogo", "yardline_100", "score_diff",
    "game_seconds_remaining", "half_seconds_remaining", "qtr",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "shotgun", "no_huddle",
    "num_rb", "num_te", "num_wr",
    "is_two_minute_drill", "is_red_zone", "is_third_or_fourth", "scoring_opp",
    "off_pass_tendency", "def_team_avg_epa", "qb_avg_epa",
    # FTN-charted pre-snap offensive presentation. The defense sees all three
    # before committing to a call, so they are ordinary confounders.
    "is_motion", "n_offense_backfield",
]
CATEGORICAL_STATE = ["formation", "qb_location"]

# Columns kept in the built frame but NEVER offered to a model as state.
#
#   n_defense_box  — pre-snap, but a DEFENSIVE choice: a sibling of the
#     treatment rather than a cause of it. Adjusting for part of the defense's
#     own decision would block the effect being estimated. Carried because it is
#     the only defensive attribute charted on ~99% of snaps (runs included), and
#     therefore the primary axis of the factored action space.
#   is_play_action / is_rpo / is_screen_pass — post-snap reveals. Carried so the
#     mediator structure can be studied, and so the leak guard in
#     `scm.identify.assert_adjustment_consistency` has real columns to reject.
#
# `test_carried_columns_never_enter_the_state` is what keeps this honest.
CARRIED_COLUMNS = (list(FTN_PRESNAP_DEFENSE) + list(FTN_POSTSNAP) + ["play_type"]
                   # The coverage as played and the rushers who rushed: the
                   # level-2 action labels, read off the level-1 frame.
                   + ["defense_coverage_type", "number_of_pass_rushers"])


# ── Level-1 action: personnel package x front, on every snap ────────────────
# The alignment the defense shows before the snap. Both halves are charted on
# ~100% of snaps (personnel in nflfastR's participation data, box count in
# FTN), runs included, and both are chosen pre-snap — so unlike coverage and
# pressure, which are measured as the play unfolds and only on dropbacks, this
# is an action a coordinator can actually call and we can actually observe.
PACKAGES = ["base", "nickel", "dime_plus"]
FRONTS = ["box_le5", "box6", "box7", "box_ge8"]
N_FRONTS = len(FRONTS)
LEVEL1_LABELS = {p * N_FRONTS + f: f"{PACKAGES[p]}_{FRONTS[f]}"
                 for p in range(len(PACKAGES)) for f in range(N_FRONTS)}

# The axes level 1 can be built from, in id order (the first is the slowest-
# varying digit). `action.level1.axes` picks which of them make up the call;
# the default is both. The box count is the axis the coordinator's read of the
# offense's intent confounds (heavy boxes face more runs than the state
# predicts; packages do not), so it is kept separable: `axes: [package]` makes
# the call the package alone, with nothing else to change.
LEVEL1_AXES = {"package": PACKAGES, "front": FRONTS}


def level1_axes(cfg: dict) -> list[str]:
    """The configured level-1 axes, validated and in canonical order."""
    axes = cfg.get("action", {}).get("level1", {}).get("axes", list(LEVEL1_AXES))
    unknown = set(axes) - set(LEVEL1_AXES)
    if unknown or not axes:
        raise ValueError(f"action.level1.axes must be a non-empty subset of "
                         f"{list(LEVEL1_AXES)}, got {axes}")
    return [a for a in LEVEL1_AXES if a in axes]


def level1_labels(cfg: dict) -> dict[int, str]:
    """Action id -> label for the configured axes (e.g. `nickel_box6`, or
    `nickel` when the call is the package alone)."""
    labels = [""]
    for axis in level1_axes(cfg):
        labels = [f"{l}_{v}" if l else v for l in labels for v in LEVEL1_AXES[axis]]
    return dict(enumerate(labels))


def level1_components(actions, cfg: dict) -> dict[str, np.ndarray]:
    """Split level-1 action ids into one integer array per configured axis."""
    rest = np.asarray(actions)
    out = {}
    for axis in reversed(level1_axes(cfg)):
        size = len(LEVEL1_AXES[axis])
        out[axis] = rest % size
        rest = rest // size
    return {a: out[a] for a in level1_axes(cfg)}

# ── Level-2 action: coverage shell x pressure, given the alignment ─────────
# Measured post-snap (the coverage as played, the rushers who rushed) and only
# on charted dropbacks, so its value is estimated there; the alignment it is
# paired with joins the state.
#
# The shell axis is configurable (`action.level2.shells`):
#   family — Cover-0 / single-high / two-high, the safety structure. Default.
#   full   — the six charted shells.
# The default is coarse because the fine labels drift between seasons in a way
# no defense could: Cover-1's share of charted dropbacks rose at 30 of 32 teams
# from 2022 to 2024 (median +18.5 pts, man coverage 29% -> 48%) and fell at all
# 32 in 2025 (median -15.7) — a charting convention moving the Cover-1 / Cover-3
# boundary, not scheme. The safety structure moves 4-7 pts. Training on 2022-24
# and evaluating on 2025 crosses that boundary, so a Cover-1-vs-Cover-3
# recommendation would rest on labels that changed meaning in between.
SHELL_FAMILIES = ["cover0", "single_high", "two_high"]
_FAMILY_OF = {"COVER_0": 0, "COVER_1": 1, "COVER_3": 1,
              "COVER_2": 2, "COVER_4": 2, "COVER_6": 2,
              # Two deep safeties, man underneath: unambiguous as a structure,
              # ambiguous as one of the six shells (dropped in `full` mode).
              "2_MAN": 2}
LEVEL2_SHELLS = {"family": SHELL_FAMILIES, "full": _SHELL_NAMES}

# The alignment as level-2 state: one ordinal column per configured level-1 axis.
L1_STATE_COLUMNS = {"package": "l1_package", "front": "l1_front"}


def level2_shell_mode(cfg: dict) -> str:
    """`family` (default) or `full` — see `LEVEL2_SHELLS`."""
    mode = cfg.get("action", {}).get("level2", {}).get("shells", "family")
    if mode not in LEVEL2_SHELLS:
        raise ValueError(f"action.level2.shells must be one of {list(LEVEL2_SHELLS)}, got {mode!r}")
    return mode


def level2_shells(cfg: dict) -> list[str]:
    return LEVEL2_SHELLS[level2_shell_mode(cfg)]


def level2_labels(cfg: dict) -> dict[int, str]:
    """Level-2 id -> label; id = 2 * shell + blitz, as in the dropback scheme."""
    return {2 * c + b: f"{name}_{'blitz' if b else 'no_blitz'}"
            for c, name in enumerate(level2_shells(cfg)) for b in (0, 1)}


def level2_alignment_columns(cfg: dict) -> list[str]:
    """The level-1 call's columns in the level-2 state, one per configured axis."""
    return [L1_STATE_COLUMNS[a] for a in level1_axes(cfg)]


def level2_config(cfg: dict) -> dict:
    """The configuration the level-2 models run under.

    Same models and OPE settings as level 1; the scheme becomes `level2` (which
    selects the coverage graph and its adjustment set), and `policy.level2`
    overrides the policy settings — the read allowance is a level-1 device.
    """
    import copy
    c = copy.deepcopy(cfg)
    c["action"]["scheme"] = "level2"
    pol = {k: v for k, v in c["policy"].items() if k not in ("level2", "read_allowance")}
    c["policy"] = {**pol, **(cfg["policy"].get("level2") or {})}
    return c


def shell_index(coverage: pd.Series, cfg: dict) -> pd.Series:
    """Charted coverage -> index into `level2_shells(cfg)`; NaN off the taxonomy."""
    cov = coverage.astype("string").str.upper()
    if level2_shell_mode(cfg) == "family":
        return cov.map(_FAMILY_OF).astype(float)
    return cov.map({name: i for i, name in enumerate(COVERAGE_SHELLS)}).astype(float)


def build_level2_dataset(ds1: Dataset, cfg: dict) -> Dataset:
    """Charted dropbacks from the level-1 dataset, with the level-2 action.

    Built from the level-1 frame rather than from raw play-by-play so every
    history feature (tendency, quality proxies) is computed over all snaps, as
    it is for level 1 — a dropback's state does not depend on which stage reads
    it. Coverage off the taxonomy (uncharted, prevent, combo, blown) is dropped
    rather than guessed. The level-1 call joins the state as ordinal columns;
    `l1_action` carries its id.
    """
    df = ds1.df[ds1.df["play_type"] == "pass"].copy()
    shell = shell_index(df["defense_coverage_type"], cfg)
    df = df[shell.notna()].copy()
    shell = shell[shell.notna()].astype(int)

    thr = cfg["action"].get("level2", {}).get("blitz_rusher_threshold", 5)
    df["l1_action"] = df[ds1.action_col].astype(int)
    for axis, comp in level1_components(df["l1_action"].to_numpy(), cfg).items():
        df[L1_STATE_COLUMNS[axis]] = comp.astype(float)
    labels = level2_labels(cfg)
    df["action"] = _shell_pressure_action(df, shell, thr)
    df["action_label"] = df["action"].map(labels)

    return Dataset(
        df=df.reset_index(drop=True),
        numeric_state=list(ds1.numeric_state) + level2_alignment_columns(cfg),
        categorical_state=list(ds1.categorical_state),
        action_col="action", reward_col=ds1.reward_col,
        n_actions=len(labels), action_labels=labels,
    )


# Team identity is a confounder in the every-snap graph: a defense's personnel
# and scheme drive both its calls and its outcomes, and so does the opponent.
TEAM_STATE = ["defteam", "posteam"]


def action_scheme(cfg: dict) -> str:
    """`shell_pressure` (charted dropbacks), `level1` (every snap), or `level2`
    (coverage x pressure given the alignment, charted dropbacks)."""
    return cfg.get("action", {}).get("scheme", "shell_pressure")


def state_columns(cfg: dict) -> tuple[list[str], list[str]]:
    """(numeric, categorical) state features for the configured action scheme.

    Must equal the adjustment set of `scm.graph_for(cfg)` — asserted by
    `scm.identify.assert_adjustment_consistency`.
    """
    scheme = action_scheme(cfg)
    if scheme in ("level1", "level2"):
        numeric = list(NUMERIC_STATE) + ["off_team_avg_epa"]
        if scheme == "level2":
            numeric += level2_alignment_columns(cfg)
        return numeric, list(CATEGORICAL_STATE) + TEAM_STATE
    return list(NUMERIC_STATE), list(CATEGORICAL_STATE)


def action_feature_matrix(cfg: dict) -> np.ndarray:
    """The feature vector each action id stands for, shape (n_actions, k).

    Both schemes factor into two ordered components, and exposing them — rather
    than 12 unrelated ids — is what lets a pooled outcome model share strength
    across cells: (package, front) for the every-snap scheme, where more
    defensive backs is lighter and more box defenders heavier; (shell, blitz)
    for the dropback scheme, with shells ordered man/press -> deep zone.
    """
    if action_scheme(cfg) == "level1":
        parts = level1_components(np.arange(len(level1_labels(cfg))), cfg)
        return np.column_stack(list(parts.values())).astype(float)
    if action_scheme(cfg) == "level2":
        ids = np.arange(len(level2_labels(cfg)))
        return np.column_stack([ids // 2, ids % 2]).astype(float)
    ids = np.arange(N_ACTIONS)
    return np.column_stack([ids // 2, ids % 2]).astype(float)


def personnel_db_count(personnel: pd.Series) -> pd.Series:
    """Defensive backs on the field, parsed from "3 CB, 2 DE, 2 DT, 1 FS, ...".

    Returns 0 where nothing parses, which callers treat as unlabeled — a real
    alignment always has at least one defensive back.
    """
    text = personnel.fillna("").astype(str)
    total = pd.Series(0.0, index=personnel.index)
    for pos in ("CB", "FS", "SS", "DB"):
        total += text.str.extract(rf"(\d+)\s*{pos}\b")[0].astype(float).fillna(0)
    return total


def _label_level1(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Construct the level-1 treatment: personnel package x front (12 actions),
    or the configured subset of those axes (`action.level1.axes`).

    Rows whose personnel does not parse, or that FTN never charted a box count
    for, carry no observed alignment and are dropped rather than guessed.
    """
    spec = cfg["action"]["level1"]
    packages = spec["personnel_packages"]
    if list(packages) != PACKAGES:
        raise ValueError(f"personnel_packages must be declared in order {PACKAGES}, "
                         f"got {list(packages)}")

    n_db = personnel_db_count(df["defense_personnel"])
    package = pd.Series(np.nan, index=df.index)
    for i, name in enumerate(PACKAGES):
        lo, hi = packages[name]
        package[(n_db >= lo) & (n_db <= hi) & (n_db > 0)] = i

    box = pd.to_numeric(df[FTN_PRESNAP_DEFENSE[0]], errors="coerce")
    # front_bins [5, 6, 7] -> <=5 | 6 | 7 | >=8
    front = pd.Series(np.searchsorted(spec["front_bins"], box.fillna(-1), side="left"),
                      index=df.index).astype(float)
    front[box.isna()] = np.nan

    # The box count is required even when it is not part of the call: the
    # sample stays the same whichever axes are configured, so a package-only
    # and a package x front run answer their questions on the same snaps.
    keep = package.notna() & front.notna()
    df = df[keep].copy()
    component = {"package": package[keep], "front": front[keep]}
    action = pd.Series(0.0, index=df.index)
    for axis in level1_axes(cfg):
        action = action * len(LEVEL1_AXES[axis]) + component[axis]
    df["action"] = action.astype(int)
    df["action_label"] = df["action"].map(level1_labels(cfg))
    return df


def _slice_mask(df: pd.DataFrame, cfg: dict, play_types: list[str]) -> pd.Series:
    """Rows in the configured slice: the play types given, the team and down in
    `cfg['data']`, a recorded EPA / distance / down, no spikes or kneels.
    `build_dataset` and `slice_report` both filter with it, so the report counts
    the same slice the models see."""
    team, down = cfg["data"]["team"], cfg["data"].get("down")
    mask = (df["play_type"].isin(play_types) & df["epa"].notna()
            & df["ydstogo"].notna() & df["down"].notna())
    if team not in (None, "all"):
        mask &= df["defteam"] == team
    if down not in (None, "all"):
        mask &= df["down"] == down
    if "play_type_nfl" in df.columns:
        mask &= ~df["play_type_nfl"].isin(["SPIKE", "KNEEL"])
    return mask


def build_dataset(pbp: pd.DataFrame, cfg: dict) -> Dataset:
    """Turn raw play-by-play into the (S, A, R) table.

    `cfg['data']['team']` names one defense, or "all" to pool every defense
    (team identity then enters the state as a confounder). The action scheme
    — `cfg['action']['scheme']` — decides what the treatment is: coverage shell
    x pressure on charted dropbacks, or personnel package x front on every snap.
    Set `cfg['data']['down']` to an int to restrict to one down, or to "all" /
    None for every down (the default).
    """
    scheme = action_scheme(cfg)
    if scheme == "level2":
        raise ValueError("action.scheme level2 is built from the level-1 dataset: "
                         "use build_level2_dataset(build_dataset(pbp, level-1 cfg), cfg)")
    df = pbp.copy()

    # ── 1. Filter to the slice ───────────────────────────────────────────────
    # Default keeps both play types so a config that omits this key behaves as
    # it did before the key existed.
    mask = _slice_mask(df, cfg, cfg["data"].get("play_types", ["pass", "run"]))
    if cfg["data"].get("use_ftn", False):
        # Declaring use_ftn means the model conditions on FTN-charted features,
        # so a play FTN never charted has none to condition on. Defaulting them
        # would invent a value (`is_motion = 0` reading as "no motion" rather
        # than "unknown") — the same silent-degradation `assert_ftn_coverage`
        # exists to prevent, just scattered instead of systematic. Small: ~0.3%
        # of the slice, but not a random 0.3% (their mean reward differs).
        mask &= df[FTN_PRESNAP_DEFENSE[0]].notna()
    df = df[mask].copy()

    # ── 2. Action labeling (treatment) ───────────────────────────────────────
    if scheme == "level1":
        df = _label_level1(df, cfg)
        labels = level1_labels(cfg)
    else:
        df = _label_actions(df, cfg)
        labels = ACTION_LABELS
    numeric_state, categorical_state = state_columns(cfg)

    # ── 3. Pre-snap state features ───────────────────────────────────────────
    halflife = cfg.get("features", {}).get("offense_quality_halflife_plays",
                                           OFFENSE_QUALITY_HALFLIFE)
    df = _build_state(df, per_offense=(scheme == "level1"), quality_halflife=halflife)

    # ── 4. Reward ────────────────────────────────────────────────────────────
    df["reward"] = -df["epa"] if cfg["reward"]["negate_epa"] else df["epa"]

    # ── 5. Final tidy ────────────────────────────────────────────────────────
    ground_truth = [c for c in df.columns if c.startswith("_")]  # synthetic only
    cols = (
        ["game_id", "play_id", "season", "week", "epa", "reward",
         "action", "action_label"]
        + numeric_state + categorical_state + CARRIED_COLUMNS + ground_truth
    )
    cols = list(dict.fromkeys(c for c in cols if c in df.columns))
    df = df[cols].dropna(subset=numeric_state + ["action"]).reset_index(drop=True)

    return Dataset(
        df=df,
        numeric_state=numeric_state,
        categorical_state=categorical_state,
        action_col="action",
        reward_col="reward",
        n_actions=len(labels),
        action_labels=dict(labels),
    )


def slice_report(pbp: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """What the slice keeps and drops, by play type — the charting-coverage story.

    `n_usable` counts plays that survive EVERY filter the dataset builder applies
    (coverage charted, and FTN charted when `use_ftn` is set), so it matches the
    final slice size rather than approximating it.

    The coverage-shell axis of the action space is only defined where nflfastR
    charted `defense_coverage_type`, and that charting is NGS dropback data. Runs
    come back charted a few percent of the time, so a bare row count makes the
    slice look like "all snaps, minus some noise" when it is really "dropbacks".
    Reporting charted-vs-total per play type puts that in the run log, next to
    the `play_types` restriction it justifies.
    """
    kept = cfg["data"].get("play_types", ["pass", "run"])
    # Both play types, whatever `play_types` keeps: the report shows what it drops.
    df = pbp[_slice_mask(pbp, cfg, ["pass", "run"])]

    if action_scheme(cfg) == "level1":
        # The alignment is labeled wherever personnel parses — coverage charting
        # is irrelevant to this treatment, which is the point of it.
        usable = personnel_db_count(df["defense_personnel"]) > 0
    else:
        usable = df["defense_coverage_type"].astype("string").str.upper().isin(COVERAGE_SHELLS)
    # Mirror every filter `build_dataset` applies, so the reported count is the
    # count that actually reaches the models. A report that says "1,907 charted"
    # beside a 1,902-play slice sends the reader hunting for the difference.
    if cfg["data"].get("use_ftn", False) and FTN_PRESNAP_DEFENSE[0] in df.columns:
        usable &= df[FTN_PRESNAP_DEFENSE[0]].notna()
    rows = [
        {
            "play_type": pt,
            "n_plays": len(grp),
            "n_usable": int(usable.loc[grp.index].sum()),
            "pct_usable": float(usable.loc[grp.index].mean()) if len(grp) else 0.0,
            "in_slice": pt in kept,
        }
        for pt, grp in df.groupby("play_type")
    ]
    return pd.DataFrame(rows).sort_values("n_plays", ascending=False).reset_index(drop=True)


def _label_actions(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Construct the 12-action treatment from charted nflfastR columns.

    coverage shell in {C0,C1,C2,C3,C4,C6} x pressure (rushers >= threshold).
    Off-taxonomy coverages (NaN, PREVENT, 2_MAN) are dropped as latent.
    """
    # Get mapping of coverage type to shell type
    shell_of = {name: i for i, name in enumerate(COVERAGE_SHELLS)}
    cov = df["defense_coverage_type"].astype("string").str.upper()
    shell = cov.map(shell_of)

    # Drop non-standard shells / coverages
    if cfg["action"].get("drop_uncharted_coverage", True):
        df = df[shell.notna()].copy()
        shell = shell[shell.notna()]
    else:
        shell = shell.fillna(3)  # default to Cover-3 (the modal shell)

    df["action"] = _shell_pressure_action(df, shell, cfg["action"]["blitz_rusher_threshold"])
    df["action_label"] = df["action"].map(ACTION_LABELS)
    return df


def _shell_pressure_action(df: pd.DataFrame, shell: pd.Series, threshold: int) -> np.ndarray:
    """Shell x pressure -> action id `shell * 2 + blitz`, blitz meaning at least
    `threshold` pass rushers (an unrecorded count reads as 4: no blitz)."""
    rushers = pd.to_numeric(df["number_of_pass_rushers"], errors="coerce").fillna(4)
    return shell.astype(int).to_numpy() * 2 + (rushers >= threshold).astype(int).to_numpy()


def _build_state(df: pd.DataFrame, per_offense: bool = False,
                 quality_halflife: float = 250) -> pd.DataFrame:
    """Engineer the pre-snap state vector. No post-snap information allowed."""

    # Extract game state data and clean
    df["down"] = pd.to_numeric(df["down"], errors="coerce")
    df["score_diff"] = pd.to_numeric(df.get("score_differential"), errors="coerce").fillna(0)
    df["yardline_100"] = pd.to_numeric(df["yardline_100"], errors="coerce").fillna(50)
    df["ydstogo"] = pd.to_numeric(df["ydstogo"], errors="coerce").clip(1, 30)

    for c in ["game_seconds_remaining", "half_seconds_remaining", "qtr",
              "posteam_timeouts_remaining", "defteam_timeouts_remaining",
              "shotgun", "no_huddle"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0)

    # Situational indicators (all pre-snap).
    df["is_two_minute_drill"] = (df["half_seconds_remaining"] <= 120).astype(int)
    df["is_red_zone"] = (df["yardline_100"] <= 20).astype(int)
    df["is_third_or_fourth"] = (df["down"] >= 3).astype(int)
    df["scoring_opp"] = (df["yardline_100"] <= 40).astype(int)

    # Offensive personnel grouping (e.g. "1 RB, 1 TE, 3 WR").
    personnel = df.get("offense_personnel", pd.Series("", index=df.index)).fillna("")
    df["num_rb"] = personnel.str.extract(r"(\d+)\s*RB").astype(float).fillna(1)
    df["num_te"] = personnel.str.extract(r"(\d+)\s*TE").astype(float).fillna(1)
    df["num_wr"] = personnel.str.extract(r"(\d+)\s*WR").astype(float).fillna(3)
    df["formation"] = df.get("offense_formation", pd.Series("UNKNOWN", index=df.index)) \
        .fillna("UNKNOWN").astype(str).str.upper()

    df = _build_ftn_state(df)
    df = _add_history_features(df, per_offense=per_offense, quality_halflife=quality_halflife)
    return df


def _build_ftn_state(df: pd.DataFrame) -> pd.DataFrame:
    """FTN-charted pre-snap features, plus the carried non-state columns.

Defaults here apply only when FTN is switched OFF. With `use_ftn` set,
    `build_dataset` has already dropped rows FTN never charted, so nothing
    reaches these defaults; without it, they keep the columns present and
    constant so a pre-FTN configuration still builds. `assert_ftn_coverage`
    turns a systematically-missing season into an error either way.
    """
    if "is_motion" in df.columns:
        df["is_motion"] = df["is_motion"].astype("boolean").fillna(False).astype(int)
    else:
        df["is_motion"] = 0
    # 1 back is the modal alignment; FTN leaves ~3% of scrimmage plays unlabeled.
    df["n_offense_backfield"] = pd.to_numeric(
        df.get("n_offense_backfield"), errors="coerce").fillna(1.0)
    df["qb_location"] = (
        df.get("qb_location", pd.Series("UNK", index=df.index))
        .astype("string").fillna("UNK").replace("0", "UNK").astype(str)
    )
    for c in CARRIED_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def assert_ftn_coverage(df: pd.DataFrame, cfg: dict) -> None:
    """Refuse to train FTN features on a season FTN does not cover.

    Silently training on a systematically-missing feature is the failure this
    prevents: with seasons 2021-2023 and FTN starting in 2022, every 2021 row
    would carry the default value, and the model would learn "no motion" as a
    property of 2021 rather than of the play. Better a loud error naming the
    seasons than a quietly season-confounded feature.
    """
    if not cfg["data"].get("use_ftn", False):
        return
    min_cov = cfg["data"].get("ftn_min_coverage", 0.80)
    holdout = cfg["data"]["holdout_season"]
    probe = FTN_PRESNAP_DEFENSE[0]
    if probe not in df.columns:
        raise ValueError(f"use_ftn is set but {probe!r} is absent — FTN never merged.")
    # Measure on scrimmage plays only. FTN charts nothing on kickoffs, punts,
    # field goals or timeouts, and counting those as misses makes a fully-covered
    # season look ~65% covered.
    cov = ftn_coverage(df, probe)
    bad = sorted(int(r.season) for _, r in cov.iterrows()
                 if r.ftn_coverage < min_cov)
    if bad:
        raise ValueError(
            f"FTN coverage below {min_cov:.0%} for season(s) {bad} "
            f"(FTN starts in 2022). Either set data.seasons to FTN-covered years "
            f"(holdout is currently {holdout}), or set data.use_ftn: false."
        )


def _add_history_features(df: pd.DataFrame, per_offense: bool = False,
                          quality_halflife: float = 250) -> pd.DataFrame:
    """Strictly-historical rolling features — offense tendency + EPA proxies.

    All use an expanding mean shifted one play within time order, so a play's
    own outcome never enters its own features. These are the `player_proxies`
    node of the SCM (QB / defensive-unit quality) plus offensive tendency.
    """
    df = df.sort_values(["season", "week", "game_id", "play_id"]).reset_index(drop=True)

    # Offensive pass tendency by distance bucket (league-avg prior pre-history).
    df["is_pass"] = (df["play_type"] == "pass").astype(int)
    df["dist_bucket"] = pd.cut(
        df["ydstogo"], bins=[0, 3, 7, 15, 100],
        labels=["short", "medium", "long", "very_long"],
    ).astype(str)
    # Pooled across offenses, a pass rate by distance is close to a league
    # constant; per offense it is an actual tendency. Grouped per offense when
    # runs are in the sample (on a pass-only sample every value would be 1).
    keys = ["posteam", "dist_bucket"] if per_offense and "posteam" in df.columns \
        else ["dist_bucket"]
    df["off_pass_tendency"] = (
        df.groupby(keys)["is_pass"]
        .transform(lambda x: x.expanding().mean().shift(1))
        .fillna(0.55)
    )

    # Player / unit EPA proxies — historical mean EPA by defense and by QB.
    df["def_team_avg_epa"] = _expanding_mean(df, "defteam", "epa")
    df["qb_on_field"] = _qb_on_field(df)
    df["qb_avg_epa"] = _expanding_mean(df, "qb_on_field", "epa")
    if per_offense:
        # Offense-unit quality: the offense's own EPA per play, recency-weighted.
        df["off_team_avg_epa"] = _recency_mean(df, "posteam", "epa", quality_halflife)
    return df


# Chosen by predictive power, not by taste: on 2022-24 training data, the
# correlation between an offense's estimate before a game and its EPA/play in
# that game was 0.284 for a plain expanding mean, 0.262 at a 1-game half-life
# (mostly noise), and peaked at 0.319 for 250 plays (~4 games) with 500 plays
# essentially tied — recent form matters, but not only the last game.
OFFENSE_QUALITY_HALFLIFE = 250


def _recency_mean(df: pd.DataFrame, group_col: str, value_col: str,
                  halflife: float) -> pd.Series:
    """Leak-free, recency-weighted mean of `value_col` within `group_col`.

    An exponentially weighted mean over the group's earlier plays (half-life in
    plays), shifted one play so a play's own outcome never enters its estimate.
    It runs across season boundaries on purpose: a team's first game is
    estimated from the end of its previous season, decayed, rather than from
    nothing. Undefined only on a group's very first play, where the league mean
    (EPA ~ 0) stands in.
    """
    if group_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return (df.groupby(group_col)[value_col]
            .transform(lambda x: x.ewm(halflife=halflife).mean().shift(1))
            .fillna(0.0))


def _qb_on_field(df: pd.DataFrame) -> pd.Series:
    """The offense's quarterback on every snap, not just on passes.

    nflfastR records `passer_player_id` only on pass plays — 100% of passes and
    0% of runs. Grouping a QB feature by it directly makes the feature missing
    (and then defaulted) on every run, so its value alone reveals whether the
    play was a run: a post-snap fact smuggled into the pre-snap state. The QB
    on the field is identity, visible before the snap, so it is carried within
    each offense's game to the snaps where he did not throw.
    """
    if "passer_player_id" not in df.columns:
        return df.get("posteam", pd.Series("UNK", index=df.index))
    if not {"game_id", "posteam"} <= set(df.columns):
        return df["passer_player_id"]
    return (df.groupby(["game_id", "posteam"])["passer_player_id"]
            .transform(lambda x: x.ffill().bfill()))


def _expanding_mean(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Leak-free expanding mean of `value_col` within `group_col` (shift 1)."""
    if group_col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return (
        df.groupby(group_col)[value_col]
        .transform(lambda x: x.expanding().mean().shift(1))
        .fillna(0.0)
    )


def time_aware_split(ds: Dataset, holdout_season: int) -> tuple[Dataset, Dataset]:
    """Split by season — never randomly across plays (avoids temporal leakage)."""
    train = ds.df["season"] < holdout_season
    return ds.subset(train), ds.subset(~train)
