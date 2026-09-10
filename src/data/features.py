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

V2 action space: 6 coverage shells {C0,C1,C2,C3,C4,C6} x pressure {no-blitz,
blitz} = 12 actions, across ALL downs (V1 was 3rd-down-only, 4 actions).

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

import pandas as pd

from .load import COVERAGE_SHELLS, N_SHELLS
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
]
CATEGORICAL_STATE = ["formation"]


def build_dataset(pbp: pd.DataFrame, cfg: dict) -> Dataset:
    """Turn raw play-by-play into the V2 (S, A, R) table for one team.

    Set `cfg['data']['down']` to an int to restrict to one down, or to "all"
    / None for every down (the V2 default).
    """
    team = cfg["data"]["team"]
    down = cfg["data"].get("down")
    # Default keeps both play types so the V1 config (which predates this key)
    # behaves exactly as before.
    play_types = cfg["data"].get("play_types", ["pass", "run"])
    df = pbp.copy()

    # ── 1. Filter to the slice ───────────────────────────────────────────────
    mask = (
        (df["defteam"] == team)
        & (df["play_type"].isin(play_types))
        & (df["epa"].notna())
        & (df["ydstogo"].notna())
        & (df["down"].notna())
    )
    if down not in (None, "all"):
        mask &= df["down"] == down
    if "play_type_nfl" in df.columns:
        mask &= ~df["play_type_nfl"].isin(["SPIKE", "KNEEL"])
    df = df[mask].copy()

    # ── 2. Action labeling (treatment) ───────────────────────────────────────
    df = _label_actions(df, cfg)

    # ── 3. Pre-snap state features ───────────────────────────────────────────
    df = _build_state(df)

    # ── 4. Reward ────────────────────────────────────────────────────────────
    df["reward"] = -df["epa"] if cfg["reward"]["negate_epa"] else df["epa"]

    # ── 5. Final tidy ────────────────────────────────────────────────────────
    ground_truth = [c for c in df.columns if c.startswith("_")]  # synthetic only
    cols = (
        ["game_id", "play_id", "season", "week", "epa", "reward",
         "action", "action_label"]
        + NUMERIC_STATE + CATEGORICAL_STATE + ground_truth
    )
    cols = [c for c in cols if c in df.columns]
    df = df[cols].dropna(subset=NUMERIC_STATE + ["action"]).reset_index(drop=True)

    return Dataset(
        df=df,
        numeric_state=list(NUMERIC_STATE),
        categorical_state=list(CATEGORICAL_STATE),
        action_col="action",
        reward_col="reward",
        n_actions=N_ACTIONS,
        action_labels=dict(ACTION_LABELS),
    )


def slice_report(pbp: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """What the slice keeps and drops, by play type — the charting-coverage story.

    The coverage-shell axis of the action space is only defined where nflfastR
    charted `defense_coverage_type`, and that charting is NGS dropback data. Runs
    come back charted a few percent of the time, so a bare row count makes the
    slice look like "all snaps, minus some noise" when it is really "dropbacks".
    Reporting charted-vs-total per play type puts that in the run log, next to
    the `play_types` restriction it justifies.
    """
    team = cfg["data"]["team"]
    kept = cfg["data"].get("play_types", ["pass", "run"])
    df = pbp[
        (pbp["defteam"] == team)
        & pbp["epa"].notna() & pbp["ydstogo"].notna() & pbp["down"].notna()
        & pbp["play_type"].isin(["pass", "run"])
    ]
    if "play_type_nfl" in df.columns:
        df = df[~df["play_type_nfl"].isin(["SPIKE", "KNEEL"])]

    charted = df["defense_coverage_type"].astype("string").str.upper().isin(COVERAGE_SHELLS)
    rows = [
        {
            "play_type": pt,
            "n_plays": len(grp),
            "n_charted": int(charted.loc[grp.index].sum()),
            "pct_charted": float(charted.loc[grp.index].mean()) if len(grp) else 0.0,
            "in_slice": pt in kept,
        }
        for pt, grp in df.groupby("play_type")
    ]
    return pd.DataFrame(rows).sort_values("n_plays", ascending=False).reset_index(drop=True)


# Backwards-compatible alias — the V1 tag imports `build_v1_dataset`. Kept so a
# `git checkout v1` codepath name still resolves if ever cross-imported.
build_v1_dataset = build_dataset


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

    # Create categorical blitz variable
    thr = cfg["action"]["blitz_rusher_threshold"]
    rushers = pd.to_numeric(df["number_of_pass_rushers"], errors="coerce").fillna(4)
    is_blitz = (rushers >= thr).astype(int)

    # Create numeric action variable based on shell coverage and blitz indicator
    df["action"] = shell.astype(int).to_numpy() * 2 + is_blitz.to_numpy()
    df["action_label"] = df["action"].map(ACTION_LABELS)
    return df


def _build_state(df: pd.DataFrame) -> pd.DataFrame:
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

    df = _add_history_features(df)
    return df


def _add_history_features(df: pd.DataFrame) -> pd.DataFrame:
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
    df["off_pass_tendency"] = (
        df.groupby("dist_bucket")["is_pass"]
        .transform(lambda x: x.expanding().mean().shift(1))
        .fillna(0.55)
    )

    # Player / unit EPA proxies — historical mean EPA by defense and by QB.
    df["def_team_avg_epa"] = _expanding_mean(df, "defteam", "epa")
    qb_key = "passer_player_id" if "passer_player_id" in df.columns else "posteam"
    df["qb_avg_epa"] = _expanding_mean(df, qb_key, "epa")
    return df


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
    train_df = ds.df[ds.df["season"] < holdout_season].reset_index(drop=True)
    test_df = ds.df[ds.df["season"] >= holdout_season].reset_index(drop=True)

    def _clone(frame: pd.DataFrame) -> Dataset:
        return Dataset(
            df=frame, numeric_state=ds.numeric_state,
            categorical_state=ds.categorical_state, action_col=ds.action_col,
            reward_col=ds.reward_col, n_actions=ds.n_actions,
            action_labels=ds.action_labels,
        )

    return _clone(train_df), _clone(test_df)
