"""
Data Layer — feature engineering: raw play-by-play -> (S, A, R) table.

Discipline enforced here (CLAUDE.md):
  * Pre-snap features only in S. Nothing derived from the play's result.
  * Action labels with charted-coverage noise are treated as the observed
    treatment; uncharted-coverage plays are dropped in V1 rather than guessed.
  * Reward R = -EPA (defense minimizes offensive EPA).

The `Dataset` returned bundles the frame with the column metadata every
downstream model needs, so feature lists are defined once, here.
"""

from __future__ import annotations

import pandas as pd

from ..schemas.dataset import Dataset

# 4-action space: coverage shell (man/zone) x pressure (blitz/no-blitz).
ACTION_LABELS = {
    0: "zone_no_blitz",
    1: "zone_blitz",
    2: "man_no_blitz",
    3: "man_blitz",
}
N_ACTIONS = 4

# Pre-snap state features S. Down is constant (3rd) in V1 so it is omitted.
NUMERIC_STATE = [
    "ydstogo", "yardline_100", "score_diff", "game_seconds_remaining",
    "half_seconds_remaining", "qtr",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "shotgun", "no_huddle",
    "num_rb", "num_te", "num_wr",
    "off_pass_tendency",
]
CATEGORICAL_STATE = ["formation"]


def build_v1_dataset(pbp: pd.DataFrame, cfg: dict) -> Dataset:
    """Turn raw play-by-play into the V1 (S, A, R) table for one team."""
    team = cfg["data"]["team"]
    down = cfg["data"]["down"]
    df = pbp.copy()

    # ── 1. Filter to the V1 slice ────────────────────────────────────────────
    mask = (
        (df["defteam"] == team)
        & (df["down"] == down)
        & (df["play_type"].isin(["pass", "run"]))
        & (df["epa"].notna())
        & (df["ydstogo"].notna())
    )
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
    cols = (
        ["game_id", "play_id", "season", "week", "epa", "reward",
         "action", "action_label"]
        + NUMERIC_STATE + CATEGORICAL_STATE
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


def _label_actions(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Construct the 4-action treatment from charted nflfastR columns."""
    # Identify man and zone coverage types
    mz = df["defense_man_zone_type"].astype("string").str.upper().fillna("")
    is_man = mz.str.contains("MAN")
    is_zone = mz.str.contains("ZONE")

    # Drop unlabeled coverages
    if cfg["action"]["drop_uncharted_coverage"]:
        df = df[is_man | is_zone].copy()
        is_man = is_man[df.index]

    # Identify blitz
    thr = cfg["action"]["blitz_rusher_threshold"]
    rushers = pd.to_numeric(df["number_of_pass_rushers"], errors="coerce").fillna(4)
    is_blitz = (rushers >= thr).astype(int)
    man = is_man.astype(int).to_numpy()

    # Pack the two binary axes into one action id: 2*man + blitz. This is the
    # exact encoding ACTION_LABELS decodes (0 zone/no, 1 zone/blitz, 2 man/no,
    # 3 man/blitz), so the id doubles as an index into the 4-action space.
    df["action"] = man * 2 + is_blitz.to_numpy()
    df["action_label"] = df["action"].map(ACTION_LABELS)
    return df


def _build_state(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer the pre-snap state vector. No post-snap information allowed."""
    # Extract and cast game state variables
    df["score_diff"] = pd.to_numeric(df.get("score_differential"), errors="coerce").fillna(0)
    df["yardline_100"] = pd.to_numeric(df["yardline_100"], errors="coerce").fillna(50)
    df["ydstogo"] = pd.to_numeric(df["ydstogo"], errors="coerce").clip(1, 30)

    for c in ["game_seconds_remaining", "half_seconds_remaining", "qtr",
              "posteam_timeouts_remaining", "defteam_timeouts_remaining",
              "shotgun", "no_huddle"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").fillna(0)

    # Offensive personnel grouping (e.g. "1 RB, 1 TE, 3 WR").
    personnel = df.get("offense_personnel", pd.Series("", index=df.index)).fillna("")
    df["num_rb"] = personnel.str.extract(r"(\d+)\s*RB").astype(float).fillna(1)
    df["num_te"] = personnel.str.extract(r"(\d+)\s*TE").astype(float).fillna(1)
    df["num_wr"] = personnel.str.extract(r"(\d+)\s*WR").astype(float).fillna(3)
    df["formation"] = df.get("offense_formation", pd.Series("UNKNOWN", index=df.index)) \
        .fillna("UNKNOWN").astype(str).str.upper()

    df = _add_pass_tendency(df)
    return df


def _add_pass_tendency(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged offensive pass rate by distance bucket — strictly historical.

    Uses an expanding mean shifted by one play so no look-ahead leaks in.
    """
    df = df.sort_values(["season", "week", "game_id", "play_id"])
    df["is_pass"] = (df["play_type"] == "pass").astype(int)
    df["dist_bucket"] = pd.cut(
        df["ydstogo"], bins=[0, 3, 7, 15, 100],
        labels=["short", "medium", "long", "very_long"],
    ).astype(str)
    df["off_pass_tendency"] = (
        df.groupby("dist_bucket")["is_pass"]
        .transform(lambda x: x.expanding().mean().shift(1))
        .fillna(0.55)  # league-average prior before any history exists
    )
    return df


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
