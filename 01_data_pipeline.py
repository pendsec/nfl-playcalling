"""
NFL Defensive Playcall Causal Inference Pipeline
Step 1: Data Loading & Feature Engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ── 1. Load nflverse play-by-play data ───────────────────────────────────────
def load_pbp(seasons: list[int]) -> pd.DataFrame:
    """
    Load nflverse play-by-play data for given seasons.
    Install via: pip install nfl-data-py
    """
    import nfl_data_py as nfl
    pbp = nfl.import_pbp_data(seasons, downcast=True, cache=False)
    return pbp


# ── 2. Filter to relevant plays ───────────────────────────────────────────────
def filter_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Keep only scrimmage plays with EPA and formation data."""
    mask = (
        pbp["play_type"].isin(["pass", "run"])
        & pbp["epa"].notna()
        & pbp["down"].notna()
        & pbp["ydstogo"].notna()
        & pbp["defteam_score"].notna()
        & pbp["posteam_score"].notna()
        # Drop spikes, kneels, penalties with no snap
        & ~pbp["play_type_nfl"].isin(["SPIKE", "KNEEL"])
    )
    return pbp[mask].copy()


# ── 3. Game state features ────────────────────────────────────────────────────
def build_game_state(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the exogenous game state variables."""
    df["score_diff"] = df["posteam_score"] - df["defteam_score"]
    df["seconds_remaining"] = df["game_seconds_remaining"]
    df["yardline_100"] = df["yardline_100"].fillna(50)
    df["down"] = df["down"].astype(int)
    df["ydstogo"] = df["ydstogo"].clip(1, 30)

    # Situational flags
    df["is_two_minute_drill"] = (df["half_seconds_remaining"] <= 120).astype(int)
    df["is_red_zone"] = (df["yardline_100"] <= 20).astype(int)
    df["is_third_or_fourth"] = df["down"].isin([3, 4]).astype(int)
    df["scoring_opp"] = (
        (df["score_diff"] <= 8) & (df["seconds_remaining"] <= 300)
    ).astype(int)

    return df


# ── 4. Offensive formation & personnel ───────────────────────────────────────
def build_offensive_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse offensive formation and personnel grouping.
    nflverse provides: offense_formation, offense_personnel
    """
    # Formation (shotgun, singleback, pistol, etc.)
    df["formation"] = df["offense_formation"].fillna("UNKNOWN").str.upper()

    # Personnel: parse "1 RB, 1 TE, 3 WR" → numerical features
    personnel = df["offense_personnel"].fillna("")
    df["num_rb"] = personnel.str.extract(r"(\d+)\s*RB").astype(float).fillna(1)
    df["num_te"] = personnel.str.extract(r"(\d+)\s*TE").astype(float).fillna(1)
    df["num_wr"] = personnel.str.extract(r"(\d+)\s*WR").astype(float).fillna(2)
    df["num_ol"] = personnel.str.extract(r"(\d+)\s*OL").astype(float).fillna(5)

    # Personnel grouping label (11, 12, 21, etc. — industry standard)
    df["personnel_group"] = (
        df["num_rb"].astype(int).astype(str)
        + df["num_te"].astype(int).astype(str)
    )

    return df


# ── 5. Player quality proxies (replaces PFF grades) ──────────────────────────
def build_player_proxies(df: pd.DataFrame, roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build proxy variables for player quality without proprietary grades.

    Proxies used:
      - Draft position (round × pick) → inversely scaled
      - Years of experience (service time in NFL)
      - Contract value proxy (from OverTheCap, merged separately)
      - Snap-count weighted stats (yards allowed per coverage snap, etc.)
    
    Here we use team-season level aggregates as a practical approximation,
    since play-level roster is expensive to join at scale.
    """
    # Example: merge team-season defensive strength as a proxy
    # In practice, join actual player snaps + stats from nfl.import_seasonal_data()
    team_def = (
        df.groupby(["defteam", "season"])["epa"]
        .mean()
        .reset_index()
        .rename(columns={"epa": "def_team_avg_epa"})  # lower = better defense
    )
    df = df.merge(team_def, on=["defteam", "season"], how="left")

    # Passer rating proxy for offensive skill (affects EPA baseline)
    qb_stats = (
        df[df["play_type"] == "pass"]
        .groupby(["passer_player_name", "season"])["epa"]
        .mean()
        .reset_index()
        .rename(columns={"epa": "qb_avg_epa"})
    )
    df = df.merge(
        qb_stats, on=["passer_player_name", "season"], how="left"
    )
    df["qb_avg_epa"] = df["qb_avg_epa"].fillna(df["qb_avg_epa"].median())

    return df


# ── 6. Offensive tendency features ───────────────────────────────────────────
def build_offensive_tendencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling historical tendency profiles per team × formation × down × distance bucket.
    These are lagged (no look-ahead) to avoid data leakage.
    """
    df = df.sort_values(["game_id", "play_id"])

    # Distance bucket
    df["dist_bucket"] = pd.cut(
        df["ydstogo"],
        bins=[0, 3, 7, 15, 100],
        labels=["short", "medium", "long", "very_long"]
    ).astype(str)

    # Pass rate by team × down × dist (rolling, lagged by 1 game)
    tendency_key = ["posteam", "season", "down", "dist_bucket", "formation"]
    df["is_pass"] = (df["play_type"] == "pass").astype(int)

    tendency = (
        df.groupby(tendency_key)["is_pass"]
        .transform(lambda x: x.expanding().mean().shift(1))
        .rename("off_pass_tendency")
    )
    df["off_pass_tendency"] = tendency.fillna(0.55)  # league average prior

    return df


# ── 7. Treatment: Defensive playcall ─────────────────────────────────────────
def build_treatment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the treatment variable from available nflverse columns.
    
    nflverse defensive call proxies:
      - defense_man_zone_type  (man / zone / unknown)
      - number_of_pass_rushers
      - defenders_in_box
      - coverage_type (if available in newer seasons)
    
    We construct a composite label capturing the key strategic dimensions:
      pressure level (box count + rushers) × coverage shell
    """
    df["defenders_in_box"] = df["defenders_in_box"].fillna(6).clip(4, 8)
    df["number_of_pass_rushers"] = df["number_of_pass_rushers"].fillna(4).clip(3, 6)

    # Pressure tier
    df["pressure_tier"] = pd.cut(
        df["number_of_pass_rushers"],
        bins=[2, 3, 4, 6],
        labels=["light", "standard", "heavy"]
    ).astype(str)

    # Box density
    df["box_density"] = pd.cut(
        df["defenders_in_box"],
        bins=[3, 5, 6, 8],
        labels=["light_box", "standard_box", "stacked_box"]
    ).astype(str)

    # Coverage (man vs zone)
    if "defense_man_zone_type" in df.columns:
        df["coverage_shell"] = df["defense_man_zone_type"].fillna("ZONE")
    else:
        df["coverage_shell"] = "ZONE"  # fallback prior

    # Composite treatment label
    df["def_playcall"] = (
        df["pressure_tier"] + "_" + df["box_density"] + "_" + df["coverage_shell"]
    )

    le = LabelEncoder()
    df["def_playcall_enc"] = le.fit_transform(df["def_playcall"])
    df["def_playcall_classes"] = [le.classes_] * len(df)  # store for decoding

    return df, le


# ── 8. Master feature set ────────────────────────────────────────────────────
GAME_STATE_COLS = [
    "down", "ydstogo", "score_diff", "seconds_remaining",
    "yardline_100", "is_two_minute_drill", "is_red_zone",
    "is_third_or_fourth", "scoring_opp"
]

FORMATION_COLS = [
    "formation", "num_rb", "num_te", "num_wr", "personnel_group"
]

TENDENCY_COLS = ["off_pass_tendency"]

PROXY_COLS = ["def_team_avg_epa", "qb_avg_epa"]

TREATMENT_COL = "def_playcall_enc"
OUTCOME_COL = "epa"

ALL_CONFOUNDERS = GAME_STATE_COLS + FORMATION_COLS + TENDENCY_COLS + PROXY_COLS


def build_model_df(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the final modeling dataframe."""
    cols = ALL_CONFOUNDERS + [TREATMENT_COL, OUTCOME_COL, "def_playcall", "play_id", "game_id"]
    model_df = df[cols].dropna(subset=GAME_STATE_COLS + [TREATMENT_COL, OUTCOME_COL])
    return model_df


if __name__ == "__main__":
    print("Pipeline module loaded. Import and call load_pbp() to begin.")
    print(f"Confounder set ({len(ALL_CONFOUNDERS)} features):")
    for c in ALL_CONFOUNDERS:
        print(f"  {c}")
