"""
Data Layer — raw play-by-play ingestion.

Two sources, same downstream schema:
  * nflfastR  — real NFL play-by-play via nfl_data_py (cached to parquet).
  * synthetic — data simulated from a *known* SCM, used by the regression
                tests and for offline skeleton runs. Because the ground-truth
                effects are known, the synthetic path is how we validate that
                the OPE / policy machinery recovers the right answer.

Only the columns the V1 feature builder needs are guaranteed here.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# Columns the V1 feature builder consumes. Pulling a subset keeps the cache
# small and makes the pre-snap/post-snap boundary explicit.
PBP_COLUMNS = [
    "game_id", "play_id", "season", "week", "defteam", "posteam",
    "down", "ydstogo", "yardline_100", "qtr",
    "game_seconds_remaining", "half_seconds_remaining",
    "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "shotgun", "no_huddle", "play_type", "play_type_nfl",
    "offense_formation", "offense_personnel",
    "defense_man_zone_type", "number_of_pass_rushers", "defenders_in_box",
    "epa",
]


# ── nflfastR ────────────────────────────────────────────────────────────────
def load_pbp(seasons: list[int], cache_dir: str = "data/raw") -> pd.DataFrame:
    """Load nflfastR play-by-play for `seasons`, caching one parquet per season.

    Network is only hit for seasons not already cached.
    """
    import nfl_data_py as nfl

    os.makedirs(cache_dir, exist_ok=True)
    frames = []
    for yr in seasons:
        path = os.path.join(cache_dir, f"pbp_{yr}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
            continue
        df = nfl.import_pbp_data([yr], downcast=False, cache=False)
        keep = [c for c in PBP_COLUMNS if c in df.columns]
        df = df[keep].copy()
        df.to_parquet(path, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── Synthetic SCM ───────────────────────────────────────────────────────────
def generate_synthetic(n_plays: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Simulate 3rd-down plays from a known structural causal model.

    Ground-truth SCM (defense perspective, reward = -EPA so higher is better):

        S  (state)      : ydstogo, yardline_100, score_diff, time, formation
        U  (confounder) : `coach_read` — a hidden pre-snap read that drives BOTH
                          the call the DC makes AND the outcome. This is the
                          selection effect that naive RL conflates with the
                          causal effect; the test checks we can see through it.
        A  (action)     : 0=zone/no-blitz 1=zone/blitz 2=man/no-blitz 3=man/blitz
        R  (reward)     : structural function of S, A, U with a KNOWN best action.

    The true best action is engineered to be *blitz on 3rd-and-long* (a > 0
    pressure effect that grows with ydstogo), while the behavior policy is
    confounded into blitzing more when `coach_read` is favorable — so the
    observational mean makes blitz look better than it causally is.
    """
    rng = np.random.default_rng(seed)
    n = n_plays

    ydstogo = rng.integers(1, 20, n).astype(float)
    yardline_100 = rng.integers(1, 99, n).astype(float)
    score_diff = rng.normal(0, 10, n).round()
    game_seconds_remaining = rng.integers(0, 3600, n).astype(float)
    is_shotgun = (rng.random(n) < (0.4 + 0.02 * ydstogo)).astype(int)

    # Hidden confounder: the coach's pre-snap read (favorable for defense > 0).
    coach_read = rng.normal(0, 1, n)

    # Behavior policy pi_b(A|S,U): confounded. Blitz propensity rises with
    # ydstogo (sensible) AND with a favorable read (the confounding).
    long_yardage = (ydstogo - 7) / 5.0
    p_blitz = _sigmoid(-0.3 + 0.6 * long_yardage + 0.8 * coach_read)
    p_man = _sigmoid(-0.2 - 0.1 * long_yardage + 0.3 * coach_read)
    blitz = (rng.random(n) < p_blitz).astype(int)
    man = (rng.random(n) < p_man).astype(int)
    action = man * 2 + blitz  # 0..3

    # Structural outcome. True causal pressure effect helps more on long downs;
    # man coverage is slightly worse by default here. coach_read also lowers EPA
    # directly (the confounding path U -> R).
    true_blitz_effect = 0.05 + 0.06 * long_yardage      # reward units (-EPA)
    true_man_effect = -0.03
    epa = (
        0.02 * (ydstogo - 7) / 5.0
        - 0.01 * (yardline_100 - 50) / 50.0
        - true_blitz_effect * blitz
        - true_man_effect * man
        - 0.20 * coach_read                              # confounding into R
        + rng.normal(0, 0.6, n)
    )

    # Map to nflfastR-like columns the feature builder understands.
    df = pd.DataFrame({
        "game_id": np.repeat(np.arange(n // 40 + 1), 40)[:n],
        "play_id": np.arange(n),
        "season": rng.choice([2021, 2022, 2023], n),
        "week": rng.integers(1, 18, n),
        "defteam": "SYN",
        "posteam": "OPP",
        "down": 3,
        "ydstogo": ydstogo,
        "yardline_100": yardline_100,
        "qtr": np.clip((3600 - game_seconds_remaining) // 900 + 1, 1, 4),
        "game_seconds_remaining": game_seconds_remaining,
        "half_seconds_remaining": game_seconds_remaining % 1800,
        "score_differential": score_diff,
        "posteam_timeouts_remaining": 3,
        "defteam_timeouts_remaining": 3,
        "shotgun": is_shotgun,
        "no_huddle": 0,
        "play_type": "pass",
        "play_type_nfl": "PASS",
        "offense_formation": np.where(is_shotgun == 1, "SHOTGUN", "SINGLEBACK"),
        "offense_personnel": "1 RB, 1 TE, 3 WR",
        "defense_man_zone_type": np.where(man == 1, "MAN_COVERAGE", "ZONE_COVERAGE"),
        "number_of_pass_rushers": np.where(blitz == 1, 5, 4),
        "defenders_in_box": np.where(blitz == 1, 7, 6),
        "epa": epa,
        # Carried only for the synthetic regression test (not a model feature):
        "_true_blitz_effect": true_blitz_effect,
        "_coach_read": coach_read,
    })
    return df


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
