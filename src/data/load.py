"""
Data Layer — raw play-by-play ingestion.

Two sources, same downstream schema:
  * nflfastR  — real NFL play-by-play via nfl_data_py (cached to parquet).
  * synthetic — data simulated from a *known* SCM, used by the regression
                tests and for offline runs. Because the ground-truth effects
                are known, the synthetic path is how we validate that the
                OPE / policy machinery recovers the right answer.

Only the columns the V2 feature builder needs are guaranteed here.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# Columns the V2 feature builder consumes. Pulling a subset keeps the cache
# small and makes the pre-snap/post-snap boundary explicit.
#
# NOTE (V2): `defense_coverage_type` is new vs V1 — it drives the 6-shell
# coverage axis. Caches built under V1 lack it; bump CACHE_VERSION to force a
# re-pull rather than silently reading a stale, column-short parquet.
PBP_COLUMNS = [
    "game_id", "play_id", "season", "week", "defteam", "posteam",
    "down", "ydstogo", "yardline_100", "qtr",
    "game_seconds_remaining", "half_seconds_remaining",
    "score_differential", "posteam_timeouts_remaining", "defteam_timeouts_remaining",
    "shotgun", "no_huddle", "play_type", "play_type_nfl",
    "offense_formation", "offense_personnel",
    "defense_coverage_type", "defense_man_zone_type",
    "number_of_pass_rushers", "defenders_in_box",
    "passer_player_id", "epa",
]

# Bump when PBP_COLUMNS changes so stale caches are transparently rebuilt.
CACHE_VERSION = "v2"

# The six canonical coverage shells the V2 action space factors over. Order is
# roughly increasing zone depth (man/press -> deep zone), which the synthetic
# SCM and the "ideal shell rises with distance" story both rely on.
COVERAGE_SHELLS = ["COVER_0", "COVER_1", "COVER_2", "COVER_3", "COVER_4", "COVER_6"]
N_SHELLS = len(COVERAGE_SHELLS)


# ── nflfastR ────────────────────────────────────────────────────────────────
def load_pbp(seasons: list[int], cache_dir: str = "data/raw") -> pd.DataFrame:
    """Load nflfastR play-by-play for `seasons`, caching one parquet per season.

    Network is only hit for seasons not already cached under the current
    CACHE_VERSION. Bumping CACHE_VERSION invalidates old caches automatically.
    """
    import nfl_data_py as nfl

    os.makedirs(cache_dir, exist_ok=True)
    frames = []
    for yr in seasons:
        path = os.path.join(cache_dir, f"pbp_{yr}_{CACHE_VERSION}.parquet")
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
# Known unobserved-confounding strength (reward units, U -> R path). The
# sensitivity analysis is validated against this: a Rosenbaum bound whose
# assumed confounding matches this should bracket the naive/true gap.
CONF_STRENGTH = 0.20


def generate_synthetic(n_plays: int = 8000, seed: int = 7) -> pd.DataFrame:
    """Simulate all-downs plays from a known 12-action structural causal model.

    Ground-truth SCM (defense perspective, reward = -EPA so higher is better):

        S  (state)      : down, ydstogo, yardline_100, score_diff, time,
                          formation, plus a QB identity with latent skill.
        U  (confounder) : `coach_read` — a hidden pre-snap read that drives BOTH
                          the DC's call AND the outcome. This is the selection
                          effect naive RL conflates with the causal effect; the
                          sensitivity analysis is what bounds it.
        A  (action)     : coverage shell c in {C0,C1,C2,C3,C4,C6} x pressure
                          b in {no-blitz, blitz}, encoded action = 2*c + b (0..11).
        R  (reward)     : structural function of S, A, U with a KNOWN best action.

    Known best response (what the Q-model / policy must recover):
      * coverage: an `ideal_shell` that RISES with distance-to-go (tight man on
        short, deep zone on long); mismatch is penalized quadratically.
      * pressure: blitz helps more on long downs and HURTS against good QBs, so
        the causal-best pressure flips sign with (distance, QB skill).

    Confounding: the DC blitzes more and plays tighter coverage when `coach_read`
    is favorable, and a favorable read also lowers EPA directly (U -> R). So the
    observational mean overstates the value of the calls made under good reads.

    A guaranteed positivity hole is baked in: Cover-0 is only ever called on
    3rd/4th-and-short, so (1st down x Cover-0) is a structurally empty stratum
    the positivity check must flag.
    """
    rng = np.random.default_rng(seed)
    n = n_plays

    # ── State ────────────────────────────────────────────────────────────────
    down = rng.integers(1, 5, n).astype(float)          # 1..4
    ydstogo = rng.integers(1, 20, n).astype(float)       # 1..19
    yardline_100 = rng.integers(1, 99, n).astype(float)
    score_diff = rng.normal(0, 10, n).round()
    game_seconds_remaining = rng.integers(0, 3600, n).astype(float)
    is_shotgun = (rng.random(n) < (0.4 + 0.02 * ydstogo)).astype(int)

    # QB identity with latent skill — an observed-proxy cause of the outcome
    # (good QB -> higher offensive EPA -> lower defensive reward). Proxied
    # downstream by a leak-free expanding `qb_avg_epa`.
    n_qbs = 12
    qb_ids = rng.integers(0, n_qbs, n)
    qb_skill = rng.normal(0, 1, n_qbs)[qb_ids]

    # Hidden confounder: the coach's pre-snap read (favorable for defense > 0).
    coach_read = rng.normal(0, 1, n)

    long_yardage = (ydstogo - 7) / 5.0
    # ideal coverage shell rises with distance: short -> 0 (tight), long -> 5 (deep).
    ideal_shell = np.clip(np.round((ydstogo - 2) / 3.0), 0, 5).astype(int)

    # ── Behavior policy pi_b(A | S, U) — confounded ──────────────────────────
    # Coverage: softmax preferring shells near `ideal_shell`, tilted toward
    # tighter coverage (low c) when the read is favorable (the confounding).
    c_idx = np.arange(N_SHELLS)
    d2 = (c_idx[None, :] - ideal_shell[:, None]) ** 2
    logits = -0.7 * d2 - 0.30 * coach_read[:, None] * (c_idx[None, :] - 2.5)
    # Structural zero: Cover-0 (all-out man) only on 3rd/4th-and-short.
    c0_allowed = (down >= 3) & (ydstogo <= 4)
    logits[~c0_allowed, 0] = -1e9
    logits -= logits.max(axis=1, keepdims=True)
    p_shell = np.exp(logits)
    p_shell /= p_shell.sum(axis=1, keepdims=True)
    shell = (rng.random(n)[:, None] > np.cumsum(p_shell, axis=1)).sum(axis=1)
    shell = np.clip(shell, 0, N_SHELLS - 1)

    # Pressure: blitz propensity rises with distance (sensible) AND with a
    # favorable read (the confounding).
    p_blitz = _sigmoid(-0.3 + 0.5 * long_yardage + 0.7 * coach_read)
    blitz = (rng.random(n) < p_blitz).astype(int)

    action = shell * 2 + blitz  # 0..11

    # ── Structural outcome ───────────────────────────────────────────────────
    coverage_reward = -0.03 * (shell - ideal_shell) ** 2          # peaks at ideal
    true_blitz_effect = 0.06 + 0.05 * long_yardage - 0.08 * qb_skill
    qb_term = -0.10 * qb_skill                                     # good QB hurts defense
    base = 0.02 * long_yardage - 0.01 * (yardline_100 - 50) / 50.0
    reward = (
        coverage_reward
        + true_blitz_effect * blitz
        + qb_term
        + base
        + CONF_STRENGTH * coach_read                              # confounding into R
        + rng.normal(0, 0.6, n)
    )
    epa = -reward

    # Map to nflfastR-like columns the feature builder understands.
    df = pd.DataFrame({
        "game_id": np.repeat(np.arange(n // 40 + 1), 40)[:n],
        "play_id": np.arange(n),
        "season": rng.choice([2021, 2022, 2023], n),
        "week": rng.integers(1, 18, n),
        "defteam": "SYN",
        "posteam": "OPP",
        "down": down,
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
        "defense_coverage_type": np.array(COVERAGE_SHELLS)[shell],
        "defense_man_zone_type": np.where(shell <= 1, "MAN_COVERAGE", "ZONE_COVERAGE"),
        "number_of_pass_rushers": np.where(blitz == 1, 5, 4),
        "defenders_in_box": np.where(blitz == 1, 7, 6),
        "passer_player_id": np.array([f"QB{i}" for i in qb_ids]),
        "epa": epa,
        # Ground-truth columns — carried through build_dataset for the tests
        # (any "_"-prefixed column is preserved), never used as a model feature.
        "_coach_read": coach_read,
        "_ideal_shell": ideal_shell,
        "_true_blitz_effect": true_blitz_effect,
        "_qb_term": qb_term,
        "_base": base,
    })
    return df


def true_policy_value(df: pd.DataFrame, actions: np.ndarray) -> float:
    """Ground-truth value V(pi) of a deterministic policy on synthetic data.

    Under do(A=a) the U->A edge is cut, so the hidden `coach_read` sits at its
    marginal mean (0) and the noise averages out. The remaining reward is a
    known function of the stored structural components — the exact target OPE
    should recover. `actions` is one action id (0..11) per row of `df`.
    """
    shell = actions // 2
    blitz = actions % 2
    coverage_reward = -0.03 * (shell - df["_ideal_shell"].to_numpy()) ** 2
    blitz_reward = df["_true_blitz_effect"].to_numpy() * blitz
    return float(
        (coverage_reward + blitz_reward
         + df["_qb_term"].to_numpy() + df["_base"].to_numpy()).mean()
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
