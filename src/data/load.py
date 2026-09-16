"""
Data Layer — raw play-by-play ingestion.

Two sources, same downstream schema:
  * nflfastR  — real NFL play-by-play via nfl_data_py (cached to parquet).
  * synthetic — data simulated from a *known* SCM, used by the regression
                tests and for offline runs. Because the ground-truth effects
                are known, the synthetic path is how we validate that the
                OPE / policy machinery recovers the right answer.

Only the columns the feature builder needs are guaranteed here.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# Columns the feature builder consumes. Pulling a subset keeps the cache
# small and makes the pre-snap/post-snap boundary explicit.
#
# NOTE: `defense_coverage_type` drives the 6-shell coverage axis. A cache built
# before this column was requested lacks it, so bump CACHE_VERSION to force a
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

# ── FTN charting ────────────────────────────────────────────────────────────
# FTN charts every play, including runs — which nflfastR's NGS coverage columns
# do not (coverage is labeled on ~94% of dropbacks and ~3% of runs, because it
# records the coverage as PLAYED and a run never develops one). FTN is the only
# free source here for a defensive attribute observed on every snap.
#
# The grouping below is the whole point of this module's FTN support: these
# columns differ in WHEN they become knowable, and mixing the groups is a
# causal error rather than a style preference.
FTN_FIRST_SEASON = 2022

# Pre-snap, offensive presentation. The defense sees these before the call, so
# they are legitimate confounders and go into the adjustment set.
FTN_PRESNAP_OFFENSE = ["is_motion", "n_offense_backfield", "qb_location"]

# Pre-snap, but a DEFENSIVE CHOICE — a sibling of the treatment, not a cause of
# it. Carried through for the factored action space; never a state feature,
# because adjusting for part of the defense's own decision would block the very
# effect being estimated. See scm.graph.DEF_FRONT.
FTN_PRESNAP_DEFENSE = ["n_defense_box"]

# Post-snap reveals. Carried so the mediator structure can be studied (and so
# the leak guard has something to guard against), NEVER used as state.
FTN_POSTSNAP = ["is_play_action", "is_rpo", "is_screen_pass"]

FTN_COLUMNS = FTN_PRESNAP_OFFENSE + FTN_PRESNAP_DEFENSE + FTN_POSTSNAP
FTN_JOIN_KEYS = ["nflverse_game_id", "nflverse_play_id"]

# FTN fills these with a literal zero / "0" on non-scrimmage plays (kicks,
# punts, timeouts) rather than leaving them null. A handful of scrimmage rows
# carry it too. Zero defenders in the box is not a real alignment, so the
# sentinel is scrubbed to NaN instead of being read as a count.
FTN_ZERO_IS_MISSING = ["n_defense_box", "qb_location"]

FTN_CACHE_VERSION = "v1"

# The six canonical coverage shells the action space factors over. Order is
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


def load_ftn(seasons: list[int], cache_dir: str = "data/raw") -> pd.DataFrame:
    """Load FTN charting for `seasons`, caching one parquet per season.

    Seasons before `FTN_FIRST_SEASON` are skipped rather than failing — FTN
    simply does not exist for them, and `merge_ftn` reports the resulting
    coverage so the gap is visible instead of silently becoming NaN features.
    """
    import nfl_data_py as nfl

    os.makedirs(cache_dir, exist_ok=True)
    frames = []
    for yr in seasons:
        if yr < FTN_FIRST_SEASON:
            continue
        path = os.path.join(cache_dir, f"ftn_{yr}_{FTN_CACHE_VERSION}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
            continue
        df = nfl.import_ftn_data([yr])
        keep = [c for c in FTN_JOIN_KEYS + FTN_COLUMNS if c in df.columns]
        df = df[keep].copy()
        df.to_parquet(path, index=False)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=FTN_JOIN_KEYS + FTN_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def merge_ftn(pbp: pd.DataFrame, ftn: pd.DataFrame) -> pd.DataFrame:
    """Left-join FTN charting onto play-by-play and scrub its zero sentinels.

    Left, not inner: an inner join would silently delete every pre-2022 season
    instead of surfacing that FTN does not cover them. `ftn_coverage` reports
    what actually matched, and the feature builder refuses to train on a season
    whose coverage is too thin.
    """
    if ftn.empty:
        out = pbp.copy()
        for c in FTN_COLUMNS:
            out[c] = np.nan
        return out

    # Normalize both sides of the key before joining. nflfastR stores play_id as
    # a float and FTN as an int; pandas coerces them, but a silent type mismatch
    # produces an all-NaN merge rather than an error, so the cast is explicit.
    # Int64 (nullable) rather than int64 so a missing play_id cannot raise.
    ftn = ftn.drop_duplicates(subset=FTN_JOIN_KEYS).copy()
    pbp = pbp.copy()
    pbp["_key_game"] = pbp["game_id"].astype(str)
    pbp["_key_play"] = pd.to_numeric(pbp["play_id"], errors="coerce").astype("Int64")
    ftn["_key_game"] = ftn[FTN_JOIN_KEYS[0]].astype(str)
    ftn["_key_play"] = pd.to_numeric(ftn[FTN_JOIN_KEYS[1]], errors="coerce").astype("Int64")

    out = pbp.merge(ftn.drop(columns=FTN_JOIN_KEYS), how="left",
                    on=["_key_game", "_key_play"])
    out = out.drop(columns=["_key_game", "_key_play"])
    for c in FTN_ZERO_IS_MISSING:
        if c in out.columns:
            out[c] = out[c].replace({0: np.nan, "0": np.nan})
    return out.drop(columns=[k for k in FTN_JOIN_KEYS if k in out.columns])


def ftn_coverage(df: pd.DataFrame, probe: str = "n_defense_box") -> pd.DataFrame:
    """Per-season share of scrimmage plays carrying FTN charting."""
    if probe not in df.columns:
        return pd.DataFrame(columns=["season", "n_plays", "ftn_coverage"])
    scrimmage = df[df["play_type"].isin(["pass", "run"]) & df["epa"].notna()]
    return (
        scrimmage.assign(_ok=scrimmage[probe].notna())
        .groupby("season")["_ok"]
        .agg(n_plays="size", ftn_coverage="mean")
        .reset_index()
    )


# ── Synthetic SCM ───────────────────────────────────────────────────────────
# Known unobserved-confounding strength (reward units, U -> R path). The
# sensitivity analysis is validated against this: a Rosenbaum bound whose
# assumed confounding matches this should bracket the naive/true gap.
CONF_STRENGTH = 0.20


def generate_synthetic(n_plays: int = 8000, seed: int = 7,
                       seasons: list[int] | None = None) -> pd.DataFrame:
    """Simulate all-downs plays from a known 12-action structural causal model.

    Ground-truth SCM (defense perspective, reward = -EPA so higher is better):

        S  (state)      : down, ydstogo, yardline_100, score_diff, time,
                          formation, pre-snap motion / backfield count / QB
                          alignment, plus a QB identity with latent skill.
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
    # Follow the configured season window rather than hard-coding one: the
    # time-aware split holds out `holdout_season`, so a generator stuck on its
    # own years silently yields an empty holdout when the config window moves.
    seasons = list(seasons) if seasons else [2021, 2022, 2023]

    # ── State ────────────────────────────────────────────────────────────────
    down = rng.integers(1, 5, n).astype(float)          # 1..4
    ydstogo = rng.integers(1, 20, n).astype(float)       # 1..19
    yardline_100 = rng.integers(1, 99, n).astype(float)
    score_diff = rng.normal(0, 10, n).round()
    game_seconds_remaining = rng.integers(0, 3600, n).astype(float)
    is_shotgun = (rng.random(n) < (0.4 + 0.02 * ydstogo)).astype(int)

    # Pre-snap offensive presentation (the FTN-charted confounders). Motion is a
    # genuine confounder here, not decoration: the DC blitzes less against it AND
    # it helps the offense, so a model that fails to adjust for it is biased —
    # which is what makes the adjustment-set tests worth running.
    is_motion = (rng.random(n) < 0.35).astype(int)
    n_backfield = np.where(is_shotgun == 1,
                           rng.choice([0, 1, 2], n, p=[0.15, 0.75, 0.10]),
                           rng.choice([1, 2], n, p=[0.8, 0.2]))

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
    p_blitz = _sigmoid(-0.3 + 0.5 * long_yardage + 0.7 * coach_read
                       - 0.4 * is_motion)
    blitz = (rng.random(n) < p_blitz).astype(int)

    action = shell * 2 + blitz  # 0..11

    # ── Structural outcome ───────────────────────────────────────────────────
    coverage_reward = -0.03 * (shell - ideal_shell) ** 2          # peaks at ideal
    true_blitz_effect = 0.06 + 0.05 * long_yardage - 0.08 * qb_skill
    qb_term = -0.10 * qb_skill                                     # good QB hurts defense
    base = (0.02 * long_yardage - 0.01 * (yardline_100 - 50) / 50.0
            - 0.05 * is_motion)          # motion helps the offense -> confounder
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
        "season": rng.choice(seasons, n),
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
        # FTN-charted columns. n_defense_box follows the defensive CALL (a
        # sibling of the treatment, never a confounder); the offensive ones are
        # pre-snap presentation; the is_* flags are post-snap reveals carried
        # only so the leak guards have something real to reject.
        "n_defense_box": np.clip(6 + blitz + rng.integers(-1, 2, n), 4, 9),
        "is_motion": is_motion.astype(bool),
        "n_offense_backfield": n_backfield.astype(float),
        "qb_location": np.where(is_shotgun == 1, "S", "U"),
        "is_play_action": (rng.random(n) < 0.20),
        "is_rpo": (rng.random(n) < 0.08),
        "is_screen_pass": (rng.random(n) < 0.06),
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
