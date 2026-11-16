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
    # Defensive personnel on the field at the snap ("3 CB, 2 DE, ..."). Charted
    # on ~100% of snaps, runs included, and chosen before the snap — the source
    # of the personnel-package axis of the level-1 action.
    "defense_personnel",
    "passer_player_id", "epa",
]

# Bump when PBP_COLUMNS changes so stale caches are transparently rebuilt.
CACHE_VERSION = "v3"

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

# ── Post-snap columns: one declaration, enforced everywhere ─────────────────
# Everything below is knowable only at or after the snap. Some are loaded because
# they are the ACTION LABEL (coverage, rushers) or the OUTCOME (epa); others are
# never loaded but sit one line away in nflverse's participation data and look
# like innocent pre-snap context. None may ever be a state feature or an
# imputation input — `scm.identify.assert_adjustment_consistency` and
# `models.imputation.features.FORBIDDEN_FEATURES` both enforce this set.
POSTSNAP_COLUMNS = frozenset({
    # participation / NGS, measured as the play unfolds
    "time_to_throw", "was_pressure", "route",
    # post-snap action labels: the coverage as PLAYED, the rushers who ACTUALLY rushed
    "defense_coverage_type", "defense_man_zone_type", "number_of_pass_rushers",
    # FTN reveals
    *FTN_POSTSNAP,
    # the outcome, and the offense's run/pass choice
    "epa", "play_type", "play_type_nfl",
})
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

    frames = _cached_seasons(seasons, cache_dir, "pbp", CACHE_VERSION, PBP_COLUMNS,
                             lambda yr: nfl.import_pbp_data([yr], downcast=False, cache=False))
    return pd.concat(frames, ignore_index=True)


def load_ftn(seasons: list[int], cache_dir: str = "data/raw") -> pd.DataFrame:
    """Load FTN charting for `seasons`, caching one parquet per season.

    Seasons before `FTN_FIRST_SEASON` are skipped rather than failing — FTN
    simply does not exist for them, and `merge_ftn` reports the resulting
    coverage so the gap is visible instead of silently becoming NaN features.
    """
    import nfl_data_py as nfl

    frames = _cached_seasons([yr for yr in seasons if yr >= FTN_FIRST_SEASON], cache_dir,
                             "ftn", FTN_CACHE_VERSION, FTN_JOIN_KEYS + FTN_COLUMNS,
                             lambda yr: nfl.import_ftn_data([yr]))
    if not frames:
        return pd.DataFrame(columns=FTN_JOIN_KEYS + FTN_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _cached_seasons(seasons, cache_dir: str, prefix: str, version: str,
                    columns: list[str], fetch) -> list[pd.DataFrame]:
    """One frame per season, read from `{prefix}_{season}_{version}.parquet` in
    `cache_dir`, or fetched with `fetch(season)`, cut to `columns` and cached
    there. The network is hit only for seasons not cached under `version`."""
    os.makedirs(cache_dir, exist_ok=True)
    frames = []
    for yr in seasons:
        path = os.path.join(cache_dir, f"{prefix}_{yr}_{version}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
            continue
        df = fetch(yr)
        df = df[[c for c in columns if c in df.columns]].copy()
        df.to_parquet(path, index=False)
        frames.append(df)
    return frames


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


# ── Synthetic SCM: personnel package x front, every snap ────────────────────
# Heaviness of an alignment on a 0..5 scale: base personnel with 8+ in the box is
# 5, dime personnel with <=5 in the box is 0. The known structure:
#
#   * the offense runs more in short-yardage / under-center / run-leaning states;
#   * DETERRENCE — a heavier alignment lowers the chance the offense runs at all;
#   * a heavy alignment is good against the run and bad against the pass;
#   * the DC's hidden read favours heavier calls AND improves the outcome, which
#     is the confounding the sensitivity analysis exists to bound.
#
# So the causal-best alignment rises with how run-likely the state is, but not
# one-for-one: loading up also pushes the offense toward the pass it is weak
# against. The value of a call is the mixture over the offense's response, which
# is exactly the estimand the every-snap pipeline targets.
_SYN_PERSONNEL = {
    0: "2 CB, 2 DE, 2 DT, 1 FS, 3 ILB, 1 SS",   # base: 4 DB
    1: "3 CB, 2 DE, 2 DT, 1 FS, 2 ILB, 1 SS",   # nickel: 5 DB
    2: "4 CB, 2 DE, 2 DT, 1 FS, 1 ILB, 1 SS",   # dime+: 6 DB
}
_ALIGN_RUN_GAIN = 0.12      # reward per unit heaviness when the offense runs
_ALIGN_PASS_COST = -0.08    # ... and when it passes
_DETERRENCE = 0.30          # logit drop in P(run) per unit heaviness


def alignment_heaviness(actions: np.ndarray) -> np.ndarray:
    """0 (dime, light box) .. 5 (base, 8+ box) for level-1 action ids."""
    actions = np.asarray(actions)
    package, front = actions // 4, actions % 4
    return (2 - package) + front


def _alignment_components(heavy: np.ndarray, p_run_logit: np.ndarray):
    """P(run | do(alignment)) and the branch rewards — shared by the generator
    and the ground-truth value so the two can never disagree."""
    centred = heavy - 2.5
    p_run = _sigmoid(p_run_logit - _DETERRENCE * centred)
    return p_run, _ALIGN_RUN_GAIN * centred, _ALIGN_PASS_COST * centred


# ── Level 2: the coverage paired with the alignment, on passes ──────────────
# Safety structure on a 0..2 scale (Cover-0, single-high, two-high). The known
# structure, active only with `coverage_effects=True`:
#
#   * the ideal structure deepens with distance to go (`_ideal_structure`);
#   * a blitz pays more from a heavier alignment (more rushers already near the
#     line) and less on long yardage;
#   * the coordinator plays lighter alignments with deeper shells, and his
#     hidden read pushes both the shell and the blitz — confounding level 2 the
#     way it confounds level 1;
#   * coverage acts on passes only (the estimand's own assumption).
_COV_MISS_COST = 0.05       # reward lost per (structure - ideal)^2
_FAMILY_SHELLS = {0: ["COVER_0"], 1: ["COVER_1", "COVER_3"],
                  2: ["COVER_2", "COVER_4", "COVER_6"]}


def _ideal_structure(ydstogo: np.ndarray) -> np.ndarray:
    return 2.0 * _sigmoid((np.asarray(ydstogo, dtype=float) - 7.0) / 3.0)


def coverage_effect(ydstogo, heavy, family, blitz) -> np.ndarray:
    """Reward from the coverage on a pass, given the alignment's heaviness —
    shared by the generator and the ground truth."""
    ydstogo = np.asarray(ydstogo, dtype=float)
    heavy = np.asarray(heavy, dtype=float)
    miss = -_COV_MISS_COST * (np.asarray(family) - _ideal_structure(ydstogo)) ** 2
    blitz_gain = 0.03 + 0.04 * (heavy - 2.5) / 2.5 - 0.04 * (ydstogo - 7.0) / 5.0
    return miss + np.asarray(blitz) * blitz_gain


def generate_synthetic_alignment(
    n_plays: int = 12000, seed: int = 11, seasons: list[int] | None = None,
    n_teams: int = 8, conf_strength: float = CONF_STRENGTH,
    coverage_effects: bool = False, effect_heterogeneity: float = 0.0,
) -> pd.DataFrame:
    """Simulate every-snap play-by-play from a known alignment SCM.

    Emits nflfastR/FTN-shaped columns so `build_dataset` consumes it unchanged,
    including the quirk that `passer_player_id` exists only on pass plays — the
    pipeline has to handle it here exactly as it does on real data.
    `conf_strength = 0` removes the hidden read's effect on the outcome, so the
    OPE machinery should then recover the TRUE value of a policy, not just the
    empirical mean.

    `coverage_effects=True` adds the level-2 structure above: the coverage and
    the pressure are called by a known behavior policy and change the reward on
    passes. It draws from its own random stream, so everything else — and the
    whole frame when it is off — is identical to the level-1 SCM.
    `true_alignment_value` ignores coverage and is exact only when it is off;
    `true_dropback_value` is the level-2 truth.

    `effect_heterogeneity > 0` makes the alignment's effect differ by
    environment: each defense-season gets its own extra reward per unit of
    heaviness, drawn N(0, effect_heterogeneity), on every snap. Also its own
    random stream; at 0 the frame is unchanged.
    """
    rng = np.random.default_rng(seed)
    seasons = list(seasons) if seasons else [2022, 2023, 2024, 2025]
    n = n_plays
    plays_per_game = 60
    n_games = n // plays_per_game + 1

    # Game-level: season, week, and which defense faces which offense.
    g_season = rng.choice(seasons, n_games)
    g_week = rng.integers(1, 18, n_games)
    g_def = rng.integers(0, n_teams, n_games)
    g_off = (g_def + rng.integers(1, n_teams, n_games)) % n_teams
    game = np.repeat(np.arange(n_games), plays_per_game)[:n]

    team_run_lean = rng.normal(0, 0.4, n_teams)        # offense identity
    team_qb_skill = rng.normal(0, 1, n_teams)          # offense's QB
    team_heavy_bias = rng.normal(0, 0.6, n_teams)      # defense's scheme
    defteam, posteam = g_def[game], g_off[game]

    # ── Pre-snap state ───────────────────────────────────────────────────────
    down = rng.integers(1, 5, n).astype(float)
    ydstogo = rng.integers(1, 20, n).astype(float)
    yardline_100 = rng.integers(1, 99, n).astype(float)
    score_diff = rng.normal(0, 10, n).round()
    game_seconds_remaining = rng.integers(0, 3600, n).astype(float)
    shotgun = (rng.random(n) < np.clip(0.25 + 0.035 * ydstogo, 0, 0.95)).astype(int)
    is_motion = (rng.random(n) < 0.4).astype(int)
    n_backfield = np.where(shotgun == 1, 1.0, rng.choice([1.0, 2.0], n, p=[0.6, 0.4]))

    p_run_logit = (0.6 - 0.22 * (ydstogo - 5) + 0.9 * (1 - shotgun)
                   + team_run_lean[posteam])
    coach_read = rng.normal(0, 1, n)                   # hidden, never recorded

    # ── Behavior: the DC's alignment call ────────────────────────────────────
    p = np.exp(_alignment_logits(p_run_logit, team_heavy_bias[defteam], coach_read))
    p /= p.sum(axis=1, keepdims=True)
    action = (rng.random(n)[:, None] > np.cumsum(p, axis=1)).sum(axis=1).clip(0, 11)
    heavy = alignment_heaviness(action)

    # ── The offense responds to the look, then the play resolves ─────────────
    p_run, r_run, r_pass = _alignment_components(heavy, p_run_logit)
    is_run = rng.random(n) < p_run
    qb_term = -0.10 * team_qb_skill[posteam]
    base = 0.02 * (ydstogo - 7) / 5.0 - 0.01 * (yardline_100 - 50) / 50.0
    reward = (np.where(is_run, r_run, r_pass) + base + qb_term
              + conf_strength * coach_read + rng.normal(0, 0.6, n))

    package, front = action // 4, action % 4
    box = np.select([front == 0, front == 1, front == 2],
                    [rng.choice([4, 5], n), 6, 7], rng.choice([8, 9], n))
    teams = np.array([f"T{i}" for i in range(n_teams)])
    shells = np.array(COVERAGE_SHELLS)[rng.integers(0, N_SHELLS, n)]
    rushers = rng.choice([4.0, 5.0], n)

    truth = {}
    if coverage_effects:
        rng2 = np.random.default_rng(seed + 7919)
        team_deep_bias = rng2.normal(0, 0.3, n_teams)
        fam_all = np.arange(3)[None, :]
        fam_logits = (-0.8 * (fam_all - _ideal_structure(ydstogo)[:, None]
                              - team_deep_bias[defteam][:, None]) ** 2
                      + 0.35 * (fam_all - 1) * coach_read[:, None]
                      + 0.40 * (fam_all - 1) * ((2.5 - heavy) / 2.5)[:, None])
        fam_p = np.exp(fam_logits - fam_logits.max(axis=1, keepdims=True))
        fam_p /= fam_p.sum(axis=1, keepdims=True)
        family = (rng2.random(n)[:, None] > np.cumsum(fam_p, axis=1)).sum(axis=1).clip(0, 2)
        p_blitz = _sigmoid(-1.0 + 0.35 * (heavy - 2.5) - 0.15 * (ydstogo - 7) / 3
                           + 0.30 * coach_read)
        blitz = (rng2.random(n) < p_blitz).astype(int)
        pick = rng2.random(n)
        shells = np.array([_FAMILY_SHELLS[f][int(u * len(_FAMILY_SHELLS[f]))]
                           for f, u in zip(family, pick)])
        rushers = np.where(blitz == 1, 5.0, 4.0)
        reward = reward + np.where(is_run, 0.0, coverage_effect(ydstogo, heavy, family, blitz))
        truth = {"_heavy": heavy}
    if effect_heterogeneity > 0:
        rng3 = np.random.default_rng(seed + 104729)
        env_slope = rng3.normal(0, effect_heterogeneity, (n_teams, len(seasons)))
        slope = env_slope[defteam, np.searchsorted(np.sort(seasons), g_season[game])]
        reward = reward + slope * (heavy - 2.5)
        truth["_env_slope"] = slope

    return pd.DataFrame({
        "game_id": np.array([f"G{g}" for g in game]),
        "play_id": np.arange(n),
        "season": g_season[game], "week": g_week[game],
        "defteam": teams[defteam], "posteam": teams[posteam],
        "down": down, "ydstogo": ydstogo, "yardline_100": yardline_100,
        "qtr": np.clip((3600 - game_seconds_remaining) // 900 + 1, 1, 4),
        "game_seconds_remaining": game_seconds_remaining,
        "half_seconds_remaining": game_seconds_remaining % 1800,
        "score_differential": score_diff,
        "posteam_timeouts_remaining": 3, "defteam_timeouts_remaining": 3,
        "shotgun": shotgun, "no_huddle": 0,
        "play_type": np.where(is_run, "run", "pass"),
        "play_type_nfl": np.where(is_run, "RUSH", "PASS"),
        "offense_formation": np.where(shotgun == 1, "SHOTGUN", "UNDER_CENTER"),
        "offense_personnel": "1 RB, 1 TE, 3 WR",
        "defense_personnel": np.vectorize(_SYN_PERSONNEL.get)(package),
        "defense_coverage_type": np.where(is_run, None, shells),
        "defense_man_zone_type": None,
        "number_of_pass_rushers": np.where(is_run, np.nan, rushers),
        "defenders_in_box": box,
        # Only on passes, as in nflfastR.
        "passer_player_id": np.where(is_run, None, np.array([f"QB_{t}" for t in teams[posteam]])),
        "epa": -reward,
        "n_defense_box": box.astype(float),
        "is_motion": is_motion.astype(bool),
        "n_offense_backfield": n_backfield,
        "qb_location": np.where(shotgun == 1, "S", "U"),
        "is_play_action": False, "is_rpo": False, "is_screen_pass": False,
        # Ground truth (any "_" column survives build_dataset; never a feature).
        "_coach_read": coach_read,
        "_p_run_logit": p_run_logit,
        "_heavy_bias": team_heavy_bias[defteam],
        "_qb_term": qb_term,
        "_base": base,
        **truth,
    })


def true_dropback_value(df: pd.DataFrame, joint_probs: np.ndarray,
                        level2_family: np.ndarray) -> float:
    """Ground-truth per-dropback value of a factored policy (coverage_effects SCM).

    The level-2 estimand: on charted dropbacks, intervene on the alignment and
    the coverage while the play stays a pass. `joint_probs` is (n, 12 * K2)
    over joint ids `a1 * K2 + a2`; `level2_family[a2]` is the safety structure
    (0..2) of level-2 call a2, whose blitz bit is `a2 % 2`. Policies differ in
    truth only through the pass branch and the coverage term; the read sits at
    its mean.
    """
    k2 = len(level2_family)
    a1, a2 = np.divmod(np.arange(12 * k2), k2)
    heavy = alignment_heaviness(a1)[None, :]
    ytg = df["ydstogo"].to_numpy()[:, None]
    value = (_ALIGN_PASS_COST * (heavy - 2.5)
             + coverage_effect(ytg, heavy, np.asarray(level2_family)[a2][None, :], (a2 % 2)[None, :]))
    return float(((joint_probs * value).sum(axis=1)
                  + df["_qb_term"].to_numpy() + df["_base"].to_numpy()).mean())


def true_alignment_propensity(df: pd.DataFrame, n_nodes: int = 40) -> np.ndarray:
    """Ground-truth pi_b(a | s) on `generate_synthetic_alignment` data, (n, 12):
    the coordinator's call probability given the OBSERVED state, i.e. his
    softmax averaged over the hidden read ~ N(0, 1) (Gauss-Hermite). This is
    what a behavior model fit on the logged state can at best recover."""
    x, w = np.polynomial.hermite_e.hermegauss(n_nodes)
    w = w / w.sum()
    p_run_logit, bias = df["_p_run_logit"].to_numpy(), df["_heavy_bias"].to_numpy()
    out = np.zeros((len(df), 12))
    for xi, wi in zip(x, w):
        e = np.exp(_alignment_logits(p_run_logit, bias, xi))
        out += wi * e / e.sum(axis=1, keepdims=True)
    return out


def _alignment_logits(p_run_logit: np.ndarray, heavy_bias: np.ndarray, read) -> np.ndarray:
    """The coordinator's logits over the 12 alignments, (n, 12), max-subtracted:
    peaked at the heaviness the run threat calls for, shifted by his defense's
    lean and by the read (per snap, or one value for every snap).
    `generate_synthetic_alignment` draws from their softmax;
    `true_alignment_propensity` averages it over the read."""
    heavy = alignment_heaviness(np.arange(12))[None, :]
    ideal = 5.0 * _sigmoid(p_run_logit)
    z = (-0.6 * (heavy - ideal[:, None] - heavy_bias[:, None]) ** 2
         + 0.30 * np.asarray(read)[..., None] * (heavy - 2.5) / 2.5)
    return z - z.max(axis=1, keepdims=True)


def true_alignment_value(df: pd.DataFrame, actions: np.ndarray) -> float:
    """Ground-truth value of a level-1 policy on `generate_synthetic_alignment` data.

    Under do(alignment) the hidden read sits at its mean (0), and the offense's
    run/pass choice RESPONDS to the intervened alignment — so the value is the
    mixture over that response, including deterrence. `actions` is one level-1
    action id per row of `df`.
    """
    q = true_alignment_q(df)
    return float(q[np.arange(len(df)), np.asarray(actions)].mean())


def true_alignment_q(df: pd.DataFrame, n_actions: int = 12) -> np.ndarray:
    """Ground-truth Q(s, a) = E[R | s, do(a)] for every level-1 call, (n, n_actions)."""
    p_logit = df["_p_run_logit"].to_numpy()
    rest = df["_qb_term"].to_numpy() + df["_base"].to_numpy()
    slope = df["_env_slope"].to_numpy() if "_env_slope" in df.columns else 0.0
    out = np.empty((len(df), n_actions))
    for a in range(n_actions):
        heavy = alignment_heaviness(np.full(len(df), a))
        p_run, r_run, r_pass = _alignment_components(heavy, p_logit)
        out[:, a] = p_run * r_run + (1 - p_run) * r_pass + slope * (heavy - 2.5) + rest
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
