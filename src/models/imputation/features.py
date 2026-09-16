"""
Declared feature set for the shell imputer — the single source of truth.

This module exists because the feature list used to live in an ad-hoc analysis
script rather than in the repo. The main pipeline has had this discipline from
the start: `data.features.NUMERIC_STATE` declares the state once,
`scm.graph.CONFOUNDER_COLUMNS` mirrors it, and
`scm.identify.assert_adjustment_consistency` raises when the two drift apart.
The imputer had no equivalent, so two callers could train on different features
and nothing downstream would notice. `assert_imputer_features` closes that gap.

Feature provenance
------------------
    game situation (5)   nflfastR pbp: down, ydstogo, yardline_100, score_diff, qtr
    offensive pre-snap   NGS personnel -> num_rb/num_te/num_wr; NGS formation;
      presentation (6)   FTN is_motion, n_offense_backfield, qb_location
    defensive pre-snap   FTN n_defense_box
      (1)
    team tendency (6)    derived: leak-free expanding share of each shell by
                         defteam, shifted one play (see add_team_shell_tendency)

Why n_defense_box is an input HERE but barred from the adjustment set
--------------------------------------------------------------------
`scm.identify.assert_adjustment_consistency` rejects n_defense_box as a state
feature, because in the outcome and OPE models it is a SIBLING of the treatment
— adjusting for part of the defense's own call blocks a slice of the effect
being estimated. The imputer is doing something different: it is not estimating
an effect, it is measuring a latent label, and box count is a legitimate PROXY
for that label (you cannot play a two-high shell with eight in the box).

Different role, different rule — but the consequence travels: the imputed shell
and the box count are correlated by construction, which matters once both become
axes of the factored action space.

Known limitation carried by these features
------------------------------------------
The tendency columns are computed from CHARTED plays, which are dropbacks. Using
them to impute run plays assumes a team's run-down shell mix resembles its
dropback mix. That is the transportability assumption, and it carries the
largest single share of this model's signal — it is what moves entropy
reduction from ~9% to ~19.6%.
"""

from __future__ import annotations

import pandas as pd

from ...data.load import COVERAGE_SHELLS, N_SHELLS

# Pre-snap only. Anything realized at or after the snap is barred — including
# the reward: the imputed variable IS the treatment whose effect on the reward
# is estimated downstream, so conditioning the imputer on the reward would let
# it write in the very association the study is trying to measure.
FORBIDDEN_FEATURES = {
    "epa", "reward", "action", "action_label", "shell", "play_type",
    "is_play_action", "is_rpo", "is_screen_pass", "off_playcall",
}

# Game situation — nflfastR play-by-play.
IMPUTER_SITUATION = ["down", "ydstogo", "yardline_100", "score_diff", "qtr"]

# Offensive pre-snap presentation — what the defense sees before calling.
IMPUTER_OFFENSE = ["num_rb", "num_te", "num_wr", "is_motion", "n_offense_backfield"]

# Defensive pre-snap — a proxy for the latent shell, not a confounder. See the
# module docstring for why this is admissible here and not in the SCM.
IMPUTER_DEFENSE = ["n_defense_box"]

IMPUTER_CATEGORICAL = ["formation", "qb_location"]

# Defaults applied when a source column is missing or unlabeled, chosen to be
# modal rather than convenient: 6 in the box and 1 back are the league's most
# common alignments, so a missing value lands on the unremarkable case.
_NUMERIC_DEFAULTS = {"n_offense_backfield": 1.0, "n_defense_box": 6.0,
                     "score_diff": 0.0}
_PERSONNEL_DEFAULTS = {"num_rb": 1.0, "num_te": 1.0, "num_wr": 3.0}


def tendency_columns(n_shells: int = N_SHELLS) -> list[str]:
    """Names of the per-team shell-tendency columns."""
    return [f"shell_tend_{k}" for k in range(n_shells)]


def add_team_shell_tendency(
    df: pd.DataFrame, shell_col: str, charted_col: str, team_col: str,
    n_shells: int = N_SHELLS,
    time_cols: tuple[str, ...] = ("season", "week", "game_id", "play_id"),
) -> pd.DataFrame:
    """Leak-free per-team shell mix: expanding share of each shell, shifted one play.

    The single most useful feature available — it is what lifts entropy
    reduction from ~9% to ~19.6% — and the easiest to leak with: a plain group
    mean would let a play's own shell inform its own prediction.
    Expanding-then-shift uses only that team's PRIOR snaps, matching the
    discipline in `data.features._add_history_features`.
    """
    out = df.sort_values([c for c in time_cols if c in df.columns]).copy()
    charted = out[charted_col].astype(bool)
    for k in range(n_shells):
        hit = ((out[shell_col] == k) & charted).astype(float)
        out[f"shell_tend_{k}"] = (
            hit.groupby(out[team_col]).transform(lambda x: x.expanding().mean().shift(1))
            .fillna(1.0 / n_shells)
        )
    return out


def imputer_features(n_shells: int = N_SHELLS) -> tuple[list[str], list[str]]:
    """`(numeric, categorical)` feature names the imputer is fit on."""
    numeric = (list(IMPUTER_SITUATION) + list(IMPUTER_OFFENSE)
               + list(IMPUTER_DEFENSE) + tendency_columns(n_shells))
    return numeric, list(IMPUTER_CATEGORICAL)


def build_imputation_frame(pbp: pd.DataFrame, n_shells: int = N_SHELLS) -> pd.DataFrame:
    """Turn FTN-merged play-by-play into the imputer's feature frame.

    Adds `shell` (0..n_shells-1, or -1 when uncharted) and `charted`, so callers
    can split labeled from unlabeled without re-deriving the taxonomy. Expects
    `merge_ftn` to have run already — the FTN columns are required, not optional.
    """
    df = pbp.copy()
    shell_txt = df["defense_coverage_type"].astype("string").str.upper()
    df["charted"] = shell_txt.isin(COVERAGE_SHELLS)
    df["shell"] = (shell_txt.map({s: i for i, s in enumerate(COVERAGE_SHELLS)})
                   .fillna(-1).astype(int))

    personnel = df.get("offense_personnel", pd.Series("", index=df.index)).fillna("")
    for col, pattern, default in [("num_rb", "RB", 1.0), ("num_te", "TE", 1.0),
                                  ("num_wr", "WR", 3.0)]:
        df[col] = (personnel.str.extract(rf"(\d+)\s*{pattern}")
                   .astype(float).fillna(default))

    df["formation"] = (df.get("offense_formation", pd.Series("UNK", index=df.index))
                       .fillna("UNK").astype(str).str.upper())
    df["qb_location"] = (df.get("qb_location", pd.Series("UNK", index=df.index))
                         .astype("string").fillna("UNK").replace("0", "UNK").astype(str))
    df["is_motion"] = (df.get("is_motion", pd.Series(False, index=df.index))
                       .astype("boolean").fillna(False).astype(int))
    df["score_diff"] = pd.to_numeric(df.get("score_differential"), errors="coerce")

    for col in ["down", "ydstogo", "yardline_100", "qtr",
                "n_defense_box", "n_offense_backfield"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    for col, default in {**_NUMERIC_DEFAULTS, **_PERSONNEL_DEFAULTS}.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    return add_team_shell_tendency(df, "shell", "charted", "defteam", n_shells)


def assert_imputer_features(df: pd.DataFrame, n_shells: int = N_SHELLS) -> list[str]:
    """Confirm every declared feature exists, and that none is post-snap.

    Mirrors `scm.identify.assert_adjustment_consistency`: the declaration and
    the builder must agree, and drift should be an error rather than a silently
    different model.
    """
    numeric, categorical = imputer_features(n_shells)
    declared = numeric + categorical
    missing = [c for c in declared if c not in df.columns]
    if missing:
        raise ValueError(
            f"Declared imputer features absent from the frame: {missing}. "
            "build_imputation_frame and imputer_features disagree — did the "
            "FTN merge run?"
        )
    leaked = sorted(set(declared) & FORBIDDEN_FEATURES)
    if leaked:
        raise ValueError(
            f"Post-snap or outcome column(s) declared as imputer features: {leaked}."
        )
    return declared
