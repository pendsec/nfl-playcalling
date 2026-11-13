"""The judge's pi_b — the propensities a holdout OPE estimate weights by.

The policy's pi_b is fit on the training seasons and stays that way: the policy
may learn nothing from the season it is judged on. The judge's pi_b answers a
different question — how did the coordinator actually mix his calls in the
scored season? — and a model fit on earlier seasons cannot know that when the
mix drifts. The nickel/8+ share went 2.1% -> 4.2% -> 1.5% -> 1.0% across
2022-25, so a train-fit pi_b weights 2025's nickel/8+ snaps at ~0.33 of what
they should be, and every calibrator fit on past seasons misses the same way
(`experiments/behavior_calibration/`).

`ope.judge_propensity`:
  * `train` — the policy's own pi_b, fit on the training seasons;
  * `pooled_crossfit` — fit on the training seasons plus the scored season's
    other games, with `season` in the state, each scored game's snaps from a
    fold that never saw that game (`ope.judge_folds`, default 5). It uses the
    scored season's states and calls, never its rewards. On real 2025 it cut
    log loss by 0.108 +/- 0.005 and moved nickel/8+'s weight balance from 0.33
    to 0.92; on the synthetic league (no drift) it cost nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .propensity import fit_behavior_model
from ...schemas.behavior import BehaviorModel
from ...schemas.dataset import Dataset

JUDGES = ("train", "pooled_crossfit")


@dataclass
class CrossFitJudge:
    """pi_b on one scored season, each game's snaps from the fold that held it out.

    Scores only the games it was built for: anything else has no out-of-fold
    model, and silently falling back to an in-sample one would let a snap's own
    call inform its propensity.
    """
    models: list[BehaviorModel]
    game_fold: dict
    n_actions: int

    def fold_of(self, df: pd.DataFrame) -> np.ndarray:
        fold = df["game_id"].map(self.game_fold)
        if fold.isna().any():
            missing = df.loc[fold.isna(), "game_id"].unique()[:3]
            raise ValueError(f"the judge was not fit for game(s) {list(missing)}; it scores "
                             "only the season it was cross-fitted on")
        return fold.to_numpy(int)

    def propensity(self, df: pd.DataFrame) -> np.ndarray:
        fold = self.fold_of(df)
        out = np.empty((len(df), self.n_actions))
        for f, model in enumerate(self.models):
            rows = fold == f
            if rows.any():
                out[rows] = model.propensity(df[rows])
        return out


def judge_mode(cfg: dict) -> str:
    mode = cfg.get("ope", {}).get("judge_propensity", "train")
    if mode not in JUDGES:
        raise ValueError(f"unknown ope.judge_propensity {mode!r}; expected one of {JUDGES}")
    return mode


def _with(ds: Dataset, df: pd.DataFrame, numeric: list[str]) -> Dataset:
    return Dataset(df=df, numeric_state=numeric, categorical_state=ds.categorical_state,
                   action_col=ds.action_col, reward_col=ds.reward_col,
                   n_actions=ds.n_actions, action_labels=ds.action_labels)


def fit_judge_behavior(train: Dataset, test: Dataset, cfg: dict,
                       beh: BehaviorModel | None = None):
    """The judge's pi_b for scoring `test` (see the module docstring).

    `beh` is the policy's train-fit pi_b, returned as is under `train`.
    """
    if judge_mode(cfg) == "train":
        return beh if beh is not None else fit_behavior_model(train, cfg, diagnose=False)[0]

    n_folds = int(cfg["ope"].get("judge_folds", 5))
    games = test.df["game_id"]
    if games.nunique() < n_folds:
        raise ValueError(f"the judge needs at least {n_folds} scored games, got {games.nunique()}")
    fold = np.empty(len(test.df), int)
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=cfg["seed"])
    for k, (_, te) in enumerate(gkf.split(test.df, groups=games)):
        fold[te] = k

    numeric = list(train.numeric_state)
    if "season" not in numeric:
        numeric.append("season")
    models = []
    for k in range(n_folds):
        rows = pd.concat([train.df, test.df[fold != k]], ignore_index=True)
        models.append(fit_behavior_model(_with(train, rows, numeric), cfg, diagnose=False)[0])
    return CrossFitJudge(models=models, game_fold=dict(zip(games, fold)),
                         n_actions=test.n_actions)
