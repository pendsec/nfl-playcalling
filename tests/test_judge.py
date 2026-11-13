"""The judge's pi_b (`src/models/behavior/judge.py`): fit where it judges."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from _config import CFG_L1
from src.data.features import build_dataset, time_aware_split
from src.data.load import generate_synthetic_alignment
from src.models.behavior.judge import CrossFitJudge, fit_judge_behavior, judge_mode
from src.models.behavior.propensity import fit_behavior_model
from src.schemas.dataset import Dataset


def _cfg(judge="pooled_crossfit", folds=3):
    c = copy.deepcopy(CFG_L1)
    c["behavior_model"].update(n_estimators=60)
    c["ope"].update(judge_propensity=judge, judge_folds=folds)
    return c


@pytest.fixture(scope="module")
def league():
    pbp = generate_synthetic_alignment(9000, seed=5, conf_strength=0.0,
                                       seasons=CFG_L1["data"]["seasons"])
    return time_aware_split(build_dataset(pbp, CFG_L1), CFG_L1["data"]["holdout_season"])


def _drifted(train: Dataset, call: int, keep: float, seed: int = 0) -> Dataset:
    """The training seasons with most of one call's snaps removed: the mix the
    scored season shows is no longer the one the policy's pi_b learned."""
    rng = np.random.default_rng(seed)
    a = train.df[train.action_col].to_numpy()
    rows = (a != call) | (rng.random(len(a)) < keep)
    return train.subset(rows)


def test_train_judge_is_the_policys_own(league):
    train, test = league
    cfg = _cfg("train")
    beh = fit_behavior_model(train, cfg, diagnose=False)[0]
    assert fit_judge_behavior(train, test, cfg, beh) is beh


def test_unknown_judge_is_rejected():
    with pytest.raises(ValueError, match="judge_propensity"):
        judge_mode(_cfg("holdout"))


def test_pooled_judge_scores_each_game_out_of_fold(league):
    train, test = league
    judge = fit_judge_behavior(train, test, _cfg(folds=3), None)
    assert isinstance(judge, CrossFitJudge) and len(judge.models) == 3
    fold = judge.fold_of(test.df)
    # whole games per fold, every fold used
    per_game = pd.Series(fold).groupby(test.df["game_id"].to_numpy()).nunique()
    assert (per_game == 1).all() and set(fold) == {0, 1, 2}
    p = judge.propensity(test.df)
    assert p.shape == (len(test.df), 12)
    np.testing.assert_allclose(p.sum(1), 1.0)
    # row order does not matter: each row goes to its own game's fold
    perm = np.random.default_rng(0).permutation(len(test.df))
    np.testing.assert_allclose(judge.propensity(test.df.iloc[perm]), p[perm])
    # each fold's model is a plain fit that never saw its games
    f0 = fold == 0
    np.testing.assert_allclose(p[f0], judge.models[0].propensity(test.df[f0]))


def test_pooled_judge_refuses_games_it_was_not_fit_for(league):
    train, test = league
    judge = fit_judge_behavior(train, test, _cfg(folds=3), None)
    with pytest.raises(ValueError, match="not fit for game"):
        judge.propensity(train.df.head(5))


def test_pooled_judge_learns_the_scored_seasons_mix(league):
    """Drop most of the most common call from the training seasons. The policy's
    pi_b under-predicts it on the scored season; the judge, which sees that
    season's other games and knows which season a snap is from, does not."""
    train, test = league
    a_te = test.df["action"].to_numpy()
    call = int(np.bincount(a_te).argmax())
    shifted = _drifted(train, call, keep=0.3)
    cfg = _cfg(folds=3)
    beh = fit_behavior_model(shifted, cfg, diagnose=False)[0]
    judge = fit_judge_behavior(shifted, test, cfg, beh)
    share = (a_te == call).mean()
    gap_policy = abs(beh.propensity(test.df)[:, call].mean() - share)
    gap_judge = abs(judge.propensity(test.df)[:, call].mean() - share)
    assert gap_policy > 0.05
    assert gap_judge < gap_policy / 3
