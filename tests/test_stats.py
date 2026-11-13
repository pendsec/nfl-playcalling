"""The shared helpers: Dataset copies, game folds, means with SEs, the gate rule."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.folds import game_folds
from src.schemas.dataset import Dataset
from src.stats import mean_se, regresses


def _ds():
    df = pd.DataFrame({"game_id": np.repeat(["a", "b", "c", "d"], 5), "x": np.arange(20.0),
                       "action": np.arange(20) % 3, "reward": np.linspace(-1, 1, 20)})
    return Dataset(df=df, numeric_state=["x"], categorical_state=[], action_col="action",
                   reward_col="reward", n_actions=3, action_labels={0: "a", 1: "b", 2: "c"})


def test_dataset_replace_and_subset_keep_the_other_fields():
    ds = _ds()
    mask = ds.df["x"].to_numpy() >= 10
    sub = ds.subset(mask)
    assert sub.df.index.tolist() == list(range(10)) and sub.df["x"].min() == 10
    assert sub.numeric_state is ds.numeric_state and sub.action_labels is ds.action_labels
    pd.testing.assert_frame_equal(ds.subset(np.flatnonzero(mask)).df, sub.df)
    assert ds.replace(numeric_state=["x", "y"]).numeric_state == ["x", "y"]
    assert ds.numeric_state == ["x"]


def test_game_folds_keep_games_whole():
    df = _ds().df
    fold = game_folds(df, 2, seed=0)
    assert (pd.Series(fold).groupby(df["game_id"]).nunique() == 1).all()
    assert set(fold) == {0, 1}


def test_mean_se_iid_and_clustered():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    m, se = mean_se(x)
    assert m == 2.5 and se == x.std(ddof=1) / 2
    assert mean_se(np.array([1.0]))[1] == float("inf")
    # perfectly correlated within clusters: the clustered SE is the cluster means' SE
    m_c, se_c = mean_se(np.array([1.0, 1.0, 3.0, 3.0]), np.array([0, 0, 1, 1]))
    assert m_c == 2.0 and se_c > mean_se(np.array([1.0, 1.0, 3.0, 3.0]))[1]
    assert np.isnan(mean_se(x, np.zeros(4))[1])


def test_regresses_needs_the_whole_interval_below_the_margin():
    assert regresses(-0.05, 0.01, 0.02)
    assert not regresses(-0.03, 0.01, 0.02)
    assert not regresses(-0.05, 0.05, 0.02)
