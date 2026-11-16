"""
The behavior-calibration experiment — the factored model, the scored-season
judges, and the scoring. Run with `pytest experiments`.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest


from _config import CFG_L1
from experiments.behavior_calibration.compare import (CANDIDATES, candidate_config,
                                                      fit_factored, scored_crossfit,
                                                      score_candidate)
from src.data.load import true_alignment_propensity
from src.ope.ipw import ipw_per_play


@pytest.fixture(scope="module")
def league(synthetic_league):
    cfg = copy.deepcopy(CFG_L1)
    cfg["behavior_model"]["n_estimators"] = 60
    train, test = synthetic_league(9000, seed=3)
    return cfg, train, test


def test_candidates_only_touch_the_behavior_model(league):
    cfg = league[0]
    for name in CANDIDATES:
        c = candidate_config(cfg, name)
        assert {k: v for k, v in c.items() if k != "behavior_model"} == \
            {k: v for k, v in cfg.items() if k != "behavior_model"}


def test_the_factored_model_is_a_full_joint_distribution(league):
    cfg, train, test = league
    p = fit_factored(train, candidate_config(cfg, "factored_dirichlet")).propensity(test.df)
    assert p.shape == (len(test.df), 12)
    np.testing.assert_allclose(p.sum(1), 1.0)
    # Its package margin is a proper distribution over the three packages.
    pkg = p.reshape(len(p), 3, 4).sum(2)
    assert (pkg > 0).all()


@pytest.mark.parametrize("pooled", [False, True])
def test_scored_season_judges_are_out_of_fold_by_game(league, pooled):
    cfg, train, test = league
    p = scored_crossfit(test, cfg, n_folds=3, train=train if pooled else None)
    assert p.shape == (len(test.df), 12)
    np.testing.assert_allclose(p.sum(1), 1.0)
    assert np.isfinite(p).all()


def test_scoring_the_true_propensity(league):
    cfg, train, test = league
    df = test.df
    pt = true_alignment_propensity(df)
    k = test.n_actions
    ctx = {"a": df["action"].to_numpy(), "r": df[test.reward_col].to_numpy(),
           "games": df["game_id"].to_numpy(), "thr": 0.05, "wclip": 20.0,
           "q_true": np.zeros((len(df), k)), "p_true": pt,
           "q_judge": np.zeros((len(df), k)),
           "policies": {"safe": pt, "greedy": np.eye(k)[pt.argmax(1)],
                        "always_5": np.tile(np.eye(k)[5], (len(df), 1))}}
    row = score_candidate("truth", pt, ctx)
    assert row["kl_true"] == pytest.approx(0.0, abs=1e-12) and row["rmse_true"] == 0.0
    assert row["safe_dr_lift"] == pytest.approx(0.0, abs=1e-12)   # the logged policy itself
    assert row["balance_rms"] < 0.15


def test_ipw_is_self_normalized():
    rng = np.random.default_rng(0)
    pi = rng.dirichlet(np.ones(4), 500)
    a = np.array([rng.choice(4, p=row) for row in pi])
    r = np.ones(500)
    assert ipw_per_play(pi, np.tile(np.eye(4)[1], (500, 1)), a, r, 20.0,
                        self_normalize=True).mean() == \
        pytest.approx(1.0)


def test_pooled_judge_accepts_a_state_that_already_has_the_season(league):
    """The season joins the state once, even when the state carries it already."""
    cfg, train, test = league
    with_season = train.numeric_state + ["season"]
    p = scored_crossfit(test.replace(numeric_state=with_season), cfg, n_folds=3,
                        train=train.replace(numeric_state=with_season))
    assert p.shape == (len(test.df), 12) and np.isfinite(p).all()
