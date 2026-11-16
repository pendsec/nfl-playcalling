"""
The outcome-model experiment — the contrast learners it tried and the criteria
it judged every candidate by. Run with `pytest experiments`.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest


pytest.importorskip("econml")

from _config import CFG_L1
from experiments.outcome_models.benchmark import (CANDIDATES, advantages, calibration_slope,
                                                  fit_candidate, pseudo_outcomes,
                                                  supported_argmax)
from src.data.load import true_alignment_q
from src.models.behavior.propensity import fit_behavior_model
from src.models.outcome.q_model import fit_q_ensemble


def _cfg(**outcome):
    c = copy.deepcopy(CFG_L1)
    c["outcome_model"].update(outcome)
    return c


@pytest.fixture(scope="module")
def league(synthetic_league):
    train, test = synthetic_league(12000, seed=3)
    pi_b = fit_behavior_model(train, CFG_L1, diagnose=False)[0].propensity(test.df)
    return dict(train=train, test=test, q_true=true_alignment_q(test.df), pi_b=pi_b)


SPECS = {"dr_learner": dict(type="dr_learner", cross_fit_folds=2, final={"type": "ridge"}),
         "r_learner": dict(type="r_learner", cross_fit_folds=2,
                           final={"n_estimators": 50, "min_samples_leaf": 100})}


@pytest.fixture(scope="module")
def fitted(league):
    return {k: fit_candidate(league["train"], _cfg(**v)) for k, v in SPECS.items()}


# ── Contrast learners ────────────────────────────────────────────────────────
@pytest.mark.parametrize("kind", list(SPECS))
def test_contrast_learners_score_every_call(fitted, league, kind):
    out = fitted[kind].predict_all_actions(league["test"].df)
    assert out.shape == (len(league["test"].df), 12)
    assert np.isfinite(out).all() and fitted[kind].q_scale > 0


def test_contrast_q_places_the_mix_at_the_conditional_mean(fitted, league):
    """Q = m(s) + tau - pi.tau: the learner's own propensity mix scores m(s)."""
    q = fitted["dr_learner"]
    df = league["test"].df.head(500)
    X = q.prep.transform(df[q.state_cols])
    pi = np.zeros((len(df), 12))
    pi[:, q.propensity.classes_] = q.propensity.predict_proba(X)
    np.testing.assert_allclose((pi * q.predict_all_actions(df)).sum(1), q.mean_model.predict(X),
                               atol=1e-9)
    assert (q.contrasts(df)[:, q.order[0]] == 0).all()   # the reference has no contrast


@pytest.mark.parametrize("kind", list(SPECS))
def test_contrast_learners_recover_the_direction_of_the_effects(fitted, league, kind):
    est = fitted[kind].predict_all_actions(league["test"].df)
    pi, sup = league["pi_b"], league["pi_b"] >= 0.05
    corr = np.corrcoef(advantages(est, pi)[sup], advantages(league["q_true"], pi)[sup])[0, 1]
    assert corr > 0.1


def test_contrast_learners_check_the_adjustment_set(league):
    t = league["train"]
    bad = type(t)(df=t.df, numeric_state=t.numeric_state + ["_coach_read"],
                  categorical_state=t.categorical_state, action_col=t.action_col,
                  reward_col=t.reward_col, n_actions=t.n_actions, action_labels=t.action_labels)
    with pytest.raises(ValueError, match="adjustment set"):
        fit_candidate(bad, _cfg(**SPECS["dr_learner"]))


def test_the_ensemble_bootstraps_a_contrast_learner(league):
    c = _cfg(**SPECS["dr_learner"], ensemble={"n_boot": 2})
    ens = fit_q_ensemble(league["train"], c, fit=fit_candidate)
    assert len(ens.members) == 2 and type(ens.members[0]).__name__ == "ContrastQModel"


def test_every_network_candidate_names_its_heads():
    """The pipeline's default head changed after this benchmark; the candidates
    pin theirs so a rerun scores the same models."""
    for name, spec in CANDIDATES.items():
        if spec.get("type") in ("tarnet", "dragonnet"):
            assert spec["neural"]["heads"] == ("factored" if "factored" in name else "separate")


# ── Benchmark criteria ───────────────────────────────────────────────────────
def test_pseudo_outcomes_are_unbiased_with_the_true_propensity():
    rng = np.random.default_rng(0)
    n, k = 200_000, 3
    pi = np.tile([0.2, 0.3, 0.5], (n, 1))
    a = (rng.random(n)[:, None] > np.cumsum(pi, 1)).sum(1)
    effect = np.array([0.0, 0.5, -0.3])
    r = effect[a] + rng.normal(0, 1, n)
    g = pseudo_outcomes(np.zeros((n, k)), pi, a, r, (0.01, 0.99), 1e9)  # a wrong Q
    np.testing.assert_allclose(g.mean(0), effect, atol=0.02)


def test_calibration_slope_recovers_a_known_slope():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 50_000)
    beta, se = calibration_slope(x, 0.5 * x + rng.normal(0, 1, len(x)), np.arange(len(x)))
    assert abs(beta - 0.5) < 3 * se and se < 0.01


def test_advantage_of_the_mix_is_zero_and_argmax_respects_support():
    q = np.array([[1.0, 2.0, 3.0]])
    pi = np.array([[0.5, 0.5, 0.0]])
    assert (pi * advantages(q, pi)).sum() == pytest.approx(0.0)
    assert supported_argmax(q, pi >= 0.05)[0] == 1
