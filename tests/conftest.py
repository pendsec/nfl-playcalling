"""Shared fixtures.

The `fitted` fixture is SESSION-scoped rather than module-scoped. It fits a
calibrated behavior model, a per-action Q-model, a cross-fit Q, and a policy —
several seconds of work — and every test file that asserts against the known SCM
needs the same one. Module scope would repeat that per file; session scope fits
it once for the whole run.

Consequence worth knowing: the objects are SHARED across tests, so a test must
not mutate them. Tests that need a variant build their own from `CFG`.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _config import CFG, N_SYNTHETIC_PLAYS, SYNTHETIC_SEED
from src.data.load import generate_synthetic
from src.data.features import build_dataset, time_aware_split
from src.models.behavior.propensity import fit_behavior_model
from src.models.outcome.q_model import fit_q_model, crossfit_q
from src.policy.conservative import learn_conservative_policy


@pytest.fixture(scope="session")
def fitted():
    """The whole pipeline fit once on data from a known SCM."""
    pbp = generate_synthetic(n_plays=N_SYNTHETIC_PLAYS, seed=SYNTHETIC_SEED)
    ds = build_dataset(pbp, CFG)
    train, test = time_aware_split(ds, CFG["data"]["holdout_season"])
    beh, diag = fit_behavior_model(train, CFG)
    q = fit_q_model(train, CFG)
    mu_train = crossfit_q(train, CFG)
    policy = learn_conservative_policy(train, q, beh, CFG)
    return dict(ds=ds, train=train, test=test, beh=beh, q=q, diag=diag,
                mu_train=mu_train, policy=policy)


def _fit_level1(conf_strength: float) -> dict:
    from _config import CFG_L1, L1_SEED, N_L1_PLAYS
    from src.data.load import generate_synthetic_alignment

    pbp = generate_synthetic_alignment(N_L1_PLAYS, seed=L1_SEED,
                                       seasons=CFG_L1["data"]["seasons"],
                                       conf_strength=conf_strength)
    ds = build_dataset(pbp, CFG_L1)
    train, test = time_aware_split(ds, CFG_L1["data"]["holdout_season"])
    beh, diag = fit_behavior_model(train, CFG_L1)
    q = fit_q_model(train, CFG_L1)
    policy = learn_conservative_policy(train, q, beh, CFG_L1)
    return dict(ds=ds, train=train, test=test, beh=beh, q=q, diag=diag, policy=policy)


@pytest.fixture(scope="session")
def fitted_l1():
    """The every-snap pipeline fit once on a known alignment SCM, WITH the
    hidden read confounding the outcome (the realistic case)."""
    return _fit_level1(conf_strength=0.20)


@pytest.fixture(scope="session")
def fitted_l1_unconfounded():
    """Same SCM with the hidden read's effect on the outcome switched off — the
    one setting where OPE should recover the TRUE policy value, not just the
    behavior policy's empirical mean."""
    return _fit_level1(conf_strength=0.0)


@pytest.fixture(scope="session")
def l1_ensemble(fitted_l1):
    """A small game-bootstrap Q ensemble on the confounded level-1 fit."""
    from _config import CFG_L1
    from src.models.outcome.q_model import fit_q_ensemble
    cfg = {**CFG_L1, "outcome_model": {**CFG_L1["outcome_model"],
                                       "ensemble": {"n_boot": 8}}}
    return fit_q_ensemble(fitted_l1["train"], cfg)


def _fit_level2(conf_strength: float) -> dict:
    """Both levels, the dropback alignment propensity and the factored
    evaluation, fit once on the alignment SCM with coverage effects."""
    from _config import CFG_L1, L2_SEED, N_L2_PLAYS
    from src.data.features import build_level2_dataset, level2_config
    from src.data.load import generate_synthetic_alignment
    from src.models.outcome.q_model import fit_q_ensemble
    from src.policy.hierarchical import evaluate_factored, fit_dropback_alignment_behavior
    from src.policy.safe import learn_safe_policy

    cfg = {**CFG_L1,
           "action": {**CFG_L1["action"], "level2": {"shells": "family",
                                                     "blitz_rusher_threshold": 5}},
           "outcome_model": {**CFG_L1["outcome_model"], "ensemble": {"n_boot": 8}},
           "policy": {"type": "safe", "lcb_kappa": 1.0, "support_threshold": 0.05}}
    pbp = generate_synthetic_alignment(N_L2_PLAYS, seed=L2_SEED,
                                       seasons=cfg["data"]["seasons"],
                                       conf_strength=conf_strength, coverage_effects=True)
    ds1 = build_dataset(pbp, cfg)
    train1, _ = time_aware_split(ds1, cfg["data"]["holdout_season"])
    beh1, _ = fit_behavior_model(train1, cfg)
    q1 = fit_q_model(train1, cfg)
    pol1 = learn_safe_policy(train1, q1, beh1, cfg, fit_q_ensemble(train1, cfg))

    cfg2 = level2_config(cfg)
    ds2 = build_level2_dataset(ds1, cfg2)
    train2, test2 = time_aware_split(ds2, cfg["data"]["holdout_season"])
    beh2, _ = fit_behavior_model(train2, cfg2)
    q2 = fit_q_model(train2, cfg2)
    pol2 = learn_safe_policy(train2, q2, beh2, cfg2, fit_q_ensemble(train2, cfg2))
    beh1_db, _ = fit_dropback_alignment_behavior(train2, cfg)
    res = evaluate_factored(test2.df, cfg, pol1, pol2, beh1_db, beh2, q2)
    return dict(pbp=pbp, cfg=cfg, cfg2=cfg2, ds1=ds1, ds2=ds2, train2=train2,
                test2=test2, beh2=beh2, q2=q2, pol1=pol1, pol2=pol2,
                beh1_db=beh1_db, eval=res)


@pytest.fixture(scope="session")
def l2():
    """Level 2 with the hidden read confounding both levels."""
    return _fit_level2(conf_strength=0.20)


@pytest.fixture(scope="session")
def l2_eval(l2):
    return l2["eval"]


@pytest.fixture(scope="session")
def l2_unconf():
    """Level 2 with the read's effect on the outcome switched off."""
    return _fit_level2(conf_strength=0.0)
