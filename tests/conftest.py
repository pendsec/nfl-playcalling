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
