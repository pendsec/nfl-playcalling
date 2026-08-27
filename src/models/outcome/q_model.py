"""
Outcome / Q-Model — Q(S, A) = E[R | S, do(A)].

V2: a per-treatment gradient-boosted outcome model — one LightGBM regressor of
reward on the SCM adjustment set, fit on the plays where each action was taken.
Nonlinear and state-dependent, so recommendations vary by situation, while rare
action cells fall back to a mean rather than extrapolating (V1's ridge blew up
off-support — the pathology DR-OPE + the conservative policy now contain).

Causal discipline (per the SCM): condition ONLY on the adjustment columns
`scm.graph.adjustment_columns()` — pre-snap confounders + player proxies. The
offensive-playcall mediator and any result-derived field are excluded so Q is the
call's total effect on reward.

Two entry points:
  * `fit_q_model`  — regressors fit on ALL of train, for scoring disjoint holdout
    states (the primary OPE use).
  * `crossfit_q`   — out-of-fold Q on the training rows themselves, so the DR
    smoke test on train is honest (no in-sample optimism).
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from ..preprocess import build_gbm_preprocessor
from ...scm import graph as scm
from ...schemas.dataset import Dataset
from ...schemas.outcome import QModel


def _make_regressor(cfg: dict):
    ocfg = cfg["outcome_model"]
    return LGBMRegressor(
        n_estimators=ocfg.get("n_estimators", 300),
        learning_rate=ocfg.get("learning_rate", 0.05),
        max_depth=ocfg.get("max_depth", 5),
        min_child_samples=ocfg.get("min_child_samples", 20),
        subsample=ocfg.get("subsample", 0.8),
        colsample_bytree=ocfg.get("colsample_bytree", 0.8),
        random_state=cfg["seed"],
        verbose=-1,
    )


def _check_conditions_on_adjustment_set(ds: Dataset) -> None:
    """Guard: the state we condition on must equal the SCM adjustment set."""
    if set(ds.state_cols) != set(scm.adjustment_columns()):
        raise ValueError(
            "Q-model features do not match the SCM adjustment set — a mediator "
            "may be leaking in or a confounder is missing."
        )


def fit_q_model(ds: Dataset, cfg: dict) -> QModel:
    """Fit per-action outcome regressors on all of `ds`."""
    _check_conditions_on_adjustment_set(ds)
    prep = build_gbm_preprocessor(ds)
    phi = prep.fit_transform(ds.df[ds.state_cols])
    T = ds.df[ds.action_col].to_numpy()
    y = ds.df[ds.reward_col].to_numpy()
    min_cell = cfg["outcome_model"].get("min_cell", 20)
    global_mean = float(y.mean())

    regressors: dict[int, object] = {}
    fallbacks: dict[int, float] = {}
    for a in range(ds.n_actions):
        m = T == a
        fallbacks[a] = float(y[m].mean()) if m.any() else global_mean
        if m.sum() >= min_cell:
            reg = _make_regressor(cfg)
            reg.fit(phi[m], y[m])
            regressors[a] = reg
        else:
            regressors[a] = None

    return QModel(prep=prep, regressors=regressors, fallbacks=fallbacks,
                  n_actions=ds.n_actions, state_cols=ds.state_cols)


def crossfit_q(ds: Dataset, cfg: dict) -> np.ndarray:
    """Out-of-fold Q(s, a) for every training row — honest in-sample mu_hat.

    For each fold, per-action regressors are fit on the other folds and used to
    score the held-out fold. Returns shape (n, n_actions).
    """
    _check_conditions_on_adjustment_set(ds)
    prep = build_gbm_preprocessor(ds)
    phi = prep.fit_transform(ds.df[ds.state_cols])
    T = ds.df[ds.action_col].to_numpy()
    y = ds.df[ds.reward_col].to_numpy()
    min_cell = cfg["outcome_model"].get("min_cell", 20)
    n_splits = cfg["outcome_model"].get("cross_fit_folds", 5)
    global_mean = float(y.mean())

    mu = np.empty((len(y), ds.n_actions))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
    for tr, va in kf.split(phi):
        for a in range(ds.n_actions):
            m = tr[T[tr] == a]
            if len(m) >= min_cell:
                reg = _make_regressor(cfg)
                reg.fit(phi[m], y[m])
                mu[va, a] = reg.predict(phi[va])
            else:
                mu[va, a] = y[m].mean() if len(m) else global_mean
    return mu
