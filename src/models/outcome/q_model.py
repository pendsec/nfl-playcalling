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


def _within_state_spread(q_all: np.ndarray) -> float:
    """Mean per-state standard deviation of Q across actions.

    Uses the *within-state* spread rather than the global one because that is
    what a policy's argmax actually compares: how far apart the calls are in a
    single situation. Floored away from zero so a degenerate Q (every action
    identical) cannot silently disable a penalty that divides by it.
    """
    return max(float(np.mean(q_all.std(axis=1))), 1e-6)


def _shrunk_cell_mean(y_cell: np.ndarray, global_mean: float,
                      prior_strength: float) -> float:
    """Empirical-Bayes mean for an action cell too thin to fit a regressor.

    The raw cell mean is a terrible constant when the cell holds a handful of
    plays: on real data one 9-play cell averaged -2.30 reward against a global
    mean near -0.09, and because that constant enters `predict_all_actions` for
    EVERY state, it dominated the policy's argmax everywhere — a phantom "never
    call this" (or, with the sign flipped, "always call this") learned from nine
    snaps of noise.

    Shrinking toward the global mean with prior weight `prior_strength` (the same
    `min_cell` threshold that declared the cell too thin) makes the fallback
    degrade gracefully: a cell just under the threshold keeps most of its own
    signal, a 1-play cell collapses to roughly the global mean, and no cell can
    manufacture an extreme Q from a sample that small.
    """
    n = len(y_cell)
    if n == 0:
        return global_mean
    return float((n * float(y_cell.mean()) + prior_strength * global_mean)
                 / (n + prior_strength))


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
        fallbacks[a] = _shrunk_cell_mean(y[m], global_mean, min_cell)
        if m.sum() >= min_cell:
            reg = _make_regressor(cfg)
            reg.fit(phi[m], y[m])
            regressors[a] = reg
        else:
            regressors[a] = None

    model = QModel(prep=prep, regressors=regressors, fallbacks=fallbacks,
                   n_actions=ds.n_actions, state_cols=ds.state_cols)
    # Measured on the training states, so every downstream policy shares one
    # scale regardless of which split it is constructed against.
    model.q_scale = _within_state_spread(model.predict_all_actions(ds.df))
    return model


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
                mu[va, a] = _shrunk_cell_mean(y[m], global_mean, min_cell)
    return mu
