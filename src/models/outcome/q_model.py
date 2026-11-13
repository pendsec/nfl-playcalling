"""
Outcome / Q-Model — Q(S, A) = E[R | S, do(A)].

A per-treatment gradient-boosted outcome model — one LightGBM regressor of
reward on the SCM adjustment set, fit on the plays where each action was taken.
Nonlinear and state-dependent, so recommendations vary by situation, while rare
action cells fall back to a shrunk mean rather than extrapolating. That fallback
matters: a regularized linear model asked to extrapolate off-support produces
implausible Q-values, which is the failure DR-OPE and the conservative policy
exist to contain.

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


def _make_regressor(cfg: dict, pooled: bool = False):
    ocfg = cfg["outcome_model"]
    extra = {"subsample_freq": 1} if pooled else {}
    return LGBMRegressor(
        **extra,
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


def _check_conditions_on_adjustment_set(ds: Dataset, cfg: dict) -> None:
    """Guard: the state we condition on must equal the SCM adjustment set."""
    # The graph is chosen by the action scheme: a different treatment means a
    # different backdoor set, so the check has to be against the right graph.
    if set(ds.state_cols) != set(scm.graph_for(cfg).adjustment_columns()):
        raise ValueError(
            "Q-model features do not match the SCM adjustment set — a mediator "
            "may be leaking in or a confounder is missing."
        )


def _model_type(cfg: dict) -> str:
    return cfg["outcome_model"].get("type", "lightgbm")


def fit_q_model(ds: Dataset, cfg: dict) -> QModel:
    """Fit the outcome model on all of `ds` — per-action regressors by default,
    or one pooled regressor when `outcome_model.structure` is `pooled`.
    `outcome_model.type` tarnet | dragonnet selects a network (`neural.py`)."""
    kind = _model_type(cfg)
    if kind != "lightgbm":
        _check_conditions_on_adjustment_set(ds, cfg)
        if kind in ("tarnet", "dragonnet"):
            from .neural import fit_neural_q_model
            return fit_neural_q_model(ds, cfg)
        hint = (" — benchmarked and not adopted; it lives in experiments/outcome_models/"
                if kind in ("dr_learner", "r_learner") else "")
        raise ValueError(f"unknown outcome_model.type '{kind}'{hint}")
    if _is_pooled(cfg):
        return fit_pooled_q_model(ds, cfg)
    _check_conditions_on_adjustment_set(ds, cfg)
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
    _check_conditions_on_adjustment_set(ds, cfg)
    if _model_type(cfg) != "lightgbm":
        return _crossfit_generic(ds, cfg)
    if _is_pooled(cfg):
        return _crossfit_pooled(ds, cfg)
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


# ── Uncertainty ──────────────────────────────────────────────────────────────
def fit_q_ensemble(ds: Dataset, cfg: dict, fit=None):
    """Bootstrap ensemble of the configured Q model, resampling whole games.

    Feeds the conservative policy's lower confidence bound. On 2022-25 the
    point Q gave the 8+ box a flat +0.048 advantage that held-out seasons did not
    bear out (residual -0.06 on base/8+), while the doubly-robust contrast for
    the same call carried a 95% CI of ±0.04: the policy was chasing an effect
    the data could not separate from zero. Penalizing a call by how much its
    advantage moves across resamples makes the policy leave the coordinator
    only where the data has established the difference.

    `fit` refits each member (default `fit_q_model`); an experiment can pass
    its own so a model outside the pipeline gets the same bootstrap.
    """
    from ...schemas.outcome import QEnsemble

    ecfg = cfg["outcome_model"].get("ensemble", {})
    n_boot = ecfg.get("n_boot", 20)
    rng = np.random.default_rng(ecfg.get("seed", cfg["seed"]))
    df = ds.df
    games = df["game_id"].to_numpy() if "game_id" in df.columns else np.arange(len(df))
    uniq, inverse = np.unique(games, return_inverse=True)
    rows_of = np.split(np.argsort(inverse, kind="stable"),
                       np.cumsum(np.bincount(inverse))[:-1])

    members = []
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[g] for g in pick])
        boot = ds.subset(idx)
        bcfg = {**cfg, "seed": int(cfg["seed"]) + 1 + b}
        members.append((fit or fit_q_model)(boot, bcfg))
    return QEnsemble(members=members, n_actions=ds.n_actions)


# ── Pooled structure ─────────────────────────────────────────────────────────
# `outcome_model.structure: pooled` fits ONE regressor on (state, action
# features) instead of one per action. Measured against a known SCM at the real
# pipeline's scale (~91k training plays, level-1 actions):
#
#                                rank corr   spread / true   conservative policy
#                                vs truth      spread        true gain over the DC
#     per-action (12 models)       0.39          2.1x              +0.016
#     pooled, package/front        0.42          0.8x              +0.028
#
# Per-action models estimate each cell's effect with independent noise, so the
# spread across actions is inflated and the policy's argmax chases the most
# overestimated cell (the optimizer's curse). Pooling shares the state structure
# across cells and lets the action's components carry the effect; the policy
# captured ~77% of the available gain against ~45% per-action.
def _pooled_design(phi: np.ndarray, actions: np.ndarray, feats: np.ndarray) -> np.ndarray:
    return np.hstack([phi, feats[np.asarray(actions)]])


def _is_pooled(cfg: dict) -> bool:
    return cfg["outcome_model"].get("structure", "per_action") == "pooled"


def fit_pooled_q_model(ds: Dataset, cfg: dict):
    """One regressor on state + the taken action's features, fit on all rows."""
    from ...data.features import action_feature_matrix
    from ...schemas.outcome import PooledQModel

    _check_conditions_on_adjustment_set(ds, cfg)
    feats = action_feature_matrix(cfg)
    prep = build_gbm_preprocessor(ds)
    phi = prep.fit_transform(ds.df[ds.state_cols])
    reg = _make_regressor(cfg, pooled=True)
    reg.fit(_pooled_design(phi, ds.df[ds.action_col].to_numpy(), feats),
            ds.df[ds.reward_col].to_numpy())
    model = PooledQModel(prep=prep, regressor=reg, action_features=feats,
                         n_actions=ds.n_actions, state_cols=ds.state_cols)
    model.q_scale = _within_state_spread(model.predict_all_actions(ds.df))
    return model


def _crossfit_generic(ds: Dataset, cfg: dict) -> np.ndarray:
    """Out-of-fold Q for any model `fit_q_model` builds: refit per fold."""
    mu = np.empty((len(ds.df), ds.n_actions))
    kf = KFold(n_splits=cfg["outcome_model"].get("cross_fit_folds", 5),
               shuffle=True, random_state=cfg["seed"])
    for tr, va in kf.split(ds.df):
        mu[va] = fit_q_model(ds.subset(tr), cfg).predict_all_actions(ds.df.iloc[va])
    return mu


def _crossfit_pooled(ds: Dataset, cfg: dict) -> np.ndarray:
    from ...data.features import action_feature_matrix

    feats = action_feature_matrix(cfg)
    prep = build_gbm_preprocessor(ds)
    phi = prep.fit_transform(ds.df[ds.state_cols])
    T = ds.df[ds.action_col].to_numpy()
    y = ds.df[ds.reward_col].to_numpy()
    mu = np.empty((len(y), ds.n_actions))
    kf = KFold(n_splits=cfg["outcome_model"].get("cross_fit_folds", 5),
               shuffle=True, random_state=cfg["seed"])
    for tr, va in kf.split(phi):
        reg = _make_regressor(cfg, pooled=True)
        reg.fit(_pooled_design(phi[tr], T[tr], feats), y[tr])
        for a in range(ds.n_actions):
            mu[va, a] = reg.predict(_pooled_design(phi[va], np.full(len(va), a), feats))
    return mu
