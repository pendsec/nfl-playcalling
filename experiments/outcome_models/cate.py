"""
Contrast-first outcome models — Q(S, A) assembled from estimated call effects.

The policy never uses Q's level, only how the calls compare in a state. A
regressor of reward on (state, action) spends its capacity on the level — EPA
varies by ~1.5 per snap with the situation, a call moves it by ~0.05 — and a
tree splits on the action only when that buys more than another state split.
These learners model the comparison directly, against a reference call (the
most common one), with econml:

  * DR-learner — fit an outcome model mu(s, a) and a propensity pi(a | s),
    cross-fitted; form the doubly-robust pseudo-outcome for each call,

        G_a = mu(s, a) + 1{A = a} / pi(a | s) * (R - mu(s, A)),

    and regress G_a - G_ref on the state. Unbiased for the contrast if either
    nuisance is right; the inverse weights make it noisy on rare calls.
  * R-learner (causal forest DML) — residualize the reward and the call
    indicators on the state (R - m(s), 1{A=a} - pi(a|s)) and find the contrasts
    tau(s) that best explain one residual with the other (Robinson's
    decomposition), with a generalized random forest. No inverse weights; rare
    calls have little residual variation, so their contrasts are weakly
    determined instead.

Both return contrasts tau_a(s) = Q(s, a) - Q(s, ref). Q itself is recovered from
the reward's conditional mean m(s) = sum_a pi(a|s) Q(s, a):

    Q(s, a) = m(s) + tau_a(s) - sum_b pi(b | s) tau_b(s)

so the coordinator's own mix scores m(s) and every call is placed by its
contrast. Same interface as `QModel` (`predict_all_actions`), so policies, the
bootstrap ensemble and OPE take it unchanged.

Not part of the pipeline: benchmarked against the pooled GBM and not adopted
(see this folder's README). Reached through `benchmark.fit_candidate`, which
also builds their bootstrap ensembles.

Causal note: the learners condition on the SCM adjustment set only (checked by
the caller, `benchmark.fit_candidate`), exactly as the regression Q does; the
unobserved read biases them the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from src.models.preprocess import build_gbm_preprocessor
from src.schemas.dataset import Dataset

LEARNERS = ("dr_learner", "r_learner")


@dataclass
class ContrastQModel:
    prep: object
    cate: object                # fitted econml estimator; contrasts vs code 0
    mean_model: object          # m(s) = E[R | s]
    propensity: object          # pi(a | s), for placing Q around m(s)
    order: np.ndarray           # econml code k -> action id (order[0] = reference)
    n_actions: int
    state_cols: list[str]
    q_scale: float = 1.0

    def contrasts(self, df: pd.DataFrame) -> np.ndarray:
        """tau_a(s) = Q(s, a) - Q(s, ref), shape (n, n_actions), by action id."""
        X = self.prep.transform(df[self.state_cols])
        tau = np.zeros((len(df), self.n_actions))
        tau[:, self.order[1:]] = np.asarray(self.cate.const_marginal_effect(X)).reshape(len(df), -1)
        return tau

    def predict_all_actions(self, df: pd.DataFrame) -> np.ndarray:
        X = self.prep.transform(df[self.state_cols])
        tau = self.contrasts(df)
        pi = np.zeros((len(df), self.n_actions))
        pi[:, self.propensity.classes_] = self.propensity.predict_proba(X)
        m = self.mean_model.predict(X)
        return m[:, None] + tau - (pi * tau).sum(axis=1, keepdims=True)

    def predict_taken(self, df: pd.DataFrame, actions: np.ndarray) -> np.ndarray:
        return self.predict_all_actions(df)[np.arange(len(df)), np.asarray(actions)]


def _nuisance_regressor(ocfg: dict, seed: int) -> LGBMRegressor:
    return LGBMRegressor(n_estimators=ocfg.get("n_estimators", 300),
                         learning_rate=ocfg.get("learning_rate", 0.05),
                         max_depth=ocfg.get("max_depth", 5),
                         min_child_samples=ocfg.get("min_child_samples", 20),
                         subsample=ocfg.get("subsample", 0.8), subsample_freq=1,
                         colsample_bytree=ocfg.get("colsample_bytree", 0.8),
                         random_state=seed, verbose=-1)


def _nuisance_classifier(ocfg: dict, seed: int) -> LGBMClassifier:
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5,
                          min_child_samples=30, random_state=seed, verbose=-1)


def _final_stage(fcfg: dict, seed: int):
    """The model of the contrasts. Pseudo-outcomes are mostly noise, so the
    default is heavily regularized: shallow trees, large leaves, few rounds."""
    kind = fcfg.get("type", "lightgbm")
    if kind == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
    if kind == "lightgbm":
        return LGBMRegressor(n_estimators=fcfg.get("n_estimators", 100),
                             learning_rate=fcfg.get("learning_rate", 0.03),
                             max_depth=fcfg.get("max_depth", 2),
                             min_child_samples=fcfg.get("min_child_samples", 1000),
                             subsample=0.8, subsample_freq=1,
                             random_state=seed, verbose=-1)
    raise ValueError(f"unknown final stage '{kind}' (ridge | lightgbm)")


def fit_contrast_q_model(ds: Dataset, cfg: dict) -> ContrastQModel:
    """Fit the configured learner (`outcome_model.type`: dr_learner | r_learner)."""
    from econml.dml import CausalForestDML
    from econml.dr import DRLearner

    ocfg, seed = cfg["outcome_model"], int(cfg["seed"])
    kind = ocfg["type"]
    fcfg = ocfg.get("final", {})
    prep = build_gbm_preprocessor(ds)
    X = prep.fit_transform(ds.df[ds.state_cols])
    y = ds.df[ds.reward_col].to_numpy()
    a = ds.df[ds.action_col].to_numpy()

    # econml takes the first (sorted) category as the reference: recode so the
    # most common call is 0. Calls never taken in `ds` have no contrast (0).
    counts = np.bincount(a, minlength=ds.n_actions)
    ref = int(counts.argmax())
    seen = [k for k in range(ds.n_actions) if counts[k] > 0 and k != ref]
    order = np.array([ref] + seen)
    code = np.full(ds.n_actions, -1)
    code[order] = np.arange(len(order))
    t = code[a]
    folds = ocfg.get("cross_fit_folds", 5)

    if kind == "dr_learner":
        est = DRLearner(model_propensity=_nuisance_classifier(ocfg, seed),
                        model_regression=_nuisance_regressor(ocfg, seed),
                        model_final=_final_stage(fcfg, seed),
                        min_propensity=cfg["ope"]["clip_propensity"][0],
                        categories=list(range(len(order))), cv=folds, random_state=seed)
    elif kind == "r_learner":
        est = CausalForestDML(model_y=_nuisance_regressor(ocfg, seed),
                              model_t=_nuisance_classifier(ocfg, seed),
                              discrete_treatment=True, categories=list(range(len(order))),
                              cv=folds, n_estimators=fcfg.get("n_estimators", 100),
                              min_samples_leaf=fcfg.get("min_samples_leaf", 200),
                              max_samples=fcfg.get("max_samples", 0.3),
                              max_depth=fcfg.get("max_depth"), inference=False,
                              n_jobs=-1, random_state=seed)
    else:
        raise ValueError(f"unknown contrast learner '{kind}' ({' | '.join(LEARNERS)})")
    est.fit(y, t, X=X)

    mean_model = _nuisance_regressor(ocfg, seed).fit(X, y)
    propensity = _nuisance_classifier(ocfg, seed).fit(X, a)
    model = ContrastQModel(prep=prep, cate=est, mean_model=mean_model, propensity=propensity,
                           order=order, n_actions=ds.n_actions, state_cols=ds.state_cols)
    from src.models.outcome.q_model import _within_state_spread
    model.q_scale = _within_state_spread(model.predict_all_actions(ds.df))
    return model
