"""
Behavior Policy Model — pi_b(A | S).

V2: a calibrated gradient-boosted classifier predicting which of the 12
defensive calls the actual DC made in state S. This is the engine of IPW/DR:
its propensities reweight logged plays, so **calibration is critical** —
miscalibrated propensities silently break inverse-weighting.

Upgrades over V1's logistic model:
  * LightGBM instead of logistic (nonlinear state -> call structure).
  * Isotonic probability calibration (`CalibratedClassifierCV`).
  * Effective-sample-size (ESS) ratio reported alongside accuracy/log-loss/ECE
    as the overlap/positivity health metric that bounds how far pi* can move.

Rare-class discipline
---------------------
A single-team 12-action slice routinely leaves one (coverage x blitz) cell with
a lone play, and `StratifiedKFold` cannot split a class with fewer members than
folds. The naive response — give up on cross-fitting whenever ANY class is
un-splittable — is what makes the diagnostics dishonest: they fall back to
in-sample predictions from a model that has memorized the training rows, and the
ESS ratio (the very number that bounds how far pi* may move) comes back roughly
twice as healthy as reality.

`_cv_splits` fixes that by scoping the fallback to the classes that need it.
Un-splittable classes are held on the *training* side of every fold — they still
inform each fit, they simply never receive an out-of-fold prediction — so the
other eleven actions get honest cross-fitted diagnostics. The excluded row count
is reported rather than absorbed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ..preprocess import build_gbm_preprocessor
from ...schemas.dataset import Dataset
from ...schemas.behavior import BehaviorModel, clip_normalize

Split = tuple[np.ndarray, np.ndarray]


def _base_classifier(cfg: dict) -> LGBMClassifier:
    bcfg = cfg["behavior_model"]
    return LGBMClassifier(
        n_estimators=bcfg.get("n_estimators", 300),
        learning_rate=bcfg.get("learning_rate", 0.05),
        max_depth=bcfg.get("max_depth", 5),
        min_child_samples=bcfg.get("min_child_samples", 30),
        subsample=bcfg.get("subsample", 0.8),
        colsample_bytree=bcfg.get("colsample_bytree", 0.8),
        random_state=cfg["seed"],
        verbose=-1,
    )


def _cv_splits(y: np.ndarray, requested: int, seed: int) -> tuple[list[Split], np.ndarray]:
    """Stratified folds that keep un-splittable classes on the training side.

    Returns `(splits, rare_mask)`. Classes with fewer members than the fold count
    cannot be stratified, so their rows are appended to every training fold and
    never appear in a test fold: they inform each fit but receive no out-of-fold
    prediction. `rare_mask` marks those rows so callers can report the exclusion.

    Returns `([], mask)` when even the splittable rows cannot support a split —
    the honest "no cross-fitting possible" signal, distinct from "one class was
    awkward".
    """
    y = np.asarray(y)
    n_splits = max(2, int(requested))
    counts = pd.Series(y).value_counts()

    rare_classes = counts.index[counts < n_splits].to_numpy()
    rare = np.isin(y, rare_classes) if len(rare_classes) else np.zeros(len(y), dtype=bool)

    idx = np.arange(len(y))
    dense_idx, rare_idx = idx[~rare], idx[rare]
    if len(dense_idx) < n_splits or pd.Series(y[dense_idx]).nunique() < 2:
        return [], rare

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = [
        (np.concatenate([dense_idx[tr], rare_idx]), dense_idx[te])
        for tr, te in skf.split(dense_idx, y[dense_idx])
    ]
    return splits, rare


def _make_classifier(cfg: dict, y: np.ndarray, seed: int) -> tuple[ClassifierMixin, bool]:
    """Isotonic-calibrated LGBM when the classes admit a split; else raw LGBM.

    Calibration folds come from `_cv_splits`, so an un-splittable class no longer
    disables calibration for every other action. Such a class simply has no
    positive example in any calibration fold, which isotonic resolves to ~0 —
    exactly the right answer for a call with a single logged play, and one the
    propensity clip then floors.
    """
    base = _base_classifier(cfg)
    splits, _ = _cv_splits(y, cfg["behavior_model"].get("calibration_folds", 3), seed)
    if splits:
        return CalibratedClassifierCV(base, cv=splits, method="isotonic"), True
    return base, False


def _fit_pipeline(ds: Dataset, cfg: dict, X: pd.DataFrame, y: np.ndarray,
                  seed: int) -> tuple[Pipeline, bool]:
    clf, calibrated = _make_classifier(cfg, y, seed)
    pipe = Pipeline([("prep", build_gbm_preprocessor(ds)), ("clf", clf)])
    pipe.fit(X, y)
    return pipe, calibrated


def fit_behavior_model(ds: Dataset, cfg: dict) -> tuple[BehaviorModel, dict]:
    """Fit pi_b on `ds` and return the model plus cross-fitted diagnostics."""
    clip = tuple(cfg["ope"]["clip_propensity"])
    seed = cfg["seed"]
    X = ds.df[ds.state_cols]
    y = ds.df[ds.action_col].to_numpy()

    pipe, calibrated = _fit_pipeline(ds, cfg, X, y, seed)
    model = BehaviorModel(
        pipeline=pipe, classes_=pipe.named_steps["clf"].classes_,
        n_actions=ds.n_actions, clip=clip,
    )

    diagnostics = _diagnose(ds, cfg, model, calibrated)
    return model, diagnostics


def _diagnose(ds: Dataset, cfg: dict, model: BehaviorModel, calibrated: bool) -> dict:
    """Accuracy, log loss, ECE, ESS ratio — cross-fitted wherever possible.

    Every metric here is computed on out-of-fold predictions, because all four
    are optimistic in-sample and the ESS ratio badly so: a model that has
    memorized its training rows assigns the taken action a near-1 propensity, so
    1/pi_b ~ 1 and overlap looks excellent no matter how thin the real support
    is. Rows whose class could not be cross-fitted are excluded from the metrics
    and counted in `n_excluded_rare`.
    """
    seed = cfg["seed"]
    X = ds.df[ds.state_cols]
    y = ds.df[ds.action_col].to_numpy()
    n = len(y)
    min_count = int(pd.Series(y).value_counts().min())

    splits, rare = _cv_splits(y, cfg["behavior_model"].get("cross_fit_folds", 3), seed)

    if splits:
        full = np.full((n, ds.n_actions), np.nan)
        for tr, te in splits:
            fold_pipe, _ = _fit_pipeline(ds, cfg, X.iloc[tr], y[tr], seed)
            block = np.zeros((len(te), ds.n_actions))
            block[:, fold_pipe.named_steps["clf"].classes_] = fold_pipe.predict_proba(X.iloc[te])
            full[te] = block
        covered = ~np.isnan(full).any(axis=1)
        n_splits, source = len(splits), "oof"
    else:
        # No cross-fitting possible at all: report in-sample numbers, clearly
        # flagged, rather than crash. These are optimistic — read them as such.
        full = np.zeros((n, ds.n_actions))
        full[:, model.classes_] = model.pipeline.predict_proba(X)
        covered = np.ones(n, dtype=bool)
        n_splits, source = 0, "in_sample"

    full = clip_normalize(np.nan_to_num(full, nan=0.0), model.clip)
    y_c, p_c = y[covered], full[covered]
    preds = p_c.argmax(axis=1)
    base_rate = pd.Series(y_c).value_counts(normalize=True).max()

    return {
        "oof_accuracy": float(accuracy_score(y_c, preds)),
        "majority_baseline": float(base_rate),
        "oof_log_loss": float(log_loss(y_c, p_c, labels=list(range(ds.n_actions)))),
        "ece": float(_expected_calibration_error(y_c, p_c)),
        "ess_ratio": float(_ess_ratio(p_c, y_c)),
        "n_splits": n_splits,
        "calibrated": calibrated,
        "diagnostics_source": source,
        "min_class_count": min_count,
        "n_diagnostic": int(covered.sum()),
        "n_excluded_rare": int((~covered).sum()),
        "rare_classes": sorted(set(y[rare].tolist())),
        "action_counts": pd.Series(y).value_counts().sort_index().to_dict(),
    }


def _expected_calibration_error(y: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """ECE on the predicted-probability of the realized action (10 bins)."""
    conf = proba[np.arange(len(y)), y]
    correct = (proba.argmax(axis=1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return ece


def _ess_ratio(proba: np.ndarray, y: np.ndarray) -> float:
    """Effective-sample-size ratio of the IPW weights — overlap quality in [0,1].

    Low ESS means a handful of low-propensity plays dominate the inverse weights,
    so downstream IPW/DR is high-variance and pi* must stay close to pi_b. Only
    meaningful on out-of-fold propensities: in-sample it measures memorization.
    """
    n = len(y)
    p_taken = proba[np.arange(n), y]
    w = 1.0 / p_taken
    w /= w.mean()
    ess = n / (w ** 2).mean()
    return ess / n
