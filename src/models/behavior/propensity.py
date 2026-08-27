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
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from ..preprocess import build_gbm_preprocessor
from ...schemas.dataset import Dataset
from ...schemas.behavior import BehaviorModel, clip_normalize


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


def _safe_folds(y: np.ndarray, requested: int) -> int:
    """Largest fold count <= requested for which every class is CV-splittable."""
    return max(2, min(requested, int(pd.Series(y).value_counts().min())))


def _make_classifier(cfg: dict, y: np.ndarray) -> Pipeline:
    """Calibrated LGBM when classes are dense enough for CV; else raw LGBM.

    Isotonic calibration needs every class to appear in each calibration fold.
    On sparse single-team slices some 12-action cells have a lone play, so we
    fall back to the uncalibrated classifier rather than crash — the run reports
    which path was taken, and the positivity check flags the same sparsity.
    """
    base = _base_classifier(cfg)
    min_count = int(pd.Series(y).value_counts().min())
    if min_count >= 2:
        folds = _safe_folds(y, cfg["behavior_model"].get("calibration_folds", 3))
        return CalibratedClassifierCV(base, cv=folds, method="isotonic"), True
    return base, False


def fit_behavior_model(ds: Dataset, cfg: dict) -> tuple[BehaviorModel, dict]:
    """Fit pi_b on `ds` and return the model plus OOF diagnostics."""
    clip = tuple(cfg["ope"]["clip_propensity"])
    X = ds.df[ds.state_cols]
    y = ds.df[ds.action_col].to_numpy()

    clf, calibrated = _make_classifier(cfg, y)
    pipe = Pipeline([("prep", build_gbm_preprocessor(ds)), ("clf", clf)])
    pipe.fit(X, y)
    model = BehaviorModel(
        pipeline=pipe, classes_=pipe.named_steps["clf"].classes_,
        n_actions=ds.n_actions, clip=clip,
    )

    diagnostics = _diagnose(X, y, cfg, model, calibrated)
    return model, diagnostics


def _diagnose(ds: Dataset, cfg: dict, model: BehaviorModel, calibrated: bool) -> dict:
    """Accuracy, log loss, ECE, ESS ratio — cross-fitted when classes allow.

    Honest out-of-fold diagnostics need StratifiedKFold with folds <= the
    smallest class. When a class is a singleton no stratified CV is possible, so
    we fall back to in-sample predictions (clearly flagged) rather than crash.
    """
    X = ds.df[ds.state_cols]
    y = ds.df[ds.action_col].to_numpy()
    min_count = int(pd.Series(y).value_counts().min())

    if min_count >= 2:
        # Safe to create striatified folds for calibration
        n_splits = _safe_folds(y, cfg["behavior_model"].get("cross_fit_folds", 3))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
        clf, _ = _make_classifier(cfg, y)
        pipe = Pipeline([("prep", build_gbm_preprocessor(ds)), ("clf", clf)])
        proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")
        source = "oof"
    else:
        # Fall back to base model
        n_splits = 0
        proba = model.pipeline.predict_proba(X)  # in-sample fallback
        source = "in_sample"

    full = np.zeros((len(y), ds.n_actions))
    full[:, model.classes_] = proba
    full = clip_normalize(full, model.clip)

    preds = full.argmax(axis=1)
    base_rate = pd.Series(y).value_counts(normalize=True).max()
    return {
        "oof_accuracy": float(accuracy_score(y, preds)),
        "majority_baseline": float(base_rate),
        "oof_log_loss": float(log_loss(y, full, labels=list(range(ds.n_actions)))),
        "ece": float(_expected_calibration_error(y, full)),
        "ess_ratio": float(_ess_ratio(full, y)),
        "n_splits": n_splits,
        "calibrated": calibrated,
        "diagnostics_source": source,
        "min_class_count": min_count,
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
    so downstream IPW/DR is high-variance and pi* must stay close to pi_b.
    """
    n = len(y)
    p_taken = proba[np.arange(n), y]
    w = 1.0 / p_taken
    w /= w.mean()
    ess = n / (w ** 2).mean()
    return ess / n
