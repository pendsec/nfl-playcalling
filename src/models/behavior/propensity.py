"""
Behavior Policy Model — pi_b(A | S).

V1: multinomial logistic regression predicting which of the 4 defensive calls
the actual DC made in state S. This is the engine of IPW/DR later; in V1 it is
used to (a) report calibration and (b) constrain the learned policy to actions
with adequate behavioral support (positivity).

Calibration is tracked alongside accuracy because miscalibrated propensities
silently break inverse-weighting downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from ..preprocess import build_state_preprocessor
from ...schemas.dataset import Dataset
from ...schemas.behavior import BehaviorModel, clip_normalize


def fit_behavior_model(ds: Dataset, cfg: dict) -> tuple[BehaviorModel, dict]:
    """Fit pi_b on `ds` and return the model plus in-sample diagnostics."""
    bcfg = cfg["behavior_model"]
    clip = tuple(cfg["ope"]["clip_propensity"])

    pipe = Pipeline([
        ("prep", build_state_preprocessor(ds)),
        ("clf", LogisticRegression(
            C=bcfg["C"], max_iter=bcfg["max_iter"],
            class_weight="balanced", random_state=cfg["seed"],
        )),
    ])

    X = ds.df[ds.state_cols]
    y = ds.df[ds.action_col].to_numpy()
    pipe.fit(X, y)
    model = BehaviorModel(
        pipeline=pipe, classes_=pipe.named_steps["clf"].classes_,
        n_actions=ds.n_actions, clip=clip,
    )

    diagnostics = _diagnose(ds, pipe, model, cfg)
    return model, diagnostics


def _diagnose(ds: Dataset, pipe: Pipeline, model: BehaviorModel, cfg: dict) -> dict:
    """Cross-fitted accuracy, log loss, and a coarse calibration (ECE) check."""
    X = ds.df[ds.state_cols]
    y = ds.df[ds.action_col].to_numpy()
    n_splits = min(cfg["behavior_model"]["cross_fit_folds"],
                   int(pd.Series(y).value_counts().min()))
    n_splits = max(n_splits, 2)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
    oof = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")
    oof_full = np.zeros((len(y), ds.n_actions))
    oof_full[:, model.classes_] = oof
    oof_full = clip_normalize(oof_full, model.clip)

    preds = oof_full.argmax(axis=1)
    base_rate = pd.Series(y).value_counts(normalize=True).max()
    return {
        "oof_accuracy": float(accuracy_score(y, preds)),
        "majority_baseline": float(base_rate),
        "oof_log_loss": float(log_loss(y, oof_full, labels=list(range(ds.n_actions)))),
        "ece": float(_expected_calibration_error(y, oof_full)),
        "n_splits": n_splits,
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
