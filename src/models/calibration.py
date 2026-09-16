"""
Probability calibration — metrics and temperature scaling.

Shared by the behavior model (pi_b) and the shell imputer, because both produce
multiclass probabilities that are consumed AS probabilities rather than as
labels: pi_b's propensities are inverted into importance weights, and the
imputer's posterior is sampled from. In both cases a miscalibrated distribution
corrupts everything downstream while accuracy looks fine.

A note on the metric, because this module exists partly to correct one
----------------------------------------------------------------------
Earlier versions binned by the predicted probability of the TRUE class and
compared it against whether the ARGMAX was correct. Those are different
quantities, and the mismatch reports a large error for a perfectly calibrated
model. Feeding it probabilities that ARE the data-generating distribution:

    6 shells, flat posterior   old metric 0.249   confidence ECE 0.005
    6 shells, sharper          old metric 0.220   confidence ECE 0.004
    2 classes, confident       old metric 0.135   confidence ECE 0.002

So a "0.27 ECE" under the old metric was mostly the metric. `confidence_ece`
and `classwise_ece` below are the standard definitions and read ~0 on
well-calibrated input.

Which metric to read
--------------------
  * `confidence_ece`  — is the model's stated confidence in its top pick right?
    The one to watch when the argmax is what gets acted on.
  * `classwise_ece`   — is EVERY class's probability right, not just the top
    one? The one that matters when the whole distribution is consumed, as it is
    for importance weights and for sampling. Strictly harder, and it moves much
    less under any calibration method.
  * `brier_score`     — a proper scoring rule; unlike ECE it cannot be gamed by
    a constant predictor, so it guards against "calibrated but useless".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _binned(values: np.ndarray, target: np.ndarray, n_bins: int) -> float:
    """Weighted mean |target - values| over equal-width bins of `values`."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (values >= lo) & (values < hi)
        if m.any():
            total += m.mean() * abs(target[m].mean() - values[m].mean())
    return float(total)


def confidence_ece(y: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Standard multiclass ECE: top-class confidence vs top-class accuracy."""
    y = np.asarray(y)
    return _binned(proba.max(axis=1), (proba.argmax(axis=1) == y).astype(float), n_bins)


def classwise_ece(y: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """One-vs-rest ECE averaged over classes — the whole distribution, not the top.

    This is the metric that matters when the probabilities are inverted into
    weights or sampled from, since those use every column rather than the max.
    """
    y = np.asarray(y)
    return float(np.mean([
        _binned(proba[:, k], (y == k).astype(float), n_bins)
        for k in range(proba.shape[1])
    ]))


def brier_score(y: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass Brier score (lower is better). A proper scoring rule."""
    y = np.asarray(y)
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((proba - onehot) ** 2).sum(axis=1).mean())


def reliability_table(y: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    """Per-bin confidence vs accuracy — the reliability diagram, as data."""
    y = np.asarray(y)
    conf, correct = proba.max(axis=1), (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                         "mean_confidence": float(conf[m].mean()),
                         "accuracy": float(correct[m].mean()),
                         "gap": float(correct[m].mean() - conf[m].mean())})
    return pd.DataFrame(rows)


@dataclass
class TemperatureScaler:
    """Single-parameter recalibration: p' proportional to p ** (1 / T).

    Chosen over isotonic/sigmoid wrappers on measured grounds (2022 train, 2023
    calibration, 2024 test, 6 coverage shells):

        raw LightGBM        confidence ECE 0.138   Brier 0.791
        isotonic CV=3                      0.070         0.767
        sigmoid  CV=5                      0.076         0.767
        temperature (T=1.87)               0.022         0.763

    Three properties earn it the default beyond the numbers: one parameter
    cannot overfit a calibration set the way a per-class isotonic fit can; the
    transform is monotone, so the ranking and therefore accuracy are untouched
    and only the confidence changes; and it needs no cross-validated refit of
    the base model. T > 1 means the base model was overconfident.
    """
    temperature: float = 1.0

    def transform(self, proba: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0) ** (1.0 / self.temperature)
        return p / p.sum(axis=1, keepdims=True)

    @classmethod
    def fit(cls, proba: np.ndarray, y: np.ndarray,
            bounds: tuple[float, float] = (0.2, 8.0)) -> "TemperatureScaler":
        """Pick T minimizing held-out negative log-likelihood.

        `proba` must come from data the base model did not train on, or T is
        fitted to memorized predictions and collapses toward 1.
        """
        from scipy.optimize import minimize_scalar

        y = np.asarray(y)
        idx = np.arange(len(y))
        logp = np.log(np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0))

        def nll(t: float) -> float:
            scaled = logp / t
            scaled -= scaled.max(axis=1, keepdims=True)
            return float(-(scaled[idx, y] - np.log(np.exp(scaled).sum(axis=1))).mean())

        best = minimize_scalar(nll, bounds=bounds, method="bounded")
        return cls(temperature=float(best.x))


def calibration_report(y: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> dict:
    """The three metrics together — never ECE alone.

    ECE is minimized by a constant predictor that reports the base rate, so a
    calibration number without a proper scoring rule beside it cannot tell
    "well calibrated" from "usefully wrong in a well-calibrated way".
    """
    return {
        "confidence_ece": confidence_ece(y, proba, n_bins),
        "classwise_ece": classwise_ece(y, proba, n_bins),
        "brier": brier_score(y, proba),
    }
