"""
Calibration metric and temperature-scaling tests.

The first test here is a regression test for a metric, not a model. The ECE
previously used by both `pi_b` and the shell imputer binned by the predicted
probability of the TRUE class while scoring whether the ARGMAX was correct.
Those are different quantities, so it reported a large error for probabilities
that were perfectly calibrated by construction — which made "our ECE is 0.27"
a statement about the metric rather than the model.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.calibration import (TemperatureScaler, brier_score,
                                    calibration_report, classwise_ece,
                                    confidence_ece, reliability_table)


def _perfect(n=30000, k=6, concentration=0.9, seed=0):
    """Probabilities that ARE the data-generating distribution, plus draws."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(k, concentration), size=n)
    y = np.array([rng.choice(k, p=row) for row in p])
    return y, p


@pytest.mark.parametrize("concentration", [0.9, 0.25])
def test_metrics_read_near_zero_on_perfectly_calibrated_input(concentration):
    """The defining property. The old metric scored ~0.25 here."""
    y, p = _perfect(concentration=concentration)
    assert confidence_ece(y, p) < 0.02
    assert classwise_ece(y, p) < 0.02


def test_old_true_class_metric_would_have_failed_that():
    """Pin the bug so it cannot quietly return.

    Reproduces the previous implementation and shows it reports a large error on
    input that is calibrated by construction.
    """
    y, p = _perfect()
    conf = p[np.arange(len(y)), y]                 # prob of the TRUE class
    correct = (p.argmax(axis=1) == y).astype(float)  # ...vs ARGMAX correctness
    edges = np.linspace(0, 1, 11)
    old = sum(
        m.mean() * abs(correct[m].mean() - conf[m].mean())
        for lo, hi in zip(edges[:-1], edges[1:])
        if (m := (conf >= lo) & (conf < hi)).any()
    )
    assert old > 0.15, "the old metric should be badly wrong here"
    assert confidence_ece(y, p) < 0.02


def test_temperature_fit_recovers_a_known_overconfidence():
    """An overconfident model sharpened by a known factor must be undone by it."""
    y, p = _perfect()
    sharpen = 2.0
    over = p ** sharpen
    over /= over.sum(axis=1, keepdims=True)

    scaler = TemperatureScaler.fit(over, y)
    assert scaler.temperature == pytest.approx(sharpen, rel=0.15)
    assert confidence_ece(y, scaler.transform(over)) < confidence_ece(y, over)


def test_temperature_scaling_never_changes_the_ranking():
    """Monotone transform: confidence moves, accuracy does not.

    This is why temperature is safe to apply to an imputer whose argmax is
    already no better than the marginal — it cannot make the point predictions
    worse, only the probabilities more honest.
    """
    y, p = _perfect()
    over = p ** 2.5
    over /= over.sum(axis=1, keepdims=True)
    scaled = TemperatureScaler(temperature=1.9).transform(over)

    assert np.array_equal(over.argmax(axis=1), scaled.argmax(axis=1))
    assert np.allclose(scaled.sum(axis=1), 1.0)
    assert (scaled >= 0).all()


def test_identity_temperature_is_a_no_op():
    _, p = _perfect(n=500)
    assert np.allclose(TemperatureScaler(temperature=1.0).transform(p), p, atol=1e-9)


def test_brier_penalises_a_calibrated_but_uninformative_predictor():
    """Why the report never ships ECE alone.

    A constant base-rate predictor is perfectly calibrated and completely
    useless. ECE cannot tell it apart from a good model; a proper scoring rule
    can, which is why `calibration_report` returns all three together.
    """
    y, p = _perfect()
    base = np.tile(np.bincount(y, minlength=p.shape[1]) / len(y), (len(y), 1))

    assert confidence_ece(y, base) < 0.05          # "perfectly calibrated"
    assert brier_score(y, base) > brier_score(y, p)  # ...but strictly worse
    assert set(calibration_report(y, p)) == {"confidence_ece", "classwise_ece", "brier"}


def test_reliability_table_bins_are_well_formed():
    y, p = _perfect(n=5000)
    t = reliability_table(y, p, n_bins=10)
    assert not t.empty and t["n"].sum() == len(y)
    assert (t["bin_lo"] < t["bin_hi"]).all()
    assert np.allclose(t["gap"], t["accuracy"] - t["mean_confidence"])
