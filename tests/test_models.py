"""
Behavior policy and outcome model: honest diagnostics and graceful degradation.

Two failure modes get most of the attention. The propensity diagnostics must be
cross-fitted even when one action is too rare to stratify, because in-sample
numbers measure memorization. And a Q-cell too thin to fit a regressor must not
contribute an extreme constant to every state.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.behavior.propensity import _cv_splits
from src.models.outcome.q_model import _shrunk_cell_mean


def test_pi_b_beats_majority_and_calibrated(fitted):
    diag = fitted["diag"]
    assert diag["oof_accuracy"] > diag["majority_baseline"]
    # Threshold tightened from 0.30 now that the metric measures calibration.
    # The old ECE binned by the TRUE class's probability while scoring ARGMAX
    # correctness, and read ~0.25 for a perfectly calibrated model — so 0.30 was
    # passing on essentially anything.
    assert diag["confidence_ece"] < 0.10
    assert diag["classwise_ece"] < 0.10
    assert diag["brier"] > 0.0
    assert 0.0 < diag["ess_ratio"] <= 1.0


def test_q_recovers_known_best_response(fitted):
    """On long downs the deep-zone blitz should out-Q a tight no-blitz call.

    True SCM: ideal_shell rises with distance and blitz helps on long downs, so
    e.g. cover4_blitz (a=9) should beat cover1_no_blitz (a=2) when ydstogo is big.
    """
    test, q = fitted["test"], fitted["q"]
    long_df = test.df[test.df["ydstogo"] >= 13]
    assert len(long_df) > 20
    q_all = q.predict_all_actions(long_df)
    assert q_all[:, 9].mean() > q_all[:, 2].mean()


def test_cv_splits_keeps_unsplittable_class_on_the_training_side():
    """One singleton class must not disable cross-fitting for every other action.

    This is what would make the propensity diagnostics dishonest: a lone play in one
    of twelve cells sent accuracy/ECE/ESS down an in-sample fallback path, and an
    in-sample ESS ratio measures memorization, not overlap.
    """
    y = np.array([0] * 50 + [1] * 50 + [2])       # class 2 has a single member
    splits, rare = _cv_splits(y, requested=3, seed=0)

    assert splits, "a single un-splittable class must not veto cross-fitting"
    assert rare.sum() == 1 and rare[-1]

    singleton = len(y) - 1
    for train_idx, test_idx in splits:
        assert singleton in train_idx      # still informs every fit
        assert singleton not in test_idx   # but never scored out-of-fold

    # Every splittable row is scored exactly once across the folds.
    scored = np.concatenate([te for _, te in splits])
    assert sorted(scored.tolist()) == list(range(len(y) - 1))


def test_behavior_diagnostics_are_out_of_fold(fitted):
    """pi_b diagnostics must be cross-fitted and account for every training row."""
    diag, train = fitted["diag"], fitted["train"]
    assert diag["diagnostics_source"] == "oof"
    assert diag["n_splits"] >= 2
    assert diag["n_diagnostic"] + diag["n_excluded_rare"] == len(train.df)
    # An in-sample ESS ratio on a memorized fit sits near 1; an honest one does not.
    assert diag["ess_ratio"] < 0.95


def test_thin_q_cells_shrink_toward_the_global_mean():
    """A cell too thin for a regressor must not contribute an extreme constant.

    On real data a 9-play cell produced a -2.30 fallback against a global mean
    near -0.09, and because a fallback is constant across every state it then
    dominated the policy's argmax everywhere.
    """
    global_mean = 0.0
    one_play = _shrunk_cell_mean(np.array([-5.0]), global_mean, 20)
    near_full = _shrunk_cell_mean(np.full(19, -5.0), global_mean, 20)

    assert _shrunk_cell_mean(np.array([]), global_mean, 20) == global_mean
    assert one_play == pytest.approx(-5.0 / 21.0)      # a lone play barely moves it
    assert abs(one_play) < abs(near_full) < 5.0        # graceful with cell size
    assert near_full > -5.0                            # never the raw cell mean
