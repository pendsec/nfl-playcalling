"""Shared synthetic-SCM configuration for the test suite.

A plain module rather than a `conftest.py` constant so every test file can
import it explicitly — the fixtures in `conftest.py` build on it, but a test
that only needs the config should not have to take a fixture to reach it.
"""

from __future__ import annotations

import os
import sys

# Repo root, so `from src...` resolves however pytest was invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Light config so the calibrated-GBM fits stay fast in CI. Deliberately mirrors
# the shape of configs/v2.yaml rather than loading it: the tests assert against a
# known SCM, and should not start failing because a real-data config was retuned.
CFG = {
    "data": {"team": "SYN", "down": "all", "holdout_season": 2023},
    "action": {"blitz_rusher_threshold": 5, "drop_uncharted_coverage": True},
    "reward": {"negate_epa": True},
    "behavior_model": {"type": "lightgbm", "n_estimators": 120, "learning_rate": 0.05,
                       "max_depth": 5, "min_child_samples": 30,
                       "calibration_folds": 3, "cross_fit_folds": 2},
    "outcome_model": {"type": "lightgbm", "n_estimators": 150, "learning_rate": 0.05,
                      "max_depth": 5, "min_child_samples": 20, "min_cell": 20,
                      "cross_fit_folds": 5},
    "policy": {"alpha": 1.0, "support_threshold": 0.05},
    "ope": {"clip_propensity": [0.01, 0.99], "weight_clip": 20.0},
    "evaluation": {"baseline_name": "greedy_baseline",
                   "candidate_name": "conservative_candidate",
                   "regression_margin": 0.02},
    "seed": 42,
}
WEIGHT_CLIP = CFG["ope"]["weight_clip"]
N_SYNTHETIC_PLAYS = 9000
SYNTHETIC_SEED = 7
