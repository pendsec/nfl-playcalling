"""
The coordinator's-read allowance — a bias benchmark for the safe policy's bound.

The coordinator sets the box partly from a read of what the offense is about to
do, and that read is not in the data (`offense_intent` -> `coach_read` in the
graph). Its footprint is: heavy boxes face more runs than the pre-snap state
predicts, light boxes fewer. Runs and passes are worth different amounts to
the defense, so part of Q's edge for a heavy box is "it faced the plays it was
set for" — a bias the game bootstrap cannot see, because it points the same
way in every resample.

Benchmark, per component c of a confounded axis (the box count):

    excess_run(c) = mean over snaps with c of [ 1{run} - P_hat(run | state) ]
    gap(c)        = mean reward on runs with c - mean reward on passes with c
    bias(c)       = excess_run(c) * gap(c)

The gap must be taken within the component: league-wide, runs and passes are
worth about the same to the defense (~0.02), but against a light box a run is
~0.14 worse for the defense than a pass and against a 7-8+ box ~0.1 better.
So both extremes are inflated — light boxes avoid the runs that hurt them,
heavy boxes get the runs they stop — at ~0.006-0.007 each, box 6 at ~0.

P_hat is cross-fit by game on the training seasons from the state alone. The
bias of a call's advantage over the coordinator's mix is then
bias(c(a)) - sum_a' pi_b(a'|s) bias(c(a')); the safe policy subtracts its
absolute value (times `scale`) from the lower bound, so a box departure has to
clear the known bias as well as the sampling noise. It is an allowance, not a
correction: the sign is not trusted, only the size.

This is the run/pass channel of the read only — the one the data shows. It is
not a bound on all confounding.

Configured by `policy.read_allowance` {axes, scale}. Axes not in the configured
action space are skipped, so dropping the box from the call
(`action.level1.axes: [package]`) turns the allowance off with no other edit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas.dataset import Dataset


def allowance_axes(cfg: dict) -> list[str]:
    """Configured allowance axes that exist in the current action space."""
    from ..data.features import action_scheme, level1_axes

    ra = cfg.get("policy", {}).get("read_allowance") or {}
    if action_scheme(cfg) != "level1" or not ra.get("scale", 0.0):
        return []
    return [a for a in ra.get("axes", []) if a in level1_axes(cfg)]


def _state_run_probability(ds: Dataset, cfg: dict) -> np.ndarray:
    """P(run | pre-snap state), cross-fit with whole games held out."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import GroupKFold
    from ..models.preprocess import build_gbm_preprocessor

    X = build_gbm_preprocessor(ds).fit_transform(ds.df[ds.state_cols])
    y = (ds.df["play_type"] == "run").to_numpy().astype(int)
    p = np.empty(len(y))
    folds = GroupKFold(n_splits=cfg["outcome_model"].get("cross_fit_folds", 5))
    for tr, va in folds.split(X, y, groups=ds.df["game_id"].to_numpy()):
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                             random_state=cfg.get("seed", 0), verbose=-1)
        clf.fit(X[tr], y[tr])
        p[va] = clf.predict_proba(X[va])[:, 1]
    return p


def estimate_read_bias(ds: Dataset, cfg: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-action read bias (reward units) and the per-component table behind it.

    Returns zeros and an empty table when no allowance axis is configured.
    Fit on training seasons only — the holdout never sets the bound it is
    judged under.
    """
    from ..data.features import LEVEL1_AXES, level1_components

    axes = allowance_axes(cfg)
    per_action = np.zeros(ds.n_actions)
    if not axes:
        return per_action, pd.DataFrame()

    df = ds.df
    is_run = (df["play_type"] == "run").to_numpy()
    reward = df[ds.reward_col].to_numpy()
    resid = is_run - _state_run_probability(ds, cfg)

    taken = level1_components(df[ds.action_col].to_numpy(), cfg)
    ids = level1_components(np.arange(ds.n_actions), cfg)
    rows = []
    for axis in axes:
        values = LEVEL1_AXES[axis]
        excess, gap = np.zeros(len(values)), np.zeros(len(values))
        for c in range(len(values)):
            m = taken[axis] == c
            if m.any() and (m & is_run).any() and (m & ~is_run).any():
                excess[c] = resid[m].mean()
                gap[c] = reward[m & is_run].mean() - reward[m & ~is_run].mean()
        bias = excess * gap
        per_action += bias[ids[axis]]
        rows += [{"axis": axis, "component": v, "n": int((taken[axis] == c).sum()),
                  "excess_run": excess[c], "run_minus_pass": gap[c], "bias": bias[c]}
                 for c, v in enumerate(values)]
    return per_action, pd.DataFrame(rows)
