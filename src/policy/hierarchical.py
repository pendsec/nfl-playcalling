"""
Policy Learning — the factored policy: alignment, then coverage.

    pi(a1, a2 | s) = pi1(a1 | s) * pi2(a2 | s, a1)

Level 1 (personnel package x front) is learned on every snap. Level 2 (coverage
shell x pressure) is learned on charted dropbacks with the alignment in its
state, so it answers "given this alignment, which coverage" — including for an
alignment the level-1 policy recommends rather than the one the coordinator
called. Each level is a safe improvement over the coordinator at that level, so
the factored policy can keep either level's mix.

Evaluated on charted dropbacks, the only sample where both levels are observed,
by DR over the joint space (ids `a1 * K2 + a2`):

    Q(s, a1, a2)      the level-2 Q with the alignment columns set to a1
    pi_b(a1, a2 | s)  P(a1 | s, dropback) * pi_b2(a2 | s, a1)

The estimand is per dropback with the play held a dropback. What an alignment
does to WHETHER the offense drops back is level 1's question, answered on every
snap; here it is outside. That is also why the alignment propensity is fit on
the dropbacks themselves: the dropback is a descendant of the alignment
(deterrence), so P(a1 | s, dropback) is not the every-snap pi_b1. A level-1
policy that defers is read as playing it — "do what the coordinator does", on
this sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.features import (L1_STATE_COLUMNS, level1_components, level1_labels,
                             state_columns)
from ..models.behavior.propensity import fit_behavior_model
from ..ope.doubly_robust import dr_per_play
from ..schemas.behavior import BehaviorModel
from ..schemas.dataset import Dataset
from ..stats import mean_se
from .safe import defer_to


def alignment_frames(df: pd.DataFrame, cfg: dict) -> list[pd.DataFrame]:
    """One copy of `df` per level-1 call, with the alignment columns set to it.

    `cfg` is the level-1 configuration. Scoring a level-2 model on frame a1 asks
    what it would say had the coordinator lined up in a1.
    """
    k1 = len(level1_labels(cfg))
    comps = level1_components(np.arange(k1), cfg)
    frames = []
    for a1 in range(k1):
        f = df.copy()
        for axis, values in comps.items():
            f[L1_STATE_COLUMNS[axis]] = float(values[a1])
        frames.append(f)
    return frames


def dropback_alignment_dataset(ds2: Dataset, cfg: dict) -> Dataset:
    """Level-2 rows relabeled with their level-1 call, over the level-1 state
    (not the alignment columns — they ARE the label). `cfg` is level 1's."""
    numeric, categorical = state_columns(cfg)
    labels = level1_labels(cfg)
    return Dataset(df=ds2.df.assign(action=ds2.df["l1_action"].to_numpy()),
                   numeric_state=numeric, categorical_state=categorical,
                   action_col="action", reward_col=ds2.reward_col,
                   n_actions=len(labels), action_labels=labels)


def fit_dropback_alignment_behavior(ds2: Dataset, cfg: dict) -> tuple[BehaviorModel, dict]:
    """P(a1 | s, dropback): the coordinator's alignment on charted dropbacks,
    fit on the level-2 training rows. `cfg` is the level-1 configuration."""
    return fit_behavior_model(dropback_alignment_dataset(ds2, cfg), cfg)


def factor(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """(n, K1) x (n, K1, K2) -> joint (n, K1 * K2) over ids a1 * K2 + a2."""
    return (p1[:, :, None] * p2).reshape(len(p1), -1)


def level1_on_dropbacks(policy1, beh1_db: BehaviorModel, df: pd.DataFrame) -> dict:
    """The level-1 policy on dropback rows, deferring to P(a1 | s, dropback)."""
    out = policy1.act(df)
    deferred = out.get("deferred")
    return {"probs": defer_to(out, beh1_db.propensity(df)),
            "deferred": np.zeros(len(df), bool) if deferred is None else deferred,
            "chosen": out["chosen"]}


def level2_under_each_alignment(policy2, df: pd.DataFrame, cfg: dict) -> dict:
    """Level-2 policy, behavior and Q under every alignment: each (n, K1, K2)."""
    pol, beh, q, deferred = [], [], [], []
    for f in alignment_frames(df, cfg):
        out = policy2.act(f)
        pol.append(out["policy_probs"])
        beh.append(out["pi_b"])
        q.append(out["q_all"])
        deferred.append(out.get("deferred", np.zeros(len(df), bool)))
    return {"policy": np.stack(pol, axis=1), "behavior": np.stack(beh, axis=1),
            "q": np.stack(q, axis=1), "deferred": np.stack(deferred, axis=1)}


def evaluate_factored(df: pd.DataFrame, cfg: dict, policy1, policy2, beh1_db: BehaviorModel,
                      weight_clip: float = 20.0,
                      judges: tuple | None = None) -> dict:
    """Paired DR of the four factorings against the coordinator, on dropbacks.

    coordinator     — his alignment, his coverage (the logging policy itself)
    alignment only  — the level-1 policy, coverage as he would pair it
    coverage only   — his alignment, the level-2 policy
    both levels     — the full factored policy

    Returns the table plus the joint arrays, so callers (the V2 -> V3 gate) can
    score other joint policies against the same nuisances.

    `judges` = (alignment judge, coverage judge) from
    `src.models.behavior.judge.fit_judge_behavior`: the coordinator's mix, and
    the DR weights, come from them instead of the policies' train-fit pi_b. The
    policies still defer to their own pi_b — they never see the scored season.
    """
    k2 = policy2.n_actions
    lvl1 = level1_on_dropbacks(policy1, beh1_db, df)
    lvl2 = level2_under_each_alignment(policy2, df, cfg)
    if judges is not None:
        judge1, judge2 = judges
        p1_b = judge1.propensity(df)
        lvl2["behavior"] = np.stack([judge2.propensity(f) for f in alignment_frames(df, cfg)],
                                    axis=1)
    else:
        p1_b = beh1_db.propensity(df)
    q_joint = lvl2["q"].reshape(len(df), -1)
    pi_b = factor(p1_b, lvl2["behavior"])
    actions = df["l1_action"].to_numpy() * k2 + df["action"].to_numpy()
    rewards = df["reward"].to_numpy()

    policies = {
        "coordinator": pi_b,
        "alignment only": factor(lvl1["probs"], lvl2["behavior"]),
        "coverage only": factor(p1_b, lvl2["policy"]),
        "both levels": factor(lvl1["probs"], lvl2["policy"]),
    }
    base = dr_per_play(q_joint, pi_b, pi_b, actions, rewards, weight_clip)
    games = df["game_id"].to_numpy()
    rows = []
    for name, probs in policies.items():
        score = dr_per_play(q_joint, pi_b, probs, actions, rewards, weight_clip)
        value, se = mean_se(score)
        lift, lift_se = mean_se(score - base, games)        # paired, games clustered
        rows.append({"policy": name, "value": value, "se": se, "lift": lift,
                     "lift_se": lift_se})
    logged = df["l1_action"].to_numpy()
    return {
        "table": pd.DataFrame(rows),
        "q_joint": q_joint, "pi_b": pi_b, "actions": actions, "rewards": rewards,
        "policies": policies, "p1_behavior": p1_b,
        "level1_departs": float((~lvl1["deferred"]).mean()),
        # level 2's departures at the alignment actually called
        "level2_departs": float((~lvl2["deferred"][np.arange(len(df)), logged]).mean()),
    }
