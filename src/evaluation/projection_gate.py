"""
V2 -> V3 regression gate, by projection into the joint space.

V2 and V3 do not answer the same question. V2 recommends a coverage shell x
pressure for one defense on charted dropbacks, blind to the alignment; V3
recommends an alignment on every snap and a coverage given it. A direct
comparison of their policy values would compare different estimands.

So both are expressed as policies over the same joint space — alignment x
coverage — on the sample where both are defined: the gate team's charted
dropbacks in the held-out season.

    V2:  P(a1 | s, dropback) * V2's coverage policy, projected onto the level-2 calls
    V3:  the factored policy, pi1(a1 | s) * pi2(a2 | s, a1)

V2's alignment follows the coordinator because V2 never chose one. Its 12-call
coverage policy is projected onto the level-2 taxonomy by summing probability
within each level-2 call (under `shells: family`, Cover-1 and Cover-3 both land
on single-high). Both are scored by paired DR with the same joint Q and joint
propensities — V3's — so the gate isolates the policies. V3 must not be
significantly worse on V2's own question; failing to prove it better is fine.

What the gate charges V3 for, by construction: its alignment departures earn
their value against the run (deterrence, run fits), and runs are outside a
per-dropback estimand. A heavier alignment can only look costly here. The
non-inferiority margin is what keeps that from blocking a policy that is better
where it is meant to be.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import yaml

from .version_compare import VersionComparison, compare_policy_arrays
from ..data.features import (_FAMILY_OF, build_dataset, level2_labels,
                             level2_shell_mode, time_aware_split)
from ..data.load import COVERAGE_SHELLS


def v2_to_level2(cfg: dict) -> np.ndarray:
    """(12, K2) 0/1 matrix: V2 call (2 * shell + blitz, six shells) -> level-2 call."""
    k2 = len(level2_labels(cfg))
    m = np.zeros((2 * len(COVERAGE_SHELLS), k2))
    for shell, name in enumerate(COVERAGE_SHELLS):
        target = _FAMILY_OF[name] if level2_shell_mode(cfg) == "family" else shell
        for b in (0, 1):
            m[2 * shell + b, 2 * target + b] = 1.0
    return m


def v2_policy_on_holdout(pbp: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Refit V2's pipeline on the training seasons; its policy on the holdout.

    V2's own configuration (`evaluation.projection_gate.baseline_config`), with
    only the seasons and holdout moved to V3's. Returns the holdout keys and V2's
    12-call policy probabilities.
    """
    from ..models.behavior.propensity import fit_behavior_model
    from ..models.outcome.q_model import fit_q_model
    from ..policy.conservative import learn_conservative_policy

    gate = cfg["evaluation"]["projection_gate"]
    with open(gate["baseline_config"]) as f:
        v2 = yaml.safe_load(f)
    v2 = copy.deepcopy(v2)
    v2["data"].update(seasons=cfg["data"]["seasons"], team=gate["team"],
                      holdout_season=cfg["data"]["holdout_season"])
    ds = build_dataset(pbp, v2)
    train, test = time_aware_split(ds, v2["data"]["holdout_season"])
    beh, _ = fit_behavior_model(train, v2)
    q = fit_q_model(train, v2)
    probs = learn_conservative_policy(train, q, beh, v2).act(test.df)["policy_probs"]
    out = test.df[["game_id", "play_id"]].copy()
    out["_v2_probs"] = list(probs)
    return out


def projection_gate(pbp: pd.DataFrame, cfg: dict, test2: pd.DataFrame, factored: dict,
                    *, graph_version: str) -> VersionComparison:
    """Gate V3's factored policy against V2's on the gate team's holdout dropbacks.

    `test2` is the level-2 holdout frame and `factored` the result of
    `policy.hierarchical.evaluate_factored` on it (its joint Q, propensities and
    policies are reused, row-aligned with `test2`).
    """
    gate = cfg["evaluation"]["projection_gate"]
    v2 = v2_policy_on_holdout(pbp, cfg)
    rows = test2[["game_id", "play_id", "defteam"]].reset_index(drop=True)
    rows["_row"] = np.arange(len(rows))
    hit = rows[rows["defteam"] == gate["team"]].merge(v2, on=["game_id", "play_id"])
    idx = hit["_row"].to_numpy()
    if len(idx) < 2:
        raise ValueError(f"no {gate['team']} holdout dropbacks shared by V2 and V3")

    proj = np.vstack(hit["_v2_probs"].to_numpy()) @ v2_to_level2(cfg)
    p1_b = factored["p1_behavior"][idx]
    v2_joint = (p1_b[:, :, None] * proj[:, None, :]).reshape(len(idx), -1)
    v3_joint = factored["policies"]["both levels"][idx]
    return compare_policy_arrays(
        factored["q_joint"][idx], factored["pi_b"][idx],
        factored["actions"][idx], factored["rewards"][idx],
        v2_joint, v3_joint,
        weight_clip=cfg["ope"].get("weight_clip", 20.0),
        margin=cfg["evaluation"].get("regression_margin", 0.02),
        baseline_name="v2_coverage", candidate_name="v3_factored",
        graph_version=graph_version, clusters=hit["game_id"].to_numpy(),
    )
