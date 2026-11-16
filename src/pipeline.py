"""The pipeline's evaluation stages, shared by `run.py` and the walkthrough notebook.

Each function computes one stage and returns what it found; neither it nor
anything it calls prints. `run.py` prints and saves the results, the notebook
narrates and plots them — so the two can never disagree about a number.

    holdout_ope       stage 6: the judge, the policy's DR lift, the DM -> IPW ->
                      DR ladder, the sensitivity bounds, the AIPW contrasts
    level2            stages 8-9: the coverage given the alignment, the
                      factored policy, the V2 -> V3 projection gate
    invariance_stage  stage 10: invariance across team-seasons (+ transfer)

The models themselves come from `policy.safe.fit_policy`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data.features import build_level2_dataset, level2_config, time_aware_split
from .models.behavior.judge import fit_judge_behavior, judge_mode
from .models.behavior.propensity import fit_behavior_model
from .models.calibration import weight_balance
from .ope.aipw import aipw_contrasts
from .ope.doubly_robust import dr_per_play, dr_policy_value
from .ope.ladder import ope_ladder
from .ope.sensitivity import attach as attach_sensitivity, sensitivity_bounds
from .policy.safe import PolicyFit, defer_to, fit_policy, is_safe
from .schemas.dataset import Dataset
from .schemas.ope import OPEResult
from .scm import graph as scm
from .scm import identify
from .stats import mean_se


def judge_report(cfg: dict, actions: np.ndarray, p_policy: np.ndarray, p_judge: np.ndarray,
                 labels: dict, games: np.ndarray) -> dict:
    """How far the policy's train-fit pi_b is from the judge's on the scored season:
    log loss (and, when they differ, the paired difference, games clustered), the
    worst calibration-in-the-large gap, and the worst weight balance (1 = the
    inverse weights average right) over calls supported on at least 200 snaps."""
    thr = cfg["policy"].get("support_threshold", 0.05)
    k = p_policy.shape[1]
    share = np.bincount(actions, minlength=k) / len(actions)
    rows = np.arange(len(actions))
    row = {"judge": judge_mode(cfg), "n": len(actions)}
    losses = {}
    for tag, p in (("train_fit", p_policy), ("judge", p_judge)):
        losses[tag] = -np.log(np.clip(p[rows, actions], 1e-12, 1.0))
        citl = np.abs(p.mean(0) - share)
        # Worst over calls supported on enough snaps to measure: a call almost
        # never supported (dime/8+) reads 0 under any pi_b and says nothing.
        wb = np.where((p >= thr).sum(0) >= 200, weight_balance(actions, p, thr), np.nan)
        worst = int(np.nanargmax(np.abs(wb - 1)))
        row.update({f"{tag}_log_loss": float(losses[tag].mean()),
                    f"{tag}_citl_max": float(citl.max()),
                    f"{tag}_citl_call": labels[int(citl.argmax())],
                    f"{tag}_balance_worst": float(wb[worst]),
                    f"{tag}_balance_call": labels[worst]})
    if row["judge"] != "train":
        row["log_loss_diff"], row["log_loss_diff_se"] = mean_se(
            losses["judge"] - losses["train_fit"], games)
    return row


# ── Stage 6: the policy on the held-out season ───────────────────────────────
@dataclass
class Holdout:
    out: dict                   # the policy's `act` on the holdout
    actions: np.ndarray
    rewards: np.ndarray
    q_all: np.ndarray           # Q on the holdout, every call
    judge: object               # the judge's pi_b model
    pi_b: np.ndarray            # the judge's pi_b on the holdout
    pi_b_policy: np.ndarray     # the policy's own (train-fit) pi_b on the holdout
    report: dict                # judge_report, plus the lifts; saved as judge.csv
    v_behavior: OPEResult
    v_policy: OPEResult         # sensitivity attached
    lifts: dict                 # "as built" / "anchored" -> (lift, game-clustered se)
    ladder: pd.DataFrame
    sensitivity: pd.DataFrame
    contrasts: pd.DataFrame


def policy_name(cfg: dict) -> str:
    return "safe" if is_safe(cfg) else "conservative"


def holdout_ope(train: Dataset, test: Dataset, cfg: dict, fit: PolicyFit,
                graph_version: str) -> Holdout:
    """Score the policy on the held-out season with the judge's pi_b."""
    clip = cfg["ope"].get("weight_clip", 20.0)
    pname = policy_name(cfg)
    df = test.df
    out = fit.policy.act(df)
    a, r = df[test.action_col].to_numpy(), df[test.reward_col].to_numpy()
    q_te = fit.q.predict_all_actions(df)
    # The judge's pi_b is not the policy's: the policy is fit on the training
    # seasons and stays that way, the judge must know how the coordinator mixed
    # his calls in the season it scores (`src/models/behavior/judge.py`).
    judge = fit_judge_behavior(train, test, cfg, fit.beh)
    pib = judge.propensity(df)
    pib_policy = fit.beh.propensity(df)
    games = df["game_id"].to_numpy()
    report = judge_report(cfg, a, pib_policy, pib, test.action_labels, games)

    v_b = dr_policy_value(q_te, pib, pib, a, r, clip, policy_name="behavior",
                          graph_version=graph_version)
    v_p = dr_policy_value(q_te, pib, out["policy_probs"], a, r, clip, policy_name=pname,
                          graph_version=graph_version)
    lifts = {}
    if is_safe(cfg) and judge is not fit.beh:
        # Deferring, the policy plays its own (train-fit) estimate of the mix,
        # which the judge no longer shares — so the lift also prices how stale
        # that estimate is. Anchored: the same departures, deferring to the
        # judge's mix, as a deployed policy would (it knows this season).
        base = dr_per_play(q_te, pib, pib, a, r, clip)
        for tag, probs in (("as built", out["policy_probs"]), ("anchored", defer_to(out, pib))):
            lifts[tag] = mean_se(dr_per_play(q_te, pib, probs, a, r, clip) - base, games)
            key = f"lift_{tag.replace(' ', '_')}"
            report[key], report[f"{key}_se"] = lifts[tag]

    ladder = pd.concat([
        ope_ladder(q_te, pib, pib, a, r, clip, policy_name="behavior",
                   graph_version=graph_version),
        ope_ladder(q_te, pib, out["policy_probs"], a, r, clip, policy_name=pname,
                   graph_version=graph_version),
    ], ignore_index=True)
    sens = sensitivity_bounds(q_te, pib, out["policy_probs"], a, r,
                              cfg["sensitivity"]["gamma_range"], clip,
                              compare_value=v_b.value, policy_name=pname)
    # Bind the bound to the estimate it qualifies, so the saved record carries
    # value + CI + sensitivity + graph version together (CLAUDE.md discipline).
    attach_sensitivity(v_p, sens)

    # Same weight clip as DR: a fixed-action contrast rests entirely on the
    # plays that took that action, so it is *more* exposed to a tiny
    # propensity than a policy value is, not less. The baseline falls back to
    # the most common call when the configured label is not in the action space
    # (e.g. `nickel_box6` after the box is dropped from it).
    by_label = {v: k for k, v in test.action_labels.items()}
    modal = int(train.df[train.action_col].mode()[0])
    baseline = by_label.get(cfg["ope"].get("aipw_baseline"), modal)
    contrasts = aipw_contrasts(q_te, pib, a, r, test.action_labels, baseline=baseline,
                               weight_clip=clip)
    return Holdout(out=out, actions=a, rewards=r, q_all=q_te, judge=judge, pi_b=pib,
                   pi_b_policy=pib_policy, report=report, v_behavior=v_b, v_policy=v_p,
                   lifts=lifts, ladder=ladder, sensitivity=sens, contrasts=contrasts)


# ── Stages 8-9: the coverage, and both levels together ───────────────────────
@dataclass
class Level2:
    cfg: dict                   # the level-2 configuration
    graph: object
    ds: Dataset                 # charted dropbacks, the alignment in the state
    train: Dataset
    test: Dataset
    n_backdoor: int             # DoWhy's adjustment variables
    beh: object
    diag: dict
    judge: object
    pi_b: np.ndarray            # the judge's pi_b2 on the holdout
    report: dict                # judge_report for level 2
    fit: PolicyFit
    out: dict                   # the level-2 policy at the logged alignment
    v_behavior: OPEResult
    v_policy: OPEResult         # sensitivity attached
    lift: tuple                 # (DR lift, game-clustered se)
    dm_lift: float
    sensitivity: pd.DataFrame
    factored: dict              # policy.hierarchical.evaluate_factored
    gate: object                # VersionComparison, or None


def level2(cfg: dict, ds1: Dataset, policy1, pbp: pd.DataFrame) -> Level2:
    """The coverage x pressure call given the alignment, scored on charted
    dropbacks; then the factored policy and, on real data, the V2 -> V3 gate."""
    from .policy.hierarchical import (dropback_alignment_dataset, evaluate_factored,
                                      fit_dropback_alignment_behavior)

    clip = cfg["ope"].get("weight_clip", 20.0)
    cfg2 = level2_config(cfg)
    graph2 = scm.graph_for(cfg2)
    gv2 = graph2.version_tag()
    ds2 = build_level2_dataset(ds1, cfg2)
    train2, test2 = time_aware_split(ds2, cfg["data"]["holdout_season"])
    est2 = identify.identify_estimand(identify.build_causal_model(train2, graph=graph2))
    bd2 = est2.get_backdoor_variables() if hasattr(est2, "get_backdoor_variables") else []

    beh2, diag2 = fit_behavior_model(train2, cfg2)
    d2 = test2.df
    a2, r2 = d2["action"].to_numpy(), d2["reward"].to_numpy()
    games = d2["game_id"].to_numpy()
    judge2 = fit_judge_behavior(train2, test2, cfg2, beh2)
    pib2 = judge2.propensity(d2)
    report2 = judge_report(cfg2, a2, beh2.propensity(d2), pib2, ds2.action_labels, games)

    fit2 = fit_policy(train2, cfg2, beh=beh2)
    out2 = fit2.policy.act(d2)
    q2_te = out2["q_all"]
    v_b2 = dr_policy_value(q2_te, pib2, pib2, a2, r2, clip, policy_name="behavior_level2",
                           graph_version=gv2)
    v_p2 = dr_policy_value(q2_te, pib2, out2["policy_probs"], a2, r2, clip,
                           policy_name="safe_level2", graph_version=gv2)
    lift = mean_se(dr_per_play(q2_te, pib2, out2["policy_probs"], a2, r2, clip)
                   - dr_per_play(q2_te, pib2, pib2, a2, r2, clip), games)
    dm_lift = float(((out2["policy_probs"] - pib2) * q2_te).sum(1).mean())
    sens2 = sensitivity_bounds(q2_te, pib2, out2["policy_probs"], a2, r2,
                               cfg["sensitivity"]["gamma_range"], clip,
                               compare_value=v_b2.value, policy_name="safe_level2")
    attach_sensitivity(v_p2, sens2)

    beh1_db, _ = fit_dropback_alignment_behavior(train2, cfg)
    judges = None
    if judge_mode(cfg) != "train":
        judge1_db = fit_judge_behavior(dropback_alignment_dataset(train2, cfg),
                                       dropback_alignment_dataset(test2, cfg), cfg, beh1_db)
        judges = (judge1_db, judge2)
    fac = evaluate_factored(d2, cfg, policy1, fit2.policy, beh1_db, clip, judges=judges)

    gate = None
    if cfg.get("evaluation", {}).get("projection_gate") and cfg["data"]["source"] != "synthetic":
        from .evaluation.projection_gate import projection_gate
        gate = projection_gate(pbp, cfg, d2, fac, graph_version=gv2)
    return Level2(cfg=cfg2, graph=graph2, ds=ds2, train=train2, test=test2,
                  n_backdoor=len(bd2), beh=beh2, diag=diag2, judge=judge2, pi_b=pib2,
                  report=report2, fit=fit2, out=out2, v_behavior=v_b2, v_policy=v_p2,
                  lift=lift, dm_lift=dm_lift, sensitivity=sens2, factored=fac, gate=gate)


# ── Stage 10: invariance across team-seasons, and transfer ───────────────────
@dataclass
class Invariance:
    n_env: int
    table: pd.DataFrame         # one row per component contrast
    per_env: pd.DataFrame
    needed: bool                # an invariance penalty is called for
    transfer: pd.DataFrame | None = None
    transfer_summary: pd.DataFrame | None = None


def invariance_stage(cfg: dict, train: Dataset, transfer: bool = False) -> Invariance:
    """Do the call's effects differ across team-seasons, and (with `transfer`)
    does the policy hold up on a defense or season it never saw? Training
    seasons only; every snap scored by nuisances cross-fitted by game."""
    from .evaluation.invariance import (cluster_ids, environment_ids, invariance_test,
                                        pooled_effects)
    from .evaluation.transfer import (crossfit_nuisances, summarize_transfer,
                                      transfer_evaluation)

    df = train.df
    env = environment_ids(df, cfg)
    nu = crossfit_nuisances(train, cfg, with_policy=transfer)
    a, r = df[train.action_col].to_numpy(), df[train.reward_col].to_numpy()
    eff = pooled_effects(nu["q_all"], nu["pi_b"], a, r, df["game_id"].to_numpy(), cfg)
    table, per_env, needed = invariance_test(nu["q_all"], nu["pi_b"], a, r, env.to_numpy(),
                                             cluster_ids(df), cfg, effects=eff)
    result = Invariance(n_env=env.nunique(), table=table, per_env=per_env, needed=needed)
    if transfer:
        result.transfer = transfer_evaluation(train, cfg, nu)
        result.transfer_summary = summarize_transfer(result.transfer)
    return result
