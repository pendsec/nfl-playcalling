#!/usr/bin/env python3
"""
V2 — "Right Tools, Simple Problem" runner.

End-to-end: load -> features -> split -> SCM identify + positivity ->
calibrated pi_b -> per-action GBM Q -> DR smoke test -> conservative policy ->
DR-OPE (pi_b vs pi*) on holdout + AIPW contrasts + sensitivity -> recommendations.

Usage:
    python run.py                          # uses configs/v2.yaml (real SF data)
    python run.py --config configs/v2.yaml
    python run.py --source synthetic       # known-SCM data, no network

V1 (the walking skeleton) is preserved at git tag `v1` — `git checkout v1`.
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)  # quiet DoWhy / LightGBM chatter

from src.scm import graph as scm
from src.scm import identify
from src.data.load import load_pbp, generate_synthetic, true_policy_value
from src.data.features import build_dataset, time_aware_split
from src.models.behavior.propensity import fit_behavior_model
from src.models.outcome.q_model import fit_q_model, crossfit_q
from src.ope.direct_method import behavior_recovery_check, dm_policy_value
from src.ope.doubly_robust import dr_policy_value
from src.ope.aipw import aipw_contrasts
from src.ope.sensitivity import sensitivity_bounds
from src.policy.conservative import learn_conservative_policy


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _rule(title: str) -> None:
    print("\n" + "=" * 70 + f"\n  {title}\n" + "=" * 70)


def main(config_path: str, source_override: str | None) -> None:
    cfg = load_config(config_path)
    if source_override:
        cfg["data"]["source"] = source_override
    np.random.seed(cfg["seed"])
    weight_clip = cfg["ope"].get("weight_clip", 20.0)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    _rule("[1/6] Data layer")
    if cfg["data"]["source"] == "synthetic":
        pbp = generate_synthetic(cfg["synthetic"]["n_plays"], cfg["synthetic"]["seed"])
        cfg["data"]["team"] = "SYN"
        print(f"  synthetic SCM: {len(pbp):,} plays")
    else:
        pbp = load_pbp(cfg["data"]["seasons"], cfg["data"]["cache_dir"])
        print(f"  nflfastR: {len(pbp):,} raw plays, seasons {cfg['data']['seasons']}")

    ds = build_dataset(pbp, cfg)
    print(f"  V2 slice ({cfg['data']['team']}, down={cfg['data']['down']}): "
          f"{len(ds.df):,} plays, {ds.n_actions} actions")
    for a, label in ds.action_labels.items():
        n = int((ds.df[ds.action_col] == a).sum())
        print(f"    {a:2d} {label:<16} {n:>5}  ({n/len(ds.df):.1%})")

    train, test = time_aware_split(ds, cfg["data"]["holdout_season"])
    print(f"  split: train={len(train.df):,} (<{cfg['data']['holdout_season']}) "
          f"holdout={len(test.df):,} (>={cfg['data']['holdout_season']})")

    # ── 2. Causal graph + identification ─────────────────────────────────────
    _rule("[2/6] SCM identification + positivity")
    print(scm.describe())
    model = identify.build_causal_model(train)
    est = identify.identify_estimand(model)
    bd = est.get_backdoor_variables() if hasattr(est, "get_backdoor_variables") else []
    print(f"  DoWhy: backdoor-identified via {len(bd)} adjustment variables")
    pos = identify.check_positivity(train, cfg.get("positivity", {}).get("min_count", 10))
    print(f"  positivity: {pos['pct_cells_supported']:.1%} of (stratum,action) cells "
          f"supported; {pos['n_absent_cells']}/{pos['n_cells']} structurally absent")
    for cell in pos["absent_cells"][:4]:
        print(f"    hole: down={cell[0]} dist={cell[1]} action={ds.action_labels[cell[-1]]}")

    # ── 3. Behavior model pi_b ───────────────────────────────────────────────
    _rule("[3/6] Behavior policy pi_b(A|S) — calibrated LightGBM")
    beh, diag = fit_behavior_model(train, cfg)
    cal = "isotonic-calibrated" if diag["calibrated"] else \
          f"UNCALIBRATED (rarest call has {diag['min_class_count']} play — too sparse to calibrate)"
    src = f"{diag['n_splits']}-fold cross-fit" if diag["diagnostics_source"] == "oof" \
          else "in-sample (classes too sparse for OOF)"
    print(f"  model: {cal}")
    print(f"  accuracy:  {diag['oof_accuracy']:.3f} "
          f"(majority baseline {diag['majority_baseline']:.3f})   [{src}]")
    print(f"  log loss:  {diag['oof_log_loss']:.3f}")
    print(f"  calibration ECE: {diag['ece']:.3f}   ESS ratio: {diag['ess_ratio']:.3f}")

    # ── 4. Outcome model Q ───────────────────────────────────────────────────
    _rule("[4/6] Outcome model Q(S,A) — per-action GBM on reward(-EPA)")
    q = fit_q_model(train, cfg)
    q_all_train = q.predict_all_actions(train.df)
    print("  mean Q by action (train):")
    for a, label in ds.action_labels.items():
        print(f"    {a:2d} {label:<16} {q_all_train[:, a].mean():+.4f}")

    # ── 5. OPE smoke test (doubly robust) ────────────────────────────────────
    _rule("[5/6] OPE — DR behavior-recovery smoke test (train, cross-fit Q)")
    mu_train = crossfit_q(train, cfg)
    rec = behavior_recovery_check(train, q, beh, q_all=mu_train, weight_clip=weight_clip)
    print(f"  empirical reward : {rec['empirical_reward']:+.4f} "
          f"(+/- {rec['empirical_se']:.4f})")
    print(f"  DM under pi_b    : {rec['dm_under_pi_b']:+.4f}")
    print(f"  DR under pi_b    : {rec['dr_under_pi_b']:+.4f}")
    print(f"  recovery gap (DR): {rec['recovery_gap']:.4f}  "
          f"-> {'PASS' if rec['passes'] else 'FAIL (propensities/OPE suspect)'}")

    # ── 6. Policy + holdout evaluation ───────────────────────────────────────
    _rule("[6/6] Conservative policy + DR evaluation on holdout")
    policy = learn_conservative_policy(train, q, beh, cfg)
    out = policy.act(test.df)
    a_te = test.df[ds.action_col].to_numpy()
    r_te = test.df[ds.reward_col].to_numpy()
    q_te = q.predict_all_actions(test.df)
    pib_te = beh.propensity(test.df)

    v_behavior = dr_policy_value(q_te, pib_te, pib_te, a_te, r_te, weight_clip)
    v_policy = dr_policy_value(q_te, pib_te, out["policy_probs"], a_te, r_te, weight_clip)
    print(f"  V_DR(pi_b) on holdout: {v_behavior.value:+.4f}  "
          f"95% CI [{v_behavior.ci95[0]:+.4f}, {v_behavior.ci95[1]:+.4f}]")
    print(f"  V_DR(pi*)  on holdout: {v_policy.value:+.4f}  "
          f"95% CI [{v_policy.ci95[0]:+.4f}, {v_policy.ci95[1]:+.4f}]")
    print(f"  estimated lift       : {v_policy.value - v_behavior.value:+.4f} reward/play")
    agree = float((out["chosen"] == a_te).mean())
    print(f"  pi* agrees with DC on {agree:.1%} of holdout plays")

    print("\n  Sensitivity to unobserved confounding (does the lift survive?):")
    sens = sensitivity_bounds(q_te, pib_te, out["policy_probs"], a_te, r_te,
                              cfg["sensitivity"]["gamma_range"], weight_clip,
                              compare_value=v_behavior.value)
    for _, row in sens.iterrows():
        flag = "robust" if row.get("beats_comparison") else "NOT robust"
        print(f"    gamma={row['gamma']:.1f}: V(pi*) in "
              f"[{row['v_lower']:+.4f}, {row['v_upper']:+.4f}]  -> {flag}")

    contrasts = aipw_contrasts(q_te, pib_te, a_te, r_te, ds.action_labels, baseline=0)

    _example_recommendations(ds, test, policy)
    _save_outputs(ds, q_all_train, v_behavior, v_policy, sens, contrasts)


def _example_recommendations(ds, test, policy) -> None:
    _rule("Example recommendations (holdout states)")
    sample = test.df.sample(min(4, len(test.df)), random_state=0).index
    for i in sample:
        row = test.df.loc[[i]]
        res = policy.act(row)
        a = int(res["chosen"][0])
        q_all = res["q_all"][0]
        supported = [ds.action_labels[j] for j in range(ds.n_actions) if res["support"][0, j]]
        s = test.df.loc[i]
        print(f"\n  {int(s['down'])} & {s['ydstogo']:.0f} | {s['formation']} | "
              f"ball on {100 - s['yardline_100']:.0f} | score {s['score_diff']:+.0f}")
        print(f"    -> recommend: {ds.action_labels[a]}  (Q={q_all[a]:+.3f})")
        print(f"       DC actually called: {ds.action_labels[int(s['action'])]}")
        print(f"       supported calls here ({len(supported)}): {', '.join(supported)}")


def _save_outputs(ds, q_all_train, v_behavior, v_policy, sens, contrasts) -> None:
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame({
        "action": list(ds.action_labels.values()),
        "mean_Q_train": q_all_train.mean(axis=0),
    }).to_csv("outputs/q_by_action.csv", index=False)
    pd.DataFrame([
        {"policy": "behavior", "value": v_behavior.value,
         "ci_lo": v_behavior.ci95[0], "ci_hi": v_behavior.ci95[1]},
        {"policy": "conservative", "value": v_policy.value,
         "ci_lo": v_policy.ci95[0], "ci_hi": v_policy.ci95[1]},
    ]).to_csv("outputs/ope_policy_values.csv", index=False)
    sens.to_csv("outputs/sensitivity.csv", index=False)
    contrasts.to_csv("outputs/aipw_contrasts.csv", index=False)
    print("\n  saved: outputs/{q_by_action,ope_policy_values,sensitivity,aipw_contrasts}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--source", default=None, choices=["nflfastR", "synthetic"])
    args = ap.parse_args()
    main(args.config, args.source)
