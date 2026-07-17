#!/usr/bin/env python3
"""
V1 — Walking Skeleton runner.

End-to-end: load -> features -> split -> pi_b -> Q -> OPE smoke test ->
greedy policy -> OPE evaluation on holdout -> example recommendations.

Usage:
    python run_v1.py                         # uses configs/v1.yaml
    python run_v1.py --config configs/v1.yaml
    python run_v1.py --source synthetic      # override data source
"""

from __future__ import annotations

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

from src.scm import graph as scm
from src.data.load import load_pbp, generate_synthetic
from src.data.features import build_v1_dataset, time_aware_split
from src.models.behavior.propensity import fit_behavior_model
from src.models.outcome.q_model import fit_q_model
from src.ope.direct_method import behavior_recovery_check, dm_policy_value
from src.policy.greedy import learn_greedy_policy


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _rule(title: str) -> None:
    print("\n" + "=" * 68 + f"\n  {title}\n" + "=" * 68)


def main(config_path: str, source_override: str | None) -> None:
    cfg = load_config(config_path)
    if source_override:
        cfg["data"]["source"] = source_override
    np.random.seed(cfg["seed"])

    # ── 1. Data ──────────────────────────────────────────────────────────────
    _rule("[1/6] Data layer")
    source = cfg["data"]["source"]
    if source == "synthetic":
        pbp = generate_synthetic(cfg["synthetic"]["n_plays"], cfg["synthetic"]["seed"])
        cfg["data"]["team"] = "SYN"  # synthetic data is labeled defteam=SYN
        print(f"  synthetic SCM: {len(pbp):,} plays")
    else:
        pbp = load_pbp(cfg["data"]["seasons"], cfg["data"]["cache_dir"])
        print(f"  nflfastR: {len(pbp):,} raw plays, seasons {cfg['data']['seasons']}")

    ds = build_v1_dataset(pbp, cfg)
    print(f"  V1 slice ({cfg['data']['team']} 3rd downs): {len(ds.df):,} plays")
    print("  action distribution:")
    for a, label in ds.action_labels.items():
        n = int((ds.df[ds.action_col] == a).sum())
        print(f"    {a} {label:<14} {n:>5}  ({n/len(ds.df):.1%})")

    train, test = time_aware_split(ds, cfg["data"]["holdout_season"])
    print(f"  split: train={len(train.df):,} (<{cfg['data']['holdout_season']}) "
          f"holdout={len(test.df):,} (>={cfg['data']['holdout_season']})")

    # ── 2. Causal graph ──────────────────────────────────────────────────────
    _rule("[2/6] SCM skeleton")
    print(scm.describe())

    # ── 3. Behavior model pi_b ───────────────────────────────────────────────
    _rule("[3/6] Behavior policy pi_b(A|S) — logistic")
    beh, diag = fit_behavior_model(train, cfg)
    print(f"  OOF accuracy:  {diag['oof_accuracy']:.3f} "
          f"(majority baseline {diag['majority_baseline']:.3f})")
    print(f"  OOF log loss:  {diag['oof_log_loss']:.3f}")
    print(f"  calibration ECE: {diag['ece']:.3f}  ({diag['n_splits']}-fold cross-fit)")

    # ── 4. Outcome model Q ───────────────────────────────────────────────────
    _rule("[4/6] Outcome model Q(S,A) — ridge on reward(-EPA)")
    q = fit_q_model(train, cfg)
    q_all_train = q.predict_all_actions(train.df)
    print("  mean Q by action (train):")
    for a, label in ds.action_labels.items():
        print(f"    {a} {label:<14} {q_all_train[:, a].mean():+.4f}")

    # ── 5. OPE smoke test + policy ───────────────────────────────────────────
    _rule("[5/6] OPE — behavior recovery smoke test (train)")
    rec = behavior_recovery_check(train, q, beh)
    print(f"  empirical reward : {rec['empirical_reward']:+.4f} "
          f"(+/- {rec['empirical_se']:.4f})")
    print(f"  DM taken-action  : {rec['dm_taken_action']:+.4f}")
    print(f"  DM under pi_b    : {rec['dm_under_pi_b']:+.4f}")
    print(f"  recovery gap     : {rec['recovery_gap']:.4f}  "
          f"-> {'PASS' if rec['passes'] else 'FAIL (Q-model suspect)'}")

    policy = learn_greedy_policy(train, q, beh, cfg)

    # ── 6. Evaluate candidate policy on HOLDOUT ──────────────────────────────
    _rule("[6/6] Policy evaluation on holdout (Direct Method)")
    out = policy.act(test.df)
    pi_b_test = beh.propensity(test.df)
    q_all_test = q.predict_all_actions(test.df)

    v_behavior = dm_policy_value(q_all_test, pi_b_test)
    v_greedy = dm_policy_value(q_all_test, out["policy_probs"])
    print(f"  V(pi_b)   on holdout: {v_behavior.value:+.4f}  "
          f"95% CI [{v_behavior.ci95[0]:+.4f}, {v_behavior.ci95[1]:+.4f}]")
    print(f"  V(greedy) on holdout: {v_greedy.value:+.4f}  "
          f"95% CI [{v_greedy.ci95[0]:+.4f}, {v_greedy.ci95[1]:+.4f}]")
    print(f"  estimated lift      : {v_greedy.value - v_behavior.value:+.4f} reward/play")
    agree = float((out["chosen"] == test.df[ds.action_col].to_numpy()).mean())
    print(f"  policy agrees with DC on {agree:.1%} of holdout plays")

    _example_recommendations(ds, test, policy)
    _save_outputs(cfg, ds, q_all_test, v_behavior, v_greedy)


def _example_recommendations(ds, test, policy) -> None:
    _rule("Example recommendations (holdout states)")
    out = policy.act(test.df)
    sample = test.df.sample(min(4, len(test.df)), random_state=0).index
    for i in sample:
        row = test.df.loc[[i]]
        res = policy.act(row)
        a = int(res["chosen"][0])
        q_all = res["q_all"][0]
        supported = [ds.action_labels[j] for j in range(ds.n_actions) if res["support"][0, j]]
        s = test.df.loc[i]
        print(f"\n  3rd & {s['ydstogo']:.0f} | {s['formation']} | "
              f"ball on {100 - s['yardline_100']:.0f} | score {s['score_diff']:+.0f}")
        print(f"    -> recommend: {ds.action_labels[a]}  (Q={q_all[a]:+.3f})")
        print(f"       DC actually called: {ds.action_labels[int(s['action'])]}")
        print(f"       supported calls here: {', '.join(supported)}")


def _save_outputs(cfg, ds, q_all_test, v_behavior, v_greedy) -> None:
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame({
        "action": list(ds.action_labels.values()),
        "mean_Q_holdout": q_all_test.mean(axis=0),
    }).to_csv("outputs/q_by_action.csv", index=False)
    pd.DataFrame([
        {"policy": "behavior", "value": v_behavior.value,
         "ci_lo": v_behavior.ci95[0], "ci_hi": v_behavior.ci95[1]},
        {"policy": "greedy", "value": v_greedy.value,
         "ci_lo": v_greedy.ci95[0], "ci_hi": v_greedy.ci95[1]},
    ]).to_csv("outputs/ope_policy_values.csv", index=False)
    print("\n  saved: outputs/q_by_action.csv, outputs/ope_policy_values.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v1.yaml")
    ap.add_argument("--source", default=None, choices=["nflfastR", "synthetic"])
    args = ap.parse_args()
    main(args.config, args.source)
