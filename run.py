#!/usr/bin/env python3
"""
Causal-RL defensive playcalling pipeline — end-to-end runner.

End-to-end: load -> features -> split -> SCM identify + positivity ->
calibrated pi_b -> per-action GBM Q -> DR smoke test -> conservative policy ->
DR-OPE (pi_b vs pi*) on holdout + AIPW contrasts + sensitivity -> recommendations.

With configs/v3.yaml it then runs level 2 (coverage x pressure given the
alignment, on charted dropbacks), scores the factored policy, gates V3 against
V2 by projection, and tests whether the call's effects are invariant across
team-seasons (with --transfer, also leave-one-team/season-out evaluation).

Usage:
    python run.py                          # uses configs/v2.yaml (real SF data)
    python run.py --config configs/v2.yaml
    python run.py --source synthetic       # known-SCM data, no network
    python run.py --config configs/v3.yaml # every defense, every snap:
                                           # personnel package x front
    python run.py --config configs/v3.yaml --transfer   # + transfer (~1 hour)
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
from src.data.load import (load_pbp, load_ftn, merge_ftn, ftn_coverage,
                           generate_synthetic_alignment,
                           generate_synthetic, true_policy_value)
from src.data.features import (action_scheme, assert_ftn_coverage, build_dataset,
                               build_level2_dataset, level2_config,
                               slice_report, time_aware_split)
from src.models.behavior.propensity import fit_behavior_model
from src.models.outcome.q_model import fit_q_model, crossfit_q, fit_q_ensemble
from src.ope.direct_method import behavior_recovery_check, dm_policy_value
from src.ope.doubly_robust import dr_policy_value
from src.ope.aipw import aipw_contrasts
from src.ope.ladder import ope_ladder
from src.ope.sensitivity import sensitivity_bounds, attach as attach_sensitivity
from src.policy.safe import learn_policy
from src.evaluation import compare_to_baseline


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _rule(title: str) -> None:
    print("\n" + "=" * 70 + f"\n  {title}\n" + "=" * 70)


def main(config_path: str, source_override: str | None, transfer: bool = False) -> None:
    cfg = load_config(config_path)
    if source_override:
        cfg["data"]["source"] = source_override
    np.random.seed(cfg["seed"])
    weight_clip = cfg["ope"].get("weight_clip", 20.0)
    # The graph is fixed by what the action IS; every estimate below is stamped
    # with its version so a saved number can't drift from the graph it assumed.
    graph = scm.graph_for(cfg)
    gv = graph.version_tag()
    every_snap = action_scheme(cfg) == "level1"

    # ── 1. Data ──────────────────────────────────────────────────────────────
    _rule("[1/10] Data layer")
    if cfg["data"]["source"] == "synthetic":
        if every_snap:
            pbp = generate_synthetic_alignment(cfg["synthetic"]["n_plays"],
                                               cfg["synthetic"]["seed"],
                                               seasons=cfg["data"]["seasons"],
                                               coverage_effects="level2" in cfg["action"])
        else:
            pbp = generate_synthetic(cfg["synthetic"]["n_plays"], cfg["synthetic"]["seed"],
                                     seasons=cfg["data"]["seasons"])
            cfg["data"]["team"] = "SYN"
        print(f"  synthetic SCM: {len(pbp):,} plays")
    else:
        pbp = load_pbp(cfg["data"]["seasons"], cfg["data"]["cache_dir"])
        print(f"  nflfastR: {len(pbp):,} raw plays, seasons {cfg['data']['seasons']}")
        if cfg["data"].get("use_ftn", False):
            pbp = merge_ftn(pbp, load_ftn(cfg["data"]["seasons"],
                                          cfg["data"]["cache_dir"]))
            cov = ftn_coverage(pbp)
            rates = "  ".join(f"{int(r.season)}:{r.ftn_coverage:.0%}"
                              for _, r in cov.iterrows())
            print(f"  FTN charting merged — coverage by season: {rates}")
            n_unmatched = int(cov["n_plays"].sum() - (cov["n_plays"] * cov["ftn_coverage"]).sum())
            if n_unmatched:
                print(f"  {n_unmatched:,} scrimmage plays carry no FTN charting "
                      f"and are excluded (use_ftn is set)")
    assert_ftn_coverage(pbp, cfg)

    for _, row in slice_report(pbp, cfg).iterrows():
        verdict = "in slice" if row["in_slice"] else "EXCLUDED (see data.play_types)"
        print(f"  {row['play_type']:>4}: {row['n_plays']:>5,} plays, "
              f"{row['n_usable']:>5,} usable ({row['pct_usable']:.0%})"
              f"  -> {verdict}")

    ds = build_dataset(pbp, cfg)
    who = "all defenses" if cfg["data"]["team"] in (None, "all") else cfg["data"]["team"]
    print(f"  slice ({who}, down={cfg['data']['down']}, "
          f"{'/'.join(cfg['data'].get('play_types', ['pass', 'run']))}): "
          f"{len(ds.df):,} plays, {ds.n_actions} actions")
    for a, label in ds.action_labels.items():
        n = int((ds.df[ds.action_col] == a).sum())
        print(f"    {a:2d} {label:<18} {n:>6,}  ({n/len(ds.df):.1%})")

    train, test = time_aware_split(ds, cfg["data"]["holdout_season"])
    print(f"  split: train={len(train.df):,} (<{cfg['data']['holdout_season']}) "
          f"holdout={len(test.df):,} (>={cfg['data']['holdout_season']})")

    # ── 2. Causal graph + identification ─────────────────────────────────────
    _rule("[2/10] SCM identification + positivity")
    print(graph.describe())
    model = identify.build_causal_model(train, graph=graph)
    est = identify.identify_estimand(model)
    bd = est.get_backdoor_variables() if hasattr(est, "get_backdoor_variables") else []
    print(f"  DoWhy: backdoor-identified via {len(bd)} adjustment variables")
    graph_checks = None
    if every_snap and cfg.get("scm", {}).get("ci_tests", False):
        # CLAUDE.md Step 2: the hand-built graph is validated with conditional-
        # independence tests. Training seasons only — the holdout stays unseen.
        from src.scm.ci_tests import alignment_node_columns, check_graph
        graph_checks = check_graph(train.df, graph, alignment_node_columns(graph))
        imp = graph_checks[graph_checks["check"] == "implied independence"]
        edg = graph_checks[graph_checks["check"] == "edge into outcome"]
        vc = imp["verdict"].value_counts()
        print(f"  graph checks: {vc.get('consistent', 0)} implied independencies consistent, "
              f"{vc.get('violated', 0)} violated, {vc.get('untestable', 0)} untestable")
        for r in imp[imp["verdict"] == "violated"].itertuples():
            print(f"    VIOLATED {r.claim} | {r.given}  (ΔR² {r.delta_r2:.4f}) -> missing edge")
        ev = edg["verdict"].value_counts()
        print(f"  edges into {graph.outcome}: {ev.get('supported', 0)} supported, "
              f"{ev.get('weak', 0)} weak, {ev.get('unsupported', 0)} unsupported")
        if cfg["scm"].get("confirm_on_holdout", False):
            # Structure confirmation ONLY: the graph was revised on training
            # seasons, so testing it there is circular. The holdout is used to
            # confirm or refute the structure — never to revise it, and never
            # for anything the policy learns from.
            confirm = check_graph(test.df, graph, alignment_node_columns(graph))
            ci = confirm[confirm["check"] == "implied independence"]["verdict"].value_counts()
            print(f"  holdout confirmation: {ci.get('consistent', 0)} consistent, "
                  f"{ci.get('violated', 0)} violated")
            graph_checks = pd.concat([graph_checks.assign(split="train"),
                                      confirm.assign(split="holdout")], ignore_index=True)
    pos = identify.check_positivity(train, cfg.get("positivity", {}).get("min_count", 10))
    print(f"  positivity: {pos['pct_cells_supported']:.1%} of (stratum,action) cells "
          f"supported; {pos['n_absent_cells']}/{pos['n_cells']} structurally absent")
    for cell in pos["absent_cells"][:4]:
        print(f"    hole: down={cell[0]} dist={cell[1]} action={ds.action_labels[cell[-1]]}")

    # ── 3. Behavior model pi_b ───────────────────────────────────────────────
    _rule("[3/10] Behavior policy pi_b(A|S) — calibrated LightGBM")
    beh, diag = fit_behavior_model(train, cfg)
    cal = "isotonic-calibrated" if diag["calibrated"] else \
          f"UNCALIBRATED (rarest call has {diag['min_class_count']} play — too sparse to calibrate)"
    src = f"{diag['n_splits']}-fold cross-fit" if diag["diagnostics_source"] == "oof" \
          else "IN-SAMPLE — optimistic (no class admits a split)"
    print(f"  model: {cal}")
    if diag["n_excluded_rare"]:
        rare = ", ".join(ds.action_labels[a] for a in diag["rare_classes"])
        print(f"  diagnostics on {diag['n_diagnostic']:,}/{len(train.df):,} plays "
              f"({diag['n_excluded_rare']} excluded: {rare} too rare to cross-fit)")
    print(f"  accuracy:  {diag['oof_accuracy']:.3f} "
          f"(majority baseline {diag['majority_baseline']:.3f})   [{src}]")
    print(f"  log loss:  {diag['oof_log_loss']:.3f}")
    print(f"  ESS ratio: {diag['ess_ratio']:.3f}")
    # Calibration as a triple, never ECE alone: a constant base-rate predictor
    # scores near-zero ECE and is useless, so a proper scoring rule sits beside it.
    print(f"  calibration: confidence ECE {diag['confidence_ece']:.3f} | "
          f"classwise ECE {diag['classwise_ece']:.3f} | Brier {diag['brier']:.3f}")

    # ── 4. Outcome model Q ───────────────────────────────────────────────────
    _rule("[4/10] Outcome model Q(S,A) — per-action GBM on reward(-EPA)")
    q = fit_q_model(train, cfg)
    q_all_train = q.predict_all_actions(train.df)
    print("  mean Q by action (train):")
    for a, label in ds.action_labels.items():
        print(f"    {a:2d} {label:<16} {q_all_train[:, a].mean():+.4f}")

    # ── 5. OPE smoke test (doubly robust) ────────────────────────────────────
    _rule("[5/10] OPE — DR behavior-recovery smoke test (train, cross-fit Q)")
    mu_train = crossfit_q(train, cfg)
    rec = behavior_recovery_check(train, q, beh, q_all=mu_train, weight_clip=weight_clip,
                                  graph_version=gv)
    print(f"  empirical reward : {rec['empirical_reward']:+.4f} "
          f"(+/- {rec['empirical_se']:.4f})")
    print(f"  DM under pi_b    : {rec['dm_under_pi_b']:+.4f}")
    print(f"  DR under pi_b    : {rec['dr_under_pi_b']:+.4f}")
    print(f"  recovery gap (DR): {rec['recovery_gap']:.4f}  "
          f"-> {'PASS' if rec['passes'] else 'FAIL (propensities/OPE suspect)'}")

    # ── 6. Policy + holdout evaluation ───────────────────────────────────────
    _rule("[6/10] Policy + DR evaluation on holdout")
    ensemble = None
    safe = cfg["policy"].get("type", "conservative") == "safe"
    pname = "safe" if safe else "conservative"
    if safe:
        ensemble = fit_q_ensemble(train, cfg)
        print(f"  Q ensemble: {len(ensemble.members)} game-bootstrap refits; the policy "
              f"keeps the DC's mix unless a call's advantage clears a "
              f"{cfg['policy']['lcb_kappa']:.2f}-sd lower bound")
    policy = learn_policy(train, q, beh, cfg, ensemble=ensemble)
    if safe and getattr(policy, "read_bias", None) is not None:
        rb_table = policy.read_bias_table
        print(f"  read allowance (x{policy.allowance_scale:g}): bias of each component from "
              f"the coordinator's read — excess runs faced x run-minus-pass reward")
        for _, row in rb_table.iterrows():
            print(f"    {row['component']:<10} excess run {row['excess_run']:+.3f}  "
                  f"x gap {row['run_minus_pass']:+.3f}  = {row['bias']:+.4f}/snap")
        os.makedirs(cfg.get("output_dir", "outputs"), exist_ok=True)
        rb_table.to_csv(os.path.join(cfg.get("output_dir", "outputs"), "read_bias.csv"), index=False)
    out = policy.act(test.df)
    a_te = test.df[ds.action_col].to_numpy()
    r_te = test.df[ds.reward_col].to_numpy()
    q_te = q.predict_all_actions(test.df)
    pib_te = beh.propensity(test.df)

    v_behavior = dr_policy_value(q_te, pib_te, pib_te, a_te, r_te, weight_clip,
                                 policy_name="behavior", graph_version=gv)
    v_policy = dr_policy_value(q_te, pib_te, out["policy_probs"], a_te, r_te, weight_clip,
                               policy_name=pname, graph_version=gv)
    print(f"  V_DR(pi_b) on holdout: {v_behavior.value:+.4f}  "
          f"95% CI [{v_behavior.ci95[0]:+.4f}, {v_behavior.ci95[1]:+.4f}]")
    print(f"  V_DR(pi*)  on holdout: {v_policy.value:+.4f}  "
          f"95% CI [{v_policy.ci95[0]:+.4f}, {v_policy.ci95[1]:+.4f}]")
    print(f"  estimated lift       : {v_policy.value - v_behavior.value:+.4f} reward/play")
    # Agreement rate alone is a poor read on conservatism: the DC plays a
    # stochastic mix whose modal call covers only ~26% of snaps, so a
    # deterministic argmax cannot agree much beyond that even when it is
    # behaving. What conservatism actually means here is that recommendations
    # land on calls the data can vouch for, so report the support directly.
    idx = np.arange(len(a_te))
    pib_chosen = pib_te[idx, out["chosen"]]
    if safe:
        dep = ~out["deferred"]
        print(f"  pi* keeps the DC's own mix on {1 - dep.mean():.1%} of holdout plays; "
              f"departs on {dep.mean():.1%}")
        for j, share in pd.Series(out["chosen"][dep]).value_counts(normalize=True).head(4).items():
            print(f"    departs to {ds.action_labels[int(j)]:<18} {share:.0%} of departures")
    else:
        print(f"  pi* agrees with DC on {float((out['chosen'] == a_te).mean()):.1%} "
              f"of holdout plays")
    print(f"  support of recommendations: mean pi_b(chosen) = {pib_chosen.mean():.3f} "
          f"(DC's own calls: {pib_te[idx, a_te].mean():.3f}); "
          f"{float((pib_chosen < 0.10).mean()):.1%} rest on pi_b < 0.10")

    # The ladder: the same policy under every estimator. DM trusts Q, IPW trusts
    # pi_b, DR needs only one of them — agreement is the thing to look for.
    ladder = pd.concat([
        ope_ladder(q_te, pib_te, pib_te, a_te, r_te, weight_clip,
                   policy_name="behavior", graph_version=gv),
        ope_ladder(q_te, pib_te, out["policy_probs"], a_te, r_te, weight_clip,
                   policy_name=pname, graph_version=gv),
    ], ignore_index=True)
    print("\n  DM -> IPW -> DR ladder on holdout:")
    for pol, grp in ladder.groupby("policy", sort=False):
        cells = "  ".join(f"{r.estimator} {r.value:+.4f}" for r in grp.itertuples())
        g0 = grp.iloc[0]
        print(f"    {pol:<13} {cells}")
        print(f"    {'':<13} weights rest on {g0.support_share:.1%} of plays, "
              f"ESS {g0.ess_ratio:.1%}, {g0.share_clipped:.1%} clipped")

    print("\n  Sensitivity to unobserved confounding (does the lift survive?):")
    sens = sensitivity_bounds(q_te, pib_te, out["policy_probs"], a_te, r_te,
                              cfg["sensitivity"]["gamma_range"], weight_clip,
                              compare_value=v_behavior.value,
                              policy_name=pname)
    # Bind the bound to the estimate it qualifies, so the saved record carries
    # value + CI + sensitivity + graph version together (CLAUDE.md discipline).
    attach_sensitivity(v_policy, sens)
    for _, row in sens.iterrows():
        flag = "robust" if row.get("beats_comparison") else "NOT robust"
        print(f"    gamma={row['gamma']:.1f}: V(pi*) in "
              f"[{row['v_lower']:+.4f}, {row['v_upper']:+.4f}]  -> {flag}")

    # Same weight clip as DR: a fixed-action contrast rests entirely on the
    # plays that took that action, so it is *more* exposed to a tiny
    # propensity than a policy value is, not less.
    by_label = {v: k for k, v in ds.action_labels.items()}
    # Falls back to the most common call when the configured label is not in
    # the action space (e.g. `nickel_box6` after the box is dropped from it).
    modal = int(train.df[ds.action_col].mode()[0])
    baseline = by_label.get(cfg["ope"].get("aipw_baseline"), modal)
    contrasts = aipw_contrasts(q_te, pib_te, a_te, r_te, ds.action_labels,
                               baseline=baseline, weight_clip=weight_clip)

    # ── 7. Policy regression gate ────────────────────────────────────────────
    _rule("[7/10] Regression gate — candidate policy vs baseline on the holdout season")
    cmp = compare_to_baseline(test, q, beh, cfg, ensemble=ensemble, candidate=policy)
    print(cmp.report())

    level2 = None
    if every_snap and "level2" in cfg["action"]:
        level2 = _level2(cfg, pbp, ds, policy, weight_clip)

    if every_snap and "invariance" in cfg:
        _invariance(cfg, train, transfer or cfg["invariance"].get("run_transfer", False))

    _example_recommendations(ds, test, policy, level2)
    _save_outputs(cfg, ds, q_all_train, v_behavior, v_policy, sens, contrasts, cmp, ladder)
    if graph_checks is not None:
        graph_checks.to_csv(f"{cfg.get('output_dir', 'outputs')}/graph_checks.csv", index=False)

    gate_v2 = level2["gate"] if level2 else None
    if cmp.regressed or (gate_v2 is not None and gate_v2.regressed):
        # CLAUDE.md: "Regressions block merge." Exit non-zero so CI fails.
        raise SystemExit(1)


def _level2(cfg, pbp, ds1, policy1, weight_clip) -> dict:
    """Stages 8-9: coverage given the alignment, the factored policy, the V2 gate."""
    from src.policy.hierarchical import evaluate_factored, fit_dropback_alignment_behavior

    cfg2 = level2_config(cfg)
    graph2 = scm.graph_for(cfg2)
    gv2 = graph2.version_tag()
    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ── 8. Level 2 ──────────────────────────────────────────────────────────
    _rule("[8/10] Level 2 — coverage x pressure given the alignment (charted dropbacks)")
    ds2 = build_level2_dataset(ds1, cfg2)
    train2, test2 = time_aware_split(ds2, cfg["data"]["holdout_season"])
    print(f"  {len(ds2.df):,} charted dropbacks ({len(ds2.df) / len(ds1.df):.0%} of snaps), "
          f"{ds2.n_actions} calls; train={len(train2.df):,} holdout={len(test2.df):,}")
    shares = pd.crosstab(ds2.df["season"], ds2.df["action_label"], normalize="index")
    print("  share of each call by season:")
    print("    " + shares.round(3).to_string().replace("\n", "\n    "))
    print(graph2.describe())
    est2 = identify.identify_estimand(identify.build_causal_model(train2, graph=graph2))
    bd2 = est2.get_backdoor_variables() if hasattr(est2, "get_backdoor_variables") else []
    print(f"  DoWhy: backdoor-identified via {len(bd2)} adjustment variables "
          f"(the alignment among them)")

    beh2, diag2 = fit_behavior_model(train2, cfg2)
    print(f"  pi_b2: accuracy {diag2['oof_accuracy']:.3f} (majority "
          f"{diag2['majority_baseline']:.3f}), ESS {diag2['ess_ratio']:.3f}, "
          f"classwise ECE {diag2['classwise_ece']:.3f}")
    a2 = test2.df["action"].to_numpy()
    r2 = test2.df["reward"].to_numpy()
    pib2 = beh2.propensity(test2.df)
    # Coverage labels are where the charting drifts, so check the propensities
    # against the season they will weight before trusting them.
    drift = np.abs(pib2.mean(0) - np.bincount(a2, minlength=ds2.n_actions) / len(a2))
    print(f"  calibration in the large on the holdout: largest gap "
          f"{drift.max():.3f} ({ds2.action_labels[int(drift.argmax())]})")

    q2 = fit_q_model(train2, cfg2)
    ens2 = fit_q_ensemble(train2, cfg2)
    policy2 = learn_policy(train2, q2, beh2, cfg2, ensemble=ens2)
    out2 = policy2.act(test2.df)
    q2_te = out2["q_all"]
    v_b2 = dr_policy_value(q2_te, pib2, pib2, a2, r2, weight_clip,
                           policy_name="behavior_level2", graph_version=gv2)
    v_p2 = dr_policy_value(q2_te, pib2, out2["policy_probs"], a2, r2, weight_clip,
                           policy_name="safe_level2", graph_version=gv2)
    dm_lift = float(((out2["policy_probs"] - pib2) * q2_te).sum(1).mean())
    dep2 = ~out2["deferred"]
    print(f"  at the coordinator's alignment, pi2 keeps his coverage mix on "
          f"{1 - dep2.mean():.1%} of holdout dropbacks (bound {cfg2['policy']['lcb_kappa']:.2f} sd)")
    for j, share in pd.Series(out2["chosen"][dep2]).value_counts(normalize=True).head(3).items():
        print(f"    departs to {ds2.action_labels[int(j)]:<22} {share:.0%} of departures")
    print(f"  V_DR(pi_b2) {v_b2.value:+.4f}   V_DR(pi2) {v_p2.value:+.4f}   "
          f"lift DR {v_p2.value - v_b2.value:+.4f} | DM {dm_lift:+.4f}")
    sens2 = sensitivity_bounds(q2_te, pib2, out2["policy_probs"], a2, r2,
                               cfg["sensitivity"]["gamma_range"], weight_clip,
                               compare_value=v_b2.value, policy_name="safe_level2")
    attach_sensitivity(v_p2, sens2)

    # ── 9. The factored policy, and the V2 -> V3 gate ───────────────────────
    _rule("[9/10] Factored policy (alignment x coverage) + V2 -> V3 projection gate")
    beh1_db, _ = fit_dropback_alignment_behavior(train2, cfg)
    fac = evaluate_factored(test2.df, cfg, policy1, policy2, beh1_db, beh2, q2, weight_clip)
    print(f"  per charted dropback, play held a dropback (n={len(test2.df):,}); "
          f"level 1 departs on {fac['level1_departs']:.1%}, level 2 on "
          f"{fac['level2_departs']:.1%}")
    for r in fac["table"].itertuples():
        print(f"    {r.policy:<15} V_DR {r.value:+.4f}   lift {r.lift:+.4f} ± {1.96 * r.lift_se:.4f}")
    print("  (level 1's departures earn their value against the run, which a "
          "per-dropback estimand cannot see)")

    gate = None
    pg = cfg.get("evaluation", {}).get("projection_gate")
    if pg and cfg["data"]["source"] != "synthetic":
        from src.evaluation.projection_gate import projection_gate
        gate = projection_gate(pbp, cfg, test2.df, fac, graph_version=gv2)
        print(f"\n  V2 -> V3 on {pg['team']}'s {cfg['data']['holdout_season']} charted dropbacks "
              f"(V2 refit on the training seasons):")
        print(gate.report())
        pd.DataFrame([gate.to_record()]).to_csv(f"{out_dir}/projection_gate.csv", index=False)
    elif pg:
        print("  V2 -> V3 gate: skipped on synthetic data (V2 is a one-team pipeline)")

    pd.DataFrame([v_b2.to_record(), v_p2.to_record()]).to_csv(
        f"{out_dir}/level2_policy_values.csv", index=False)
    sens2.to_csv(f"{out_dir}/level2_sensitivity.csv", index=False)
    fac["table"].assign(graph_version=gv2).to_csv(f"{out_dir}/factored_policy.csv", index=False)
    print(f"\n  saved: {out_dir}/{{level2_policy_values,level2_sensitivity,factored_policy"
          f"{',projection_gate' if gate else ''}}}.csv")
    return {"cfg2": cfg2, "ds2": ds2, "policy": policy2, "gate": gate}


# Named situations rather than random rows: the recommendations a coordinator
# most needs help with are the ones a dropback-only sample cannot see at all.
_SITUATIONS = [
    ("3rd/4th & short", lambda d: (d["down"] >= 3) & (d["ydstogo"] <= 2)),
    ("goal-to-go inside the 5", lambda d: d["yardline_100"] <= 5),
    ("1st & 10", lambda d: (d["down"] == 1) & (d["ydstogo"] == 10)),
    ("3rd & long", lambda d: (d["down"] == 3) & (d["ydstogo"] >= 8)),
]


def _invariance(cfg, train, transfer: bool) -> None:
    """Stage 10: do the call's effects differ across team-seasons, and does the
    policy transfer to a defense or season it never saw? Training seasons only."""
    from src.evaluation.invariance import (cluster_ids, environment_ids, invariance_test,
                                           pooled_effects)
    from src.evaluation.transfer import (crossfit_nuisances, summarize_transfer,
                                         transfer_evaluation)

    _rule("[10/10] Invariance across team-seasons" + (" + transfer" if transfer else ""))
    df = train.df
    n_env = environment_ids(df, cfg).nunique()
    print(f"  environments: {n_env} ({cfg['invariance'].get('environment', 'team_season')}), "
          f"training seasons only; nuisances cross-fitted by game")
    nu = crossfit_nuisances(train, cfg, with_policy=transfer)
    a, r = df[train.action_col].to_numpy(), df[train.reward_col].to_numpy()
    eff = pooled_effects(nu["q_all"], nu["pi_b"], a, r, df["game_id"].to_numpy(), cfg)
    table, per_env, needed = invariance_test(nu["q_all"], nu["pi_b"], a, r,
                                             environment_ids(df, cfg).to_numpy(),
                                             cluster_ids(df), cfg, effects=eff)
    print("  component contrasts (DR over all environments; spread across them):")
    for row in table.itertuples():
        print(f"    {row.contrast:<20} effect {row.effect:+.4f} ± {row.effect_se:.4f} | "
              f"Q varies it sd {row.modeled_sd:.4f} | unmodeled tau {row.unmodeled_tau:.4f} "
              f"(p {row.p:.3f}, phi {row.phi:.2f}, {row.n_env} env) -> {row.verdict}")
    print(f"  invariance penalty {'NEEDED' if needed else 'not needed'}: "
          f"{'a contrast varies' if needed else 'no contrast'} beyond tau "
          f"{cfg['invariance'].get('min_tau', cfg['evaluation']['regression_margin']):.3f} "
          f"at the Bonferroni level")

    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(f"{out_dir}/invariance.csv", index=False)
    per_env.to_csv(f"{out_dir}/invariance_by_environment.csv", index=False)
    saved = "invariance,invariance_by_environment"
    if transfer:
        tr = transfer_evaluation(train, cfg, nu)
        summary = summarize_transfer(tr)
        print("\n  transfer: the policy refit without the group, scored on it "
              "(lift over the coordinator; anchored = the same departures, deferring "
              "to the group's own mix; gap vs the in-domain policy)")
        for row in summary.itertuples():
            print(f"    {row.split:<22} {row.groups:>2} groups  departs {row.departs:.1%}  "
                  f"lift {row.lift:+.4f} ± {row.lift_se:.4f} (tau {row.lift_tau:.4f})  "
                  f"anchored {row.lift_anchored:+.4f} ± {row.lift_anchored_se:.4f}  "
                  f"gap {row.gap:+.4f} ± {row.gap_se:.4f}  "
                  f"regress {row.n_regress}  worst {row.worst} {row.worst_lift:+.4f}")
        tr.to_csv(f"{out_dir}/transfer.csv", index=False)
        summary.to_csv(f"{out_dir}/transfer_summary.csv", index=False)
        saved += ",transfer,transfer_summary"
    else:
        print("  transfer (leave one team / season out, ~1 hour): run with --transfer")
    print(f"\n  saved: {out_dir}/{{{saved}}}.csv")


def _example_recommendations(ds, test, policy, level2=None) -> None:
    _rule("Example recommendations (holdout states)")
    for name, where in _SITUATIONS:
        pool = test.df[where(test.df)]
        if pool.empty:
            continue
        i = pool.sample(1, random_state=0).index[0]
        row = test.df.loc[[i]]
        res = policy.act(row)
        a = int(res["chosen"][0])
        q_all = res["q_all"][0]
        supported = [ds.action_labels[j] for j in range(ds.n_actions) if res["support"][0, j]]
        s = test.df.loc[i]
        print(f"\n  {name}: {int(s['down'])} & {s['ydstogo']:.0f} | {s['formation']} | "
              f"ball on {100 - s['yardline_100']:.0f} | score {s['score_diff']:+.0f}")
        if res.get("deferred") is not None and res["deferred"][0]:
            print(f"    -> no call clears the bound: keep the usual mix "
                  f"(most often {ds.action_labels[a]})")
        elif res.get("q_sd") is not None:
            print(f"    -> recommend: {ds.action_labels[a]}  (advantage over the usual mix "
                  f"{res['advantage'][0][a]:+.3f} ± {res['q_sd'][0][a]:.3f})")
        else:
            print(f"    -> recommend: {ds.action_labels[a]}  (Q={q_all[a]:+.3f})")
        print(f"       DC actually called: {ds.action_labels[int(s['action'])]}")
        print(f"       supported calls here ({len(supported)}): {', '.join(supported)}")
        if level2 is not None:
            # The coverage is called before the offense commits, so it is
            # recommended on every snap — for the alignment just recommended.
            from src.policy.hierarchical import alignment_frames
            frame = alignment_frames(row, level2["cfg2"])[a]
            r2 = level2["policy"].act(frame)
            c = int(r2["chosen"][0])
            labels2 = level2["ds2"].action_labels
            if r2["deferred"][0]:
                print(f"       coverage: keep the usual mix with it "
                      f"(most often {labels2[c]})")
            else:
                print(f"       coverage: {labels2[c]}  (advantage {r2['advantage'][0][c]:+.3f} "
                      f"± {r2['q_sd'][0][c]:.3f})")


def _save_outputs(cfg, ds, q_all_train, v_behavior, v_policy, sens, contrasts,
                  cmp, ladder) -> None:
    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({
        "action": list(ds.action_labels.values()),
        "mean_Q_train": q_all_train.mean(axis=0),
    }).to_csv(f"{out_dir}/q_by_action.csv", index=False)
    # One self-describing row per estimate: point value, CI, n, estimator,
    # graph version, and whether a sensitivity bound backs it (CLAUDE.md:
    # "every OPE estimate ships with ... the assumed causal graph version").
    pd.DataFrame([
        v_behavior.to_record(),
        v_policy.to_record(),
    ]).to_csv(f"{out_dir}/ope_policy_values.csv", index=False)
    ladder.to_csv(f"{out_dir}/ope_ladder.csv", index=False)
    sens.to_csv(f"{out_dir}/sensitivity.csv", index=False)
    contrasts.to_csv(f"{out_dir}/aipw_contrasts.csv", index=False)
    pd.DataFrame([cmp.to_record()]).to_csv(f"{out_dir}/version_comparison.csv", index=False)
    print(f"\n  saved: {out_dir}/{{q_by_action,ope_policy_values,ope_ladder,sensitivity,"
          "aipw_contrasts,version_comparison}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2.yaml")
    ap.add_argument("--source", default=None, choices=["nflfastR", "synthetic"])
    ap.add_argument("--transfer", action="store_true",
                    help="also run leave-one-team-out / leave-one-season-out (~1 hour)")
    args = ap.parse_args()
    main(args.config, args.source, args.transfer)
