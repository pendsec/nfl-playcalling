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
    python run.py                          # configs/v3.yaml: every defense, every
                                           # snap, personnel package x front
    python run.py --source synthetic       # known-SCM data, no network
    python run.py --transfer               # + transfer (~1 hour)
    python run.py --config configs/v2.yaml # one defense's coverage, dropbacks only
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
                           generate_synthetic)
from src.data.features import (action_scheme, assert_ftn_coverage, build_dataset,
                               slice_report, time_aware_split)
from src.models.behavior.propensity import fit_behavior_model
from src.models.outcome.q_model import fit_q_model, crossfit_q
from src.ope.direct_method import behavior_recovery_check
from src.pipeline import holdout_ope, invariance_stage, level2
from src.policy.safe import fit_policy, is_safe
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
    src = f"{diag['n_splits']}-fold cross-fit by game" if diag["diagnostics_source"] == "oof" \
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
    safe = is_safe(cfg)
    fit = fit_policy(train, cfg, beh=beh, q=q)
    policy = fit.policy
    if safe:
        print(f"  Q ensemble: {len(fit.ensemble.members)} game-bootstrap refits; the policy "
              f"keeps the DC's mix unless a call's advantage clears a "
              f"{cfg['policy']['lcb_kappa']:.2f}-sd lower bound")
    out_dir = cfg.get("output_dir", "outputs")
    if safe and getattr(policy, "read_bias", None) is not None:
        rb_table = policy.read_bias_table
        print(f"  read allowance (x{policy.allowance_scale:g}): bias of each component from "
              f"the coordinator's read — excess runs faced x run-minus-pass reward")
        for _, row in rb_table.iterrows():
            print(f"    {row['component']:<10} excess run {row['excess_run']:+.3f}  "
                  f"x gap {row['run_minus_pass']:+.3f}  = {row['bias']:+.4f}/snap")
        os.makedirs(out_dir, exist_ok=True)
        rb_table.to_csv(os.path.join(out_dir, "read_bias.csv"), index=False)
    h = holdout_ope(train, test, cfg, fit, gv)
    out, a_te, pib_te = h.out, h.actions, h.pi_b
    _print_judge(cfg, h.report)
    print(f"  V_DR(pi_b) on holdout: {h.v_behavior.value:+.4f}  "
          f"95% CI [{h.v_behavior.ci95[0]:+.4f}, {h.v_behavior.ci95[1]:+.4f}]")
    print(f"  V_DR(pi*)  on holdout: {h.v_policy.value:+.4f}  "
          f"95% CI [{h.v_policy.ci95[0]:+.4f}, {h.v_policy.ci95[1]:+.4f}]")
    print(f"  estimated lift       : {h.v_policy.value - h.v_behavior.value:+.4f} reward/play")
    for tag, (d, se) in h.lifts.items():
        print(f"    lift {tag:<9}: {d:+.4f} ± {1.96 * se:.4f} (paired, games clustered)")
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
    print("\n  DM -> IPW -> DR ladder on holdout:")
    for pol, grp in h.ladder.groupby("policy", sort=False):
        cells = "  ".join(f"{r.estimator} {r.value:+.4f}" for r in grp.itertuples())
        g0 = grp.iloc[0]
        print(f"    {pol:<13} {cells}")
        print(f"    {'':<13} weights rest on {g0.support_share:.1%} of plays, "
              f"ESS {g0.ess_ratio:.1%}, {g0.share_clipped:.1%} clipped")

    print("\n  Sensitivity to unobserved confounding (does the lift survive?):")
    for _, row in h.sensitivity.iterrows():
        flag = "robust" if row.get("beats_comparison") else "NOT robust"
        print(f"    gamma={row['gamma']:.1f}: V(pi*) in "
              f"[{row['v_lower']:+.4f}, {row['v_upper']:+.4f}]  -> {flag}")

    # ── 7. Policy regression gate ────────────────────────────────────────────
    _rule("[7/10] Regression gate — candidate policy vs baseline on the holdout season")
    cmp = compare_to_baseline(test, q, beh, cfg, ensemble=fit.ensemble, candidate=policy,
                              judge=h.judge)
    print(cmp.report())

    l2 = None
    if every_snap and "level2" in cfg["action"]:
        l2 = _level2(cfg, pbp, ds, policy)

    if every_snap and "invariance" in cfg:
        _invariance(cfg, train, transfer or cfg["invariance"].get("run_transfer", False))

    _example_recommendations(ds, test, policy, l2)
    _save_outputs(cfg, ds, q_all_train, h, cmp)
    if graph_checks is not None:
        graph_checks.to_csv(f"{out_dir}/graph_checks.csv", index=False)

    gate_v2 = l2.gate if l2 else None
    if cmp.regressed or (gate_v2 is not None and gate_v2.regressed):
        # CLAUDE.md: "Regressions block merge." Exit non-zero so CI fails.
        raise SystemExit(1)


def _level2(cfg, pbp, ds1, policy1):
    """Stages 8-9: coverage given the alignment, the factored policy, the V2 gate."""
    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ── 8. Level 2 ──────────────────────────────────────────────────────────
    _rule("[8/10] Level 2 — coverage x pressure given the alignment (charted dropbacks)")
    l2 = level2(cfg, ds1, policy1, pbp)
    ds2 = l2.ds
    print(f"  {len(ds2.df):,} charted dropbacks ({len(ds2.df) / len(ds1.df):.0%} of snaps), "
          f"{ds2.n_actions} calls; train={len(l2.train.df):,} holdout={len(l2.test.df):,}")
    shares = pd.crosstab(ds2.df["season"], ds2.df["action_label"], normalize="index")
    print("  share of each call by season:")
    print("    " + shares.round(3).to_string().replace("\n", "\n    "))
    print(l2.graph.describe())
    print(f"  DoWhy: backdoor-identified via {l2.n_backdoor} adjustment variables "
          f"(the alignment among them)")
    diag2 = l2.diag
    print(f"  pi_b2: accuracy {diag2['oof_accuracy']:.3f} (majority "
          f"{diag2['majority_baseline']:.3f}), ESS {diag2['ess_ratio']:.3f}, "
          f"classwise ECE {diag2['classwise_ece']:.3f}")
    # Coverage labels are where the charting drifts, so check the propensities
    # against the season they will weight before trusting them.
    _print_judge(l2.cfg, l2.report)
    out2 = l2.out
    dep2 = ~out2["deferred"]
    print(f"  at the coordinator's alignment, pi2 keeps his coverage mix on "
          f"{1 - dep2.mean():.1%} of holdout dropbacks (bound {l2.cfg['policy']['lcb_kappa']:.2f} sd)")
    for j, share in pd.Series(out2["chosen"][dep2]).value_counts(normalize=True).head(3).items():
        print(f"    departs to {ds2.action_labels[int(j)]:<22} {share:.0%} of departures")
    print(f"  V_DR(pi_b2) {l2.v_behavior.value:+.4f}   V_DR(pi2) {l2.v_policy.value:+.4f}   "
          f"lift DR {l2.v_policy.value - l2.v_behavior.value:+.4f} | DM {l2.dm_lift:+.4f}")

    # ── 9. The factored policy, and the V2 -> V3 gate ───────────────────────
    _rule("[9/10] Factored policy (alignment x coverage) + V2 -> V3 projection gate")
    fac = l2.factored
    print(f"  per charted dropback, play held a dropback (n={len(l2.test.df):,}); "
          f"level 1 departs on {fac['level1_departs']:.1%}, level 2 on "
          f"{fac['level2_departs']:.1%}")
    for r in fac["table"].itertuples():
        print(f"    {r.policy:<15} V_DR {r.value:+.4f}   lift {r.lift:+.4f} ± {1.96 * r.lift_se:.4f}")
    print("  (level 1's departures earn their value against the run, which a "
          "per-dropback estimand cannot see)")

    pg = cfg.get("evaluation", {}).get("projection_gate")
    if l2.gate is not None:
        print(f"\n  V2 -> V3 on {pg['team']}'s {cfg['data']['holdout_season']} charted dropbacks "
              f"(V2 refit on the training seasons):")
        print(l2.gate.report())
        print(f"  (one defense's season, {l2.gate.n_games} games: the game-clustered SE may "
              "understate the noise, which errs toward blocking, never toward passing)")
        pd.DataFrame([l2.gate.to_record()]).to_csv(f"{out_dir}/projection_gate.csv", index=False)
    elif pg:
        print("  V2 -> V3 gate: skipped on synthetic data (V2 is a one-team pipeline)")

    pd.DataFrame([l2.v_behavior.to_record(), l2.v_policy.to_record()]).to_csv(
        f"{out_dir}/level2_policy_values.csv", index=False)
    l2.sensitivity.to_csv(f"{out_dir}/level2_sensitivity.csv", index=False)
    fac["table"].assign(graph_version=l2.graph.version_tag()).to_csv(
        f"{out_dir}/factored_policy.csv", index=False)
    print(f"\n  saved: {out_dir}/{{level2_policy_values,level2_sensitivity,factored_policy"
          f"{',projection_gate' if l2.gate else ''}}}.csv")
    return l2


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
    _rule("[10/10] Invariance across team-seasons" + (" + transfer" if transfer else ""))
    inv = invariance_stage(cfg, train, transfer)
    print(f"  environments: {inv.n_env} ({cfg['invariance'].get('environment', 'team_season')}), "
          f"training seasons only; nuisances cross-fitted by game")
    print("  component contrasts (DR over all environments; spread across them):")
    for row in inv.table.itertuples():
        print(f"    {row.contrast:<20} effect {row.effect:+.4f} ± {row.effect_se:.4f} | "
              f"Q varies it sd {row.modeled_sd:.4f} | unmodeled tau {row.unmodeled_tau:.4f} "
              f"(p {row.p:.3f}, phi {row.phi:.2f}, {row.n_env} env) -> {row.verdict}")
    needed = inv.needed
    print(f"  invariance penalty {'NEEDED' if needed else 'not needed'}: "
          f"{'a contrast varies' if needed else 'no contrast'} beyond tau "
          f"{cfg['invariance'].get('min_tau', cfg['evaluation']['regression_margin']):.3f} "
          f"at the Bonferroni level")

    out_dir = cfg.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    inv.table.to_csv(f"{out_dir}/invariance.csv", index=False)
    inv.per_env.to_csv(f"{out_dir}/invariance_by_environment.csv", index=False)
    saved = "invariance,invariance_by_environment"
    if transfer:
        print("\n  transfer: the policy refit without the group, scored on it "
              "(lift over the coordinator; anchored = the same departures, deferring "
              "to the group's own mix; gap vs the in-domain policy)")
        for row in inv.transfer_summary.itertuples():
            print(f"    {row.split:<22} {row.groups:>2} groups  departs {row.departs:.1%}  "
                  f"lift {row.lift:+.4f} ± {row.lift_se:.4f} (tau {row.lift_tau:.4f})  "
                  f"anchored {row.lift_anchored:+.4f} ± {row.lift_anchored_se:.4f}  "
                  f"gap {row.gap:+.4f} ± {row.gap_se:.4f}  "
                  f"regress {row.n_regress}  worst {row.worst} {row.worst_lift:+.4f}")
        inv.transfer.to_csv(f"{out_dir}/transfer.csv", index=False)
        inv.transfer_summary.to_csv(f"{out_dir}/transfer_summary.csv", index=False)
        saved += ",transfer,transfer_summary"
    else:
        print("  transfer (leave one team / season out, ~1 hour): run with --transfer")
    print(f"\n  saved: {out_dir}/{{{saved}}}.csv")


def _example_recommendations(ds, test, policy, l2=None) -> None:
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
        if l2 is not None:
            # The coverage is called before the offense commits, so it is
            # recommended on every snap — for the alignment just recommended.
            from src.policy.hierarchical import alignment_frames
            frame = alignment_frames(row, l2.cfg)[a]
            r2 = l2.fit.policy.act(frame)
            c = int(r2["chosen"][0])
            labels2 = l2.ds.action_labels
            if r2["deferred"][0]:
                print(f"       coverage: keep the usual mix with it "
                      f"(most often {labels2[c]})")
            else:
                print(f"       coverage: {labels2[c]}  (advantage {r2['advantage'][0][c]:+.3f} "
                      f"± {r2['q_sd'][0][c]:.3f})")


def _print_judge(cfg, row: dict, indent: str = "  ") -> None:
    """The judge report (`pipeline.judge_report`) for the run log."""
    if row["judge"] == "train":
        print(f"{indent}judge's pi_b: the policy's own (train-fit) — holdout log loss "
              f"{row['train_fit_log_loss']:.3f}; largest calibration-in-the-large gap "
              f"{row['train_fit_citl_max']:.3f} ({row['train_fit_citl_call']})")
        return
    print(f"{indent}judge's pi_b: training seasons + the holdout's other games, season in "
          f"the state, {cfg['ope'].get('judge_folds', 5)} folds by game (calls, never rewards)")
    for tag, name in (("train_fit", "train-fit (the policy's)"), ("judge", "judge")):
        print(f"{indent}  {name:<25} log loss {row[f'{tag}_log_loss']:.4f} · largest "
              f"share gap {row[f'{tag}_citl_max']:.3f} ({row[f'{tag}_citl_call']}) · worst "
              f"weight balance {row[f'{tag}_balance_worst']:.2f} ({row[f'{tag}_balance_call']})")
    print(f"{indent}  log loss, judge - train-fit: {row['log_loss_diff']:+.4f} "
          f"± {1.96 * row['log_loss_diff_se']:.4f}")


def _save_outputs(cfg, ds, q_all_train, h, cmp) -> None:
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
        h.v_behavior.to_record(),
        h.v_policy.to_record(),
    ]).to_csv(f"{out_dir}/ope_policy_values.csv", index=False)
    h.ladder.to_csv(f"{out_dir}/ope_ladder.csv", index=False)
    h.sensitivity.to_csv(f"{out_dir}/sensitivity.csv", index=False)
    h.contrasts.to_csv(f"{out_dir}/aipw_contrasts.csv", index=False)
    pd.DataFrame([cmp.to_record()]).to_csv(f"{out_dir}/version_comparison.csv", index=False)
    print(f"\n  saved: {out_dir}/{{q_by_action,ope_policy_values,ope_ladder,sensitivity,"
          "aipw_contrasts,version_comparison}.csv")
    pd.DataFrame([h.report]).to_csv(f"{out_dir}/judge.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3.yaml")
    ap.add_argument("--source", default=None, choices=["nflfastR", "synthetic"])
    ap.add_argument("--transfer", action="store_true",
                    help="also run leave-one-team-out / leave-one-season-out (~1 hour)")
    args = ap.parse_args()
    main(args.config, args.source, args.transfer)
