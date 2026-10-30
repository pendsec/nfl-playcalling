# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Causal RL for Defensive Playcalling

## Project Overview

This project builds a **causal reinforcement learning** system that recommends defensive playcalls in football given pre-snap game state. The deliverable is a decision support tool - not autopilot - that gives coaches credibly-evaluated recommendations along with the causal reasoning behind them. 

### Why causal, not vanilla offline RL
Football play-by-play is observational data with heavy confounding: coaches choose calls based on hidden reads (matchup intuition, film study, recent tendencies) that also influence outcomes. Naive offline RL conflates the *causal* effect of a call with the *selection effect* of when coaches choose it, producing biased policies that fail under distribution shift (new opponents, new seasons, rare states). 

Causal methods (deconfounded OPE, instrumental variables, structural counterfactuals, invariant policy learning) are how we get unbiased policy evaluation and transferable policies from this kind of data.

---

## Problem Definition

- **State (S)**: pre-snap context - down, distance, field position, score differential, time remaining, timeouts, personnel, formation, motion, opponent rolling tendencies, QB/OC identity, weather.
- **Action (A)**: defensive call - personnel package x front x coverage shell x pressure (x disguise, which needs pre-snap alignment data not available from free sources). Discretized and factored hierarchically — see V3.
- **Reward (R)**: EPA-against pre play (primary); drive-level outcomes as secondary signal.
- **Confounders**: coach's hidden read, matchup intuition, recent film study - proxied by tracking data once available.

---

## Pipeline Architecture

Eight components, each with a phased build path. **Always ship a walking skeleton end-to-end before deepening any single component.**

### Step 0: Tractable Slice
Scope down before modeling. Start with a slice small enough to debug end-to-end and sanity-check against coach intuition.

### Step 1: Data Layer
Play-level dataset: state features, action taken, outcome. Includes feature engineering and (later) tracking-derived features. Schema discipline matters - pre-snap state must not leak post-snap info.

### Step 2: Causal Graph (SCM Skeleton)
Hand-built DAG over project variables encoding confounders, mediators, colliders, and instruments. Disciplines what to condition on in every downstream model. Validated with conditional independence tests.

**CI testing (`src/scm/ci_tests.py`)**: missing edges are the strong claims — each implies a conditional independence that data can refute — so every one is tested (regression-based, with an effect size, since at 10^5 snaps almost anything is "significant"); present edges into the outcome get an edge-relevance check instead. Rules that came out of building it:
- A claim with no *observed* separating set is reported **untestable**, never skipped. The coach's unrecorded read makes the alignment a collider, so e.g. "offensive tendency has no direct effect on EPA" cannot be checked by any data we have.
- Paths running entirely through unobserved nodes count as adjacency (latent projection): `x -> execution -> epa` is observationally `x -> epa`, not a claim of independence.
- **Keep an edge unless the data can refute its absence.** Edge relevance conditions on the treatment (a collider with the latent read) and cannot split credit between parents that measure overlapping things (team identity vs quality proxies), so "unsupported" is weak evidence. Dropping an edge is the stronger, often uncheckable, claim.
- Revise the graph on training seasons; the held-out season may **confirm** the structure (`scm.confirm_on_holdout`) but never revise it and never feed the policy.

**V2 note**: the DAG also declares what the sample is SELECTED on, not just what is adjusted for. `coverage_charted` is a selection node with both `off_playcall` and `epa` as parents, because coverage labels exist on ~94% of dropbacks and ~3% of runs. That makes V2's estimand conditional on a charted dropback — run/pass deterrence is outside it — and it is hashed into the graph fingerprint so no estimate can quietly shed the assumption.

### Step 3: Behavior Policy Model - π_b(A|S)
Classifier predicting what defensive call the actual DC made in state S. Heart of IPW reweighting and doubly robust OPE. Calibration is critical; positivity violations bound how aggressive the learned policy can be.

### Step 4: Outcome / Q-Model
Predicts E[R | S, do(A)] - the *interventional* outcome, not observational. Built using only confounders + state features per the SCM (mediators excluded). Eventually grows into a structural causal model supporting counterfactual rollouts.

### Step 5: Off-Policy Evaluation (OPE)
Unbiased estimator of any candidate policy's value from logged data. Direct Method -> IPW -> Doubly Robust -> sensitivity analysis for unobserved confounders. The judge of every candidate policy. Validate by recovering the behavior policy's known average reward.

**V2 note**: V2 ships DM and DR (whose IPW correction term does the reweighting) but no *standalone* IPW estimator. On single-team data with an ESS ratio this low, a pure-IPW number would be dominated by a handful of tiny propensities and would add variance without adding a decision. The standalone rung is deferred to V3, where multi-team pooling makes it informative and the notebook can show the full DM -> IPW -> DR ladder side by side. 

### Step 6: Policy Learning
The actual π*(A|S). Behavior-constrained greedy -> CQL/IQL -> causal-pessimism -> invariant policy learning across teams/seasons -> hierarchical factored policy.

### Step 7: Counterfactual Data Augmentation (optional)
Structural simulator generating plausible counterfactual outcomes under alternative defensive calls. Multiplies effective dataset for rare states. Dangerous if the SCM is wrong - defer until later phases.

### Step 8: Validation, Monitoring, Deployment
Shadow mode in practice -> shadow model in games -> live decision support. Per-state confidence surfacing, drift detection, explainability ("recommend Cover 3 because: third and medium against 11 personnel, slot WR has 0.6 separation vs man, RB is poor blocker"). 

---

## Phased Build Plan

The walking skeleton at V1 is the highest-priority step. A pipeline that gives obviously-wrong answers in week 2 is far better than a "correct" pipeline that ships in month 18. 

Every version ships a narrated demo notebook (`notebooks/vN_pipeline_walkthrough.ipynb`) as an explicit deliverable — see the Demo bullets below and the demo-notebook convention.

### V1 - Walking Skeleton
- **Scope**: 3rd downs only, one team's defense, 3 seasons, action space = {man, zone, blitz yes/no} (4 actions).
- **Models**: logistic π_b, ridge Q-regression on EPA, Direct Method OPE, behavior-constrained greedy policy.
- **Goal**: end-to-end pipeline that produces and evaluates a policy. Probably wrong in places, but debuggable.
- **Demo**: `notebooks/v1_pipeline_walkthrough.ipynb` — narrated end-to-end walkthrough, including the deliberately-wrong parts (off-support DM blow-up). Preserved at git tag `v1`.

### V2 - Right Tools, Simple Problem
- **Scope**: all downs, one team, 12 actions (coverage shell x pressure). Charted dropbacks only — the coverage axis comes from NGS pass charting, so runs are ~3% labeled and are excluded explicitly via `data.play_types` rather than incidentally by the uncharted-coverage drop.
- **Models**: GBM π_b with calibration, GBM Q-model with proper feature selection from SCM, doubly robust OPE, CQL/IQL policy.
- **Add**: sensitivity analysis bounding impact of unobserved confounders; OPE provenance (every estimate stamped with estimator + causal-graph version + its sensitivity bound); the version-over-version regression gate in `src/evaluation/`.
- **Demo**: `notebooks/v2_pipeline_walkthrough.ipynb` — real-data narrative (action sparsity, DAG + DoWhy identification, positivity heatmap, calibration curve, DR recovery, sensitivity band) plus a synthetic ground-truth validation aside.

### V3 - Scale and Transfer
- **Scope**: all 32 defenses pooled, **all snaps (runs included)**, seasons 2022–2024 for training with **2025 held out**. FTN charting (2022+) and nflfastR's `defense_personnel` (charted on ~100% of snaps) supply the fully-observed action axes.
- **Action space — two-level hierarchical factorization (12 x 12 = 144)**:
  - **Level 1 — personnel package (base <=4 DB / nickel 5 / dime+ 6+) x front (box <=5 / 6 / 7 / 8+) = 12.** Genuinely pre-snap and observed on ~100% of runs and passes. No dropback selection, so short-yardage and goal-line states are in the data and the run/pass *deterrence* term is estimable (run rate by package: base 64% / nickel 40% / dime+ 14%). This is the headline.
  - **Level 2 — coverage shell (6) x pressure (2) = 12.** The shell is *called* pre-snap on every play, so the policy recommends one on every snap — but its value is *estimated* on dropbacks only, where it is observed, and both components are measured post-snap (played coverage, actual rushers). A dropback-conditional refinement to level 1, labeled as such.
  - **Disguise dropped.** Disguise is the gap between the shown pre-snap shell and the played coverage; free data carries only the played half. Needs PFF-grade pre-snap alignment — V4+ at the earliest.
- **Models**: GBM π_b and Q (the V2 estimators plus team features) are the baseline. Neural π_b and Q with embeddings for **defense team, offense team, and opposing QB**, temperature-calibrated and **gated against the GBM baseline on held-out-season OPE** — they ship only if they beat it. No OC/DC embedding: free data carries only head coaches, one per defense per season, i.e. equivalent to team-season.
- **Invariance**: environments are **team-seasons** (96 in training). Test first — do action-component effects differ significantly across environments? — then an invariance-penalized Q (IRM / REx-style) only if the test says it is needed. Transfer is judged by leave-one-team-out and leave-one-season-out OPE.
- **Add**: standalone IPW estimator completing the DM -> IPW -> DR ladder (deferred from V2 — see Step 5); econml CATE estimators; explainability layer surfacing top causal features per recommendation (Step 8 format); interpretable-Γ sensitivity (benchmark the bound against observed confounders, Cinelli–Hazlett style); **Dirichlet / vector scaling** for the shell imputer.
- **Shell imputation is a sensitivity analysis, not core** (`src/models/imputation/`, built at V2). It answers one question: *does imputing shell on run plays change any level-1 conclusion?* It stays out of the core path because:
  - it would contaminate level 1, the one result that is assumption-light (observed treatment, every snap, no transportability assumption);
  - it adds little: ~19.6% of shell uncertainty removed, two draws agree on ~32% of plays, ~43% of pooled variance comes from the unknown label;
  - excluding the outcome from the imputer attenuates run-play shell effects toward zero *by construction* — Rubin's rules fix the variance, not that bias — so in the core path the policy would learn "shell doesn't matter on runs" because the data was built to say so;
  - its transportability assumption cannot be validated: the only NGS-charted runs are runs defended as passes (mean reward −0.39 vs +0.07 on ordinary runs).
  - Revisit if PFF-grade shell labels on runs become available, if the sensitivity analysis flips a level-1 conclusion, or if the imputer's information rises materially.
  - **Imputer calibration (still open)**: cross-fitted temperature scaling fixes confidence ECE (0.0321 -> 0.0199) but barely moves classwise ECE (0.0378 -> 0.0372), which is what matters for a posterior that is sampled from. Dirichlet (K^2+K params) or vector scaling (2K) is the next rung.
- **Known limitation**: the level-2 estimand assumes the shell's effect on run plays is negligible or absorbed by level 1. The model cannot recommend a shell *for its run-fit value* ("Cover-3 because its safeties help against the run") — a real gap, since the shell is chosen before the defense knows which branch it is on. State it in the V3 docs, don't leave it implicit.
- **Regression gate V2 -> V3 (Phase 2)**: V3 changes scope, action space and estimand, so the two policies don't answer the same question. Gate by **projection into the joint space**: on SF charted dropbacks in the 2025 holdout, express both policies over (alignment x coverage) — V2's as its shell x pressure policy with the alignment following behavior, V3's as its alignment policy with the coverage following the level-2 policy — and compare them by paired DR with a joint Q. V3 must not regress on V2's own question; later versions gate against V3.
  - *Why not in Phase 1*: V2's Q is a function of shell x pressure only, blind to personnel and box count — exactly what level 1 changes — so projecting a level-1-only policy onto V2's action space would pass trivially while measuring nothing. The comparison needs the level-2 models Phase 2 builds. Phase 1 gates the level-1 policy against its own greedy baseline.
- **Build phases** (walking-skeleton first):
  0. Housekeeping — `v2` tag, `configs/v3.yaml`, `defense_personnel` in the data layer with a cache bump, guards for post-snap participation columns.
  1. **Level 1 end to end (done)** — multi-team data, package x front on all snaps, GBM π_b and a pooled GBM Q with team features, the alignment graph (`scm.graph.ALIGNMENT_GRAPH`, v3.0), DM -> IPW -> DR ladder, conservative level-1 policy, 2025-holdout OPE, level-1 regression gate, notebook.
  2. **Level 2 (done)** — shell x pressure policy given the level-1 call; combined DR over the factored policy; the V2 -> V3 projection gate.
  3. Transfer & invariance — leave-one-team/season-out OPE; invariance test, then an invariance-penalized Q if needed.
  4. Neural π_b/Q with embeddings, gated against GBM.
  5. Explainability — econml CATE, per-recommendation causal explanations, interpretable Γ, pressure mediation (does the box effect run through `was_pressure`?).
  6. Refinements — Dirichlet/vector scaling; the imputation sensitivity analysis.
- **Phase 1 findings** (each changed the build):
  - **Q structure.** One regressor per alignment estimates each cell with independent noise, inflating the spread across actions (2.1x the truth on a known SCM at real scale), so the policy's argmax chases the most overestimated cell. A **pooled** Q over (state, package, front) keeps the spread honest (0.8x) and let the conservative policy capture ~77% of the available true gain vs ~45% per-action. `outcome_model.structure: pooled`; the dropback pipeline keeps per-action.
  - **The positivity floor does real work.** Raw argmax-Q *loses* to the coordinator in truth; the same argmax restricted to π_b >= 0.05 beats it.
  - **QB leak.** `passer_player_id` exists on 100% of passes and 0% of runs, so a QB feature grouped on it is defaulted on every run and encodes play type — a post-snap fact in the state. The QB on the field is carried within each offense's game (`features._qb_on_field`).
  - **Offensive tendency** is now per offense and distance bucket; pooled over offenses it is close to a league constant, and on a pass-only sample it is identically 1.
  - **DR noise on small samples.** On the ~2k-play synthetic holdout DR's error on the policy value (±0.03) is as large as the true gain (+0.017), flipping sign between runs — noise, not bias. The real holdout is ~16x larger.
- **Level-1 gain diagnosis and the safe policy** (after Phase 1). The conservative policy's DM lift (+0.0125) never survived DR (−0.002 ± 0.023): one season resolves only ~±0.02. The policy chased an 8+ box edge Q held as a flat main effect; per-season AIPW contrasts are stable within noise (Cochran's Q p=0.64 for base/8+; only nickel/7 and base/≤5 vary by season), so season-pessimism was tried and rejected. Box count is partly *selected*: heavy boxes face more runs than the state predicts (nickel/8+ +10.5 pts) — the read of offensive intent, which the graph does not yet encode. Fixes:
  - **Pessimism must anchor on the coordinator's mix, not its mode.** A deterministic argmax pushed by an LCB collapses to the modal call, stops mixing, and lost ~0.01 in DR. `policy.type: safe` (`src/policy/safe.py`) keeps π_b unless a supported call's advantage over the mix clears `lcb_kappa` bootstrap sds (whole games resampled, `fit_q_ensemble`). κ=2.326 chosen on 2022-23 -> 2024 only: departs ~11% of snaps, 2024 DR +0.0085, 2025 −0.008, pooled −0.000 ± 0.010 — the tightest interval yet, centered on zero.
  - The surviving departures are mostly 8+ boxes: the bootstrap measures sampling noise, not the read's bias. The next lever is the confounding, not more pessimism.
  - Any in-sample policy evaluation must cross-fit the *policy* too: a policy built from in-sample Q showed a spurious +0.078 DR lift on training data.
- **Graph v3.1** (after Phase 1): the first alignment graph failed 11 of its 12 testable independence claims, all for missing edges *among confounders* — so the adjustment set, and every estimate, was unaffected. Added 10 edges (team identity -> tendency/personnel/formation/game state; quality proxies -> game state and the offense's presentation; tendency -> formation; game state -> run/pass), and **execution** as an unobserved mediator (the call, the coverage, the run/pass choice and player quality all reach EPA through it; EPA's parents are now game state and execution). A single-child latent adds no testable claim — it explains why the direct edges existed. Result: every testable claim consistent on 2022–24 **and on 2025**, where the dependencies got weaker, not stronger.
- **Package vs box, and the read allowance** (before Phase 2). Package-only check (treatment = package, box unadjusted): effects season-stable (heterogeneity p=0.70 base, 0.26 dime+), ≈0 on average; the one clear effect is base vs nickel in run-likely states (+0.049 ± 0.031); "dime on passing downs" not supported. No read footprint by package (excess run share within ±0.016); it lives in the box (−0.05 at ≤5 -> +0.06 at 8+, within every package). Chosen: keep the box in the call (option C) but guard it:
  - `policy.read_allowance` (`src/policy/read_bias.py`): bias(box) = excess runs faced (vs a game-cross-fit state-only P(run)) x run-minus-pass reward *within that box* (league-wide the gap is ~0.02 and the bias vanishes; within box it is −0.14 at ≤5, +0.09–0.10 at 7/8+). Bias ≈ +0.007 (≤5), 0 (6), +0.002 (7), +0.006–0.008 (8+). The LCB subtracts scale x |bias(a) − Σ π_b bias|. An allowance, not a correction — only the size is trusted. Run/pass channel of the read only.
  - scale=2 chosen on 2022-23 -> 2024 with κ held at 2.326 (DR +0.0104 there); 2025 −0.0034 ± 0.0096 (vs −0.0084 without), departs ~6% (was ~11%), share of departures to 8+ ~46% (was ~56%).
  - **Reversibility**: `action.level1.axes: [package]` gives the package-only call on the same snaps; the allowance switches itself off and the AIPW baseline falls back to the modal call. The gate takes the train-built `candidate` (building it on the holdout would estimate the bias there).
- **Graph v3.2 — the offense's plan** (`offense_intent`, unobserved): parents game state, team identity, player proxies; children offensive personnel, formation, the run/pass call, and the coach's read. Names the confounding path alignment <- read <- intent -> run/pass -> execution -> EPA behind the heavy-box selection finding. No estimate changes (nothing new to adjust for; the read was already unobserved). Consequences: never split an alignment's effect by run vs pass (run/pass is now a descendant of the confounder); the run/pass ⊥ player-quality claim becomes untestable, leaving one testable claim (personnel ⊥ tendency). `tendency -> intent` is deliberately absent — it would erase that claim, which the data confirms on 2022–24 and 2025. The excess-run bias now feeds the policy's read allowance; not yet built: it as a standing per-season diagnostic (declined for now) and as the benchmark for interpretable Γ.
- **Phase 2 findings** (level 2, the factored policy, the V2 -> V3 gate):
  - **Coverage labels drift with the charting.** Cover-1's share of charted dropbacks rose at 30/32 defenses 2022->24 (median +18.5 pts; man 29% -> 48%) and fell at 32/32 in 2025 (median −15.7) — the Cover-1/Cover-3 boundary moving, not scheme. Safety structure holds within ~4 pts. So `action.level2.shells: family` (Cover-0 / single-high / two-high x pressure = 6; `2_MAN` -> two-high) is the default; `full` (6 shells x 2 = 12, the original spec) is one config line. Off-taxonomy coverage (uncharted, prevent, combo, blown, Cover-9) is dropped. Blitz rate is stable (27–28%).
  - **`full` checked on real data, and rejected as the default.** Same pipeline, `shells: full` (71,794 dropbacks; `2_MAN` dropped): π_b2 accuracy 0.31 vs majority 0.20, worst calibration-in-the-large gap 0.076 (Cover-1 no blitz; family 0.043). Level 2 departs 7.5% (family 2.2%), and 82% of departures go to Cover-1 — the label whose share the charting inflated in the training seasons. No gain for it: coverage-only DR lift −0.002 ± 0.015 (family −0.004 ± 0.007; DM +0.005), both levels −0.019 ± 0.025. V2 -> V3 gate −0.183, CI [−0.61, +0.24] — passes, but resolves nothing. Revisit `full` only with labels that are stable across seasons (e.g. PFF).
  - **Data**: `build_level2_dataset` takes charted dropbacks from the level-1 frame (history features stay computed over all snaps); the level-1 call joins the state as `l1_package` / `l1_front` per configured axes. 75,705 dropbacks (54% of snaps). π_b2 accuracy 0.46 vs majority 0.36, classwise ECE 0.012, next-season calibration-in-the-large within 4.3 pts.
  - **Graph**: `scm.graph.coverage_graph(cfg)` — the alignment graph with treatment def_coverage, the alignment adjusted for (its columns follow `level1.axes`; under package-only the box is a sibling of the coverage and guarded out), selection on coverage_charted, no coverage -> run/pass edge (stated assumption).
  - **κ for level 2 is NOT tuned on DR.** The tuning rule picked κ=0 (departs 95%, 2024 DR +0.073 vs DM +0.020; 2025 +0.028 ± 0.06). DR far above DM is consistent with the read but not unique to it: on the synthetic SCM (notebook config, κ=0) with the read in the result DR reads +0.115 where truth is −0.010; without it DR still overshoots (+0.048 vs truth +0.002, DM +0.025) — noise on ~2k holdout dropbacks, ~2.5x smaller than the read's overshoot. Fixed at level 1's 99% bound (`policy.level2.lcb_kappa: 2.326`): departs 2.2%, 2025 DR lift −0.004 ± 0.007, DM +0.001.
  - **Factored DR** (`src/policy/hierarchical.py`): joint Q = level-2 Q with alignment columns swept; joint π_b = P(a1 | s, dropback) (fit on dropbacks — the dropback is a descendant of the alignment) x π_b2; a deferring level-1 policy plays P(a1|s,dropback). 2025: alignment only −0.017 ± 0.020, coverage only −0.004 ± 0.007, both −0.023 ± 0.021. The alignment row is structurally pessimistic: holding the play a dropback hides the value of heavier looks against the run.
  - **V2 -> V3 gate** (`src/evaluation/projection_gate.py`): V2 refit on SF 2022–24, its 12-call policy projected onto the level-2 calls, with the coordinator's dropback alignment, vs V3 factored; paired DR with V3's joint Q on SF 2025 dropbacks (n=619): −0.055, 95% CI [−0.215, +0.104] -> passes (no regression shown). Underpowered: one defense's season cannot show a difference smaller than ~0.16.
  - Synthetic: `generate_synthetic_alignment(coverage_effects=True)` (separate RNG stream; the level-1 frame is unchanged), `true_dropback_value` for the per-dropback truth.
- **Offense-unit quality proxy**: `off_team_avg_epa`, the offense's EPA/play exponentially weighted with a 250-play half-life (~4 games), carried across seasons so week 1 starts from last season's form rather than zero. Chosen by predicting next-game EPA on 2022–24: 0.319 correlation vs 0.284 for an unweighted history (1-game half-life: 0.262). The QB and defense proxies are still unweighted expanding means — a candidate for the same treatment.
- **Execution in the data**: realized execution (pressure, time to throw, separation) is post-snap and a mediator — never state, never adjusted for; its legitimate use is mediation analysis, e.g. `was_pressure` to test whether the box effect runs through pressure (Phase 5; moved from Phase 2 — it explains a recommendation, it changes no estimate). Expected execution enters only through pre-snap quality proxies.
- **Demo**: `notebooks/v3_pipeline_walkthrough.ipynb` — level-1 recommendations on every snap (short yardage included), the DM -> IPW -> DR ladder, cross-team/season OPE and invariance checks, hierarchical-policy drill-down, per-recommendation causal explanations, and the imputation sensitivity result.

### V4 - Power-Ups
- **Scope**: tracking-data features, structural simulator, full deployment loop. 
- **Models**: neural SCM, counterfactual data augmentation validated against real-data OPE.
- **Add**: shadow-mode deployment with per-recommendation confidence and drift monitoring.
- **Demo**: `notebooks/v4_pipeline_walkthrough.ipynb` — tracking-derived features, structural counterfactual rollouts validated against real-data OPE, and a shadow-mode deployment walkthrough with confidence + drift panels.

---

## Repository Layout (target)

```
data/
  raw/              # nflfastR/CFBfastR/PFF dumps
  processed/        # play-level (S, A, R) tables
  tracking/         # NGS / participation data (V3+)
src/                
  data/             # ingestion, features engineering, action labeling
  scm/              # causal graph, identification analysis, CI tests
  models/           
    behavior/       # π_b - propensity model
    outcome/        # Q / outcome model
    structural/     # generative SCM (V4+)
  ope/              # DM, IPW, DR estimators, sensitivity analysis
  policy/           # policy learning (greedy, CQL, IPL, hierarchical)
  augmentation/     # counterfactual data generation (V4)
  monitoring/       # shadow-mode, drift, explainability
  evaluation/       # holdout validation, synthetic SCM checks
notebooks/          # exploratory analysis, sanity checks
configs/            # version-pinned configs per phase (V1, V2, ...)
tests/              # unit + integration; synthetic SCM regression tests
```

---

## Conventions

### Causal discipline
- Every model that estimates a causal quantity (Q, OPE) must declare which SCM variables it conditions on and why. Mediators must NOT be conditioned on when estimating A -> R effects.
- Every OPE estimate ships with: point estimate, confidence interval, sensitivity bound for unobserved confounding, and the assumed causal graph version.
- Recovering the behavior policy's average reward via OPE is the smoke test before trusting any candidate policy estimate. 

### Data discipline
- Pre-snap features only in S. Any post-snap leakage invalidates the entire pipeline.
- Action labels with known noise (inferred coverage labels) treated as a latent variable, not ground truth.
- Time-aware splits - never random splits across plays. Hold out by season (or week, for in-season evaluation).

### Modeling discipline
- Walking-skeleton-first. No component goes deeper than peers until V_n is end-to-end working.
- Every new model version evaluation against the previous via OPE on a held-out season. Regressions block merge.
- Calibration tracked alongside the accuracy for π_b. Miscalibrated propensities silently break IPW.

### Validation discipline
- Synthetic SCM regression tests: simulate data from a known SCM, run the pipeline, verify recovered effects match truth. Run on every PR to OPE / Q-model / propensity code.
- Coach-in-the-loop sanity checks for V2+ recommendations. If a domain expert can't construct the causal story the recommendation isn't ready.

### Demo-notebook discipline
- Each version ships `notebooks/vN_pipeline_walkthrough.ipynb` as an explicit deliverable: a narrated, plotted, end-to-end run of that version's pipeline with a per-stage explanation. It is not optional polish — a version isn't done until its demo notebook exists.
- The notebook must execute clean top-to-bottom before merge (`python -m nbconvert --to notebook --execute --inplace notebooks/vN_pipeline_walkthrough.ipynb`); a failing or stale notebook blocks merge like a failing test.
- Real-data narrative with a synthetic ground-truth validation aside (V2+), so every headline OPE claim is checkable against a known SCM.
- When a version is superseded, freeze its notebook at that version's git tag rather than letting it rot against `HEAD`.
- Notebooks and the README are **standalone**: no references to earlier versions, upgrades, or what changed. Version history is internal — it lives here in CLAUDE.md and in git tags.

---

## Data Sources

- **nflfastR** (NFL play-by-play, free). Its participation merge carries `defense_personnel` on ~100% of snaps, runs included — pre-snap, and the source of the V3 personnel-package axis. Only head coaches are recorded (no DC/OC identity).
- **FTN charting** (free, 2022+, via `nfl_data_py.import_ftn_data`) - play-level charting on ~99% of scrimmage plays **including runs**: `n_defense_box`, `is_motion`, `n_offense_backfield`, `qb_location`, plus post-snap `is_play_action` / `is_rpo` / `is_screen_pass`. This is the only free source of a defensive attribute observed on every snap, since NGS coverage labels are ~94% on dropbacks and ~3% on runs.
- **CFBfastR** (college play-by-play, free)
- **NFL Next Gen Stats** (tracking, public release - limited)
- **PFF** (charted coverages, participation - paid; needed for cleaner defensive labels)
- **Sportradar** (alternative paid feed)

V1 uses nflfastR + manual inferred / PFF-charted defensive labels for one team. V3+ integrates tracking data. 

---

## Open Questions (revisit per phase)

- How to handle multi-agent dynamics (offense adapts to your tendencies). Single-agent RL is V1-V3; game-theoretic treatment may belong in V4+.
- Reward shaping beyond per-play EPA - drive-level, multi-drive momentum, opponent fatigue.
- Action-space discretization vs. structured/factored representation. Hierarchical policy (V3) is the hedge.
- Deployment ceiling - likely shadow mode and post-game review, not in-game autonomy. Confirm with stakeholders early.

---

## Out of Scope

- Offensive playcalling (could mirror this pipeline, but separate project)
- Real-time in-game autonomous decision-making (acceptable ceiling: live decision-support with human-in-the-loop). 
- Player-level evaluation or contract optimization.

---

### Reading List

- Bareinboim & Pearl - causal bandits, counterfactual policy evaluation
- Lu et al. - *Sample-Efficient RL via Counterfactual-Based Data Augmentation*
- Zhang & Bareinboim - *Designing Optimal Dynamic Treatment Regimes*
- Zeng et al. (2023) - *Causal Reinforcement Learning: A Survey*
- Cervone et al. - EPV in basketball (closest sports CRL precedent)