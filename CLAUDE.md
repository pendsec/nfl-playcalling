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
- **Action (A)**: defensive call - coverage shell x pressure x front x disguise. Discretized and (eventually) factored hierarchically.
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

### Step 3: Behavior Policy Model - π_b(A|S)
Classifier predicting what defensive call the actual DC made in state S. Heart of IPW reweighting and doubly robust OPE. Calibration is critical; positivity violations bound how aggressive the learned policy can be.

### Step 4: Outcome / Q-Model
Predicts E[R | S, do(A)] - the *interventional* outcome, not observational. Built using only confounders + state features per the SCM (mediators excluded). Eventually grows into a structural causal model supporting counterfactual rollouts.

### Step 5: Off-Policy Evaluation (OPE)
Unbiased estimator of any candidate policy's value from logged data. Direct Method -> IPW -> Doubly Robust -> sensitivity analysis for unobserved confounders. The judge of every candidate policy. Validate by recovering the behavior policy's known average reward. 

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
- **Scope**: all downs, one team, 12 actions (coverage shell x pressure).
- **Models**: GBM π_b with calibration, GBM Q-model with proper feature selection from SCM, doubly robust OPE, CQL/IQL policy.
- **Add**: sensitivity analysis bounding impact of unobserved confounders.
- **Demo**: `notebooks/v2_pipeline_walkthrough.ipynb` — real-data narrative (action sparsity, DAG + DoWhy identification, positivity heatmap, calibration curve, DR recovery, sensitivity band) plus a synthetic ground-truth validation aside.

### V3 - Scale and Transfer
- **Scope**: multi-team, factored 50+ action space.
- **Models**: neural π_b and Q-model with embeddings (team, QB, OC), invariant policy learning across seasons/teams, hierarchical factored policy.
- **Add**: explainability layer surfacing top causal features per recommendation.
- **Demo**: `notebooks/v3_pipeline_walkthrough.ipynb` — multi-team transfer story: cross-team/season OPE, invariance checks, hierarchical-policy drill-down, and the per-recommendation causal-feature explanations.

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

---

## Data Sources

- **nflfastR** (NFL play-by-play, free)
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