# Graph Report - nfl-playcalling  (2026-08-27)

## Corpus Check
- Corpus is ~12,827 words - fits in a single context window. You may not need a graph.

## Summary
- 312 nodes · 598 edges · 19 communities (10 shown, 9 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 29 edges (avg confidence: 0.89)
- Token cost: 73,273 input · 0 output

## Community Hubs (Navigation)
- OPE & Policy Doctrine
- Problem Framing & Data Sources
- OPE Estimators & Data Schemas
- Pipeline Entrypoint & Ingestion
- Causal Graph & Identification
- Feature Engineering & SCM Tests
- Behavior Policy Fitting
- Q-Model & Preprocessing
- Q Prediction Interface
- Policy Action Interface
- Data Package Namespace
- Evaluation Package Namespace
- Source Package Root
- Behavior Package Namespace
- Models Package Namespace
- Outcome Package Namespace
- OPE Package Namespace
- Policy Package Namespace
- Step 0: Tractable Slice

## God Nodes (most connected - your core abstractions)
1. `Dataset` - 28 edges
2. `main()` - 21 edges
3. `BehaviorModel` - 15 edges
4. `QModel` - 15 edges
5. `V2 Run Config` - 13 edges
6. `fit_behavior_model()` - 12 edges
7. `_diagnose()` - 12 edges
8. `behavior_recovery_check()` - 12 edges
9. `Causal RL for Defensive Playcalling` - 12 edges
10. `fit_q_model()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Conservative Behavior-Regularized Policy` --semantically_similar_to--> `Behavior-Constrained Greedy Policy`  [INFERRED] [semantically similar]
  README.md → CLAUDE.md
- `networkx` --conceptually_related_to--> `Step 2: Causal Graph (SCM Skeleton)`  [INFERRED]
  requirements.txt → CLAUDE.md
- `V1 OPE Propensity Clipping` --conceptually_related_to--> `Inverse Propensity Weighting`  [INFERRED]
  configs/v1.yaml → CLAUDE.md
- `V2 Positivity Min-Count Threshold` --shares_data_with--> `DoWhy Backdoor Identification + Positivity Check`  [INFERRED]
  configs/v2.yaml → README.md
- `V2 Calibrated LightGBM Behavior Model` --references--> `lightgbm`  [INFERRED]
  configs/v2.yaml → requirements.txt

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **OPE Estimator Progression (DM -> IPW -> DR -> Sensitivity)** — claude_direct_method_ope, claude_ipw, claude_doubly_robust_ope, claude_sensitivity_analysis_unobserved_confounding, claude_behavior_policy_recovery_smoke_test [EXTRACTED 1.00]
- **V2 Pipeline Components** — readme_calibrated_lightgbm_propensity, readme_per_treatment_gbm_q, readme_doubly_robust_policy_value, readme_conservative_policy, readme_dowhy_backdoor_identification, readme_rosenbaum_sensitivity [EXTRACTED 1.00]
- **Causal Validity Guardrails** — claude_mediator_exclusion, claude_pre_snap_only_no_leakage, claude_propensity_calibration, claude_positivity_violations, claude_time_aware_splits, claude_synthetic_scm_regression_test [INFERRED 0.85]

## Communities (19 total, 9 thin omitted)

### Community 0 - "OPE & Policy Doctrine"
Cohesion: 0.07
Nodes (55): Behavior-Constrained Greedy Policy, Behavior-Policy Recovery Smoke Test, Causal Discipline Convention, Coach-in-the-Loop Sanity Check, CQL / IQL Conservative Offline RL, Demo-Notebook Discipline, Direct Method OPE, Doubly Robust OPE (+47 more)

### Community 1 - "Problem Framing & Data Sources"
Cohesion: 0.05
Nodes (51): Action (A): Defensive Call, Open Question: Action-Space Discretization vs Factoring, Bareinboim & Pearl - Causal Bandits, Causal RL for Defensive Playcalling, Cervone et al. - EPV in Basketball, CFBfastR Data Source, Data Discipline Convention, Decision Support, Not Autopilot (+43 more)

### Community 2 - "OPE Estimators & Data Schemas"
Cohesion: 0.11
Nodes (30): behavior_recovery_check(), dm_policy_value(), ndarray, Off-Policy Evaluation — Direct Method + the mandatory recovery smoke test. The…, Direct-Method value of a (possibly stochastic) policy. q_all, policy_probs:…, Smoke test: does OPE recover the behavior policy's mean reward? Numbers that…, dr_policy_value(), ndarray (+22 more)

### Community 3 - "Pipeline Entrypoint & Ingestion"
Cohesion: 0.09
Nodes (31): _example_recommendations(), load_config(), main(), _rule(), _save_outputs(), Split by season — never randomly across plays (avoids temporal leakage)., time_aware_split(), generate_synthetic() (+23 more)

### Community 4 - "Causal Graph & Identification"
Cohesion: 0.08
Nodes (28): adjustment_columns(), adjustment_set(), describe(), Causal Graph (SCM Skeleton) — V2. A hand-built DAG over the project variables.…, Human-readable summary for the run log., Abstract confounder nodes to condition on (backdoor set given observed)., Concrete feature columns forming the adjustment set (mediators excluded). This…, GML serialization of the abstract DAG (for inspection / DoWhy). (+20 more)

### Community 5 - "Feature Engineering & SCM Tests"
Cohesion: 0.10
Nodes (22): Series, _add_history_features(), build_dataset(), _build_state(), _expanding_mean(), _label_actions(), DataFrame, Data Layer — feature engineering: raw play-by-play -> (S, A, R) table.… (+14 more)

### Community 6 - "Behavior Policy Fitting"
Cohesion: 0.14
Nodes (22): LGBMClassifier, Pipeline, _base_classifier(), _diagnose(), _ess_ratio(), _expected_calibration_error(), fit_behavior_model(), _make_classifier() (+14 more)

### Community 7 - "Q-Model & Preprocessing"
Cohesion: 0.16
Nodes (17): ColumnTransformer, fixture, _check_conditions_on_adjustment_set(), crossfit_q(), fit_q_model(), _make_regressor(), ndarray, Outcome / Q-Model — Q(S, A) = E[R | S, do(A)]. V2: a per-treatment gradient-… (+9 more)

### Community 8 - "Q Prediction Interface"
Cohesion: 0.47
Nodes (4): DataFrame, ndarray, Q(s, a) for every action, shape (n, n_actions). Counterfactual sweep: hold…, Q(s, a) only for the action actually taken in each row.

### Community 9 - "Policy Action Interface"
Cohesion: 0.40
Nodes (3): DataFrame, Return per-state Q values, support mask, and chosen action., Return per-state Q, propensities, conservative scores, and choice.

## Knowledge Gaps
- **9 isolated node(s):** `Step 0: Tractable Slice`, `Coach-in-the-Loop Sanity Check`, `Target Repository Layout`, `CFBfastR Data Source`, `Sportradar Feed` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Dataset` connect `OPE Estimators & Data Schemas` to `Pipeline Entrypoint & Ingestion`, `Causal Graph & Identification`, `Feature Engineering & SCM Tests`, `Behavior Policy Fitting`, `Q-Model & Preprocessing`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `QModel` connect `OPE Estimators & Data Schemas` to `Q Prediction Interface`, `Q-Model & Preprocessing`?**
  _High betweenness centrality (0.030) - this node is a cross-community bridge._
- **Why does `main()` connect `Pipeline Entrypoint & Ingestion` to `OPE Estimators & Data Schemas`, `Causal Graph & Identification`, `Feature Engineering & SCM Tests`, `Behavior Policy Fitting`, `Q-Model & Preprocessing`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `BehaviorModel` (e.g. with `ConservativePolicy` and `GreedyPolicy`) actually correct?**
  _`BehaviorModel` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `QModel` (e.g. with `ConservativePolicy` and `GreedyPolicy`) actually correct?**
  _`QModel` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Step 0: Tractable Slice`, `Coach-in-the-Loop Sanity Check`, `Target Repository Layout` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `OPE & Policy Doctrine` be split into smaller, more focused modules?**
  _Cohesion score 0.06734006734006734 - nodes in this community are weakly interconnected._