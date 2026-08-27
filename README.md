# nfl-playcalling

Evaluating NFL playcalling using causal inference techniques and building a
defensive-playcall recommendation agent using causal reinforcement learning.

See [CLAUDE.md](CLAUDE.md) for the full project design and phased build plan.

## V2 — Right Tools, Simple Problem (current)

An end-to-end causal-RL pipeline that produces and **credibly evaluates** a
defensive policy. V2 keeps the problem small (one team) but replaces V1's
deliberately-wrong skeleton parts with the right tools: calibrated propensities,
a doubly-robust off-policy evaluator, a conservative policy, and a sensitivity
analysis that bounds unobserved confounding.

**Scope:** all downs · one team's defense (SF) · 2021–2023 · **12 actions**
= 6 coverage shells {C0, C1, C2, C3, C4, C6} × {blitz, no-blitz}.

**Pipeline:**

| Step | Component | V2 model |
|------|-----------|----------|
| Data | `src/data/` | nflfastR loader + synthetic-SCM generator; 12-action (S, A, R) builder, leak-free confounders + player proxies |
| SCM | `src/scm/` | hand-built DAG (`graph.py`) + DoWhy backdoor identification & positivity check (`identify.py`) |
| πb | `src/models/behavior/` | calibrated LightGBM, cross-fitted; ECE + ESS-ratio diagnostics |
| Q | `src/models/outcome/` | per-treatment GBM on reward(−EPA), SCM-selected features, out-of-fold path for honest DR |
| OPE | `src/ope/` | **doubly-robust** policy value (self-normalized, switch-clipped) + DR recovery smoke test; AIPW contrasts; Rosenbaum sensitivity |
| Policy | `src/policy/` | conservative behavior-regularized (CQL/IQL-style) policy |

### Install

```bash
pip install --only-binary=:all: -r requirements.txt
```

(`--only-binary=:all:` avoids source builds on the py3.13 / sklearn 1.8 /
pandas 3 stack.)

### Run

```bash
python run.py                     # real SF nflfastR data (configs/v2.yaml)
python run.py --source synthetic  # known-SCM data, no network
pytest tests/                     # synthetic-SCM regression tests
```

For a narrated, step-by-step walkthrough with plots (action sparsity, the DAG,
positivity heatmap, calibration curve, DR recovery, and the sensitivity band),
open [`notebooks/v2_pipeline_walkthrough.ipynb`](notebooks/v2_pipeline_walkthrough.ipynb).
It runs the story on real SF data with a synthetic ground-truth validation aside.

All V2 scope lives in [`configs/v2.yaml`](configs/v2.yaml) (team, seasons,
action taxonomy, model hyperparameters, sensitivity γ range).

### What V2 fixes vs V1

- **Off-support blow-up → doubly-robust OPE.** V1's ridge-Q Direct Method
  extrapolated to implausible off-distribution values, so greedy lift looked
  absurd (+1.x EPA/play). DR's IPW correction on logged plays pulls estimates
  back to reality; the DR recovery smoke test gates every candidate policy.
- **Uncalibrated propensities → calibrated LightGBM.** Isotonic calibration with
  ECE tracked, since miscalibrated propensities silently break inverse-weighting.
- **Unaddressed confounding → sensitivity analysis.** A Rosenbaum / marginal-
  sensitivity-model bound reports whether the estimated lift survives an
  unobserved confounder of odds-ratio Γ (validated against the synthetic SCM's
  known confounding).
- **Naive greedy → conservative policy.** A behavior-regularized argmax that
  only departs from the DC when Q is confidently higher and the call is
  well-supported.

### Known V2 characteristics (honest limitations)

- **Sparse action space for one team.** 12 actions over ~1,300 charted training
  plays leaves several (coverage × blitz) cells with a handful — or a single —
  play. The pipeline **surfaces this** rather than hiding it: the positivity
  report flags unsupported strata, and πb falls back to *uncalibrated* when a
  call is too rare to cross-validate. Holdout policy-value CIs are correspondingly
  wide, and the sensitivity analysis typically rates the lift *not robust* to
  even mild confounding — the appropriately-humble answer this data supports.
  Denser labels (PFF charting) or multi-team pooling (V3) are the fixes.
- **econml CATE / explainability** is deferred to V3, per the phased plan.

## V1 — Walking Skeleton (archived)

V1 (3rd downs only, 4 actions, logistic πb, ridge Q, Direct-Method OPE, greedy
policy) is preserved at git tag **`v1`**:

```bash
git checkout v1        # original skeleton + notebooks/v1_pipeline_walkthrough.ipynb
```
