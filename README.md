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

**Scope:** all downs · one team's defense (SF) · 2021–2023 · **charted
dropbacks** · **12 actions** = 6 coverage shells {C0, C1, C2, C3, C4, C6} ×
{blitz, no-blitz}.

The dropback restriction is not incidental: the coverage axis comes from
nflfastR's `defense_coverage_type`, which is NGS charting on pass plays, so only
~3% of runs carry a label. `data.play_types` declares that explicitly and the
run log reports the split, so a recommendation reads *"given the offense drops
back"* rather than unconditionally.

**Pipeline:**

| Step | Component | V2 model |
|------|-----------|----------|
| Data | `src/data/` | nflfastR loader + synthetic-SCM generator; 12-action (S, A, R) builder, leak-free confounders + player proxies |
| SCM | `src/scm/` | hand-built DAG (`graph.py`) + DoWhy backdoor identification & positivity check (`identify.py`) |
| πb | `src/models/behavior/` | calibrated LightGBM, cross-fitted around un-splittable classes; ECE + ESS-ratio diagnostics |
| Q | `src/models/outcome/` | per-treatment GBM on reward(−EPA), SCM-selected features, out-of-fold path for honest DR |
| OPE | `src/ope/` | **doubly-robust** policy value (self-normalized, switch-clipped) + DR recovery smoke test; AIPW contrasts (same clipping, reported with support counts); Rosenbaum sensitivity |
| Policy | `src/policy/` | conservative behavior-regularized (CQL/IQL-style) policy |
| Eval | `src/evaluation/` | version-over-version regression gate: paired DR comparison of the V1 vs V2 policy on the held-out season |

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

### Causal bookkeeping

Two disciplines from [CLAUDE.md](CLAUDE.md) are enforced in code rather than by
convention:

- **Every OPE estimate declares its assumptions.** `OPEResult` carries the
  estimator, the policy valued, and the causal-graph version it assumed
  (`scm.graph.version_tag()` — a hand-maintained `GRAPH_VERSION` plus a content
  fingerprint of the DAG, so an unversioned edit to the graph still shows up on
  every stored estimate). A sensitivity table is *attached* to the estimate it
  qualifies rather than living in a separate variable, and `attach` refuses a
  bound computed for a different policy. `outputs/ope_policy_values.csv` is one
  self-describing row per estimate: value, CI, n, estimator, graph version, and
  the Γ up to which the conclusion survives.

- **Version regressions block merge.** `src/evaluation/version_compare.py`
  scores V1's greedy policy and V2's conservative policy by DR on the *same*
  held-out plays, so the difference is **paired** — the shared Q/πb noise
  cancels instead of adding, which matters because each policy's own CI is far
  wider than the gap between them. The gate blocks only on a *significant
  deterioration* beyond a configured non-inferiority margin
  (`evaluation.regression_margin`); failing to prove an improvement is an
  acceptable outcome on this data, and `run.py` exits non-zero on a regression.

### Known V2 characteristics (honest limitations)

- **Sparse action space for one team.** 12 actions over ~1,300 charted training
  plays leaves several (coverage × blitz) cells with a handful — or a single —
  play. The pipeline **surfaces this** rather than hiding it: the positivity
  report flags unsupported strata, and a class too thin to stratify is held on
  the training side of every cross-fit fold and counted in the diagnostics
  rather than quietly disabling them. Holdout policy-value CIs are
  correspondingly wide, and the sensitivity analysis typically rates the lift
  *not robust* to even mild confounding — the appropriately-humble answer this
  data supports. Denser labels (PFF charting) or multi-team pooling (V3) are
  the fixes.

- **πb is only slightly better than guessing the modal call.** Cross-fitted
  accuracy clears the majority baseline by a few points, and the ESS ratio sits
  near 0.28 — a minority of plays carry most of the inverse weight. Measured
  in-sample the same model reports ~0.98 accuracy and a 0.93 ESS ratio; both are
  memorisation. The out-of-fold numbers are the ones that bound how far π\* may
  move, so they are the ones reported.
- **V2's policy does not measurably beat V1's on the holdout.** The regression
  gate reports *no change detected*: the conservative policy scores slightly
  above the V1 greedy policy, with a paired-difference CI comfortably straddling
  zero. On ~700 holdout plays the data cannot resolve the difference either way,
  and the width is honest — the two policies now make genuinely different calls,
  where a near-identical pair would have produced a precise-looking interval
  around nothing. V2's contribution is that the estimate is *credible*, not that
  the number is higher.

- **Conservatism is a scaled knob, not a magic constant.** The pessimism penalty
  is expressed in units of the fitted Q model's typical within-state spread
  (`QModel.q_scale`), so `policy.alpha` is comparable across fits instead of
  depending on the reward scale. It is set from a support criterion — at
  `alpha: 1.0` the mean πb of recommended calls (0.220) exceeds that of the DC's
  own calls (0.165), and recommendations resting on πb < 0.10 fall from 36%
  (greedy) to 16% — not from the OPE number, which varies well inside noise
  across the range.
- **No standalone IPW estimator.** V2 ships DM and DR; the pure-IPW rung of the
  DM → IPW → DR ladder is deferred to V3, where multi-team pooling makes it
  informative. At V2's ESS ratio a pure-IPW estimate would be dominated by a
  few tiny propensities and would add variance without changing a decision.
- **econml CATE / explainability** is deferred to V3, per the phased plan.

## V1 — Walking Skeleton (archived)

V1 (3rd downs only, 4 actions, logistic πb, ridge Q, Direct-Method OPE, greedy
policy) is preserved at git tag **`v1`**:

```bash
git checkout v1        # original skeleton + notebooks/v1_pipeline_walkthrough.ipynb
```
