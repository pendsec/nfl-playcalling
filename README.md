# nfl-playcalling

Evaluating NFL playcalling using causal inference techniques and building a
defensive-playcall recommendation agent using causal reinforcement learning.

See [CLAUDE.md](CLAUDE.md) for the full project design and phased build plan.

## The pipeline

An end-to-end causal-RL pipeline that produces and **credibly evaluates** a
defensive policy. The problem is kept deliberately small — one team — so that
every component can be checked: calibrated propensities, a doubly-robust
off-policy evaluator, a conservative policy, and a sensitivity analysis that
bounds unobserved confounding.

**Scope:** all downs · one team's defense (SF) · 2022–2024 · **charted
dropbacks** · **12 actions** = 6 coverage shells {C0, C1, C2, C3, C4, C6} ×
{blitz, no-blitz}.

The season window starts at 2022 because FTN charting does — see
[FTN charting](#ftn-charting) below.

The dropback restriction is not incidental, and it is more than a scope note —
see [What the estimand actually is](#what-the-estimand-actually-is) below.

**Pipeline:**

| Step | Component | Model |
|------|-----------|----------|
| Data | `src/data/` | nflfastR + FTN loaders and synthetic-SCM generator; 12-action (S, A, R) builder, leak-free confounders + player proxies |
| SCM | `src/scm/` | hand-built DAG (`graph.py`) with declared mediator + selection nodes, content-fingerprinted; DoWhy backdoor identification & positivity check (`identify.py`) |
| πb | `src/models/behavior/` | calibrated LightGBM, cross-fitted around un-splittable classes; confidence/classwise ECE + Brier + ESS-ratio diagnostics |
| Q | `src/models/outcome/` | per-treatment GBM on reward(−EPA), SCM-selected features, out-of-fold path for honest DR |
| OPE | `src/ope/` | **doubly-robust** policy value (self-normalized, switch-clipped) + DR recovery smoke test; AIPW contrasts (same clipping, reported with support counts); Rosenbaum sensitivity |
| Policy | `src/policy/` | conservative behavior-regularized (CQL/IQL-style) policy |
| Eval | `src/evaluation/` | policy regression gate: paired DR comparison of the candidate against the baseline policy on the held-out season |

**Built but not wired in:** `src/models/imputation/` treats the coverage shell as
a *latent* treatment label — charted on ~94% of dropbacks but ~3% of runs — and
imputes it from a calibrated pre-snap posterior, sampling `m` draws and pooling
them with Rubin's rules so the interval reflects what the imputation does not
know. It has no entry point in `run.py`, the notebook, or `configs/`: it is
groundwork for the factored action space, and running it changes the estimand.
Read `RubinResult.fraction_missing_information` before trusting anything built
on it — on this data roughly **43%** of the pooled variance comes from not
knowing the label, and two independent draws agree on only ~32% of plays.

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

All scope lives in [`configs/v2.yaml`](configs/v2.yaml) (team, seasons,
action taxonomy, model hyperparameters, sensitivity γ range).

### Design decisions, and the failure each one prevents

- **Doubly-robust OPE, not the Direct Method.** A plug-in Q model asked to score
  actions it has no support for extrapolates to implausible values, and a greedy
  policy chasing them reports absurd lift (+1.x EPA/play is achievable this way,
  and meaningless). DR's importance-weighted correction on logged plays anchors
  the estimate to observed reality, and the DR recovery smoke test gates every
  candidate policy before it is believed.
- **Calibrated propensities.** Isotonic calibration with calibration error
  tracked, since miscalibrated propensities silently break inverse-weighting.

  Calibration metrics live in `src/models/calibration.py` and are reported as a
  triple: **confidence ECE** (is the top-class confidence right?), **classwise
  ECE** (is *every* class's probability right? — the one that matters, since
  propensities are inverted into weights that use every column), and **Brier**,
  a proper scoring rule. ECE is never reported alone: it is minimized by a
  constant base-rate predictor, so on its own it cannot separate "well
  calibrated" from "uninformative but well calibrated".

  A note on the metric, since the obvious implementation is wrong: binning by
  the predicted probability of the **true** class while scoring whether the
  **argmax** was correct mixes two different quantities, and reads ~0.25 on
  probabilities that are perfectly calibrated by construction. The definitions
  in `calibration.py` read ~0 on such input, which is what makes πb's measured
  0.022 meaningful.
- **Sensitivity analysis on every headline number.** A Rosenbaum / marginal-
  sensitivity-model bound reports whether the estimated lift survives an
  unobserved confounder of odds-ratio Γ (validated against the synthetic SCM's
  known confounding).
- **A conservative policy rather than a naive greedy one.** A behavior-regularized
  argmax that departs from the DC only when Q is confidently higher and the call
  is well-supported. The unregularized greedy policy is kept as the baseline the
  regression gate measures against.

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
  scores the greedy baseline and the conservative candidate by DR on the *same*
  held-out plays, so the difference is **paired** — the shared Q/πb noise
  cancels instead of adding, which matters because each policy's own CI is far
  wider than the gap between them. The gate blocks only on a *significant
  deterioration* beyond a configured non-inferiority margin
  (`evaluation.regression_margin`); failing to prove an improvement is an
  acceptable outcome on this data, and `run.py` exits non-zero on a regression.

### FTN charting

`data.use_ftn` merges FTN's play-level charting (2022+, ~99% of scrimmage plays)
alongside nflfastR. Its columns are split by **when they become knowable**, which
is a causal distinction rather than a stylistic one:

| FTN column | role | in the adjustment set? |
|---|---|---|
| `is_motion`, `n_offense_backfield`, `qb_location` | pre-snap offensive presentation — the defense sees these before calling | **yes** |
| `n_defense_box` | pre-snap, but a *defensive choice* | **no** — carried only |
| `is_play_action`, `is_rpo`, `is_screen_pass` | post-snap reveals | **no** — carried only |

`n_defense_box` is the interesting one. It is pre-snap, which makes it look like
an ideal confounder — but it is part of the defense's own decision, a *sibling*
of the treatment rather than a cause of it, so adjusting for it would block a
slice of the effect being estimated. It is carried because it is the only
defensive attribute charted on ~99% of **all** snaps (runs included, where
coverage is charted on ~3%), which makes it the primary axis of the planned factored
action space. The DAG declares it as `def_front`
(`scm.graph.DEFENSIVE_CHOICE`), and `assert_adjustment_consistency` raises if it
— or any post-snap flag — turns up among the state features.

Two guards keep this from degrading quietly:

- **`features.assert_ftn_coverage`** refuses to train FTN features on a season
  FTN does not cover. Without it, a 2021–2023 window would default every 2021
  row, and the model would learn "no motion" as a property of *2021* rather than
  of the play — a feature confounded with season across half of training. The
  season window is 2022–2024 rather than letting that happen.
- **Zero-sentinel scrubbing.** FTN writes `0` / `"0"` into `n_defense_box` and
  `qb_location` where it charts nothing (kickoffs, punts, timeouts, and ~0.2% of
  scrimmage plays). Zero defenders in the box is not an alignment, so the
  sentinel becomes `NaN` instead of being read as a count.

### What the estimand actually is

This pipeline estimates **the effect of a coverage call on EPA, given a charted dropback** —
not the unconditional effect of the call. That restriction is forced by the data
and is now encoded in the DAG (`scm.graph.SELECTION`) rather than left in prose,
so it is hashed into the graph fingerprint every estimate carries.

Why it is forced: the coverage axis comes from nflfastR's
`defense_coverage_type`, which is NGS charting on dropbacks. Runs carry a shell
label on **3.0%** of snaps (43 of 1,411), and only in 2023 — 2021 and 2022 are
at 0.0%. Coarsening to a man/zone axis does not rescue them: after discarding
empty-string placeholders, runs carry a man/zone label on the same 3.0%. There
is no cheaper action axis hiding in the run plays.

Two consequences, both real:

- **Selection on a mediator.** The defense calls its play pre-snap; run-vs-pass
  is realized after. So filtering to dropbacks conditions on a *post-treatment*
  variable, and `def_playcall → off_playcall` is a genuine edge — the look the
  defense shows drives audibles and RPO reads. A call's ability to *deter* a run
  is therefore invisible to this pipeline. Descriptively, SF's opponents ran on
  14% of snaps against a ≤5-man box and 65% against 8+; whatever share of that
  is causal, none of it is measured here.

- **Informative missingness among the dropbacks themselves.** 242 pass plays are
  dropped for uncharted coverage, and they are the defense's *best* outcomes —
  mean reward **+0.92** against **−0.04** for the plays kept. Sacks, scrambles
  and throwaways resolve no shell to label. The full pass population averages
  +0.067 reward/play; the modelled slice averages −0.036, so every absolute
  policy value here is biased low by roughly 0.10 reward/play, and the DR
  recovery smoke test recovers the *slice's* mean rather than the team's.

  What keeps this from invalidating the comparison: the drop rate is **9.3% on
  blitzes vs 9.4% on non-blitzes** — near-identical, so the bias shifts every
  policy's level together instead of distorting the contrast between them, which
  is what OPE is actually asked to judge. Treat the *lift* as the meaningful
  quantity and the absolute level as an underestimate.

The fix for both is denser charting (PFF) or a treatment definition observable
on every snap — `defenders_in_box` is charted on ~98% of runs and passes alike,
which is what makes a box-count action axis the natural next step.

### Known characteristics (honest limitations)

- **Sparse action space for one team.** 12 actions over ~1,300 charted training
  plays leaves several (coverage × blitz) cells with a handful — or a single —
  play. The pipeline **surfaces this** rather than hiding it: the positivity
  report flags unsupported strata, and a class too thin to stratify is held on
  the training side of every cross-fit fold and counted in the diagnostics
  rather than quietly disabling them. Holdout policy-value CIs are
  correspondingly wide, and the sensitivity analysis typically rates the lift
  *not robust* to even mild confounding — the appropriately-humble answer this
  data supports. Denser labels (PFF charting) or multi-team pooling are
  the fixes.

- **πb is only slightly better than guessing the modal call.** Cross-fitted
  accuracy clears the majority baseline by a few points, and the ESS ratio sits
  near 0.28 — a minority of plays carry most of the inverse weight. Measured
  in-sample the same model reports ~0.98 accuracy and a 0.93 ESS ratio; both are
  memorisation. The out-of-fold numbers are the ones that bound how far π\* may
  move, so they are the ones reported.
- **The conservative policy does not measurably beat the greedy baseline.** The
  regression gate reports *no change detected*: the conservative policy scores
  slightly above the baseline, with a paired-difference CI comfortably straddling
  zero. On ~700 holdout plays the data cannot resolve the difference either way,
  and the width is honest — the two policies now make genuinely different calls,
  where a near-identical pair would have produced a precise-looking interval
  around nothing. The contribution here is that the estimate is *credible*, not
  that the number is higher.

- **Conservatism is a scaled knob, not a magic constant.** The pessimism penalty
  is expressed in units of the fitted Q model's typical within-state spread
  (`QModel.q_scale`), so `policy.alpha` is comparable across fits instead of
  depending on the reward scale. It is set from a support criterion — at
  `alpha: 1.0` the mean πb of recommended calls (0.220) exceeds that of the DC's
  own calls (0.165), and recommendations resting on πb < 0.10 fall from 36%
  (greedy) to 16% — not from the OPE number, which varies well inside noise
  across the range.
- **No standalone IPW estimator.** The pipeline ships DM and DR; the pure-IPW
  rung of the DM → IPW → DR ladder is deferred until multi-team pooling makes it
  informative. At this ESS ratio a pure-IPW estimate would be dominated by a
  few tiny propensities and would add variance without changing a decision.
- **The DAG is a hand-built hypothesis, not a discovered graph.** It is
  versioned (`GRAPH_VERSION`) and content-fingerprinted, so an estimate can never
  drift from the structure that produced it: edit an edge, move a variable
  between observed/mediator/selection, and the fingerprint on every stored
  estimate changes even if nobody remembered to bump the label. What that
  machinery does *not* do is tell you the graph is
  right; the conditional-independence tests that would challenge it are thin at
  this sample size.

- **econml CATE / explainability** is deferred, per the phased plan in CLAUDE.md.
