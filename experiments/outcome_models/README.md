# Outcome models: which Q serves the policy best?

**Question.** The policy never uses Q's level. It uses each call's *advantage* over
the coordinator's mix, `A(s, a) = Q(s, a) − Σ_b π_b(b|s) Q(s, b)`, and only for the
calls it may choose (π_b ≥ 0.05). The situation moves EPA by about 1.5 per snap,
and a call by about 0.05. Can a model built for causal contrasts estimate that
small number better than the pooled gradient-boosted regressor?

**Verdict.**
- **The pooled GBM stays the default.** No candidate beats it on any of the three
  benchmarks.
- **The factored DragonNet is the neural candidate for Phase 4.** It ties the GBM
  on the real holdout's doubly-robust loss and is second on the synthetic truth.
- **It is not yet usable in the safe policy.** Its game-bootstrap spread is too
  tight: it departs from the coordinator about 2.5× as often and loses
  0.012 ± 0.008 on 2025. It needs its own uncertainty estimate (an ensemble of
  independently trained networks, or MC dropout) before it can be gated against
  the GBM.

## Candidates

All eight expose `predict_all_actions(df) → (n, 12)`, so the safe policy, the
bootstrap ensemble and OPE take any of them unchanged. All condition on the SCM
adjustment set only.

| Name | What it is | Where |
|---|---|---|
| `gbm_pooled` | One LightGBM regressor over state plus the call's (package, box). Q comes from swapping the call's features and re-predicting. | `src/models/outcome/q_model.py` |
| `dr_learner_ridge` / `dr_learner_gbm` | econml DR-learner, explained below. Ridge or shallow-GBM final stage. | `cate.py` (this folder) |
| `r_learner_forest` | econml causal forest DML (Robinson's decomposition), explained below. | `cate.py` |
| `tarnet` / `tarnet_factored` | PyTorch: shared state representation φ(s), with separate or factored heads. | `src/models/outcome/neural.py` |
| `dragonnet` / `dragonnet_factored` | TARNet plus a propensity head and targeted regularization. | `src/models/outcome/neural.py` |

**DR-learner**
1. Cross-fits an outcome model μ(s, a) and a propensity π(a|s).
2. Forms the doubly robust pseudo-outcome
   `G_a = μ(s,a) + 1{A=a}/π(a|s)·(R − μ(s,A))` for each call.
3. Regresses `G_a − G_ref` on the state, with the most common call as the reference.

It is unbiased if either nuisance model is right. The inverse weights make it
noisy on rarely made calls.

**R-learner**
1. Residualizes the reward, `R − m(s)`, and the call indicators, `1{A=a} − π(a|s)`.
2. Fits the contrasts τ(s) that best explain one residual with the other, using
   a generalized random forest.

It uses no inverse weights. A rare call's indicator barely varies after
residualization, so its contrast is only weakly determined.

For both econml learners, Q is rebuilt as `m(s) + τ_a(s) − Σ π(b|s) τ_b(s)`, so
the coordinator's mix scores m(s).

**TARNet**
- **Representation:** an MLP over standardized numerics plus learned embeddings
  for defense, offense, formation and QB alignment. Embedding 0 is reserved for
  an unseen category and stays at zero.
- **Heads:** one per call, read off φ(s). Each snap's loss uses the head of the
  call actually made.
- **Early stopping:** on held-out whole games.

**DragonNet**
- **Propensity head π(a|φ), trained jointly.** This makes φ keep what predicts
  the coordinator's call, i.e. the confounders.
- **Targeted regularization:** a learnable ε_a per call and the extra loss
  `(R − Q(s,A) − ε_A/π(A|s))²`. This is the doubly robust estimating equation
  built into training, extended from the binary original to 12 calls.

**Factored heads** replace the 12 per-call heads with one head over (φ, the
call's package and box, its one-hot). This is the neural counterpart of the
pooled GBM, and it is the default in `neural.py`.

## Method

Every candidate is fit on 2022–24 and scored on the 2025 holdout, on the calls
with π_b ≥ 0.05. The safe policy uses 10 game-bootstrap refits (the pipeline
uses 20).

- **Synthetic leagues** (140k plays, 32 teams; truth from
  `src.data.load.true_alignment_q`), run with the coordinator's hidden read
  (`synthetic_with_read`) and without it (`synthetic_no_read`, pure estimation
  error). Scored on:
  - advantage RMSE and correlation against the truth;
  - spread ratio (estimated sd / true sd; above 1 is noise the argmax chases);
  - the true lift of the greedy and safe policies over the coordinator's mix.
- **Real 2025** (`real_2025`), with no truth. Scored on:
  - **DR loss:** mean squared gap between the estimated advantages and doubly
    robust pseudo-outcomes. The pseudo-outcomes are built only from 2025, with
    nuisances cross-fitted by game, and the loss is reported relative to the GBM.
    The pseudo-outcome is unbiased for the true advantage, so differences in this
    loss estimate differences in true error.
  - **Calibration slope** of the pseudo-outcomes on the estimated advantages.
  - **Greedy and safe-policy DR lift,** scored with the pipeline's own judge.
- **Checking the real-data judge:** the same real-data criteria also run on the
  synthetic leagues. Without the read, the DR loss ranks the eight models in
  almost exactly their true-error order. With the read, it ties the factored
  DragonNet with the GBM where the truth slightly favours the GBM.

## Results

Error is the advantage RMSE against the truth, on supported calls. Lift is the
safe policy's true lift. The real-data column is the DR loss relative to the
GBM (lower is better).

| Model | Error, no read | Error, with read | Spread, no read | Lift, no read / with read | Real DR loss vs GBM |
|---|---|---|---|---|---|
| **gbm_pooled** | **0.0216** | **0.0247** | 0.78 | **+0.026** / +0.025 | — |
| dragonnet_factored | 0.0231 | 0.0288 | 0.59 | +0.020 / +0.025 | +0.0001 ± 0.0007 |
| tarnet_factored | 0.0249 | 0.0282 | 0.75 | +0.025 / +0.024 | +0.0011 ± 0.0007 |
| r_learner_forest | 0.0252 | 0.0321 | 1.05 | +0.025 / +0.024 | +0.0029 ± 0.0012 |
| dr_learner_gbm | 0.0360 | 0.0411 | 1.22 | +0.017 / +0.020 | +0.0070 ± 0.0028 |
| tarnet | 0.0392 | 0.0402 | 1.42 | +0.006 / +0.010 | +0.0101 ± 0.0021 |
| dr_learner_ridge | 0.0399 | 0.0438 | 1.21 | +0.013 / +0.018 | +0.0109 ± 0.0027 |
| dragonnet | 0.0445 | 0.0488 | 1.57 | +0.006 / +0.009 | +0.0202 ± 0.0034 |

Real 2025, safe policy:

| Model | Departs | DR lift | Calibration slope |
|---|---|---|---|
| gbm_pooled | 7.5% | −0.001 ± 0.005 | 0.32 ± 0.59 |
| dragonnet_factored | 19.4% | −0.012 ± 0.008 | 0.69 ± 0.63 |
| tarnet_factored | 48.2% | −0.023 ± 0.013 | −0.03 ± 0.43 |
| r_learner_forest | 7.4% | +0.001 ± 0.006 | −0.12 ± 0.29 |
| dr_learner_gbm | 10.8% | +0.012 ± 0.009 | −0.06 ± 0.26 |
| dr_learner_ridge | 18.8% | −0.001 ± 0.009 | −0.54 ± 0.28 |
| tarnet | 2.0% | +0.002 ± 0.003 | −0.25 ± 0.17 |
| dragonnet | 4.0% | −0.000 ± 0.003 | −0.17 ± 0.12 |

Full tables, including fit times, are in `results/`. The network fit times there
predate the `SnapBatches` speed-up and were measured with three runs sharing the
CPU.

## Why they landed where they did

- **The contrast learners treat the 12 calls as unrelated categories.** A rare
  call cannot borrow from calls that share its package or its box, which the
  pooled design gives for free. The DR-learner's inverse weights also inflate the
  spread 1.2–1.5×. The causal forest is accurate on synthetic data but worse on
  real data, and about 45× slower to fit than the GBM.
- **Separate heads repeat the per-action optimizer's curse.** The spread runs
  1.4–1.9× the truth (the same failure as one GBM per call, found earlier). The
  factored head is required.
- **The factored networks' bootstrap spread is too tight.** In the safe policy
  they depart far more often than the GBM and lose on 2025.
- **The DR-learner's +0.012 safe lift is not evidence for it.** Its DR loss is
  clearly worse, and on the league with the read, DR overstates every policy.
- **Off-support extrapolation.** Where coordinators almost never make a call, the
  factored networks extrapolate in the wrong direction (correlation with the
  truth about −0.2 to −0.3). The policy never chooses those calls.
- **The real call effects sit near the noise floor.** Calibration slopes on 2025
  are weak for every model.
- **The networks were not tuned.** One default architecture was used throughout.
  That is the obvious lever still untried.

## What broke along the way

- **OpenMP deadlock.** torch and LightGBM each bundle their own libomp on macOS.
  Once torch had run a multi-threaded op, the next LightGBM fit hung.
  `neural.py` pins torch to one thread at import, which costs about 30% per epoch
  at this size.
- **MPS crash.** Torch 2.15's MPS backend crashes on Darwin 23.2, so training
  runs on the CPU.
- **DataLoader overhead.** `DataLoader(TensorDataset)` indexed rows one at a time
  and dominated the epoch time. `SnapBatches` slices tensors directly and replays
  the loader's generator draws (a per-epoch base seed, then the permutation, then
  a discarded remainder permutation). Its batches are bit-identical and 3–10×
  faster.
- **econml category order.** econml needs sorted treatment categories, so the
  reference call is recoded to 0.

## Rerun

From the repository root:

```bash
python -m experiments.outcome_models.benchmark --source nflfastR --n-boot 10                   # results/real_2025.csv
python -m experiments.outcome_models.benchmark --source synthetic --n-boot 10                  # results/synthetic_with_read.csv
python -m experiments.outcome_models.benchmark --source synthetic --conf-strength 0 --n-boot 10  # results/synthetic_no_read.csv
python -m experiments.outcome_models.benchmark --source synthetic --n-plays 12000 --n-boot 2 \
    --models gbm_pooled dragonnet_factored --out /tmp/smoke.csv                                # quick check
pytest experiments/outcome_models
```

The full runs took about 3 hours each when run three at once, mostly in the
bootstrap ensembles of the econml learners. Each run rewrites its CSV after every
candidate.

## Files

- `benchmark.py`: the candidates, the criteria, and the CLI.
- `cate.py`: the DR- and R-learners. They are experiment-only; `fit_q_model`
  rejects them and points here.
- `results/`: the three result tables.
- `test_benchmark.py`: the contrast learners and the criteria (run with
  `pytest experiments`).

The networks and their tests live in the pipeline: `src/models/outcome/neural.py`
and `tests/test_neural_q.py`.
