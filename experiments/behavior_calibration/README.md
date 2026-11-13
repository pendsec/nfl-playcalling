# Behavior-model calibration: which π_b gives weights IPW and DR can trust?

**Question.** π_b is the shipped LightGBM classifier with sklearn's one-vs-rest
isotonic calibration. That calibrates each call's column alone, then
renormalizes the row. Does a joint multiclass calibrator do better: temperature,
vector scaling, or Dirichlet calibration (Kull et al. 2019)? What about a
factored model, P(package | s) × P(box | s, package)? "Better" means better for
what IPW and DR consume: every call's probability on a held-out season.

**Verdict.**
- **Dirichlet calibration is not adopted for π_b.** On the synthetic league,
  where the miscalibration is real calibration error, it cuts the KL divergence
  from the true propensity by 64%. On real data it does not help: log loss
  −0.003 ± 0.003 on 2024 and +0.003 ± 0.002 on 2025. It now exists in
  `src/models/calibration.py` (`MatrixScaler`) behind
  `behavior_model.calibration: dirichlet`. Its next use is the shell imputer
  (Phase 6), where classwise calibration is the open gap.
- **Calibration folds matter more than the calibration method.**
  - Folds by snap put a game's other snaps in the fold's training data, so the
    out-of-fold probabilities look better than they will on new games. A
    calibrator fit to them learns the wrong map.
  - Dirichlet with snap folds is the worst candidate on real data (+0.030 /
    +0.023); with game folds it ties isotonic.
  - Isotonic with game folds (`calibration_groups: game`) is the best
    calibrator on both real folds (−0.016 ± 0.002, −0.006 ± 0.002) and neutral
    on synthetic. **Adopted**: it is now the default in `configs/v3.yaml`.
- **The real miss is drift, not calibration.**
  - Within the training seasons, out-of-fold weight balance is close to 1 for
    most calls (within ±0.06; the two rarest dime calls are off by 0.2–0.3 on
    few snaps).
  - Applied to the next season, every candidate misses the same calls by the
    same amount: nickel/8+ balances at 0.33, base/6 at 1.2–1.7. The call mix
    moves between seasons (nickel/8+ 2.1% → 4.2% → 1.5% → 1.0% from 2022 to
    2025; base/7 jumps to 16.3% in 2025), and no calibrator fit on past seasons
    can learn next season's mix.
- **The fix is the judge's π_b, not the policy's.** Fit the judge's π_b on the
  training seasons plus the scored season's other games, with the season as a
  feature, out of fold by game (`pooled_crossfit`):
  - log loss −0.108 ± 0.005 on 2025, against −0.006 for the best calibrator;
  - nickel/8+ balance 0.92 (was 0.33);
  - no cost where nothing drifts: −0.005 ± 0.001 on synthetic.
  
  It uses only the scored season's states and calls, never its rewards. The
  policy stays train-only.
- **The policy verdicts survive the better judge.** Safe-policy DR lift:
  - 2024 (the fold κ and the read allowance were tuned on): +0.0095 ± 0.008,
    against +0.0104 with the train-fit judge;
  - 2025: −0.0069 ± 0.006, against −0.0034.
  
- **Adopted.** The pipeline's holdout judge is now this model
  (`ope.judge_propensity: pooled_crossfit`, `src/models/behavior/judge.py`),
  used for every holdout weight in `run.py` and the notebook. With it and game
  calibration folds, the full pipeline on 2025:
  - judge log loss −0.100 ± 0.009 against the policy's π_b;
  - safe-policy DR lift −0.0078 ± 0.0114 (95%), or −0.0041 ± 0.0104 with the
    policy's deferrals anchored to the judge's mix;
  - level-2 coverage judge: log loss −0.024 ± 0.004, worst weight balance
    0.54 → 0.93;
  - both regression gates pass.

## Candidates

| Name | π_b | Where |
|---|---|---|
| `raw` | LightGBM, no calibration | `calibration: none` |
| `isotonic` | One-vs-rest isotonic over 3 snap-stratified folds | `calibration_groups: play` |
| `isotonic_game` | The same, folds by game (now the default) | `calibration_groups: game` |
| `temperature` | One temperature on out-of-fold log-probabilities | `calibration: temperature` |
| `vector` | Per-class temperature and shift, 2K parameters | `calibration: vector` |
| `dirichlet` | softmax(W log p + b), K²+K parameters, ODIR-regularized (off-diagonal W and b shrunk; λ by inner CV over 1e-4…1) | `calibration: dirichlet` |
| `dirichlet_game` | The same, folds by game | plus `calibration_groups: game` |
| `factored_dirichlet` | P(package \| s) × P(box \| s, package), each part Dirichlet over game folds | `compare.py` only |
| `scored_crossfit` | The shipped π_b fit on the scored season only, 5 game folds | `compare.py` only (a judge) |
| `pooled_crossfit` | The same, each fold also fit on the training seasons, with `season` in the state | a judge; the pipeline's is `ope.judge_propensity: pooled_crossfit` |

The four matrix-type calibrators (`temperature`, `vector`, `dirichlet`,
`dirichlet_game`) are fit on the base model's out-of-fold predictions, then
applied to the model refit on every row (`CrossFitCalibratedClassifier` in
`propensity.py`).

## Method

Every candidate is scored on the season it did not train on. The scores use the
clipped, renormalized propensities the pipeline uses. Folds:

| Fold | Fit → score | Use |
|---|---|---|
| `--fold tune` | 2022–23 → 2024 | where a choice is made |
| `--fold holdout` | 2022–24 → 2025 | confirmation only |

Criteria:
- **Log loss.** The primary score: a proper scoring rule. Reported with a
  game-clustered paired SE against the shipped isotonic.
- **Brier, confidence ECE and classwise ECE.** Classwise is the one that
  matters for weights.
- **`citl_max`.** The worst per-call gap between the mean predicted
  probability and the observed share.
- **Weight balance.** Per call, the sum of 1{A=a}/π(a|s) over the snaps where
  the call is supported (π ≥ 0.05), divided by the number of those snaps. It is
  1 in expectation for the true propensity, and it is the check closest to what
  IPW relies on. It is restricted to supported calls because unrestricted, the
  synthetic coordinator's near-zero propensities make it useless as a check.
- **ESS ratio** of the inverse weights.
- **Synthetic only.** These use `src.data.load.true_alignment_propensity`: the
  coordinator's softmax averaged over his hidden read by Gauss–Hermite
  quadrature, which is the best any model of the observed state can do.
  - KL(true ‖ estimated) and RMSE on the supported cells;
  - the IPW and DR error of every always-call-a policy against
    `true_alignment_q`. This is reported but not used: it is dominated by the
    weight clip on off-support snaps, and the true propensity scores 0.065 on
    it too.
- **OPE.** The shipped safe policy and the supported greedy GBM argmax are
  built once per fold, on the training seasons with the shipped π_b. Each
  candidate is then only the judge's π_b. Reported: their paired DR lift over
  the coordinator, and on synthetic the error against the truth.

## Results

Log loss is relative to the shipped isotonic (lower is better). Balance is
nickel/8+, the call that drifts most.

| | Real 2024: log loss | Real 2025: log loss | Real 2025: classwise ECE | Real 2025: balance | Synthetic: log loss | Synthetic: KL to truth |
|---|---|---|---|---|---|---|
| true propensity | | | | | −0.042 | 0 |
| isotonic (shipped) | 0 | 0 | 0.019 | 0.33 | 0 | 0.039 |
| raw | +0.006 ± 0.002 | +0.002 ± 0.002 | 0.019 | 0.34 | −0.006 | 0.035 |
| isotonic_game | −0.016 ± 0.002 | −0.006 ± 0.002 | 0.019 | 0.35 | +0.001 | 0.040 |
| temperature | +0.010 ± 0.002 | +0.004 ± 0.002 | 0.018 | 0.34 | −0.005 | 0.035 |
| vector | +0.024 ± 0.003 | +0.020 ± 0.002 | 0.021 | 0.33 | −0.005 | 0.035 |
| dirichlet | +0.030 ± 0.004 | +0.023 ± 0.003 | 0.022 | 0.33 | **−0.027** | **0.014** |
| dirichlet_game | −0.003 ± 0.003 | +0.003 ± 0.002 | 0.021 | 0.33 | **−0.027** | **0.014** |
| factored_dirichlet | −0.004 ± 0.003 | +0.005 ± 0.003 | 0.021 | 0.37 | −0.023 | 0.018 |
| scored_crossfit | −0.131 ± 0.010 | −0.106 ± 0.008 | 0.009 | 1.00 | +0.028 | 0.066 |
| **pooled_crossfit** | **−0.135 ± 0.008** | **−0.108 ± 0.005** | **0.008** | 0.92 | −0.005 ± 0.001 | 0.035 |

Two notes on the table:
- The worst per-call share gap on 2025 is 0.04–0.06 for every calibrator
  (0.044 for the shipped one), against 0.009 for `scored_crossfit` and 0.015
  for `pooled_crossfit`.
- Weight balance with `pooled_crossfit` is within 0.12 of 1 on 2025 for every
  call except the two rarest dime calls (dime/7 0.47, dime/8+ with 14 taken
  snaps 0).

**Does the judge change the policy verdict?** Safe-policy DR lift over the
coordinator:

| Judge's π_b | 2024 (tuning fold) | 2025 |
|---|---|---|
| isotonic, train-fit (shipped) | +0.0104 ± 0.0063 | −0.0034 ± 0.0052 |
| isotonic_game | +0.0086 ± 0.0059 | −0.0008 ± 0.0048 |
| dirichlet_game | +0.0034 ± 0.0051 | +0.0004 ± 0.0052 |
| scored_crossfit | −0.0017 ± 0.0092 | −0.0147 ± 0.0087 |
| pooled_crossfit | +0.0095 ± 0.0084 | −0.0069 ± 0.0060 |

On the synthetic league without the read, the safe policy's DR error against
the truth is 0.003–0.005 under every train-fit candidate. It is 0.006 under the
true propensity and 0.006 under `pooled_crossfit`, and 0.010 under the
small-sample `scored_crossfit`. Calibration is not what limits OPE accuracy
there.

**Fit time** (one fit on 2022–24, idle machine):
- isotonic: about 25 s;
- `dirichlet_game`: 100–160 s (the inner CV for λ is 15 L-BFGS fits);
- the pooled judge: about 140 s (5 folds).

## Why

- **Dirichlet fixes what it can see.** The synthetic coordinator's call
  probabilities are a smooth softmax over how heavy each look is, and
  LightGBM's errors there run across calls in a structured way. That is
  exactly the cross-class correction a full W can make and a one-vs-rest
  isotonic cannot.
- **On real data the dominant error is the mix moving between seasons.** That
  is out of reach of anything fit on earlier seasons. The 156 parameters fit
  to the training seasons' errors carry their drift into the scored season.
- **Snap-level folds hand the calibrator an optimistic picture.** Snaps in the
  same game share the script, the opponent and the score, so a fold model has
  effectively seen its test snaps' games. On data with no game structure
  (synthetic) the two schemes agree.
- **The scored season's own games are the only place its mix can be learned.**
  Alone, one season is too little data: synthetic KL 0.066, and the 2024 lift
  swings to −0.002. Pooled with the training seasons and told the season, the
  model learns the shared state structure from every season and the mix from
  the scored one.

## What this changes (not yet adopted)

1. **The judge.** `run.py` scores the holdout with the train-fit π_b
   (`pib_te = beh.propensity(test.df)`). Replacing it with the pooled cross-fit
   π_b changes every holdout IPW/DR estimate and its provenance. The safe
   policy's 2025 lift moves within its SE (−0.003 → −0.007). The experiments'
   judges (`outcome_models`, `dragonnet_tuning`) use the same train-fit π_b
   and would follow.
2. **`calibration_groups: game`** for the policy's own π_b: small, consistent
   gains on real data, neutral on synthetic.

## Rerun

From the repository root:

```bash
python -m experiments.behavior_calibration.compare --source nflfastR --fold tune
python -m experiments.behavior_calibration.compare --source nflfastR --fold holdout
python -m experiments.behavior_calibration.compare --source synthetic --conf-strength 0 --fold holdout
python -m experiments.behavior_calibration.compare --source synthetic --fold holdout
# the scored-season judges, as run here (subset + its own file):
python -m experiments.behavior_calibration.compare --source nflfastR --fold holdout \
    --models isotonic pooled_crossfit --out experiments/behavior_calibration/results/holdout_real_pooled.csv
pytest experiments/behavior_calibration
```

Run them one at a time. Four LightGBM processes at once oversubscribe the CPU
badly: fits slowed 10–100×, which is why `fit_s` in the four full-candidate
CSVs is not a timing.

## Files

- `compare.py`: the candidates, the factored model, the scored-season judges,
  the criteria, and the CLI.
- `results/`:
  - `{tune,holdout}_{real,synthetic_*}.csv`: every calibrator;
  - `*_scored.csv`: adds `scored_crossfit`, with honest fit times;
  - `*_pooled.csv`: adds `pooled_crossfit`.
- `test_compare.py`: the factored model, the judges, and the scoring
  (`pytest experiments`).

The calibrators and their tests live in the pipeline:
- `src/models/calibration.py` (`MatrixScaler`, `weight_balance`,
  `calibration_in_the_large`) and `tests/test_calibration.py`;
- `src/models/behavior/propensity.py` (`behavior_model.calibration`,
  `calibration_groups`) and `tests/test_models.py`;
- `src/data/load.py` (`true_alignment_propensity`).
