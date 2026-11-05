# DragonNet tuning: do better hyperparameters make it the Q?

**Question.** The outcome-model benchmark (`../outcome_models/`) ran the factored
DragonNet with one untuned default architecture. It tied the pooled GBM on the
real holdout's doubly-robust loss. Does tuning it pull it ahead?

**Verdict.**
- **No. The shipped defaults stay.** The search's winner beat the default on the
  tuning season (2024) by about 1.2 SE. On 2025 the gain was gone
  (−0.00015 ± 0.00037). On the synthetic league with no read it was clearly
  worse: true advantage error 0.0259 against the default's 0.0214.
- **On real data the judge rewards flat advantages.** The real call effects sit
  near the noise floor, so the DR loss on one season is lowest for a Q that
  says the calls barely differ. Across the 48 configs, the advantage spread and
  the loss on 2024 have a rank correlation of 0.77.
- **The winner "fixed" the safe policy by turning it off.** Its advantages are
  so flat that the policy departs from the coordinator on 0.1% of 2025 snaps
  (+0.001 ± 0.001). Where real gains exist (the synthetic league), it captures
  about half of what the GBM does.
- **What remains for Phase 4 is uncertainty, not architecture.** The DragonNet
  needs its own uncertainty estimate (a deep ensemble or MC dropout) for the
  safe policy.
- **Two knobs were added to `neural.py` and kept,** both off by default:
  `embed_max` (the widest embedding) and `early_stop: outcome` (stop on Q's
  error alone, not the whole DragonNet loss).

## Method

**Training seasons choose; the holdout only confirms.**

| Stage | Fit → score | What |
|---|---|---|
| `search` | 2022–23 → 2024 | The default plus 47 random draws from the space below, 3 seeds each |
| `refine` | 2022–23 → 2024 | The search's top 6 and the default, rerun with 5 fresh seeds. The lowest mean loss is chosen (`results/chosen.json`) |
| `confirm` | 2022–24 → 2025 | Chosen vs default vs GBM, 5 seeds. Then the outcome benchmark's harness on all three, safe policy included (10 game-bootstrap refits) |

**Score.** The outcome benchmark's real-data judge:
- the mean squared gap between each call's estimated advantage over the
  coordinator's mix and a doubly robust pseudo-outcome;
- the pseudo-outcomes come from the scored season alone, with nuisances
  cross-fitted by game there;
- only supported calls count (π_b ≥ 0.05);
- reported relative to the pooled GBM fit on the same seasons, so lower is
  better and the GBM is 0.

**Checking the judge.** The search also ran on the synthetic league without the
read (140k plays, 32 teams), where the true advantages are known. This tests
whether the judge ranks configs by their true error at the tuning fold's size.

**Space.**

| Knob | Values | Default |
|---|---|---|
| `hidden` | [64,32] · [128,64] · [256,128] · [128,128,64] · [256,256,128] | [128,64] |
| `head_hidden` | 16 · 32 · 64 | 32 |
| `dropout` | 0 · 0.1 · 0.2 · 0.3 | 0.1 |
| `lr` | 3e-4 · 1e-3 · 3e-3 | 1e-3 |
| `weight_decay` | 1e-5 · 1e-4 · 1e-3 · 1e-2 | 1e-4 |
| `batch_size` | 256 · 512 · 1024 · 2048 · 4096 | 1024 |
| `alpha` (propensity head) | 0.1 · 0.3 · 1 · 3 | 1 |
| `beta` (targeted regularization) | 0 · 0.3 · 1 · 3 | 1 |
| `embed_max` | 4 · 8 · 16 | 8 |
| `early_stop` | loss · outcome | loss |

Sampled configs get 100 epochs; the default keeps its 60. Early stopping
(patience 5, on held-out games) ends nearly every run long before either cap.

## Results

**Search (2024).**
- All 48 DragonNets beat the GBM on DR loss: −0.0009 to −0.0000, against
  −0.00055 for the default.
- The noise is as large as the gaps. The seed sd of a config's loss is about
  0.0002, and the paired SE against the default about 0.0003.
- The winners share a profile: lr 3e-3, stopping around epoch 10, and an
  advantage spread of 0.007–0.018 against the default's 0.027.

**The judge on synthetic data.**
- Its ranking agrees with the true error: Spearman 0.70 across the 48 configs.
  Its pick ranks 4th by truth.
- The real and synthetic rankings are unrelated (Spearman −0.07). The synthetic
  league's call effects are stronger, so the best configs differ between them.
- By true error, the shipped default ranks 3rd of 48 and the chosen config 46th.

**Refine (2024, fresh seeds).** Loss relative to the GBM.

| Config | DR loss | vs default | Spread | Calibration slope |
|---|---|---|---|---|
| default (0) | −0.00022 | — | 0.030 | 0.19 |
| **42 (chosen)** | **−0.00083** | −0.00061 ± 0.00049 | 0.011 | 0.71 |
| 15 | −0.00078 | −0.00057 ± 0.00037 | 0.017 | 0.59 |
| 4 | −0.00072 | −0.00050 ± 0.00041 | 0.015 | 0.31 |
| 5 | −0.00071 | −0.00050 ± 0.00033 | 0.023 | 0.53 |

With fresh seeds, the default's loss moved from −0.00055 to −0.00022.

Config 42 settings:
- `hidden` [256, 256, 128], `head_hidden` 32;
- `dropout` 0.3, `weight_decay` 1e-5;
- `lr` 3e-3, `batch_size` 256;
- `alpha` 1, `beta` 0.3;
- `embed_max` 8, `early_stop: outcome`.

**Confirm (2025, 5 seeds; synthetic truth).**

| | Real DR loss vs GBM | vs default | Synthetic error, no read | Synthetic error, with read |
|---|---|---|---|---|
| GBM | 0 | | 0.0216 | **0.0247** |
| default | +0.00026 ± 0.00064 | — | **0.0214** | 0.0279 |
| tuned (42) | +0.00012 ± 0.00059 | −0.00015 ± 0.00037 | 0.0259 | 0.0269 |

On 2025 the tuned net's advantages are flat (sd 0.008) and uncalibrated
(slope −0.63).

**Safe policy (single seed, 10 game-bootstrap refits).**

| | Real 2025: departs / DR lift | Synthetic true lift, no read / with read |
|---|---|---|
| GBM | 7.5% / −0.001 ± 0.005 | **+0.026** / **+0.025** |
| default DragonNet | 19.4% / −0.012 ± 0.008 | +0.020 / **+0.025** |
| tuned DragonNet | 0.1% / +0.001 ± 0.001 | +0.012 / +0.020 |

## What this says about tuning on this problem

- **One season's DR loss cannot pick hyperparameters when the effects are this
  small.** At this signal level the loss mostly measures how much a model
  shrinks, and the best amount of shrinkage changes between seasons.
- **A selector that does work would need more than one season of judging.**
  Two options:
  - rolling folds (2022→2023, 2022–23→2024) with the loss pooled across them;
  - a criterion that penalizes lost signal, such as requiring a calibration
    slope near 1.
- **The default's place in the benchmark was not luck.** It ranks 3rd of 48 on
  the synthetic truth. Hyperparameters are not what keeps the DragonNet behind
  the GBM in the safe policy.

## What broke along the way

- **Seed noise.** A single seed made the search's ordering unreliable: the
  default alone moved by 0.0003 between seed sets, as much as the gaps being
  ranked. Hence 3 seeds to search, 5 to refine and to confirm.
- **Parallelism.** torch is pinned to one thread (see `neural.py`), so configs
  run in a spawn-started process pool, one network per process. Fork would
  share LightGBM's OpenMP runtime with the workers.

## Rerun

From the repository root:

```bash
python -m experiments.dragonnet_tuning.tune search  --source nflfastR --workers 4                   # ~18 min
python -m experiments.dragonnet_tuning.tune search  --source synthetic --conf-strength 0 --workers 4
python -m experiments.dragonnet_tuning.tune check   --source synthetic --conf-strength 0            # the judge vs truth
python -m experiments.dragonnet_tuning.tune refine  --source nflfastR --workers 7                   # ~4 min
python -m experiments.dragonnet_tuning.tune confirm --source nflfastR --workers 2                   # ~25 min
python -m experiments.dragonnet_tuning.tune confirm --source synthetic [--conf-strength 0] --workers 2
pytest experiments/dragonnet_tuning
```

Each CSV is rewritten as each config finishes.

## Files

- `tune.py`: the space, the judge, the worker pool, and the four stages.
- `results/`:
  - `search_*.csv`, `refine_real.csv`, `chosen.json`: the tuning-fold stages;
  - `confirm_*.csv`: multi-seed DR loss and truth on the holdout;
  - `policy_*.csv`: the outcome benchmark's rows for the GBM, the default and
    the tuned net.
- `test_tune.py`: covers three things:
  - the search's config 0 is exactly what `fit_neural_q_model` does with no
    `neural` block;
  - sampling is reproducible;
  - the judge's zero is the GBM.
