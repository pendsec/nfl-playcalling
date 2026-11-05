# Experiments

Trials of alternatives to the pipeline's choices, kept so a question is not
asked twice. Each one has a verdict. What was adopted lives in `src/`; what was
not stays here or is described below.

## Conventions

- **One folder per experiment.** Each holds:
  - a `README.md`: the question, method, results, verdict, what broke, and how
    to rerun;
  - the code;
  - `results/` (tracked in git, unlike `outputs/`);
  - tests, which run with `pytest experiments`.
- **Experiments import `src`; `src` never imports `experiments`.** A piece that
  is adopted moves into `src/` with its tests.
- **Judge every alternative by what the pipeline uses,** on the held-out season,
  and beside a synthetic league whose answer is known. Use the same rule as
  `CLAUDE.md`: training seasons choose, the holdout only confirms.

## With code

| Experiment | Question | Verdict |
|---|---|---|
| [`outcome_models/`](outcome_models/) | Do causal-contrast learners (econml DR-learner, causal-forest R-learner) or PyTorch TARNet / DragonNet estimate the call advantages better than the pooled GBM? | **Pooled GBM stays.** Factored DragonNet ties it on real 2025 DR loss and is the Phase 4 neural candidate, but needs its own uncertainty before it can be gated. Separate per-call heads and the contrast learners lose. |
| [`dragonnet_tuning/`](dragonnet_tuning/) | Does tuning the factored DragonNet (48-config random search, 2022–23 → 2024, refined on fresh seeds) make it beat the GBM? | **Defaults stay.** The winner's 2024 gain vanished on 2025 (−0.00015 ± 0.00037 vs default) and it was worse on the synthetic truth (0.0259 vs 0.0214). One season's DR loss rewards flat advantages when the call effects are this small. The default ranks 3rd of 48 by true error. |

## Earlier trials (no saved code)

These ran as one-off scripts while the pipeline was being built. Their numbers
and reasoning are recorded in `CLAUDE.md` (V3 findings), and each can be
reproduced through the config switch named.

| Trial | Result | Kept | Switch |
|---|---|---|---|
| One Q regressor per alignment vs one pooled over (state, package, box) | Per-action spread 2.1× the truth; pooled 0.8×. Captured about 77% of the available gain vs about 45%. | pooled | `outcome_model.structure` |
| Raw argmax-Q vs argmax restricted to π_b ≥ 0.05 | Raw argmax loses to the coordinator in truth; the restricted one beats it. | support floor | `policy.support_threshold` |
| Deterministic conservative policy vs safe (mix-anchored) policy | The argmax collapsed to the modal call and lost about 0.01 in DR. | safe | `policy.type` |
| Safe-policy bound κ ∈ {0, 0.5, 1, 1.645, 2.326}, tuned on 2022-23 → 2024 | Best DR there at 2.326 (2024 +0.0085). 2025: −0.008; pooled over both, −0.000 ± 0.010. | 2.326 | `policy.lcb_kappa` |
| Season pessimism (worst season's contrast) | Per-season contrasts stable within noise (Cochran's Q p = 0.64). | rejected | — |
| Read allowance scale ∈ {0, 1, 2, 3} at κ = 2.326 | 2 best on 2024. 2025: −0.003 vs −0.008 without; departs 6% vs 11%. | 2 | `policy.read_allowance.scale` |
| Call = package only vs package × box | Package effects stable and free of the read; the box carries the read. | box kept, guarded | `action.level1.axes` |
| Level-2 κ tuned on DR | Picked κ = 0 (departs 95%). DR far above DM, the read's signature. | rejected; κ fixed at 2.326 | `policy.level2.lcb_kappa` |
| Coverage: six charted shells vs safety-structure families | Full labels drift with the charting. `full` departs 3× as often (82% to Cover-1) for no DR gain. | family | `action.level2.shells` |
| Offense quality half-life: 1 game, 250 plays, unweighted | Next-game EPA correlation 0.262 / 0.319 / 0.284. | 250 plays | `features.offense_quality_halflife_plays` |
| Invariance statistic: per-team-season DR contrasts, chi² | DR SEs about 2× too small; clustered SEs still 1.5–2× too small. | residual contrasts + game-permutation null | `src/evaluation/invariance.py` |
| Leave-one-team-out: defer to the fold's π_b vs the team's own mix | −0.013 ± 0.003 vs −0.002 ± 0.002: the loss is the unknown mix, not the departures. | both reported | `lift_anchored` in `transfer.csv` |
| Shell imputer calibration: cross-fitted temperature scaling | Confidence ECE 0.032 → 0.020; classwise ECE barely moves. | open (Dirichlet / vector scaling next) | `src/models/imputation/` |
