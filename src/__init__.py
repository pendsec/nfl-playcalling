"""
Causal RL for defensive playcalling — source package.

Each subpackage is one concern from the eight-step architecture in CLAUDE.md:

    data/       Step 1    ingestion + feature engineering: raw pbp -> (S, A, R)
    scm/        Step 2    hand-built causal graph disciplining what to condition on
    models/     Step 3+4  behavior policy pi_b(A|S), outcome / Q-model Q(S,A),
                          shared calibration, and latent-treatment imputation
    ope/        Step 5    off-policy evaluation (DM, IPW, doubly-robust, AIPW
                          contrasts, sensitivity)
    policy/     Step 6    policy learning (greedy, conservative, safe, and the
                          factored alignment x coverage policy)
    evaluation/           regression gates, invariance across team-seasons, and
                          leave-one-team/season-out transfer
    schemas/              the dataclasses passed between stages (Dataset,
                          BehaviorModel, QModel, OPEResult, the policies)
    folds, stats          game-grouped fold assignment; means with (clustered)
                          standard errors and the gates' non-inferiority rule

The phased plan for deepening each component lives in CLAUDE.md. `run.py` at the
repo root wires these packages together into a single runnable pipeline.
"""
