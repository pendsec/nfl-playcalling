"""
Causal RL for defensive playcalling — source package.

Each subpackage is one concern from the eight-step architecture in CLAUDE.md:

    data/       Step 1    ingestion + feature engineering: raw pbp -> (S, A, R)
    scm/        Step 2    hand-built causal graph disciplining what to condition on
    models/     Step 3+4  behavior policy pi_b(A|S), outcome / Q-model Q(S,A),
                          shared calibration, and latent-treatment imputation
    ope/        Step 5    off-policy evaluation (DM, doubly-robust, sensitivity)
    policy/     Step 6    policy learning (greedy and behavior-regularized)
    evaluation/           the paired-DR policy regression gate
    schemas/              the @dataclass shapes passed between stages (Dataset,
                          BehaviorModel, QModel, OPEResult, GreedyPolicy)

The phased plan for deepening each component lives in CLAUDE.md. `run.py` at the
repo root wires these packages together into a single runnable pipeline.
"""
