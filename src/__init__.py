"""
Causal RL for defensive playcalling — source package.

Each subpackage is one concern from the eight-step architecture in CLAUDE.md:

    data/       Step 1    ingestion + feature engineering: raw pbp -> (S, A, R)
    scm/        Step 2    hand-built causal graph disciplining what to condition on
    models/     Step 3+4  behavior policy pi_b(A|S) and outcome / Q-model Q(S,A)
    ope/        Step 5    off-policy evaluation (Direct Method in V1)
    policy/     Step 6    policy learning (behavior-constrained greedy in V1)
    evaluation/           holdout validation and synthetic-SCM regression checks
    schemas/              the @dataclass shapes passed between stages (Dataset,
                          BehaviorModel, QModel, OPEResult, GreedyPolicy)

V1 is a deliberately simple, end-to-end "walking skeleton"; the phased plan for
deepening each component (V2..V4) lives in CLAUDE.md. `run_v1.py` at the repo
root wires these packages together into a single runnable pipeline.
"""
