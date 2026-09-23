"""Causal graph (Step 2): the hand-built structural-causal-model skeleton.

    graph     — the DAG over project variables: treatment (defensive call),
                outcome (EPA), the offensive-playcall *mediator*, the unobserved
                `coach_read` *confounder*, the `coverage_charted` node the sample
                is *selected on*, and `def_front`, a sibling of the treatment
                that is carried but never adjusted for. Exposes the backdoor
                adjustment set every downstream causal estimate must obey, plus a
                content fingerprint stamped onto each estimate.
    identify  — DoWhy backdoor confirmation, the per-stratum positivity check,
                and the guards that reject a mediator, a selection variable, or
                the defense's own front turning up among the state features.
"""
