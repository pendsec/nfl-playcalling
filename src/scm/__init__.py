"""Causal graph (Step 2): the hand-built structural-causal-model skeleton.

    graph  — the DAG over V1 variables: treatment (defensive call), outcome
             (EPA), the offensive-playcall *mediator*, and the unobserved
             `coach_read` *confounder*. Exposes the backdoor adjustment set that
             every downstream causal estimate (Q-model, OPE) is required to obey.
"""
