"""Off-policy evaluation (Step 5): value any candidate policy from logged data.

    direct_method  — V_DM(pi) = E_s [ sum_a pi(a|s) Q(s,a) ], plus the mandatory
                     behavior-recovery smoke test (OPE must reproduce pi_b's own
                     mean reward before any off-policy estimate is trusted).
    ipw            — V_IPW / V_SNIPW: the logged rewards reweighted by
                     pi / pi_b, no outcome model; also the clipped importance
                     weights every estimator shares and their diagnostics.
    doubly_robust  — V_DR(pi): the direct method plus an importance-weighted
                     correction on the taken action, self-normalized and weight-
                     clipped. Consistent if EITHER Q or pi_b is right, and the
                     estimator every policy claim is judged by.
    aipw           — AIPW contrasts of fixed actions against a baseline call
                     (the DR score of "always call d"). Interpretability for the
                     causal story, never a policy judge.
    ladder         — DM, IPW, SNIPW and DR of one policy side by side.
    sensitivity    — marginal-sensitivity-model bounds on V_DR as unobserved
                     confounding of odds-ratio Gamma is allowed to tilt the
                     propensities. This is what qualifies every headline number.

Every estimate carries the version of the causal graph its models were fit
under (`graph_version`, required).
"""
