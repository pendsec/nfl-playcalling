"""Off-policy evaluation (Step 5): value any candidate policy from logged data.

    direct_method  — V_DM(pi) = E_s [ sum_a pi(a|s) Q(s,a) ], plus the mandatory
                     behavior-recovery smoke test (OPE must reproduce pi_b's own
                     mean reward before any off-policy estimate is trusted).
    doubly_robust  — V_DR(pi): the direct method plus an importance-weighted
                     correction on the taken action, self-normalized and weight-
                     clipped. Consistent if EITHER Q or pi_b is right, and the
                     estimator every V2 policy claim is judged by.
    aipw           — AIPW contrasts of fixed actions against a baseline call.
                     Interpretability for the causal story, never a policy judge.
    sensitivity    — marginal-sensitivity-model bounds on V_DR as unobserved
                     confounding of odds-ratio Gamma is allowed to tilt the
                     propensities. This is what qualifies every headline number.

A standalone IPW estimator is deliberately absent at V2 (see CLAUDE.md Step 5):
DR's correction term already does the reweighting, and at this slice's ESS ratio
a pure-IPW figure would be dominated by a few plays on the propensity floor
without changing any decision. It returns in V3 with multi-team pooling, where
the full DM -> IPW -> DR ladder becomes worth showing side by side.
"""
