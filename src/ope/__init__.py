"""Off-policy evaluation (Step 5): value any candidate policy from logged data.

    direct_method  — V_DM(pi) = E_s [ sum_a pi(a|s) Q(s,a) ], plus the mandatory
                     behavior-recovery smoke test (OPE must reproduce pi_b's own
                     mean reward before any off-policy estimate is trusted).

Biased when Q is misspecified; V2 upgrades this to IPW / doubly-robust estimation
with sensitivity bounds for the unobserved `coach_read` confounder.
"""
