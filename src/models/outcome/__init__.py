"""Outcome / Q-model (Step 4): Q(S, A) = E[R | S, do(A)].

Estimates the *interventional* reward of a defensive call — what happens if we
intervene and make call A, not merely what was observed when the DC chose it.
Conditions only on the SCM's adjustment set (pre-snap state) and deliberately
excludes the offensive-playcall mediator, so it captures the call's total effect.
"""
