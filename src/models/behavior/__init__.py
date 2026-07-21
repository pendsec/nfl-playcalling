"""Behavior policy pi_b(A|S) (Step 3): models which call the DC actually made.

This propensity model is the engine of IPW / doubly-robust reweighting in later
versions. In V1 it earns its place two ways: it reports calibration (miscalibrated
propensities silently break inverse-weighting) and it supplies the positivity
constraint that keeps the learned policy from recommending unsupported calls.
"""
