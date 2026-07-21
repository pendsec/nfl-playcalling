"""Models: the two estimators the causal pipeline is built on.

    behavior/   pi_b(A|S) — what call the DC actually made (Step 3)
    outcome/    Q(S, A)   — interventional reward of a call (Step 4)
    preprocess  — the shared state transformer (scale numeric, one-hot
                  categorical) that BOTH models fit on, so they see identical
                  features.
"""
