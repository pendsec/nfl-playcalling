"""Models: the estimators the causal pipeline is built on.

    behavior/    pi_b(A|S) — what call the DC actually made (Step 3)
    outcome/     Q(S, A)   — interventional reward of a call (Step 4)
    imputation/  the coverage shell as a LATENT treatment label, imputed with
                 its uncertainty carried forward rather than dropped
    preprocess   — the shared state transformer (scale numeric, one-hot
                 categorical) that both estimators fit on, so they see identical
                 features.
    calibration  — calibration metrics and temperature scaling, shared by pi_b
                 and the imputer because both produce probabilities that are
                 CONSUMED as probabilities (inverted into weights, sampled from)
                 rather than read as labels.
"""
