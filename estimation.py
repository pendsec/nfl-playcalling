"""
NFL Defensive Playcall Causal Inference Pipeline
Step 3: Propensity Model, Outcome Model, AIPW Estimation & Causal Forest CATE
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier, LGBMRegressor
from econml.dr import ForestDRLearner
from econml.cate_interpreter import SingleTreeCATEInterpreter
import warnings
warnings.filterwarnings("ignore")


# ── Column sets (mirror 01_data_pipeline.py) ─────────────────────────────────
NUMERIC_CONFOUNDERS = [
    "down", "ydstogo", "score_diff", "seconds_remaining", "yardline_100",
    "is_two_minute_drill", "is_red_zone", "is_third_or_fourth", "scoring_opp",
    "num_rb", "num_te", "num_wr",
    "off_pass_tendency", "def_team_avg_epa", "qb_avg_epa",
]
CATEGORICAL_CONFOUNDERS = ["formation", "personnel_group"]
ALL_CONFOUNDERS = NUMERIC_CONFOUNDERS + CATEGORICAL_CONFOUNDERS
TREATMENT_COL = "def_playcall_enc"
OUTCOME_COL = "epa"


# ── 1. Preprocessing ──────────────────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    """
    Preprocessing pipeline:
    - Numeric: pass-through (already engineered)
    - Categorical: one-hot encode formation and personnel group
    """
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_CONFOUNDERS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_CONFOUNDERS),
        ]
    )


# ── 2. Propensity Model ───────────────────────────────────────────────────────
def fit_propensity_model(
    X: np.ndarray, T: np.ndarray, n_splits: int = 5
) -> np.ndarray:
    """
    Fit a multi-class propensity model P(DefPlaycall | Confounders).
    Uses cross-fitting (out-of-fold predictions) to avoid overfitting
    the propensity scores to the same data used for outcome estimation.

    Returns: propensity_scores of shape (n, n_treatment_classes)
    """
    clf = CalibratedClassifierCV(
        LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        ),
        cv=3, method="isotonic"
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    propensity = cross_val_predict(clf, X, T, cv=skf, method="predict_proba")

    # Clip to avoid extreme weights (positivity enforcement)
    propensity = np.clip(propensity, 0.01, 0.99)
    propensity = propensity / propensity.sum(axis=1, keepdims=True)  # renormalize

    print(f"Propensity model: mean min prob = {propensity.min(axis=1).mean():.3f}")
    print(f"Effective sample size (ESS) ratio: {_ess_ratio(propensity, T):.3f}")
    return propensity


def _ess_ratio(propensity: np.ndarray, T: np.ndarray) -> float:
    """Effective sample size ratio — a measure of propensity overlap quality."""
    n = len(T)
    p_t = propensity[np.arange(n), T]
    weights = 1.0 / p_t
    weights /= weights.mean()
    ess = n / (weights ** 2).mean()
    return ess / n


# ── 3. Outcome Model ──────────────────────────────────────────────────────────
def fit_outcome_model(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    n_treatment_classes: int, n_splits: int = 5
) -> np.ndarray:
    """
    Fit a separate outcome model E[EPA | DefPlaycall=d, Confounders] for each
    treatment level d, using cross-fitting.

    Returns: mu_hat of shape (n, n_treatment_classes)
             Each column is the predicted potential outcome under playcall d.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mu_hat = np.zeros((len(Y), n_treatment_classes))

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        T_train, Y_train = T[train_idx], Y[train_idx]

        for d in range(n_treatment_classes):
            # Train on plays where treatment == d
            mask_d = T_train == d
            if mask_d.sum() < 20:
                # Not enough data for this call in fold — use global mean
                mu_hat[val_idx, d] = Y_train[mask_d].mean() if mask_d.any() else Y_train.mean()
                continue

            reg = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
            reg.fit(X_train[mask_d], Y_train[mask_d])
            mu_hat[val_idx, d] = reg.predict(X_val)

    return mu_hat


# ── 4. AIPW (Doubly Robust) Estimator ────────────────────────────────────────
def aipw_ate(
    Y: np.ndarray,
    T: np.ndarray,
    propensity: np.ndarray,
    mu_hat: np.ndarray,
    treatment_classes: list,
    baseline: int = 0,
) -> pd.DataFrame:
    """
    Augmented Inverse Probability Weighted (AIPW / Doubly Robust) estimator.
    Estimates the Average Treatment Effect (ATE) of each defensive playcall
    relative to the baseline call.

    AIPW score for treatment d:
        psi_d(i) = mu_hat[i,d] + (1{T==d} / P(T=d|X)) * (Y - mu_hat[i,d])

    ATE(d vs baseline) = E[psi_d] - E[psi_baseline]

    Doubly robust: consistent if EITHER propensity OR outcome model is correct.
    """
    n = len(Y)
    n_treatments = len(treatment_classes)
    psi = np.zeros((n, n_treatments))

    for d in range(n_treatments):
        indicator = (T == d).astype(float)
        p_d = propensity[:, d]
        psi[:, d] = mu_hat[:, d] + (indicator / p_d) * (Y - mu_hat[:, d])

    results = []
    psi_base = psi[:, baseline]

    for d in range(n_treatments):
        if d == baseline:
            continue
        ate_d = (psi[:, d] - psi_base).mean()
        se_d = (psi[:, d] - psi_base).std() / np.sqrt(n)
        ci_lo = ate_d - 1.96 * se_d
        ci_hi = ate_d + 1.96 * se_d
        results.append({
            "def_playcall": treatment_classes[d],
            "baseline": treatment_classes[baseline],
            "ate_vs_baseline": ate_d,     # negative = better for defense
            "se": se_d,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "significant": (ci_lo < 0 < ci_hi) == False,
        })

    return pd.DataFrame(results).sort_values("ate_vs_baseline")


# ── 5. CATE via Causal Forest (econml) ───────────────────────────────────────
def fit_causal_forest(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
) -> ForestDRLearner:
    """
    Fit a Doubly Robust Causal Forest to estimate Conditional Average Treatment
    Effects (CATEs): how the effect of each defensive call varies across contexts.

    Uses econml's ForestDRLearner:
      - model_propensity: LightGBM classifier (P(T|X))
      - model_regression: LightGBM regressor  (E[Y|T,X))
      - final estimator: causal forest (heterogeneous effect)

    Output is a trained model exposing:
      .effect(X)          → CATE estimates per sample
      .effect_interval(X) → confidence intervals
    """
    model = ForestDRLearner(
        model_propensity=LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            min_child_samples=20, verbose=-1, random_state=42
        ),
        model_regression=LGBMRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            min_child_samples=20, verbose=-1, random_state=42
        ),
        n_estimators=500,
        min_samples_leaf=20,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(Y, T, X=X)
    return model


def interpret_cate_tree(
    model: ForestDRLearner,
    X: np.ndarray,
    feature_names: list,
    treatment_idx: int = 1,
    max_depth: int = 3,
) -> None:
    """
    Fit a shallow decision tree to the CATE estimates to produce
    human-interpretable playcalling rules.

    e.g.: "In 3rd & long + shotgun + losing: blitz heavy reduces EPA by 0.3"
    """
    interpreter = SingleTreeCATEInterpreter(
        include_model_uncertainty=True,
        max_depth=max_depth,
        min_samples_leaf=50,
    )
    interpreter.interpret(model, X)
    interpreter.plot(feature_names=feature_names, treatment_index=treatment_idx,
                     figsize=(16, 8), fontsize=10)
    print("CATE decision tree plotted.")


# ── 6. Sensitivity Analysis ───────────────────────────────────────────────────
def sensitivity_analysis(
    ate_estimates: pd.DataFrame,
    gamma_range: list = [1.0, 1.5, 2.0, 2.5],
) -> pd.DataFrame:
    """
    Rosenbaum-style sensitivity analysis.
    
    For each gamma (odds ratio of hidden confounding), bounds how much
    the ATE could shift if there were an unobserved confounder of that strength.
    
    gamma = 1.0 → no hidden confounding (baseline)
    gamma = 2.0 → hidden confounder can double the odds of treatment
    
    If the effect sign is stable up to gamma=2, it's robust.
    """
    rows = []
    for _, r in ate_estimates.iterrows():
        for gamma in gamma_range:
            # Simple Rosenbaum bound: shift estimate by ±log(gamma)*se
            shift = np.log(gamma) * r["se"]
            rows.append({
                "def_playcall": r["def_playcall"],
                "gamma": gamma,
                "ate_lower_bound": r["ate_vs_baseline"] - shift,
                "ate_upper_bound": r["ate_vs_baseline"] + shift,
                "sign_stable": (
                    (r["ate_vs_baseline"] - shift < 0) ==
                    (r["ate_vs_baseline"] + shift < 0)
                )
            })
    return pd.DataFrame(rows)


# ── 7. Full pipeline runner ───────────────────────────────────────────────────
def run_estimation_pipeline(df: pd.DataFrame, label_encoder: LabelEncoder):
    """
    End-to-end estimation pipeline.
    
    1. Preprocess
    2. Propensity model (cross-fit)
    3. Outcome model   (cross-fit)
    4. AIPW ATE estimates
    5. Causal Forest CATE
    6. Sensitivity analysis
    """
    print("\n" + "="*60)
    print("NFL CAUSAL ESTIMATION PIPELINE")
    print("="*60)

    # --- Preprocess
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df[ALL_CONFOUNDERS])
    T = df[TREATMENT_COL].values.astype(int)
    Y = df[OUTCOME_COL].values.astype(float)
    n_calls = len(label_encoder.classes_)

    feature_names = (
        NUMERIC_CONFOUNDERS
        + list(preprocessor.named_transformers_["cat"]
               .get_feature_names_out(CATEGORICAL_CONFOUNDERS))
    )

    print(f"\nData: {len(Y):,} plays | {n_calls} defensive call types")
    print(f"Feature matrix: {X.shape}")

    # --- Step 2: Propensity
    print("\n[1/4] Fitting propensity model...")
    propensity = fit_propensity_model(X, T)

    # --- Step 3: Outcome
    print("\n[2/4] Fitting outcome models (per treatment level)...")
    mu_hat = fit_outcome_model(X, T, Y, n_treatment_classes=n_calls)

    # --- Step 4: AIPW ATE
    print("\n[3/4] Computing AIPW (doubly robust) ATE estimates...")
    ate_df = aipw_ate(Y, T, propensity, mu_hat,
                      treatment_classes=list(label_encoder.classes_),
                      baseline=0)
    print("\nATE vs Baseline (negative = defense wins):")
    print(ate_df.to_string(index=False))

    # --- Step 4b: Sensitivity
    print("\n[3b/4] Sensitivity analysis...")
    sens_df = sensitivity_analysis(ate_df)
    robust = sens_df[sens_df["gamma"] == 2.0]["sign_stable"].mean() * 100
    print(f"Effect sign stable under gamma=2.0 for {robust:.0f}% of playcalls")

    # --- Step 5: Causal Forest
    print("\n[4/4] Fitting Causal Forest (CATE)...")
    cf_model = fit_causal_forest(X, T, Y)
    cate_estimates = cf_model.effect(X)
    print(f"CATE shape: {cate_estimates.shape} (n_plays × n_treatment_pairs)")
    print(f"Mean CATE across plays: {cate_estimates.mean(axis=0)}")

    return {
        "preprocessor": preprocessor,
        "propensity": propensity,
        "mu_hat": mu_hat,
        "ate": ate_df,
        "sensitivity": sens_df,
        "causal_forest": cf_model,
        "cate": cate_estimates,
        "X": X,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    print("Estimation module loaded.")
    print("Call run_estimation_pipeline(df, label_encoder) to run.")
