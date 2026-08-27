"""Shared state preprocessing.

Two preprocessors so linear and tree models each get what they need:

  * `build_state_preprocessor`  — standardized numerics + one-hot categoricals,
    for regularized *linear* models (kept for baselines / the V1 tag).
  * `build_gbm_preprocessor`    — passthrough numerics + one-hot categoricals,
    for gradient-boosted trees (scale-invariant, so no StandardScaler).

Both expose the same fit/transform interface (a fitted ColumnTransformer), so
pi_b and the Q-model can swap preprocessors without touching downstream code.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..schemas.dataset import Dataset


def build_state_preprocessor(ds: Dataset) -> ColumnTransformer:
    """Standardize numeric state, one-hot the categorical state (linear models)."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ds.numeric_state),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ds.categorical_state),
        ]
    )


def build_gbm_preprocessor(ds: Dataset) -> ColumnTransformer:
    """Passthrough numeric state, one-hot the categorical state (tree models).

    Trees are scale-invariant, so scaling numerics buys nothing and would only
    obscure the raw feature values in the split log.
    """
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", ds.numeric_state),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ds.categorical_state),
        ]
    )
