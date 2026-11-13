"""Shared state preprocessing for the gradient-boosted models: numeric state
passed through, categorical state one-hot encoded (trees are scale-invariant,
so nothing is standardized)."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from ..schemas.dataset import Dataset


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
