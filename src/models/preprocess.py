"""Shared state preprocessing — one fitted transformer for pi_b and Q."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..schemas.dataset import Dataset


def build_state_preprocessor(ds: Dataset) -> ColumnTransformer:
    """Standardize numeric state, one-hot the categorical state.

    Scaling matters here: logistic pi_b and ridge Q are both regularized
    linear models, so unscaled features would distort the penalty.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ds.numeric_state),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ds.categorical_state),
        ]
    )
