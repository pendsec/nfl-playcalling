"""`Dataset` — the built (S, A, R) table plus the column metadata models need.

Pure data container. Constructed by `src.data.features.build_v1_dataset`; every
downstream model reads its `state_cols` / `action_col` / `reward_col` rather than
hard-coding column names, so feature lists are defined once (in the data layer)
and flow through the pipeline on this object.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class Dataset:
    """A built (S, A, R) table plus the metadata models need."""
    df: pd.DataFrame
    numeric_state: list[str]
    categorical_state: list[str]
    action_col: str
    reward_col: str
    n_actions: int
    action_labels: dict[int, str]

    @property
    def state_cols(self) -> list[str]:
        return self.numeric_state + self.categorical_state
