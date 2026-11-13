"""Fold assignment for cross-fitting: whole games stay together.

Plays within a game share the matchup, the weather and the game script, so a
model scored on a play must not have seen that play's game. Shared by the
holdout judge (`models.behavior.judge`) and the transfer / invariance
nuisances (`evaluation.transfer`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def game_folds(df: pd.DataFrame, n_folds: int, seed: int) -> np.ndarray:
    """Fold id per row, whole games together."""
    fold = np.empty(len(df), int)
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(gkf.split(df, groups=df["game_id"])):
        fold[te] = k
    return fold
