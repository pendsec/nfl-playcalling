"""`BehaviorModel` — a fitted behavior policy pi_b(A | S).

Wraps the fitted sklearn pipeline and turns it into calibrated, clipped,
full-width propensities P(A=a | S) over the whole action space. Fitted by
`src.models.behavior.propensity.fit_behavior_model`.

`clip_normalize` lives here too because it is intrinsic to how a BehaviorModel
produces propensities (it is the positivity guard); the fit-time diagnostics in
propensity.py import it back from here so both paths clip identically.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass
class BehaviorModel:
    pipeline: Pipeline
    classes_: np.ndarray          # action ids the model can output, sorted
    n_actions: int
    clip: tuple[float, float]

    def propensity(self, df: pd.DataFrame) -> np.ndarray:
        """P(A=a | S) for every action a, shape (n, n_actions).

        Classes absent from training get probability 0; the row is then
        renormalized after clipping so each row sums to 1.
        """
        # predict_proba only returns columns for classes seen in training, in
        # `classes_` order. Scatter them back into a full n_actions-wide matrix
        # so column j always means action j; any action never seen in training
        # stays 0 here and is handled by the clip+renormalize below.
        proba = self.pipeline.predict_proba(df)
        full = np.zeros((len(df), self.n_actions))
        full[:, self.classes_] = proba
        return clip_normalize(full, self.clip)


def clip_normalize(p: np.ndarray, clip: tuple[float, float]) -> np.ndarray:
    """Clip propensities into `clip`, then renormalize each row to sum to 1.

    The clip is the positivity guard that keeps downstream inverse-weighting from
    exploding on near-zero propensities. Shared by `BehaviorModel.propensity` and
    the fit-time calibration diagnostics.
    """
    p = np.clip(p, clip[0], clip[1])
    return p / p.sum(axis=1, keepdims=True)
