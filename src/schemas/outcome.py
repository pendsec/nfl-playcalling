"""`QModel` — a fitted outcome / Q-model, Q(S, A) = E[R | S, do(A)].

One gradient-boosted regressor per action level (a per-treatment outcome model),
sharing a fitted preprocessor. Evaluates Q for the taken action or,
counterfactually, for every action in each state.

Causal note: conditions only on the SCM adjustment set (pre-snap confounders +
player proxies); the offensive-playcall mediator and any result-derived field are
excluded, so Q captures the call's *total* effect on reward. Fitted by
`src.models.outcome.q_model.fit_q_model`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer


@dataclass
class QModel:
    prep: ColumnTransformer
    regressors: dict[int, object]   # action id -> fitted regressor (or None)
    fallbacks: dict[int, float]     # action id -> constant used when no regressor
    n_actions: int
    state_cols: list[str]

    # Typical within-state spread of Q across actions, measured on the training
    # states at fit time. This is the natural unit for anything that trades off
    # against Q — notably the conservative policy's pessimism penalty, whose
    # `alpha` would otherwise mean something different for every fitted model
    # (and, on real data, was ~13x too small to influence a single argmax).
    # Lives on the model rather than the policy so that every policy built from
    # this Q — including the ones the regression gate constructs on the holdout —
    # shares one scale derived from training data.
    q_scale: float = 1.0

    def predict_all_actions(self, df: pd.DataFrame) -> np.ndarray:
        """Q(s, a) for every action, shape (n, n_actions).

        Counterfactual sweep: hold state fixed and score it under each action's
        own regressor. Actions with too few training plays fall back to their
        (or the global) mean reward — a flat estimate the conservative policy is
        designed not to chase.
        """
        phi = self.prep.transform(df[self.state_cols])
        n = phi.shape[0]
        out = np.empty((n, self.n_actions))
        for a in range(self.n_actions):
            reg = self.regressors.get(a)
            out[:, a] = reg.predict(phi) if reg is not None else self.fallbacks[a]
        return out

    def predict_taken(self, df: pd.DataFrame, actions: np.ndarray) -> np.ndarray:
        """Q(s, a) only for the action actually taken in each row."""
        q_all = self.predict_all_actions(df)
        return q_all[np.arange(len(df)), np.asarray(actions)]


@dataclass
class PooledQModel:
    """One regressor over (state, action features) rather than one per action.

    The action enters as a small feature vector — for the every-snap scheme,
    (package, front) — so the model shares strength across action cells: a
    rarely-called alignment borrows from every alignment that shares its package
    and every one that shares its front. Scoring the counterfactual is then a
    matter of swapping the action features and re-predicting.

    Same interface as `QModel`, so policies and OPE take either.
    """
    prep: ColumnTransformer
    regressor: object
    action_features: np.ndarray     # (n_actions, k): the features for action a
    n_actions: int
    state_cols: list[str]
    q_scale: float = 1.0

    def predict_all_actions(self, df: pd.DataFrame) -> np.ndarray:
        phi = self.prep.transform(df[self.state_cols])
        n = phi.shape[0]
        return np.column_stack([
            self.regressor.predict(np.hstack([phi, np.tile(self.action_features[a], (n, 1))]))
            for a in range(self.n_actions)
        ])

    def predict_taken(self, df: pd.DataFrame, actions: np.ndarray) -> np.ndarray:
        q_all = self.predict_all_actions(df)
        return q_all[np.arange(len(df)), np.asarray(actions)]
