"""
Outcome / Q-Model — Q(S, A) = E[R | S, do(A)].

V1: ridge regression of reward on a design matrix that blocks the (scaled,
one-hot) state by action. Concretely the model learns a per-action intercept
and a per-action state-effect vector under a single shared L2 penalty — i.e.
partially-pooled per-action ridge. This keeps action effects state-dependent
(so recommendations vary by situation) while staying robust to small action
cells like man+blitz.

Causal note: per the V1 SCM we condition only on pre-snap state S. The
offensive playcall is a post-snap *mediator* of A -> R and is deliberately
excluded; conditioning on it would block part of the effect we want.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

from ..preprocess import build_state_preprocessor
from ...data.features import Dataset


@dataclass
class QModel:
    prep: ColumnTransformer
    ridge: Ridge
    n_actions: int
    state_cols: list[str]

    def predict_all_actions(self, df: pd.DataFrame) -> np.ndarray:
        """Q(s, a) for every action, shape (n, n_actions)."""
        phi = self.prep.transform(df[self.state_cols])
        n = phi.shape[0]
        out = np.zeros((n, self.n_actions))
        # Counterfactual sweep: hold the state fixed and force every row to each
        # action in turn. Column a is Q(s, do(A=a)) for all states — this is what
        # OPE and the greedy argmax consume to compare calls in the same state.
        for a in range(self.n_actions):
            actions = np.full(n, a)
            out[:, a] = self.ridge.predict(_design(phi, actions, self.n_actions))
        return out

    def predict_taken(self, df: pd.DataFrame, actions: np.ndarray) -> np.ndarray:
        """Q(s, a) only for the action actually taken in each row."""
        phi = self.prep.transform(df[self.state_cols])
        return self.ridge.predict(_design(phi, actions, self.n_actions))


def fit_q_model(ds: Dataset, cfg: dict) -> QModel:
    prep = build_state_preprocessor(ds)
    phi = prep.fit_transform(ds.df[ds.state_cols])
    actions = ds.df[ds.action_col].to_numpy()
    y = ds.df[ds.reward_col].to_numpy()

    ridge = Ridge(alpha=cfg["outcome_model"]["alpha"], fit_intercept=False)
    ridge.fit(_design(phi, actions, ds.n_actions), y)

    return QModel(prep=prep, ridge=ridge, n_actions=ds.n_actions,
                  state_cols=ds.state_cols)


def _design(phi: np.ndarray, actions: np.ndarray, n_actions: int) -> np.ndarray:
    """Block design: [per-action intercepts | action-blocked state].

    Columns 0..K-1 are action indicators (intercepts); the remaining K*d
    columns hold the state vector placed in the block for the row's action and
    zero elsewhere — so each action gets its own linear state response.
    """
    n, d = phi.shape
    intercepts = np.zeros((n, n_actions))
    intercepts[np.arange(n), actions] = 1.0
    blocked = np.zeros((n, n_actions * d))
    rows = np.arange(n)
    for a in range(n_actions):
        m = actions == a
        if m.any():
            blocked[np.ix_(rows[m], range(a * d, (a + 1) * d))] = phi[m]
    return np.hstack([intercepts, blocked])
