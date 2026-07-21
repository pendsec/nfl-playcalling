"""`QModel` — a fitted outcome / Q-model, Q(S, A) = E[R | S, do(A)].

Holds the fitted preprocessor + ridge and evaluates Q for the taken action or,
counterfactually, for every action in each state. Fitted by
`src.models.outcome.q_model.fit_q_model`.

Causal note: conditions only on pre-snap state S; the offensive-playcall mediator
is excluded so Q captures the call's *total* effect on reward.

`design_matrix` lives here because it is how a QModel lays out its features (a
per-action block design); the fit function in q_model.py imports it back from
here so training and prediction build identical matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge


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
            out[:, a] = self.ridge.predict(design_matrix(phi, actions, self.n_actions))
        return out

    def predict_taken(self, df: pd.DataFrame, actions: np.ndarray) -> np.ndarray:
        """Q(s, a) only for the action actually taken in each row."""
        phi = self.prep.transform(df[self.state_cols])
        return self.ridge.predict(design_matrix(phi, actions, self.n_actions))


def design_matrix(phi: np.ndarray, actions: np.ndarray, n_actions: int) -> np.ndarray:
    """Block design: [per-action intercepts | action-blocked state].

    Columns 0..K-1 are action indicators (intercepts); the remaining K*d
    columns hold the state vector placed in the block for the row's action and
    zero elsewhere — so each action gets its own linear state response. Shared by
    `QModel`'s predict methods and fit-time `fit_q_model`.
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
