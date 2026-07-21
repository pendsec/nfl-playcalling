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

from sklearn.linear_model import Ridge

from ..preprocess import build_state_preprocessor
from ...schemas.dataset import Dataset
from ...schemas.outcome import QModel, design_matrix


def fit_q_model(ds: Dataset, cfg: dict) -> QModel:
    prep = build_state_preprocessor(ds)
    phi = prep.fit_transform(ds.df[ds.state_cols])
    actions = ds.df[ds.action_col].to_numpy()
    y = ds.df[ds.reward_col].to_numpy()

    ridge = Ridge(alpha=cfg["outcome_model"]["alpha"], fit_intercept=False)
    ridge.fit(design_matrix(phi, actions, ds.n_actions), y)

    return QModel(prep=prep, ridge=ridge, n_actions=ds.n_actions,
                  state_cols=ds.state_cols)
