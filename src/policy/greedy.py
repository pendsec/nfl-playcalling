"""
Policy Learning — behavior-constrained greedy (V1).

Decision rule:

    pi*(s) = argmax_a  Q(s, a)   subject to   pi_b(a | s) >= support_threshold

The behavior constraint is the V1 stand-in for the pessimism/positivity
discipline that CQL and causal-pessimism formalize later: we refuse to
recommend a call the DC effectively never makes in this kind of state, because
the Q-model has no support there and OPE cannot vouch for it.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from ..models.outcome.q_model import QModel
from ..models.behavior.propensity import BehaviorModel
from ..data.features import Dataset


@dataclass
class GreedyPolicy:
    q: QModel
    beh: BehaviorModel
    support_threshold: float
    n_actions: int

    def act(self, df: pd.DataFrame) -> dict:
        """Return per-state Q values, support mask, and chosen action."""
        q_all = self.q.predict_all_actions(df)
        support = self.beh.propensity(df) >= self.support_threshold

        # Mask unsupported actions to -inf before the argmax. If a state has no
        # supported action (shouldn't happen after clipping), fall back to the
        # unconstrained argmax so we always return a call.
        masked = np.where(support, q_all, -np.inf)
        chosen = masked.argmax(axis=1)
        no_support = ~support.any(axis=1)
        chosen[no_support] = q_all[no_support].argmax(axis=1)

        probs = np.zeros_like(q_all)
        probs[np.arange(len(df)), chosen] = 1.0
        return {"q_all": q_all, "support": support,
                "chosen": chosen, "policy_probs": probs}


def learn_greedy_policy(ds: Dataset, q: QModel, beh: BehaviorModel,
                        cfg: dict) -> GreedyPolicy:
    return GreedyPolicy(
        q=q, beh=beh,
        support_threshold=cfg["policy"]["support_threshold"],
        n_actions=ds.n_actions,
    )
