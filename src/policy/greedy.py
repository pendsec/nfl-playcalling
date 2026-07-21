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

from ..schemas.dataset import Dataset
from ..schemas.behavior import BehaviorModel
from ..schemas.outcome import QModel
from ..schemas.policy import GreedyPolicy


def learn_greedy_policy(ds: Dataset, q: QModel, beh: BehaviorModel,
                        cfg: dict) -> GreedyPolicy:
    return GreedyPolicy(
        q=q, beh=beh,
        support_threshold=cfg["policy"]["support_threshold"],
        n_actions=ds.n_actions,
    )
