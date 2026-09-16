"""
Policy Learning — behavior-constrained greedy.

Decision rule:

    pi*(s) = argmax_a  Q(s, a)   subject to   pi_b(a | s) >= support_threshold

The behavior constraint is the simplest form of the pessimism/positivity
discipline that CQL and causal-pessimism formalize: we refuse to recommend a
call the DC effectively never makes in this kind of state, because the Q-model
has no support there and OPE cannot vouch for it. It is kept as the unregularized
baseline the conservative policy is measured against.
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
