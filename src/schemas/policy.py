"""`GreedyPolicy` — the behavior-constrained greedy policy pi*(A | S).

Recommends argmax_a Q(s, a) restricted to calls with adequate behavioral support
(pi_b(a|s) >= support_threshold). The support constraint is V1's stand-in for the
pessimism/positivity discipline CQL formalizes later: it refuses to recommend a
call the DC effectively never makes in that kind of state. Built by
`src.policy.greedy.learn_greedy_policy`.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .behavior import BehaviorModel
from .outcome import QModel


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
