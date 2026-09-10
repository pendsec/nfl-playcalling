"""Policies pi*(A | S).

`GreedyPolicy` — behavior-constrained argmax_a Q(s, a) over calls with adequate
support (pi_b(a|s) >= threshold). Kept as V1's baseline: the policy the V2
conservative policy is compared against via DR-OPE.

`ConservativePolicy` — the V2 policy. In this single-step contextual-bandit
setting, CQL/IQL reduce to a behavior-regularized (KL-to-pi_b) argmax:

    pi*(s) = argmax_a  [ Q(s, a) + alpha * q_scale * log pi_b(a | s) ]
             subject to the support floor

The penalty term is the conservatism/pessimism penalty: low-support actions
carry a large negative log-propensity, so the policy is pulled toward calls the
DC actually makes unless Q is confidently higher. alpha -> 0 recovers greedy;
alpha -> inf recovers the behavior mode. Full *sequential* CQL/IQL (bootstrapped
targets, expectile value) only becomes meaningful in V3+ once drive-level
transitions enter — here there is a single step, so no bootstrap.

`q_scale` is what makes alpha portable. Q is measured in reward(-EPA) units whose
spread depends entirely on the data and the fit, while log pi_b is in nats — so a
bare `alpha` silently means something different for every model. Expressing the
penalty in units of the typical within-state Q spread (`QModel.q_scale`) fixes
that: alpha is then "how many standard deviations of within-state Q advantage a
call must show to justify one nat of departure from the DC". At alpha=0.5 a call
sitting an order of magnitude less likely than the DC's (about 2.3 nats) has to
beat it by more than a full sigma of Q to be recommended.
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


@dataclass
class ConservativePolicy:
    """Behavior-regularized (CQL/IQL-style) conservative policy.

    pi*(s) = argmax_a [ Q(s,a) + alpha*log pi_b(a|s) ] over supported actions.
    Built by `src.policy.conservative.learn_conservative_policy`.
    """
    q: QModel
    beh: BehaviorModel
    alpha: float
    support_threshold: float
    n_actions: int
    q_scale: float = 1.0   # units of Q per nat; see module docstring

    def act(self, df: pd.DataFrame) -> dict:
        """Return per-state Q, propensities, conservative scores, and choice."""
        q_all = self.q.predict_all_actions(df)
        pi_b = self.beh.propensity(df)
        support = pi_b >= self.support_threshold

        # Conservative score: Q plus a KL-to-behavior penalty, expressed in Q's
        # own units so alpha is comparable across fits. log of the (clipped)
        # propensity is a large negative for low-support calls.
        penalty = self.alpha * self.q_scale * np.log(np.clip(pi_b, 1e-12, 1.0))
        scores = q_all + penalty

        # Restrict to supported actions; fall back to unconstrained score if a
        # state has no supported call (shouldn't happen after propensity clip).
        masked = np.where(support, scores, -np.inf)
        chosen = masked.argmax(axis=1)
        no_support = ~support.any(axis=1)
        chosen[no_support] = scores[no_support].argmax(axis=1)

        probs = np.zeros_like(q_all)
        probs[np.arange(len(df)), chosen] = 1.0
        return {"q_all": q_all, "pi_b": pi_b, "scores": scores,
                "support": support, "chosen": chosen, "policy_probs": probs}
