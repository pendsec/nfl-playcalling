"""Policies pi*(A | S).

`GreedyPolicy` — behavior-constrained argmax_a Q(s, a) over calls with adequate
support (pi_b(a|s) >= threshold). Kept as the unregularized baseline that the
conservative policy is compared against via DR-OPE.

`ConservativePolicy` — the shipped policy. In this single-step contextual-bandit
setting, CQL/IQL reduce to a behavior-regularized (KL-to-pi_b) argmax:

    pi*(s) = argmax_a  [ Q(s, a) + alpha * q_scale * log pi_b(a | s) ]
             subject to the support floor

The penalty term is the conservatism/pessimism penalty: low-support actions
carry a large negative log-propensity, so the policy is pulled toward calls the
DC actually makes unless Q is confidently higher. alpha -> 0 recovers greedy;
alpha -> inf recovers the behavior mode. Full *sequential* CQL/IQL (bootstrapped
targets, expectile value) only becomes meaningful once drive-level
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
from .outcome import QEnsemble, QModel


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
        return {"q_all": q_all, "pi_b": pi_b, "support": support, "chosen": chosen,
                "policy_probs": probs}


@dataclass
class SafeImprovementPolicy:
    """Defer to the coordinator unless the data establishes a better call.

    In each state, keep the DC's own mix pi_b(.|s) — unless some supported call
    clears a lower confidence bound on its advantage over that mix:

        LCB(s, a) = [Q(s,a) - sum_a' pi_b(a'|s) Q(s,a')] - kappa * sd_boot(s, a)

    and then switch entirely to the call with the highest LCB. This is the
    SPIBB idea (safe policy improvement with baseline bootstrapping) in the
    one-step setting: uncertainty sends the policy back to what the DC does,
    not to the DC's most frequent call. That distinction matters — a
    deterministic argmax pushed toward the modal call stops mixing, and on
    2022-25 that alone cost ~0.01 EPA/play in DR, so the "most pessimistic"
    deterministic policy was not the safe one.

    Where it defers, the policy IS the behavior policy, so OPE scores those
    states at the DC's own value with no extrapolation; the estimate rests only
    on the departures. Built by `src.policy.safe.learn_safe_policy`.

    `chosen` is the headline call: the switched-to call where the policy
    departs, the DC's most frequent call where it defers. `deferred` says which.

    `read_bias` (per action, reward units; `src.policy.read_bias`) adds a
    confounding allowance to the bound:

        LCB(s, a) -= allowance_scale * |read_bias[a] - sum_a' pi_b(a'|s) read_bias[a']|

    The bootstrap sd covers sampling noise only; this covers the known bias of
    the coordinator's read, so a box departure must clear both. None or zeros
    leave the rule as above.
    """
    q: QModel
    beh: BehaviorModel
    ensemble: QEnsemble
    kappa: float
    support_threshold: float
    n_actions: int
    read_bias: np.ndarray | None = None
    allowance_scale: float = 1.0

    def act(self, df: pd.DataFrame) -> dict:
        q_all = self.q.predict_all_actions(df)
        pi_b = self.beh.propensity(df)
        support = pi_b >= self.support_threshold
        adv = q_all - (pi_b * q_all).sum(axis=1, keepdims=True)
        q_sd = self.ensemble.advantage_sd(df, pi_b)
        allowance = np.zeros_like(adv)
        if self.read_bias is not None:
            rb = np.asarray(self.read_bias)[None, :]
            allowance = self.allowance_scale * np.abs(rb - (pi_b * rb).sum(axis=1, keepdims=True))
        lcb = np.where(support, adv - self.kappa * q_sd - allowance, -np.inf)

        best = lcb.argmax(axis=1)
        switch = lcb.max(axis=1) > 0
        probs = pi_b.copy()
        probs[switch] = 0.0
        probs[np.where(switch)[0], best[switch]] = 1.0
        chosen = np.where(switch, best, pi_b.argmax(axis=1))
        return {"q_all": q_all, "pi_b": pi_b, "advantage": adv, "q_sd": q_sd,
                "allowance": allowance, "lcb": lcb, "support": support, "deferred": ~switch,
                "chosen": chosen, "policy_probs": probs}
