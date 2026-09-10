"""
Policy Learning — conservative, behavior-regularized policy (V2).

Single-step contextual-bandit reduction of CQL/IQL: the learned policy maximizes
a pessimism-penalized value

    pi*(s) = argmax_a  [ Q(s, a) + alpha * q_scale * log pi_b(a | s) ]

subject to the hard support floor pi_b(a|s) >= support_threshold. `q_scale` is
the fitted Q model's own typical within-state spread, which makes alpha a
scale-free knob instead of one that has to be retuned for every fit. The penalty
term is the conservatism penalty — it pulls the policy toward
well-supported calls and only departs from the DC when Q is confidently higher,
which is what keeps DR-OPE able to vouch for the recommendation. See
`schemas.policy.ConservativePolicy` for the decision rule itself.
"""

from __future__ import annotations

from ..schemas.dataset import Dataset
from ..schemas.behavior import BehaviorModel
from ..schemas.outcome import QModel
from ..schemas.policy import ConservativePolicy


def learn_conservative_policy(ds: Dataset, q: QModel, beh: BehaviorModel,
                              cfg: dict) -> ConservativePolicy:
    pcfg = cfg["policy"]
    return ConservativePolicy(
        q=q, beh=beh,
        alpha=pcfg.get("alpha", 0.5),
        support_threshold=pcfg.get("support_threshold", 0.05),
        n_actions=ds.n_actions,
        # Taken from the Q model (measured on ITS training states), not
        # recomputed from `ds` — so the policy the regression gate builds on the
        # holdout is the same policy, not one with a holdout-derived scale.
        q_scale=getattr(q, "q_scale", 1.0),
    )
