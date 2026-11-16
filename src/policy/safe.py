"""
Policy Learning — safe improvement over the coordinator.

    pi*(.|s) = pi_b(.|s)                        if no supported call has LCB > 0
             = 1{a = argmax_a LCB(s, a)}        otherwise
    LCB(s, a) = advantage of a over the DC's mix - kappa * bootstrap sd
                - read allowance (box count only; see `policy.read_bias`)

The bootstrap sd comes from `fit_q_ensemble` (whole games resampled). It
measures how far Q's comparison between calls moves with the sample — sampling
uncertainty only. It cannot see confounding bias: a call whose apparent edge
comes from the coordinator's hidden read is "established" to the bootstrap
however wrong it is. The read allowance covers the one such bias the data
shows — heavy boxes facing more runs than the state predicts — and nothing else.

kappa is chosen on an out-of-time fold inside the training seasons (fit
2022-23, evaluate 2024), never on the held-out season. See
`schemas.policy.SafeImprovementPolicy` for the decision rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models.behavior.propensity import fit_behavior_model
from ..models.outcome.q_model import fit_q_ensemble, fit_q_model
from ..schemas.dataset import Dataset
from ..schemas.behavior import BehaviorModel
from ..schemas.outcome import QEnsemble, QModel
from ..schemas.policy import SafeImprovementPolicy


def is_safe(cfg: dict) -> bool:
    return cfg["policy"].get("type", "conservative") == "safe"


def learn_safe_policy(ds: Dataset, q: QModel, beh: BehaviorModel, cfg: dict,
                      ensemble: QEnsemble) -> SafeImprovementPolicy:
    """`ds` must be the training set: the read allowance is estimated on it."""
    from .read_bias import allowance_axes, estimate_read_bias

    pcfg = cfg["policy"]
    read_bias, table = estimate_read_bias(ds, cfg) if allowance_axes(cfg) else (None, None)
    policy = SafeImprovementPolicy(
        q=q, beh=beh, ensemble=ensemble,
        kappa=pcfg.get("lcb_kappa", 2.326),
        support_threshold=pcfg.get("support_threshold", 0.05),
        n_actions=ds.n_actions,
        read_bias=read_bias,
        allowance_scale=(pcfg.get("read_allowance") or {}).get("scale", 1.0),
    )
    policy.read_bias_table = table      # per-component breakdown, for reporting
    return policy


def learn_policy(ds: Dataset, q: QModel, beh: BehaviorModel, cfg: dict,
                 ensemble: QEnsemble | None = None):
    """The configured candidate policy: `policy.type` safe | conservative."""
    if is_safe(cfg):
        if ensemble is None:
            raise ValueError("policy.type 'safe' needs a Q ensemble "
                             "(models.outcome.q_model.fit_q_ensemble)")
        return learn_safe_policy(ds, q, beh, cfg, ensemble)
    from .conservative import learn_conservative_policy
    return learn_conservative_policy(ds, q, beh, cfg)


@dataclass
class PolicyFit:
    """The configured policy and the models it was built from."""
    beh: BehaviorModel
    q: QModel
    ensemble: QEnsemble | None
    policy: object


def fit_policy(ds: Dataset, cfg: dict, beh: BehaviorModel | None = None,
               q: QModel | None = None) -> PolicyFit:
    """The configured policy, end to end, from `ds` alone: pi_b and Q (unless
    given), the bootstrap ensemble when the policy is safe, then the policy."""
    beh = beh if beh is not None else fit_behavior_model(ds, cfg, diagnose=False)[0]
    q = q if q is not None else fit_q_model(ds, cfg)
    ensemble = fit_q_ensemble(ds, cfg) if is_safe(cfg) else None
    return PolicyFit(beh=beh, q=q, ensemble=ensemble,
                     policy=learn_policy(ds, q, beh, cfg, ensemble=ensemble))


def defer_to(out: dict, pi_b: np.ndarray) -> np.ndarray:
    """A policy's probabilities with its deferrals playing `pi_b` instead of its
    own estimate of the coordinator's mix. `out` is the policy's `act` output."""
    deferred = out.get("deferred")
    if deferred is None:            # a policy that never defers
        return out["policy_probs"]
    return np.where(deferred[:, None], pi_b, out["policy_probs"])
