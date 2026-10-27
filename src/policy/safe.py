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

from ..schemas.dataset import Dataset
from ..schemas.behavior import BehaviorModel
from ..schemas.outcome import QEnsemble, QModel
from ..schemas.policy import SafeImprovementPolicy


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
    if cfg["policy"].get("type", "conservative") == "safe":
        if ensemble is None:
            raise ValueError("policy.type 'safe' needs a Q ensemble "
                             "(models.outcome.q_model.fit_q_ensemble)")
        return learn_safe_policy(ds, q, beh, cfg, ensemble)
    from .conservative import learn_conservative_policy
    return learn_conservative_policy(ds, q, beh, cfg)
