"""
Pipeline data structures — every @dataclass the pipeline passes between stages,
collected in one place so the shapes of the data are easy to find and read.

    Dataset        the built (S, A, R) table + column metadata   (data layer)
    BehaviorModel  a fitted behavior policy pi_b(A|S)            (Step 3)
    QModel, PooledQModel, QEnsemble
                   fitted outcome models Q(S, A) and the bootstrap ensemble (Step 4)
    OPEResult      one off-policy-evaluation estimate + 95% CI   (Step 5)
    GreedyPolicy, ConservativePolicy, SafeImprovementPolicy
                   the policies, each with its decision rule `act`   (Step 6)

The operations that build them (build_dataset, fit_behavior_model, fit_q_model,
dr_policy_value, learn_policy, ...) stay in their component packages and import
these classes from here.
"""

from .dataset import Dataset
from .behavior import BehaviorModel, clip_normalize
from .outcome import PooledQModel, QEnsemble, QModel
from .ope import OPEResult
from .policy import ConservativePolicy, GreedyPolicy, SafeImprovementPolicy

__all__ = [
    "Dataset",
    "BehaviorModel", "clip_normalize",
    "QModel", "PooledQModel", "QEnsemble",
    "OPEResult",
    "GreedyPolicy", "ConservativePolicy", "SafeImprovementPolicy",
]
