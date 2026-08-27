"""
Pipeline data structures — every @dataclass the pipeline passes between stages,
collected in one place so the shapes of the data are easy to find and read.

    Dataset        the built (S, A, R) table + column metadata   (data layer)
    BehaviorModel  a fitted behavior policy pi_b(A|S)            (Step 3)
    QModel         a fitted outcome / Q-model Q(S, A)            (Step 4)
    OPEResult      one off-policy-evaluation estimate + 95% CI   (Step 5)
    GreedyPolicy   the behavior-constrained greedy policy pi*    (Step 6)

Only the *shapes* live here. The operations that build them (build_dataset,
fit_behavior_model, fit_q_model, dr_policy_value, learn_conservative_policy, ...)
stay in their component packages and import these classes from here. You can
import a shape either from its component module (unchanged) or straight from
`src.schemas`.
"""

from .dataset import Dataset
from .behavior import BehaviorModel, clip_normalize
from .outcome import QModel
from .ope import OPEResult
from .policy import GreedyPolicy, ConservativePolicy

__all__ = [
    "Dataset",
    "BehaviorModel", "clip_normalize",
    "QModel",
    "OPEResult",
    "GreedyPolicy", "ConservativePolicy",
]
