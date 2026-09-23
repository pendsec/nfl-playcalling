"""Evaluation: holdout validation and policy regression gates.

`version_compare` implements CLAUDE.md's modeling discipline — "every new model
version evaluated against the previous via OPE on a held-out season; regressions
block merge". It scores the incumbent baseline policy and the candidate policy
by DR on the *same* holdout plays and gates on the paired difference.

The synthetic-SCM regression suite (simulate from a known SCM, assert the
pipeline recovers the truth) lives in tests/, where pytest can gate on it.
"""

from .version_compare import (
    VersionComparison,
    compare_policies,
    compare_to_baseline,
    to_frame,
)

__all__ = ["VersionComparison", "compare_policies", "compare_to_baseline", "to_frame"]
