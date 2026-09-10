"""Evaluation: holdout validation and version-over-version regression gates.

`version_compare` implements CLAUDE.md's modeling discipline — "every new model
version evaluated against the previous via OPE on a held-out season; regressions
block merge". It scores the previous version's policy and the candidate policy
by DR on the *same* holdout plays and gates on the paired difference.

The synthetic-SCM regression suite (simulate from a known SCM, assert the
pipeline recovers the truth) lives in tests/, where pytest can gate on it.
"""

from .version_compare import (
    VersionComparison,
    compare_policies,
    compare_v1_to_v2,
    to_frame,
)

__all__ = ["VersionComparison", "compare_policies", "compare_v1_to_v2", "to_frame"]
