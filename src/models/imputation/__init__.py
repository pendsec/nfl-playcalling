"""Latent-treatment imputation (V3).

The coverage shell is charted on ~94% of dropbacks and ~3% of runs, because
NGS records the coverage as PLAYED and a run never develops one. The shell was
still CALLED on those snaps — it is latent, not absent — so this package treats
it as a missing treatment label to be imputed with its uncertainty carried
forward, rather than a row to be deleted.

    features.imputer_features      — the declared feature set (single source of truth)
    features.build_imputation_frame— raw FTN-merged pbp -> those features
    shell.fit_shell_imputer        — calibrated P(shell | pre-snap state) on charted plays
    shell.ShellImputer.draw        — weighted sampling: m draws from that posterior
    pooling.pool_rubin             — Rubin's rules, so m draws become one estimate whose
                                     interval reflects what the imputation does NOT know

Read `pooling.RubinResult.fraction_missing_information` before trusting anything
downstream: it reports how much of the final variance came from not knowing the
label, and on this data it is large (~43% on SF run plays).
"""

from .features import (FORBIDDEN_FEATURES, IMPUTER_CATEGORICAL, IMPUTER_DEFENSE,
                       IMPUTER_OFFENSE, IMPUTER_SITUATION, add_team_shell_tendency,
                       assert_imputer_features, build_imputation_frame,
                       imputer_features, tendency_columns)
from .pooling import RubinResult, pool_rubin
from .shell import ShellImputer, fit_shell_imputer

__all__ = [
    "FORBIDDEN_FEATURES", "IMPUTER_CATEGORICAL", "IMPUTER_DEFENSE",
    "IMPUTER_OFFENSE", "IMPUTER_SITUATION", "add_team_shell_tendency",
    "assert_imputer_features", "build_imputation_frame", "imputer_features",
    "tendency_columns", "RubinResult", "pool_rubin", "ShellImputer",
    "fit_shell_imputer",
]
