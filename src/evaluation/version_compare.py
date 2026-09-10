"""
Version-over-version policy comparison — the regression gate (Step 8).

CLAUDE.md's modeling discipline: *"Every new model version evaluated against the
previous via OPE on a held-out season. Regressions block merge."* This module is
that gate. V2's conservative policy has to earn its place against V1's
behavior-constrained greedy policy on the held-out season, judged by the same
doubly-robust estimator that judges everything else.

Two things make the comparison honest:

  * **Paired scoring.** Both policies are valued on the *same* holdout plays with
    the same fitted Q and pi_b, so we compare per-play DR scores pairwise. The
    standard error of the paired difference is much smaller than what you would
    get by differencing two independent CIs — a real regression is detectable
    even though each policy's own CI is wide on this sparse single-team data.

  * **An explicit non-inferiority margin, not a point comparison.** With noisy
    holdout data the new policy will essentially never tie exactly. The gate asks
    whether the new version is *significantly worse* — the upper end of the
    difference's CI falling below `-margin` — rather than whether it happened to
    score lower. That fails on genuine regressions without blocking on noise.

The gate is deliberately asymmetric: failing to prove an *improvement* is fine
and expected on this data (V2's honest finding is a wide, confounding-sensitive
band). Proving a *deterioration* is what blocks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..schemas.dataset import Dataset
from ..schemas.behavior import BehaviorModel
from ..schemas.outcome import QModel
from ..schemas.ope import OPEResult
from ..ope.doubly_robust import dr_per_play, dr_policy_value


@dataclass
class VersionComparison:
    """Result of scoring a candidate policy against the previous version's."""
    baseline_name: str
    candidate_name: str
    baseline: OPEResult
    candidate: OPEResult
    diff: float               # mean paired DR difference (candidate - baseline)
    diff_se: float            # standard error of the paired difference
    margin: float             # non-inferiority margin used by the gate
    n: int

    @property
    def diff_ci95(self) -> tuple[float, float]:
        return (self.diff - 1.96 * self.diff_se, self.diff + 1.96 * self.diff_se)

    @property
    def regressed(self) -> bool:
        """True when the candidate is *significantly worse* than the baseline.

        Significantly worse = the whole 95% CI of the paired difference sits
        below the negative margin. A merely-lower point estimate with a CI that
        still touches -margin is noise, not a regression.
        """
        return self.diff_ci95[1] < -self.margin

    @property
    def improved(self) -> bool:
        """True when the candidate is significantly better (CI entirely > 0)."""
        return self.diff_ci95[0] > 0.0

    @property
    def verdict(self) -> str:
        if self.regressed:
            return "REGRESSION"
        if self.improved:
            return "IMPROVED"
        return "NO CHANGE DETECTED"

    def to_record(self) -> dict:
        lo, hi = self.diff_ci95
        return {
            "baseline": self.baseline_name,
            "candidate": self.candidate_name,
            "v_baseline": self.baseline.value,
            "v_candidate": self.candidate.value,
            "paired_diff": self.diff,
            "paired_diff_se": self.diff_se,
            "diff_ci_lo": lo,
            "diff_ci_hi": hi,
            "margin": self.margin,
            "n": self.n,
            "verdict": self.verdict,
            "blocks_merge": self.regressed,
            "graph_version": self.candidate.graph_version,
        }

    def report(self) -> str:
        """Human-readable block for the run log."""
        lo, hi = self.diff_ci95
        gate = "BLOCKS MERGE" if self.regressed else "gate passes"
        w = max(len(self.baseline_name), len(self.candidate_name))
        return (
            f"  V_DR({self.baseline_name:<{w}}) = {self.baseline.value:+.4f}\n"
            f"  V_DR({self.candidate_name:<{w}}) = {self.candidate.value:+.4f}\n"
            f"  paired difference     = {self.diff:+.4f}  "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]  (n={self.n})\n"
            f"  non-inferiority margin= {self.margin:.4f}\n"
            f"  verdict: {self.verdict}  ->  {gate}"
        )


def compare_policies(
    ds: Dataset,
    q: QModel,
    beh: BehaviorModel,
    baseline_probs: np.ndarray,
    candidate_probs: np.ndarray,
    weight_clip: float = 20.0,
    margin: float = 0.02,
    baseline_name: str = "v1_greedy",
    candidate_name: str = "v2_conservative",
) -> VersionComparison:
    """Score two policies by DR on the same holdout rows and gate on the diff.

    `ds` must be the held-out split (a season the models never saw) — passing
    training rows would compare the two policies on data they were fit to, which
    is exactly the self-congratulation this gate exists to prevent.

    `margin` is in reward (−EPA) units per play: differences smaller than this
    are treated as not worth blocking on regardless of significance.
    """
    df = ds.df
    actions = df[ds.action_col].to_numpy()
    rewards = df[ds.reward_col].to_numpy()
    q_all = q.predict_all_actions(df)
    pi_b = beh.propensity(df)

    base_scores = dr_per_play(q_all, pi_b, baseline_probs, actions, rewards,
                              weight_clip=weight_clip)
    cand_scores = dr_per_play(q_all, pi_b, candidate_probs, actions, rewards,
                              weight_clip=weight_clip)

    # Paired difference: same play, same Q, same propensities on both sides, so
    # the shared estimation noise cancels instead of adding.
    delta = cand_scores - base_scores
    n = len(delta)
    diff_se = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")

    return VersionComparison(
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        baseline=dr_policy_value(q_all, pi_b, baseline_probs, actions, rewards,
                                 weight_clip=weight_clip, policy_name=baseline_name),
        candidate=dr_policy_value(q_all, pi_b, candidate_probs, actions, rewards,
                                  weight_clip=weight_clip, policy_name=candidate_name),
        diff=float(delta.mean()),
        diff_se=diff_se,
        margin=margin,
        n=n,
    )


def compare_v1_to_v2(
    test: Dataset,
    q: QModel,
    beh: BehaviorModel,
    cfg: dict,
) -> VersionComparison:
    """The concrete V1 -> V2 gate: greedy (V1) vs conservative (V2) on holdout.

    Both policies are built from the *same* V2 models — the comparison isolates
    the decision rule, which is the thing V2 changed. Comparing against a
    separately-refit V1 stack would confound the policy change with the model
    change and tell us nothing about either.
    """
    from ..policy.greedy import learn_greedy_policy
    from ..policy.conservative import learn_conservative_policy

    v1 = learn_greedy_policy(test, q, beh, cfg)
    v2 = learn_conservative_policy(test, q, beh, cfg)

    ecfg = cfg.get("evaluation", {})
    return compare_policies(
        test, q, beh,
        baseline_probs=v1.act(test.df)["policy_probs"],
        candidate_probs=v2.act(test.df)["policy_probs"],
        weight_clip=cfg["ope"].get("weight_clip", 20.0),
        margin=ecfg.get("regression_margin", 0.02),
        baseline_name=ecfg.get("baseline_name", "v1_greedy"),
        candidate_name=ecfg.get("candidate_name", "v2_conservative"),
    )


def to_frame(comparisons: list[VersionComparison]) -> pd.DataFrame:
    """One row per comparison, for `outputs/version_comparison.csv`."""
    return pd.DataFrame([c.to_record() for c in comparisons])
