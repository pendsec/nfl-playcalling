"""
Sensitivity analysis — bounding unobserved confounding (V2 headline "Add").

The observed adjustment set leaves residual confounding through `coach_read` (the
unobserved read). We bound its impact with a Rosenbaum / marginal-sensitivity
model: an unobserved confounder could tilt each play's TRUE propensity away from
the estimated pi_b by at most an odds ratio Gamma, so the true importance weight
lies in a band around the estimated one. We then take the worst case over that
band for the doubly-robust value.

Concretely, only the IPW correction term of V_DR is confounding-sensitive (the
Direct-Method term is held at the observed-data fit). The correction is a
self-normalized weighted average of residuals g_i = r_i - q(s_i,a_i); under the
MSM each weight may be multiplied by lambda_i in [1/Gamma, Gamma]. The extreme of
a self-normalized weighted average is a threshold rule on the sorted residuals,
solved here in closed form via prefix sums.

This is NOT the reference's `log(Gamma)*se` shortcut (which conflates sampling
error with confounding bias). At Gamma=1 the band collapses and the bound equals
the point DR estimate; as Gamma grows the interval widens monotonically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _msm_correction_bound(g: np.ndarray, w: np.ndarray, gamma: float, want_max: bool) -> float:
    """Worst-case self-normalized weighted mean of g under weights w*[1/G, G].

    Minimizing puts the heavy factor (Gamma) on the smallest residuals and the
    light factor (1/Gamma) on the largest; maximizing swaps them. The optimum
    over lambda_i in [1/Gamma, Gamma] is a threshold on sorted g (linear-
    fractional program), so we scan the n+1 breakpoints.
    """
    order = np.argsort(g)
    g = g[order]
    w = w[order]
    hi, lo = gamma, 1.0 / gamma

    wg = w * g
    # Prefix sums: first k plays get one factor, the rest the other.
    pre_w = np.concatenate([[0.0], np.cumsum(w)])
    pre_wg = np.concatenate([[0.0], np.cumsum(wg)])
    tot_w, tot_wg = pre_w[-1], pre_wg[-1]

    # To MINIMIZE: heavy factor (hi) on the smallest-g prefix, light (lo) on rest.
    # To MAXIMIZE: light factor (lo) on the smallest-g prefix, heavy (hi) on rest.
    a, b = (lo, hi) if want_max else (hi, lo)
    num = a * pre_wg + b * (tot_wg - pre_wg)
    den = a * pre_w + b * (tot_w - pre_w)
    vals = num / den
    return float(vals.max() if want_max else vals.min())


def sensitivity_bounds(
    q_all: np.ndarray,
    pi_b: np.ndarray,
    policy_probs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    gamma_range: list[float],
    weight_clip: float = 20.0,
    compare_value: float | None = None,
) -> pd.DataFrame:
    """DR value bounds for a policy over a range of confounding strengths Gamma.

    If `compare_value` (e.g. V(pi_b)) is given, each row flags whether the policy
    still beats it under that Gamma — i.e. whether the improvement is robust.
    """
    n = len(rewards)
    idx = np.arange(n)
    actions = np.asarray(actions)

    dm_mean = float((policy_probs * q_all).sum(axis=1).mean())
    w = np.clip(policy_probs[idx, actions] / pi_b[idx, actions], 0.0, weight_clip)
    g = rewards - q_all[idx, actions]

    rows = []
    for gamma in gamma_range:
        if gamma <= 1.0:
            corr = float(np.average(g, weights=w)) if w.sum() > 0 else 0.0
            lo = hi = dm_mean + corr
        else:
            lo = dm_mean + _msm_correction_bound(g, w, gamma, want_max=False)
            hi = dm_mean + _msm_correction_bound(g, w, gamma, want_max=True)
        row = {"gamma": gamma, "v_lower": lo, "v_upper": hi, "width": hi - lo}
        if compare_value is not None:
            row["beats_comparison"] = bool(lo > compare_value)
        rows.append(row)
    return pd.DataFrame(rows)
