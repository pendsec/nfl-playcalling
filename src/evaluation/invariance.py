"""
Invariance — do the call's effects differ across environments?

An environment is a team-season (`invariance.environment`), 96 of them in the
training seasons. The pooled Q assumes one model of how the call changes the
outcome serves every defense in every season, with the state and team features
carrying the differences. This module tests that assumption before anything is
built to enforce it.

The effects tested are the call's components, not its 12 cells: one team-
season holds ~1,100 snaps, which estimates "base instead of nickel" but not
"base/8+ instead of nickel/6". A component contrast compares two policies that
differ only on that axis and keep the coordinator's choice on the other:

    pi_c(a | s) = pi_b(a | s) 1{a's component = c} / pi_b(c | s)

against the most common component (nickel; a 6-man box). Both terms carry the
coordinator's mix on the other axis, so a team that rarely plays dime moves
`pi_b(dime | s)`, not the contrast's meaning.

Two readings per contrast, across environments:

  * modeled:   the contrast as the pooled Q reads it on each environment's
               states. It varies when teams face different states — a base
               package is worth more to a defense that sees more short
               yardage — and that variation the pooled Q already carries.
  * unmodeled: Q's miss on the snaps that took the component minus its miss on
               the snaps that took the reference, per environment. If Q's
               effects transfer, the two misses are alike in every
               environment up to noise; a spread across environments is effect
               variation Q does not carry. This is the decision.

The unmodeled reading uses Q's residuals on the calls actually made, not a DR
contrast per environment: one team-season calls a component a few dozen to a
few hundred times, and inverse-propensity weights on samples that small have
standard errors that understate the noise ~2x (on the known SCM with no
heterogeneity at all, every front contrast read as "varies"). The residuals
carry no weights. The pooled contrast — one number over all environments, where
DR's weights are well behaved — is reported beside it by `pooled_effects`.

Heterogeneity is Cochran's Q, calibrated by permuting games across
environments (see `invariance_test`), with the DerSimonian-Laird between-
environment sd tau beside it, because significance alone is not the question:
at 96 environments a real but negligible spread can be "significant". An
invariance penalty is called for only when the residual spread is both
significant (Bonferroni over the contrasts) and at least `invariance.min_tau` —
by default the regression margin, the smallest per-snap difference the project
acts on.

Standard errors are clustered by game: plays in one game share an opponent, a
script and a scoreboard.

The nuisances must be out of fold for every snap scored — see
`evaluation.transfer.crossfit_nuisances` — or the residual measures Q's
in-sample fit, not its transfer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..data.features import LEVEL1_AXES, level1_axes, level1_components
from ..ope.doubly_robust import dr_per_play


def environment_ids(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """One label per row: `defteam_season` (team_season), `defteam`, or `season`."""
    kind = cfg.get("invariance", {}).get("environment", "team_season")
    if kind == "team_season":
        return df["defteam"].astype(str) + "_" + df["season"].astype(str)
    if kind in ("team", "defteam"):
        return df["defteam"].astype(str)
    if kind == "season":
        return df["season"].astype(str)
    raise ValueError(f"invariance.environment must be team_season | team | season, got {kind!r}")


def component_policy(pi_b: np.ndarray, comp_of_action: np.ndarray, c: int) -> np.ndarray:
    """pi_b restricted to the actions whose component is `c`, renormalized."""
    mask = (comp_of_action == c).astype(float)[None, :]
    p = pi_b * mask
    return p / p.sum(axis=1, keepdims=True)


def clustered_mean_se(x: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    """Mean of `x` and its standard error with observations clustered."""
    x = np.asarray(x, float)
    mean = float(x.mean())
    _, inv = np.unique(clusters, return_inverse=True)
    g = np.bincount(inv, weights=x - mean)
    n_g = len(g)
    if n_g < 2:
        return mean, float("nan")
    se = np.sqrt((g ** 2).sum() * n_g / (n_g - 1)) / len(x)
    return mean, float(se)


def reference_components(actions: np.ndarray, cfg: dict) -> dict[str, int]:
    """The most common value of each configured axis — the contrasts' baseline."""
    comps = level1_components(actions, cfg)
    return {axis: int(np.bincount(v).argmax()) for axis, v in comps.items()}


def pooled_effects(q_all: np.ndarray, pi_b: np.ndarray, actions: np.ndarray,
                   rewards: np.ndarray, games: np.ndarray, cfg: dict,
                   reference: dict[str, int] | None = None) -> pd.DataFrame:
    """Each component contrast by paired DR over all environments together."""
    clip = cfg["ope"].get("weight_clip", 20.0)
    k = q_all.shape[1]
    reference = reference or reference_components(actions, cfg)
    comp_of_action = level1_components(np.arange(k), cfg)
    rows = []
    for axis in level1_axes(cfg):
        names = LEVEL1_AXES[axis]
        ref = reference[axis]
        pol_ref = component_policy(pi_b, comp_of_action[axis], ref)
        base = dr_per_play(q_all, pi_b, pol_ref, actions, rewards, clip)
        for c in range(len(names)):
            if c == ref:
                continue
            pol_c = component_policy(pi_b, comp_of_action[axis], c)
            diff = dr_per_play(q_all, pi_b, pol_c, actions, rewards, clip) - base
            est, se = clustered_mean_se(diff, games)
            rows.append({"axis": axis, "component": names[c], "reference": names[ref],
                         "effect": est, "effect_se": se})
    return pd.DataFrame(rows)


def heterogeneity(est: np.ndarray, se: np.ndarray) -> dict:
    """Cochran's Q, I^2 and the DerSimonian-Laird tau across groups."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[ok], se[ok]
    k = len(est)
    if k < 2:
        return {"k": k, "pooled": float("nan"), "pooled_se": float("nan"), "q": float("nan"),
                "p": float("nan"), "i2": float("nan"), "tau": float("nan")}
    w = 1.0 / se ** 2
    pooled = float((w * est).sum() / w.sum())
    q = float((w * (est - pooled) ** 2).sum())
    df = k - 1
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - df) / c)
    return {"k": k, "pooled": pooled, "pooled_se": float(np.sqrt(1.0 / w.sum())),
            "q": q, "p": float(stats.chi2.sf(q, df)),
            "i2": max(0.0, (q - df) / q) if q > 0 else 0.0, "tau": float(np.sqrt(tau2))}


# ── The residual contrast, per environment ──────────────────────────────────
# Everything is aggregated to clusters first — one cluster is one defense's
# side of one game, the unit the permutation moves — so a permutation is a
# relabelling of clusters and costs a few bincounts.
def cluster_ids(df: pd.DataFrame) -> np.ndarray:
    """One id per (game, defense): a game holds both teams' defensive snaps."""
    return pd.factorize(df["game_id"].astype(str) + "|" + df["defteam"].astype(str))[0]


def _cluster_sums(miss, in_c, in_ref, cl, n_cl):
    return (np.bincount(cl, miss * in_c, n_cl), np.bincount(cl, in_c, n_cl),
            np.bincount(cl, miss * in_ref, n_cl), np.bincount(cl, in_ref, n_cl))


def _env_residuals(sums, env_of_cl, n_env, min_taken):
    """Residual contrast per environment, its clustered se, and testability."""
    s_c, n_c, s_r, n_r = sums
    tot = [np.bincount(env_of_cl, v, n_env) for v in sums]
    with np.errstate(divide="ignore", invalid="ignore"):
        m_c, m_r = tot[0] / tot[1], tot[2] / tot[3]
        psi = ((s_c - m_c[env_of_cl] * n_c) / tot[1][env_of_cl]
               - (s_r - m_r[env_of_cl] * n_r) / tot[3][env_of_cl])
        g = np.bincount(env_of_cl, minlength=n_env).astype(float)
        se = np.sqrt(np.bincount(env_of_cl, psi ** 2, n_env) * g / (g - 1))
    ok = (tot[1] >= min_taken) & (tot[3] >= min_taken) & (g > 1)
    return m_c - m_r, se, ok, tot[1], tot[3]


def _cochran_q(est, se) -> tuple[float, int]:
    w = 1.0 / se ** 2
    pooled = (w * est).sum() / w.sum()
    return float((w * (est - pooled) ** 2).sum()), len(est)


def invariance_test(q_all: np.ndarray, pi_b: np.ndarray, actions: np.ndarray,
                    rewards: np.ndarray, env: np.ndarray, clusters: np.ndarray, cfg: dict,
                    reference: dict[str, int] | None = None,
                    effects: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Per contrast: how much Q says the effect varies, and how much it misses.

    Returns (summary, per-environment table, penalty needed).

    The null is calibrated by permutation, not by the chi-square: clusters are
    reassigned across environments at random (`invariance.n_permutations`
    times), which keeps every environment's size and breaks any link between
    environment and residual. The clustered standard errors are too small when
    an environment holds ~17 clusters — on the known SCM with environments
    assigned at random, 45% of contrasts were "significant" at 5% by the
    chi-square — so

      * the p-value is the share of permutations whose Cochran's Q reaches the
        observed one;
      * `phi`, the permutations' mean Q over its degrees of freedom, is how much
        the standard errors understate the noise; tau is the DerSimonian-Laird
        spread with the standard errors scaled up by sqrt(phi).

    `modeled_sd` is the spread of Q's own contrast across environments (state
    mix and team features). A penalty is needed when any contrast's residual
    spread has p < `alpha / n_contrasts` AND `unmodeled_tau >= invariance.min_tau`.
    """
    icfg = cfg.get("invariance", {})
    alpha = icfg.get("alpha", 0.05)
    min_tau = icfg.get("min_tau", cfg.get("evaluation", {}).get("regression_margin", 0.02))
    min_taken = icfg.get("min_taken", 30)
    n_perm = icfg.get("n_permutations", 500)
    rng = np.random.default_rng(icfg.get("seed", cfg.get("seed", 0)))

    actions = np.asarray(actions)
    k = q_all.shape[1]
    env_labels, env_idx = np.unique(np.asarray(env), return_inverse=True)
    n_env = len(env_labels)
    cl = np.asarray(clusters)
    n_cl = cl.max() + 1
    env_of_cl = np.zeros(n_cl, int)
    env_of_cl[cl] = env_idx
    if np.any(np.bincount(cl, env_idx, n_cl) != env_of_cl * np.bincount(cl, minlength=n_cl)):
        raise ValueError("every cluster must sit inside one environment")

    reference = reference or reference_components(actions, cfg)
    comp_of_action = level1_components(np.arange(k), cfg)
    taken = level1_components(actions, cfg)
    miss = np.asarray(rewards) - q_all[np.arange(len(actions)), actions]
    perms = [rng.permutation(env_of_cl) for _ in range(n_perm)]

    groups = []
    for axis in level1_axes(cfg):
        ref = reference[axis]
        groups += [(axis, c, ref) for c in range(len(LEVEL1_AXES[axis])) if c != ref]
    level = alpha / max(len(groups), 1)

    rows, per_env = [], []
    for axis, c, ref in groups:
        names = LEVEL1_AXES[axis]
        dm_play = ((component_policy(pi_b, comp_of_action[axis], c)
                    - component_policy(pi_b, comp_of_action[axis], ref)) * q_all).sum(axis=1)
        dm_env = np.bincount(env_idx, dm_play, n_env) / np.bincount(env_idx, minlength=n_env)
        sums = _cluster_sums(miss, taken[axis] == c, taken[axis] == ref, cl, n_cl)
        est, se, ok, n_c, n_r = _env_residuals(sums, env_of_cl, n_env, min_taken)
        q_obs, k_obs = _cochran_q(est[ok], se[ok])
        q_null, ratio = [], []
        for perm in perms:
            e_p, s_p, ok_p, _, _ = _env_residuals(sums, perm, n_env, min_taken)
            q_p, k_p = _cochran_q(e_p[ok_p], s_p[ok_p])
            q_null.append(q_p)
            ratio.append(q_p / max(k_p - 1, 1))
        phi = max(1.0, float(np.mean(ratio)))
        het = heterogeneity(est[ok], se[ok] * np.sqrt(phi))
        p = (1 + int((np.asarray(q_null) >= q_obs).sum())) / (1 + n_perm)
        varies = bool(p < level and het["tau"] >= min_tau)
        contrast = f"{names[c]} vs {names[ref]}"
        rows.append({
            "axis": axis, "contrast": contrast,
            "n_env": int(ok.sum()), "n_env_untestable": int((~ok).sum()),
            "modeled_mean": float(dm_env[ok].mean()), "modeled_sd": float(dm_env[ok].std(ddof=1)),
            "resid_mean": het["pooled"], "unmodeled_tau": het["tau"], "phi": phi,
            "q": q_obs, "p": p, "verdict": "varies" if varies else "invariant",
        })
        per_env.append(pd.DataFrame({
            "axis": axis, "contrast": contrast, "environment": env_labels,
            "n_taken": n_c.astype(int), "n_taken_ref": n_r.astype(int), "testable": ok,
            "modeled": dm_env, "resid": np.where(ok, est, np.nan),
            "resid_se": np.where(ok, se * np.sqrt(phi), np.nan)}))
    table = pd.DataFrame(rows)
    if effects is not None and len(table):
        eff = effects.assign(contrast=effects["component"] + " vs " + effects["reference"])
        table = table.merge(eff[["axis", "contrast", "effect", "effect_se"]],
                            on=["axis", "contrast"], how="left")
    return table, pd.concat(per_env, ignore_index=True), bool((table["verdict"] == "varies").any())
