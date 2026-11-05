"""
Outcome-model benchmark — which Q serves the policy best?

Run from the repository root:

    python -m experiments.outcome_models.benchmark --source nflfastR --n-boot 10
    python -m experiments.outcome_models.benchmark --source synthetic [--conf-strength 0]

Results land in `results/` beside this file (tracked in git, unlike outputs/).

Every candidate is fit on the training seasons and judged on the held-out one
by what the policy uses: each call's ADVANTAGE over the coordinator's mix,

    A(s, a) = Q(s, a) - sum_b pi_b(b | s) Q(s, b),

on the calls the policy may choose (pi_b >= the support threshold). Q's level,
and how well it predicts the reward, barely matter: the state moves EPA by ~1.5
per snap, a call by ~0.05.

Known truth (the synthetic league, `true_alignment_q`):
  * adv_rmse     — error of the advantages on supported calls
  * adv_corr     — their correlation with the true advantages
  * spread_ratio — sd of the estimated advantages / sd of the true ones; >1 is
                   the noise the policy's argmax chases (the optimizer's curse)
  * true lift of the supported argmax (capture = share of the oracle's lift) and
    of the safe policy built on the candidate, both over the coordinator's mix
    (pi_b) in truth — the same baseline the DR lifts use.

No truth (real data) — the same quantities against doubly-robust
pseudo-outcomes built on the HOLDOUT with nuisances cross-fitted by game there,
so the judge never saw the candidates' training data:

    G_a(s) = mu(s, a) + 1{A = a} / pi(a | s) * (R - mu(s, A))
    A_G(s, a) = G_a(s) - sum_b pi_b(b | s) G_b(s)

  * dr_loss  — mean (A_hat - A_G)^2 over supported calls, minus the baseline's.
               A_G is unbiased for the true advantage, so differences in this
               loss estimate differences in true error (the noise term is
               common to every candidate).
  * cal_slope — slope of A_G on A_hat: 1 = the advantages are on the right
               scale, <1 = overstated, ~0 = no signal.
  * dr_lift of the supported argmax and of the safe policy, by the pipeline's
    own judge, paired against the coordinator, game-clustered.

The pseudo-outcome criteria run on the synthetic league too, beside the truth,
so the real-data ranking can be checked against one where the answer is known.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Callable

import numpy as np
import pandas as pd

from src.evaluation.invariance import clustered_mean_se
from src.models.behavior.propensity import fit_behavior_model
from src.models.outcome.q_model import (_check_conditions_on_adjustment_set, fit_q_ensemble,
                                        fit_q_model)
from src.ope.doubly_robust import dr_per_play
from src.policy.safe import learn_policy
from src.schemas.dataset import Dataset

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# name -> outcome_model overrides on top of the config's (pooled GBM) block.
CANDIDATES: dict[str, dict] = {
    "gbm_pooled": {},
    "dr_learner_ridge": {"type": "dr_learner", "final": {"type": "ridge"}},
    "dr_learner_gbm": {"type": "dr_learner", "final": {"type": "lightgbm"}},
    "r_learner_forest": {"type": "r_learner"},
    "tarnet": {"type": "tarnet", "neural": {"heads": "separate"}},
    "tarnet_factored": {"type": "tarnet", "neural": {"heads": "factored"}},
    "dragonnet": {"type": "dragonnet", "neural": {"heads": "separate"}},
    "dragonnet_factored": {"type": "dragonnet", "neural": {"heads": "factored"}},
}
CONTRAST_LEARNERS = ("dr_learner", "r_learner")


def fit_candidate(ds: Dataset, cfg: dict):
    """Any candidate's Q: the experiment-only contrast learners (`cate.py`) here,
    everything else through the pipeline's own `fit_q_model`."""
    if cfg["outcome_model"].get("type") in CONTRAST_LEARNERS:
        from .cate import fit_contrast_q_model
        _check_conditions_on_adjustment_set(ds, cfg)
        return fit_contrast_q_model(ds, cfg)
    return fit_q_model(ds, cfg)


def candidate_config(cfg: dict, name: str, overrides: dict | None = None) -> dict:
    c = copy.deepcopy(cfg)
    c["outcome_model"].update(copy.deepcopy(overrides if overrides is not None
                                            else CANDIDATES[name]))
    return c


def advantages(q: np.ndarray, pi_b: np.ndarray) -> np.ndarray:
    return q - (pi_b * q).sum(axis=1, keepdims=True)


def supported_argmax(q: np.ndarray, support: np.ndarray) -> np.ndarray:
    return np.where(support, q, -np.inf).argmax(axis=1)


def pseudo_outcomes(mu: np.ndarray, pi: np.ndarray, a: np.ndarray, r: np.ndarray,
                    clip: tuple[float, float], weight_clip: float) -> np.ndarray:
    """Doubly-robust G_a(s) for every call, (n, n_actions)."""
    n = len(a)
    w = np.minimum(1.0 / np.clip(pi[np.arange(n), a], *clip), weight_clip)
    g = mu.copy()
    g[np.arange(n), a] += w * (r - mu[np.arange(n), a])
    return g


def calibration_slope(a_hat: np.ndarray, target: np.ndarray, clusters: np.ndarray
                      ) -> tuple[float, float]:
    """OLS slope of target on a_hat, with a cluster-robust standard error."""
    x = a_hat - a_hat.mean()
    sxx = float((x ** 2).sum())
    if sxx <= 0:
        return np.nan, np.nan
    beta = float((x * (target - target.mean())).sum() / sxx)
    resid = target - target.mean() - beta * x
    s = pd.Series(x * resid).groupby(clusters).sum().to_numpy()
    return beta, float(np.sqrt((s ** 2).sum()) / sxx)


def _paired_lift(q_judge, pi_judge, probs, a, r, clip, games):
    base = dr_per_play(q_judge, pi_judge, pi_judge, a, r, clip)
    return clustered_mean_se(dr_per_play(q_judge, pi_judge, probs, a, r, clip) - base, games)


def benchmark(train: Dataset, test: Dataset, cfg: dict, candidates: list[str] | None = None,
              q_true: np.ndarray | None = None, n_boot: int | None = None,
              progress: Callable[[str], None] | None = None,
              checkpoint: str | None = None) -> pd.DataFrame:
    """One row per candidate. `q_true` (test rows x calls) switches on the truth
    columns; `checkpoint` is a CSV rewritten after every candidate."""
    from src.evaluation.transfer import crossfit_nuisances

    say = progress or (lambda m: None)
    candidates = candidates or list(CANDIDATES)
    clip, wclip = cfg["ope"]["clip_propensity"], cfg["ope"].get("weight_clip", 20.0)
    thr = cfg["policy"].get("support_threshold", 0.05)
    df = test.df
    a, r = df[test.action_col].to_numpy(), df[test.reward_col].to_numpy()
    games, n, rows_ = df["game_id"].to_numpy(), len(df), np.arange(len(df))

    # Shared across candidates: the policy's pi_b (fit on train), the pipeline's
    # judge (train-fit GBM Q + pi_b), and the holdout-cross-fitted pseudo-outcomes.
    t0 = time.time()
    beh = fit_behavior_model(train, cfg, diagnose=False)[0]
    pi_b = beh.propensity(df)
    support = pi_b >= thr
    q_judge = fit_q_model(train, cfg).predict_all_actions(df)
    hold = crossfit_nuisances(test, cfg)
    target = advantages(pseudo_outcomes(hold["q_all"], hold["pi_b"], a, r, clip, wclip), pi_b)
    cell_games = np.broadcast_to(games[:, None], support.shape)[support]
    say(f"shared nuisances {time.time() - t0:.0f}s")
    if q_true is not None:
        adv_true = advantages(q_true, pi_b)
        v_mix = (pi_b * q_true).sum(1).mean()      # the coordinator's mix, in truth
        oracle = q_true[rows_, supported_argmax(q_true, support)].mean() - v_mix

    out, base_loss = [], None
    for name in candidates:
        c = candidate_config(cfg, name)
        if n_boot is not None:
            c["outcome_model"].setdefault("ensemble", {})["n_boot"] = n_boot
        t = time.time()
        q = fit_candidate(train, c)
        fit_s = time.time() - t
        q_hat = q.predict_all_actions(df)
        adv = advantages(q_hat, pi_b)
        greedy = supported_argmax(q_hat, support)
        loss_rows = np.where(support, (adv - target) ** 2, 0.0).sum(1) / support.sum(1)
        if base_loss is None:
            base_loss = loss_rows
        d_loss, d_loss_se = clustered_mean_se(loss_rows - base_loss, games)
        slope, slope_se = calibration_slope(adv[support], target[support], cell_games)
        g_lift, g_se = _paired_lift(q_judge, pi_b, np.eye(test.n_actions)[greedy], a, r,
                                    wclip, games)
        row = {"model": name, "fit_s": fit_s,
               "rmse_factual": float(np.sqrt(np.mean((q_hat[rows_, a] - r) ** 2))),
               "adv_sd": float(adv[support].std()),
               "dr_loss_vs_base": d_loss, "dr_loss_vs_base_se": d_loss_se,
               "cal_slope": slope, "cal_slope_se": slope_se,
               "greedy_departs": float((greedy != a).mean()),
               "greedy_dr_lift": g_lift, "greedy_dr_lift_se": g_se}

        t = time.time()
        ens = (fit_q_ensemble(train, c, fit=fit_candidate)
               if c["policy"].get("type") == "safe" else None)
        pol = learn_policy(train, q, beh, c, ensemble=ens).act(df)
        probs = pol["policy_probs"]
        s_lift, s_se = _paired_lift(q_judge, pi_b, probs, a, r, wclip, games)
        departs = ~pol["deferred"] if "deferred" in pol else np.ones(n, bool)
        row.update({"ensemble_s": time.time() - t, "safe_departs": float(departs.mean()),
                    "safe_dr_lift": s_lift, "safe_dr_lift_se": s_se})

        if q_true is not None:
            at = adv_true[support]
            row.update({
                "adv_rmse": float(np.sqrt(np.mean((adv[support] - at) ** 2))),
                "adv_corr": float(np.corrcoef(adv[support], at)[0, 1]),
                "spread_ratio": float(adv[support].std() / at.std()),
                "greedy_true_lift": float(q_true[rows_, greedy].mean() - v_mix),
                "oracle_true_lift": float(oracle),
                "safe_true_lift": float((probs * q_true).sum(1).mean() - v_mix),
            })
            row["greedy_capture"] = row["greedy_true_lift"] / row["oracle_true_lift"]
        out.append(row)
        if checkpoint:
            pd.DataFrame(out).to_csv(checkpoint, index=False)
        say(f"{name}: fit {fit_s:.0f}s, ensemble+policy {row['ensemble_s']:.0f}s")
    return pd.DataFrame(out)


def _load(cfg: dict, source: str, n_plays: int, conf_strength: float | None):
    """(train, test, q_true or None) for the configured split."""
    from src.data.features import build_dataset, time_aware_split
    from src.data.load import (generate_synthetic_alignment, load_ftn, load_pbp, merge_ftn,
                               true_alignment_q)

    if source == "synthetic":
        kw = {} if conf_strength is None else {"conf_strength": conf_strength}
        pbp = generate_synthetic_alignment(n_plays, cfg["synthetic"]["seed"], n_teams=32,
                                           seasons=cfg["data"]["seasons"], **kw)
    else:
        seasons, cache = cfg["data"]["seasons"], cfg["data"]["cache_dir"]
        pbp = merge_ftn(load_pbp(seasons, cache), load_ftn(seasons, cache))
    ds = build_dataset(pbp, cfg)
    train, test = time_aware_split(ds, cfg["data"]["holdout_season"])
    q_true = true_alignment_q(test.df) if source == "synthetic" else None
    return train, test, q_true


if __name__ == "__main__":
    import argparse
    import logging
    import warnings

    import yaml

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    p = argparse.ArgumentParser(description="Benchmark outcome models for the policy.")
    p.add_argument("--config", default="configs/v3.yaml")
    p.add_argument("--source", choices=["nflfastR", "synthetic"], default="nflfastR")
    p.add_argument("--n-plays", type=int, default=140_000, help="synthetic league size")
    p.add_argument("--conf-strength", type=float, default=None,
                   help="synthetic: the hidden read's effect on the reward (0 = none)")
    p.add_argument("--n-boot", type=int, default=None, help="ensemble size for the safe policy")
    p.add_argument("--models", nargs="*", default=None, help=f"subset of {list(CANDIDATES)}")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    t0 = time.time()
    log = lambda m: print(f"  [{time.time() - t0:6.0f}s] {m}", flush=True)
    train, test, q_true = _load(cfg, args.source, args.n_plays, args.conf_strength)
    log(f"{args.source}: train {len(train.df):,}, test {len(test.df):,}")
    tag = ("real_2025" if args.source == "nflfastR" else
           "synthetic_with_read" if args.conf_strength is None else
           "synthetic_no_read" if args.conf_strength == 0 else
           f"synthetic_conf{args.conf_strength:g}")
    out = args.out or os.path.join(RESULTS, f"{tag}.csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    table = benchmark(train, test, cfg, args.models, q_true, args.n_boot, progress=log,
                      checkpoint=out)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table.round(4).to_string(index=False))
    print(f"saved {out}")
