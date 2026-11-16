"""
Behavior-model calibration — which pi_b gives the weights IPW and DR can trust?

Run from the repository root:

    python -m experiments.behavior_calibration.compare --source nflfastR --fold tune
    python -m experiments.behavior_calibration.compare --source nflfastR --fold holdout
    python -m experiments.behavior_calibration.compare --source synthetic --conf-strength 0 --fold holdout

`--fold tune` fits on 2022-23 and scores 2024 (where the choice is made);
`--fold holdout` fits on 2022-24 and scores 2025 (confirmation only).

Candidates: the LightGBM classifier under each calibration in
`src.models.behavior.propensity` (`behavior_model.calibration`, and
`calibration_groups` for folds by game), plus a factored model built here,
P(package | s) x P(front | s, package), each part Dirichlet-calibrated.

Scored on the held-out season, on the clipped, renormalized propensities the
pipeline actually uses:
  * log_loss (with a game-clustered paired SE vs the shipped isotonic), Brier,
    confidence and classwise ECE;
  * citl_max — the worst per-call gap between mean predicted and observed share;
  * weight balance — per call, sum 1{A=a}/pi(a|s) over the snaps where the call
    is supported (pi >= 0.05), over their count: 1 for the true propensity.
    Reported as the worst and the RMS miss over calls;
  * ESS ratio of the inverse weights.
Synthetic only (`true_alignment_propensity`: the coordinator's softmax averaged
over his hidden read, the best any model of the observed state can do):
  * kl_true — mean KL(true || estimated); rmse_true on the supported cells;
  * the IPW and DR error of the value of every "always call a" policy (and the
    safe and greedy policies), against `true_alignment_q`.
`scored_crossfit` is not a calibrator: it is the shipped pi_b fit on the scored
season itself, out of fold by game — the judge's propensity estimated where the
judging happens, for when the call mix drifts between seasons.
Every candidate scores the same fixed target policies — built once on the
training seasons with the shipped pi_b — so only the judge's pi_b changes.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.stats import mean_se
from src.models.behavior.judge import fit_judge_behavior
from src.models.behavior.propensity import fit_behavior_model
from src.models.calibration import (calibration_in_the_large,
                                    calibration_report, weight_balance)
from src.models.outcome.q_model import fit_q_model
from src.ope.doubly_robust import dr_per_play
from src.ope.ipw import ipw_per_play
from src.schemas.behavior import clip_normalize
from src.schemas.dataset import Dataset

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TUNE_HOLDOUT = 2024

# name -> behavior_model overrides; "factored" is built in this file.
CANDIDATES: dict[str, dict] = {
    "raw": {"calibration": "none"},
    "isotonic": {},                                             # shipped
    "isotonic_game": {"calibration_groups": "game"},
    "temperature": {"calibration": "temperature"},
    "vector": {"calibration": "vector"},
    "dirichlet": {"calibration": "dirichlet"},
    "dirichlet_game": {"calibration": "dirichlet", "calibration_groups": "game"},
    "factored_dirichlet": {"calibration": "dirichlet", "calibration_groups": "game"},
    # Not a calibrator: the shipped pi_b fit on the SCORED season, out of fold
    # by game. Uses that season's states and calls, never its rewards.
    "scored_crossfit": {},
    # The same, but each fold also trains on every training season, with the
    # season as a feature: the scored season's own mix, without giving up data.
    "pooled_crossfit": {},
}
FACTORED = ("factored_dirichlet",)
SCORED = ("scored_crossfit", "pooled_crossfit")
SCORED_FOLDS = 5
N_PACKAGES, N_FRONTS = 3, 4          # level-1 id = package * 4 + front


# ── The factored behavior model ──────────────────────────────────────────────
def _raw_proba(model, df: pd.DataFrame) -> np.ndarray:
    out = np.zeros((len(df), model.n_actions))
    out[:, model.classes_] = model.pipeline.predict_proba(df)
    return out


@dataclass
class FactoredBehavior:
    """pi_b(a | s) = P(package | s) * P(front | s, package)."""
    package: object
    front: object
    clip: tuple[float, float]

    def propensity(self, df: pd.DataFrame) -> np.ndarray:
        p_pkg = _raw_proba(self.package, df)
        joint = np.zeros((len(df), N_PACKAGES * N_FRONTS))
        for k in range(N_PACKAGES):
            p_front = _raw_proba(self.front, df.assign(l1_package_id=float(k)))
            joint[:, k * N_FRONTS:(k + 1) * N_FRONTS] = p_pkg[:, [k]] * p_front
        return clip_normalize(joint, self.clip)


def fit_factored(train: Dataset, cfg: dict) -> FactoredBehavior:
    a = train.df[train.action_col].to_numpy()
    base = dict(numeric_state=train.numeric_state, categorical_state=train.categorical_state,
                reward_col=train.reward_col, action_labels={})
    pkg_ds = Dataset(df=train.df.assign(pkg_action=a // N_FRONTS), action_col="pkg_action",
                     n_actions=N_PACKAGES, **base)
    front_ds = Dataset(df=train.df.assign(front_action=a % N_FRONTS,
                                          l1_package_id=(a // N_FRONTS).astype(float)),
                       action_col="front_action", n_actions=N_FRONTS,
                       **{**base, "numeric_state": train.numeric_state + ["l1_package_id"]})
    return FactoredBehavior(package=fit_behavior_model(pkg_ds, cfg, diagnose=False)[0],
                            front=fit_behavior_model(front_ds, cfg, diagnose=False)[0],
                            clip=tuple(cfg["ope"]["clip_propensity"]))


def scored_crossfit(test: Dataset, cfg: dict, n_folds: int = SCORED_FOLDS,
                    train: Dataset | None = None) -> np.ndarray:
    """pi_b on the scored season, each game's snaps from a model that never saw
    that game. With `train`, every fold also fits on the training seasons, and
    the season joins the state so the model can tell the mixes apart — the
    pipeline's pooled judge (`fit_judge_behavior`), which this experiment chose."""
    if train is not None:
        jcfg = {**cfg, "ope": {**cfg["ope"], "judge_propensity": "pooled_crossfit",
                               "judge_folds": n_folds}}
        return fit_judge_behavior(train, test, jcfg).propensity(test.df)
    from src.folds import game_folds
    fold = game_folds(test.df, n_folds, cfg["seed"])
    p = np.empty((len(test.df), test.n_actions))
    for f in range(n_folds):
        te = fold == f
        beh = fit_behavior_model(test.subset(~te), cfg, diagnose=False)[0]
        p[te] = beh.propensity(test.df[te])
    return p


def candidate_config(cfg: dict, name: str) -> dict:
    c = copy.deepcopy(cfg)
    c["behavior_model"].update(copy.deepcopy(CANDIDATES[name]))
    return c


def fit_candidate(train: Dataset, cfg: dict, name: str):
    c = candidate_config(cfg, name)
    return fit_factored(train, c) if name in FACTORED else fit_behavior_model(train, c,
                                                                              diagnose=False)[0]


# ── Scoring ──────────────────────────────────────────────────────────────────
def row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))


def ess_ratio(y: np.ndarray, p: np.ndarray) -> float:
    w = 1.0 / p[np.arange(len(y)), y]
    w = w / w.mean()
    return float(1.0 / (w ** 2).mean())


def target_policies(train: Dataset, test: Dataset, cfg: dict, k: int) -> dict[str, np.ndarray]:
    """Fixed policies to evaluate, (n, k) probabilities each: the shipped safe
    policy, the supported greedy argmax of the GBM Q, and 'always call a'."""
    from src.policy.safe import fit_policy
    beh = fit_behavior_model(train, cfg, diagnose=False)[0]
    q = fit_q_model(train, cfg)
    df = test.df
    pol = {"safe": fit_policy(train, cfg, beh, q).policy.act(df)["policy_probs"]}
    support = beh.propensity(df) >= cfg["policy"].get("support_threshold", 0.05)
    greedy = np.where(support, q.predict_all_actions(df), -np.inf).argmax(axis=1)
    pol["greedy"] = np.eye(k)[greedy]
    for a in range(k):
        pol[f"always_{a}"] = np.tile(np.eye(k)[a], (len(df), 1))
    return pol


def score_candidate(name: str, p: np.ndarray, ctx: dict) -> dict:
    y, games, thr = ctx["a"], ctx["games"], ctx["thr"]
    rl = row_log_loss(y, p)
    wb = weight_balance(y, p, thr)
    row = {"model": name, "log_loss": float(rl.mean()),
           **calibration_report(y, p),
           "citl_max": float(np.abs(calibration_in_the_large(y, p)).max()),
           "balance_worst": float(np.nanmax(np.abs(wb - 1))),
           "balance_rms": float(np.sqrt(np.nanmean((wb - 1) ** 2))),
           "ess_ratio": ess_ratio(y, p),
           "support_share": float((p >= thr).mean())}
    if "base_rows" in ctx:
        row["log_loss_vs_isotonic"], row["log_loss_vs_isotonic_se"] = mean_se(
            rl - ctx["base_rows"], games)
    for c in range(p.shape[1]):
        row[f"balance_{c}"] = float(wb[c])

    # OPE of the fixed policies with this pi_b as the judge's.
    a, r, wclip, q = ctx["a"], ctx["r"], ctx["wclip"], ctx["q_judge"]
    ipw_err, dr_err = [], []
    for pname, probs in ctx["policies"].items():
        ipw = float(ipw_per_play(p, probs, a, r, wclip, self_normalize=True).mean())
        dr = float(dr_per_play(q, p, probs, a, r, wclip).mean())
        if pname in ("safe", "greedy"):
            base = dr_per_play(q, p, p, a, r, wclip)
            lift, se = mean_se(dr_per_play(q, p, probs, a, r, wclip) - base, games)
            row[f"{pname}_dr_lift"], row[f"{pname}_dr_lift_se"] = lift, se
            row[f"{pname}_ipw"], row[f"{pname}_dr"] = ipw, dr
        if ctx["q_true"] is not None:
            truth = float((probs * ctx["q_true"]).sum(axis=1).mean())
            if pname in ("safe", "greedy"):
                row[f"{pname}_true"] = truth
                row[f"{pname}_ipw_err"], row[f"{pname}_dr_err"] = ipw - truth, dr - truth
            else:
                ipw_err.append(ipw - truth)
                dr_err.append(dr - truth)
    if ctx["q_true"] is not None:
        pt = ctx["p_true"]
        sup = pt >= thr
        row.update({
            "kl_true": float((pt * (np.log(np.clip(pt, 1e-12, 1)) - np.log(p))).sum(1).mean()),
            "rmse_true": float(np.sqrt(np.mean((p - pt)[sup] ** 2))),
            "always_ipw_rmse": float(np.sqrt(np.mean(np.square(ipw_err)))),
            "always_dr_rmse": float(np.sqrt(np.mean(np.square(dr_err)))),
        })
    return row


def compare(train: Dataset, test: Dataset, cfg: dict, candidates: list[str] | None = None,
            q_true: np.ndarray | None = None, p_true: np.ndarray | None = None,
            progress=None, checkpoint: str | None = None) -> pd.DataFrame:
    say = progress or (lambda m: None)
    candidates = candidates or list(CANDIDATES)
    df = test.df
    clip = tuple(cfg["ope"]["clip_propensity"])
    ctx = {"a": df[test.action_col].to_numpy(), "r": df[test.reward_col].to_numpy(),
           "games": df["game_id"].to_numpy(), "thr": cfg["policy"].get("support_threshold", 0.05),
           "wclip": cfg["ope"].get("weight_clip", 20.0), "q_true": q_true,
           "p_true": None if p_true is None else clip_normalize(p_true, clip)}
    t = time.time()
    ctx["q_judge"] = fit_q_model(train, cfg).predict_all_actions(df)
    ctx["policies"] = target_policies(train, test, cfg, test.n_actions)
    say(f"judge Q and target policies {time.time() - t:.0f}s")

    rows = []
    if p_true is not None:
        rows.append(score_candidate("true_propensity", ctx["p_true"], ctx))
    order = ["isotonic"] + [c for c in candidates if c != "isotonic"]
    for name in order:
        t = time.time()
        p = (scored_crossfit(test, candidate_config(cfg, name),
                             train=train if name == "pooled_crossfit" else None)
             if name in SCORED else fit_candidate(train, cfg, name).propensity(df))
        fit_s = time.time() - t
        if name == "isotonic":
            ctx["base_rows"] = row_log_loss(ctx["a"], p)
        rows.append({**score_candidate(name, p, ctx), "fit_s": fit_s})
        if checkpoint:
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
        say(f"{name}: log loss {rows[-1]['log_loss']:.4f}, classwise ECE "
            f"{rows[-1]['classwise_ece']:.4f}, balance worst {rows[-1]['balance_worst']:.3f} "
            f"({fit_s:.0f}s)")
    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    import logging
    import warnings

    import yaml

    from experiments.outcome_models.benchmark import _load
    from src.data.features import time_aware_split
    from src.data.load import true_alignment_propensity, true_alignment_q

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    p = argparse.ArgumentParser(description="Compare behavior-model calibrations.")
    p.add_argument("--config", default="configs/v3.yaml")
    p.add_argument("--source", choices=["nflfastR", "synthetic"], default="nflfastR")
    p.add_argument("--fold", choices=["tune", "holdout"], default="tune")
    p.add_argument("--n-plays", type=int, default=140_000)
    p.add_argument("--conf-strength", type=float, default=None)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    t0 = time.time()
    log = lambda m: print(f"  [{time.time() - t0:6.0f}s] {m}", flush=True)
    train, test, _ = _load(cfg, args.source, args.n_plays, args.conf_strength)
    if args.fold == "tune":
        train, test = time_aware_split(train, TUNE_HOLDOUT)
    synth = args.source == "synthetic"
    q_true = true_alignment_q(test.df) if synth else None
    p_true = true_alignment_propensity(test.df) if synth else None
    tag = ("real" if not synth else "synthetic_with_read" if args.conf_strength is None
           else "synthetic_no_read" if args.conf_strength == 0
           else f"synthetic_conf{args.conf_strength:g}")
    os.makedirs(RESULTS, exist_ok=True)
    out = args.out or os.path.join(RESULTS, f"{args.fold}_{tag}.csv")
    log(f"{tag} {args.fold}: fit {len(train.df):,}, score {len(test.df):,}")
    table = compare(train, test, cfg, args.models, q_true, p_true, progress=log, checkpoint=out)
    table.to_csv(out, index=False)
    cols = [c for c in ["model", "log_loss", "log_loss_vs_isotonic", "log_loss_vs_isotonic_se",
                        "classwise_ece", "confidence_ece", "citl_max", "balance_worst",
                        "balance_rms", "ess_ratio", "kl_true", "rmse_true", "always_ipw_rmse",
                        "always_dr_rmse", "safe_dr_lift", "safe_dr_lift_se", "safe_dr_err",
                        "greedy_dr_lift", "fit_s"] if c in table]
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table[cols].round(4).to_string(index=False))
    log(f"saved {out}")


if __name__ == "__main__":
    main()
