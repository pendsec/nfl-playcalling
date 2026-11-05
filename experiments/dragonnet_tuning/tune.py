"""
Factored DragonNet tuning — which hyperparameters give the most accurate advantages?

Run from the repository root, in order:

    python -m experiments.dragonnet_tuning.tune search  --source nflfastR
    python -m experiments.dragonnet_tuning.tune search  --source synthetic --conf-strength 0
    python -m experiments.dragonnet_tuning.tune refine  --source nflfastR
    python -m experiments.dragonnet_tuning.tune confirm --source nflfastR
    python -m experiments.dragonnet_tuning.tune confirm --source synthetic [--conf-strength 0]

Training seasons choose; the holdout only confirms.

  * search  — random search over the space below, on the TUNING fold: fit on
              2022-23, score on 2024. Every config with 3 seeds.
  * refine  — the search's best configs and the default, rerun with 5 fresh
              seeds (the search's best are partly lucky draws), still on the
              tuning fold. The lowest mean loss is chosen -> results/chosen.json.
  * confirm — the chosen config against the default and the GBM, fit on
              2022-24 and scored once on 2025 (real) or against the truth
              (synthetic).

The score is the outcome-model benchmark's real-data judge: the mean squared
gap between the estimated advantages A(s, a) = Q(s, a) - sum_b pi_b Q(s, b) and
doubly-robust pseudo-outcomes built on the scored season alone (nuisances
cross-fitted by game there), over supported calls (pi_b >= 0.05), relative to
the pooled GBM. The search runs on the synthetic league too, where the true
advantages are known, to check that this judge ranks configs by their true
error at the tuning fold's size.
"""

from __future__ import annotations

import copy
import itertools
import json
import os
import pickle
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd

from experiments.outcome_models.benchmark import (CANDIDATES, _load, advantages, benchmark,
                                                  calibration_slope, pseudo_outcomes)
from src.data.features import time_aware_split
from src.data.load import true_alignment_q
from src.evaluation.invariance import clustered_mean_se
from src.schemas.dataset import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
TUNE_HOLDOUT = 2024            # the tuning fold: 2022-23 -> 2024

# The shipped defaults (src/models/outcome/neural.py), the search's config 0.
DEFAULT = {"hidden": [128, 64], "head_hidden": 32, "dropout": 0.1, "lr": 1e-3,
           "weight_decay": 1e-4, "batch_size": 1024, "alpha": 1.0, "beta": 1.0,
           "embed_max": 8, "early_stop": "loss", "max_epochs": 60}

SPACE = {
    "hidden": [[64, 32], [128, 64], [256, 128], [128, 128, 64], [256, 256, 128]],
    "head_hidden": [16, 32, 64],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3, 1e-2],
    "batch_size": [256, 512, 1024, 2048, 4096],
    "alpha": [0.1, 0.3, 1.0, 3.0],      # weight of the propensity head
    "beta": [0.0, 0.3, 1.0, 3.0],       # weight of targeted regularization
    "embed_max": [4, 8, 16],            # widest embedding (teams, QB alignment, ...)
    "early_stop": ["loss", "outcome"],  # stop on the whole loss or on Q's error alone
}
SEARCH_MAX_EPOCHS = 100        # the sampled configs get room; `epochs` shows who used it


def sample_configs(n: int, seed: int = 0) -> list[dict]:
    """The default, then n - 1 distinct random draws from SPACE."""
    rng = np.random.default_rng(seed)
    seen, out = {json.dumps(DEFAULT, sort_keys=True)}, [dict(DEFAULT)]
    while len(out) < n:
        c = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
        c["max_epochs"] = SEARCH_MAX_EPOCHS
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def neural_config(cfg: dict, params: dict, seed: int) -> dict:
    c = copy.deepcopy(cfg)
    c["seed"] = int(seed)
    c["outcome_model"].update({"type": "dragonnet",
                               "neural": {"heads": "factored", **copy.deepcopy(params)}})
    return c


# ── The judge ────────────────────────────────────────────────────────────────
def build_context(train: Dataset, test: Dataset, cfg: dict,
                  q_true: np.ndarray | None = None) -> dict:
    """Everything shared by the configs: the policy's pi_b (fit on `train`), the
    supported calls, the scored season's pseudo-outcome advantages, and the
    GBM's per-snap loss — the zero every config is measured from."""
    from src.evaluation.transfer import crossfit_nuisances
    from src.models.behavior.propensity import fit_behavior_model
    from src.models.outcome.q_model import fit_q_model

    clip, wclip = cfg["ope"]["clip_propensity"], cfg["ope"].get("weight_clip", 20.0)
    df = test.df
    a, r = df[test.action_col].to_numpy(), df[test.reward_col].to_numpy()
    pi_b = fit_behavior_model(train, cfg, diagnose=False)[0].propensity(df)
    support = pi_b >= cfg["policy"].get("support_threshold", 0.05)
    hold = crossfit_nuisances(test, cfg)
    target = advantages(pseudo_outcomes(hold["q_all"], hold["pi_b"], a, r, clip, wclip), pi_b)
    games = df["game_id"].to_numpy()
    ctx = {"train": train, "test_df": df, "cfg": cfg, "pi_b": pi_b, "support": support,
           "target": target, "games": games, "q_true": q_true,
           "cell_games": np.broadcast_to(games[:, None], support.shape)[support]}
    gbm = fit_q_model(train, cfg).predict_all_actions(df)
    ctx["base_loss"] = row_loss(advantages(gbm, pi_b), target, support)
    ctx["gbm"] = score(gbm, ctx)[0]
    return ctx


def row_loss(adv: np.ndarray, target: np.ndarray, support: np.ndarray) -> np.ndarray:
    return np.where(support, (adv - target) ** 2, 0.0).sum(1) / support.sum(1)


def score(q_hat: np.ndarray, ctx: dict) -> tuple[dict, np.ndarray]:
    """Metrics for one Q on the scored season, and its per-snap DR loss."""
    sup, pi_b = ctx["support"], ctx["pi_b"]
    adv = advantages(q_hat, pi_b)
    loss = row_loss(adv, ctx["target"], sup)
    d, se = (clustered_mean_se(loss - ctx["base_loss"], ctx["games"])
             if "base_loss" in ctx else (0.0, 0.0))
    slope, _ = calibration_slope(adv[sup], ctx["target"][sup], ctx["cell_games"])
    out = {"dr_loss": d, "dr_loss_se": se, "cal_slope": slope, "adv_sd": float(adv[sup].std())}
    if ctx["q_true"] is not None:
        at = advantages(ctx["q_true"], pi_b)[sup]
        out.update({"adv_rmse": float(np.sqrt(np.mean((adv[sup] - at) ** 2))),
                    "adv_corr": float(np.corrcoef(adv[sup], at)[0, 1]),
                    "spread_ratio": float(adv[sup].std() / at.std())})
    return out, loss


# ── Workers ──────────────────────────────────────────────────────────────────
_CTX: dict | None = None


def _init(path: str) -> None:
    global _CTX
    import warnings
    warnings.filterwarnings("ignore")
    with open(path, "rb") as f:
        _CTX = pickle.load(f)


def run_config(cid: int, params: dict, seeds: list[int], ctx: dict | None = None) -> dict:
    """Fit one config once per seed. Returns the seed-mean metrics, their seed
    sd, the 'ensemble' (the seeds' Qs averaged) metrics, and the seed-averaged
    per-snap loss (for paired comparisons between configs)."""
    from src.models.outcome.q_model import fit_q_model

    ctx = ctx or _CTX
    per_seed, qs, losses = [], [], []
    for s in seeds:
        t = time.time()
        m = fit_q_model(ctx["train"], neural_config(ctx["cfg"], params, s))
        q = m.predict_all_actions(ctx["test_df"])
        sc, loss = score(q, ctx)
        per_seed.append({**sc, "epochs": len(m.history), "fit_s": time.time() - t})
        qs.append(q)
        losses.append(loss)
    seed_df = pd.DataFrame(per_seed)
    row = {"config": cid, "params": json.dumps(params, sort_keys=True), "n_seeds": len(seeds)}
    for k in seed_df.columns:
        row[k] = float(seed_df[k].mean())
    row["dr_loss_seed_sd"] = float(seed_df["dr_loss"].std(ddof=1)) if len(seeds) > 1 else np.nan
    mean_loss = np.mean(losses, axis=0)
    row["dr_loss_paired_se"] = clustered_mean_se(mean_loss - ctx["base_loss"], ctx["games"])[1]
    ens = score(np.mean(qs, axis=0), ctx)[0]
    row.update({f"ens_{k}": v for k, v in ens.items() if not k.endswith("_se")})
    return {"row": row, "loss": mean_loss.astype(np.float32)}


def run_all(ctx: dict, configs: list[tuple[int, dict]], seeds: list[int], workers: int,
            out_csv: str, log) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Every config through a spawn pool (torch and LightGBM must not share a
    forked OpenMP runtime); the CSV is rewritten as each one finishes."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    pickle.dump(ctx, tmp)
    tmp.close()
    rows, losses = [], {}
    try:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                 initializer=_init, initargs=(tmp.name,)) as pool:
            futs = {pool.submit(run_config, cid, p, seeds): cid for cid, p in configs}
            for f in as_completed(futs):
                res = f.result()
                rows.append(res["row"])
                losses[res["row"]["config"]] = res["loss"]
                pd.DataFrame(rows).sort_values("config").to_csv(out_csv, index=False)
                r = res["row"]
                log(f"config {r['config']:>2}: dr_loss {r['dr_loss']:+.5f} "
                    f"(seed sd {r['dr_loss_seed_sd']:.5f}), epochs {r['epochs']:.0f}, "
                    f"{r['fit_s']:.0f}s/fit  [{len(rows)}/{len(configs)}]")
    finally:
        os.unlink(tmp.name)
    return pd.DataFrame(rows).sort_values("config").reset_index(drop=True), losses


# ── Stages ───────────────────────────────────────────────────────────────────
def tag_for(source: str, conf_strength: float | None) -> str:
    if source == "nflfastR":
        return "real"
    return ("synthetic_with_read" if conf_strength is None else
            "synthetic_no_read" if conf_strength == 0 else f"synthetic_conf{conf_strength:g}")


def tuning_fold(train: Dataset, source: str) -> tuple[Dataset, Dataset, np.ndarray | None]:
    fit, scored = time_aware_split(train, TUNE_HOLDOUT)
    return fit, scored, (true_alignment_q(scored.df) if source == "synthetic" else None)


def paired_vs(losses: dict[int, np.ndarray], ref: int, games: np.ndarray) -> dict[int, tuple]:
    return {c: clustered_mean_se(l - losses[ref], games) for c, l in losses.items()}


def selector_check(search_csv: str) -> dict:
    """On a league with known truth: does the DR loss rank configs like their
    true advantage error does?"""
    from scipy.stats import spearmanr
    d = pd.read_csv(search_csv)
    d = d[d["config"] >= 0]                     # the GBM's reference row is not a config
    rho = spearmanr(d["dr_loss"], d["adv_rmse"]).statistic
    best_dr = d.loc[d["dr_loss"].idxmin()]
    return {"spearman": float(rho), "n_configs": len(d),
            "dr_pick_config": int(best_dr["config"]),
            "dr_pick_true_rank": int((d["adv_rmse"] < best_dr["adv_rmse"]).sum() + 1),
            "best_true_rmse": float(d["adv_rmse"].min()),
            "dr_pick_true_rmse": float(best_dr["adv_rmse"]),
            "default_true_rmse": float(d.loc[d["config"] == 0, "adv_rmse"].iloc[0])}


def main() -> None:
    import argparse
    import logging
    import warnings

    import yaml

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    p = argparse.ArgumentParser(description="Tune the factored DragonNet.")
    p.add_argument("stage", choices=["search", "refine", "confirm", "check"])
    p.add_argument("--config", default="configs/v3.yaml")
    p.add_argument("--source", choices=["nflfastR", "synthetic"], default="nflfastR")
    p.add_argument("--n-plays", type=int, default=140_000)
    p.add_argument("--conf-strength", type=float, default=None)
    p.add_argument("--n-configs", type=int, default=48)
    p.add_argument("--seeds", type=int, default=None, help="search 3, refine/confirm 5")
    p.add_argument("--top", type=int, default=6, help="refine: configs carried from search")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--n-boot", type=int, default=10, help="confirm: safe-policy ensemble")
    args = p.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    log = lambda m: print(f"  [{time.time() - t0:6.0f}s] {m}", flush=True)
    tag = tag_for(args.source, args.conf_strength)

    if args.stage == "check":
        print(json.dumps(selector_check(os.path.join(RESULTS, f"search_{tag}.csv")), indent=2))
        return

    cfg = yaml.safe_load(open(args.config))
    train, test, q_true = _load(cfg, args.source, args.n_plays, args.conf_strength)

    if args.stage in ("search", "refine"):
        fit, scored, q_tune = tuning_fold(train, args.source)
        log(f"{tag} tuning fold: fit {len(fit.df):,} (<{TUNE_HOLDOUT}), score {len(scored.df):,}")
        ctx = build_context(fit, scored, cfg, q_tune)
        log(f"judge built; GBM {ctx['gbm']}")
        if args.stage == "search":
            configs = list(enumerate(sample_configs(args.n_configs)))
            seeds = [cfg["seed"] + i for i in range(args.seeds or 3)]
            out = os.path.join(RESULTS, f"search_{tag}.csv")
        else:
            search = pd.read_csv(os.path.join(RESULTS, f"search_{tag}.csv"))
            top = search[search["config"] != 0].nsmallest(args.top, "dr_loss")
            configs = [(0, dict(DEFAULT))] + [(int(c), json.loads(pr)) for c, pr in
                                               zip(top["config"], top["params"])]
            seeds = [1000 + i for i in range(args.seeds or 5)]
            out = os.path.join(RESULTS, f"refine_{tag}.csv")
        table, losses = run_all(ctx, configs, seeds, args.workers, out, log)
        vs_default = paired_vs(losses, 0, ctx["games"])
        table["vs_default"] = table["config"].map(lambda c: vs_default[c][0])
        table["vs_default_se"] = table["config"].map(lambda c: vs_default[c][1])
        gbm_row = {"config": -1, "params": "gbm_pooled", **ctx["gbm"]}
        table = pd.concat([pd.DataFrame([gbm_row]), table], ignore_index=True)
        table.to_csv(out, index=False)
        if args.stage == "refine":
            best = table[table["config"] >= 0].nsmallest(1, "dr_loss").iloc[0]
            chosen = {"config": int(best["config"]), "params": json.loads(best["params"]),
                      "tuning_dr_loss": float(best["dr_loss"]),
                      "tuning_vs_default": float(best["vs_default"]),
                      "tuning_vs_default_se": float(best["vs_default_se"])}
            json.dump(chosen, open(os.path.join(RESULTS, "chosen.json"), "w"), indent=2)
            log(f"chosen: {chosen}")
    else:  # confirm
        chosen = json.load(open(os.path.join(RESULTS, "chosen.json")))
        ctx = build_context(train, test, cfg, q_true)
        log(f"{tag} holdout: fit {len(train.df):,}, score {len(test.df):,}; GBM {ctx['gbm']}")
        configs = [(0, dict(DEFAULT)), (chosen["config"], chosen["params"])]
        seeds = [2000 + i for i in range(args.seeds or 5)]
        out = os.path.join(RESULTS, f"confirm_{tag}.csv")
        table, losses = run_all(ctx, configs, seeds, args.workers, out, log)
        vs_default = paired_vs(losses, 0, ctx["games"])
        table["vs_default"] = table["config"].map(lambda c: vs_default[c][0])
        table["vs_default_se"] = table["config"].map(lambda c: vs_default[c][1])
        table = pd.concat([pd.DataFrame([{"config": -1, "params": "gbm_pooled", **ctx["gbm"]}]),
                           table], ignore_index=True)
        table.to_csv(out, index=False)
        # The safe policy on the tuned net, through the outcome benchmark's harness.
        CANDIDATES["dragonnet_tuned"] = {"type": "dragonnet",
                                         "neural": {"heads": "factored", **chosen["params"]}}
        bench = benchmark(train, test, cfg, ["gbm_pooled", "dragonnet_factored", "dragonnet_tuned"],
                          q_true, args.n_boot, progress=log,
                          checkpoint=os.path.join(RESULTS, f"policy_{tag}.csv"))
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(bench.round(4).to_string(index=False))
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_colwidth", 60):
        print(table.round(5).to_string(index=False))
    log(f"saved {out}")


if __name__ == "__main__":
    main()
