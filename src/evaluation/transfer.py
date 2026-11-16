"""
Transfer — does the policy hold up on a defense or a season it never saw?

The 2025 holdout asks whether a policy learned on 2022-24 survives the next
season. Two more questions, both inside the training seasons so the holdout
stays untouched:

  * leave one team out   — learn the policy on 31 defenses, apply it to the 32nd.
                           Its team features are unseen; only the state and the
                           league's behavior speak for it.
  * leave one season out — learn on two of 2022-24, apply to the third.

Each fold refits the whole policy — pi_b, Q, the bootstrap ensemble and the
read allowance — without the held-out group. The judge is not refit with it:
every snap is scored by nuisances cross-fitted by GAME over all training
seasons (`crossfit_nuisances`), which know the held-out defense from its
other games. Evaluating a team with models that never saw it would confuse
"the policy does not transfer" with "the evaluator does not".

Each fold is reported against the in-domain policy on the same snaps — the
policy refit on the game folds, so it saw the group's other games but not the
snap itself. The paired difference is the transfer gap: what not having seen
the defense (or season) costs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..models.behavior.propensity import fit_behavior_model
from ..models.outcome.q_model import fit_q_model
from ..ope.doubly_robust import dr_per_play
from ..policy.safe import defer_to, fit_policy
from ..folds import game_folds
from ..schemas.dataset import Dataset
from ..stats import heterogeneity, mean_se, regresses

SPLITS = {"leave_one_team_out": "defteam", "leave_one_season_out": "season"}


def crossfit_nuisances(ds: Dataset, cfg: dict, n_folds: int | None = None,
                       with_policy: bool = False,
                       progress: Callable[[str], None] | None = None) -> dict:
    """Out-of-fold Q and pi_b for every row, folds by game.

    With `with_policy`, also the in-domain policy's probabilities: the policy
    refit on each fold's training games and applied to the fold, reusing that
    fold's Q and pi_b.
    """
    n_folds = n_folds or cfg.get("invariance", {}).get("crossfit_folds", 5)
    fold = game_folds(ds.df, n_folds, cfg["seed"])
    n, k = len(ds.df), ds.n_actions
    q_all, pi_b = np.empty((n, k)), np.empty((n, k))
    probs = np.empty((n, k)) if with_policy else None
    departs = np.zeros(n, bool) if with_policy else None
    for f in range(n_folds):
        te = fold == f
        train = ds.subset(~te)
        beh = fit_behavior_model(train, cfg, diagnose=False)[0]
        q = fit_q_model(train, cfg)
        test_df = ds.df[te]
        q_all[te], pi_b[te] = q.predict_all_actions(test_df), beh.propensity(test_df)
        if with_policy:
            out = fit_policy(train, cfg, beh, q).policy.act(test_df)
            probs[te] = out["policy_probs"]
            departs[te] = ~out["deferred"] if "deferred" in out else True
        if progress:
            progress(f"game fold {f + 1}/{n_folds}")
    return {"q_all": q_all, "pi_b": pi_b, "fold": fold,
            "policy_probs": probs, "departs": departs}


def transfer_evaluation(ds: Dataset, cfg: dict, nuisances: dict,
                        splits: list[str] | None = None,
                        progress: Callable[[str], None] | None = None) -> pd.DataFrame:
    """One row per held-out group: the transferred policy's lift and its gap.

    `ds` is the training seasons; `nuisances` from `crossfit_nuisances(...,
    with_policy=True)` on the same rows.
    """
    splits = splits or cfg.get("invariance", {}).get("transfer_splits", list(SPLITS))
    clip = cfg["ope"].get("weight_clip", 20.0)
    margin = cfg.get("evaluation", {}).get("regression_margin", 0.02)
    df = ds.df
    a, r = df[ds.action_col].to_numpy(), df[ds.reward_col].to_numpy()
    games = df["game_id"].to_numpy()
    rows = []
    for split in splits:
        col = SPLITS[split]
        for g in sorted(df[col].unique()):
            held = (df[col] == g).to_numpy()
            out = fit_policy(ds.subset(~held), cfg).policy.act(df[held])
            q, pb, aa, rr = nuisances["q_all"][held], nuisances["pi_b"][held], a[held], r[held]
            base = dr_per_play(q, pb, pb, aa, rr, clip)
            moved = dr_per_play(q, pb, out["policy_probs"], aa, rr, clip)
            inside = dr_per_play(q, pb, nuisances["policy_probs"][held], aa, rr, clip)
            departs = ~out["deferred"] if "deferred" in out else np.ones(held.sum(), bool)
            # The same departures, deferring to the group's own mix rather than
            # the fold's estimate of it: separates "the departures do not
            # transfer" from "the coordinator's mix is not known".
            anchored = defer_to(out, pb)
            anch = dr_per_play(q, pb, anchored, aa, rr, clip)
            lift, lift_se = mean_se(moved - base, games[held])
            lift_a, lift_a_se = mean_se(anch - base, games[held])
            lift_in, lift_in_se = mean_se(inside - base, games[held])
            gap, gap_se = mean_se(moved - inside, games[held])
            dm = float(((out["policy_probs"] - pb) * q).sum(axis=1).mean())
            rows.append({
                "split": split, "held_out": str(g), "n": int(held.sum()),
                "departs": float(departs.mean()),
                "departs_in_domain": float(nuisances["departs"][held].mean()),
                "lift": lift, "lift_se": lift_se, "lift_dm": dm,
                "lift_anchored": lift_a, "lift_anchored_se": lift_a_se,
                "lift_in_domain": lift_in, "lift_in_domain_se": lift_in_se,
                "gap": gap, "gap_se": gap_se,
                "regresses": bool(regresses(lift, lift_se, margin)),
            })
            if progress:
                progress(f"{split}: {g}")
    return pd.DataFrame(rows)


def summarize_transfer(table: pd.DataFrame) -> pd.DataFrame:
    """Per split: pooled lift and gap, their spread across groups, regressions."""
    rows = []
    for split, t in table.groupby("split", sort=False):
        lift = heterogeneity(t["lift"], t["lift_se"])
        gap = heterogeneity(t["gap"], t["gap_se"])
        anch = heterogeneity(t["lift_anchored"], t["lift_anchored_se"])
        worst = t.loc[t["lift"].idxmin()]
        rows.append({
            "split": split, "groups": len(t), "departs": float(t["departs"].mean()),
            "lift": lift["pooled"], "lift_se": lift["pooled_se"],
            "lift_tau": lift["tau"], "lift_p": lift["p"],
            "lift_anchored": anch["pooled"], "lift_anchored_se": anch["pooled_se"],
            "gap": gap["pooled"], "gap_se": gap["pooled_se"],
            "gap_tau": gap["tau"], "gap_p": gap["p"],
            "n_regress": int(t["regresses"].sum()),
            "worst": worst["held_out"], "worst_lift": float(worst["lift"]),
            "worst_lift_se": float(worst["lift_se"]),
        })
    return pd.DataFrame(rows)
