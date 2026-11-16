"""
The DragonNet tuning harness — the search space, the judge, one config's run.
Run with `pytest experiments`.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest


pytest.importorskip("torch")

from _config import CFG_L1
from experiments.dragonnet_tuning.tune import (DEFAULT, SPACE, build_context, neural_config,
                                               run_config, sample_configs, score,
                                               selector_check, tuning_fold)


def test_the_search_starts_from_the_shipped_default():
    """Config 0 is what `fit_neural_q_model` does with no `neural` block: every
    key's fallback there equals DEFAULT here."""
    import ast
    import inspect
    from src.models.outcome import neural
    tree = ast.parse(inspect.getsource(neural.fit_neural_q_model))
    fallbacks = {c.args[0].value: ast.literal_eval(c.args[1]) for c in ast.walk(tree)
                 if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "get"
                 and getattr(c.func.value, "id", "") == "ncfg" and len(c.args) == 2}
    assert sample_configs(12, seed=0)[0] == DEFAULT
    assert {k: fallbacks[k] for k in DEFAULT} == DEFAULT


def test_sampled_configs_are_distinct_reproducible_and_in_the_space():
    a, b = sample_configs(20, seed=1), sample_configs(20, seed=1)
    assert a == b
    assert len({json.dumps(c, sort_keys=True) for c in a}) == 20
    for c in a[1:]:
        assert all(c[k] in v for k, v in SPACE.items())


def test_neural_config_pins_the_factored_dragonnet():
    c = neural_config(CFG_L1, {"dropout": 0.3}, seed=9)
    assert c["seed"] == 9 and c["outcome_model"]["type"] == "dragonnet"
    assert c["outcome_model"]["neural"] == {"heads": "factored", "dropout": 0.3}
    assert CFG_L1["outcome_model"].get("type", "lightgbm") == "lightgbm"   # not mutated


@pytest.fixture(scope="module")
def ctx(synthetic_league):
    cfg = copy.deepcopy(CFG_L1)
    cfg["invariance"] = {**cfg.get("invariance", {}), "crossfit_folds": 2}
    train, _ = synthetic_league(9000, seed=3)
    fit, scored, q_true = tuning_fold(train, "synthetic")
    assert set(fit.df["season"]) == {2022, 2023} and set(scored.df["season"]) == {2024}
    return build_context(fit, scored, cfg, q_true)


def test_the_gbm_is_the_zero_of_the_loss(ctx):
    assert ctx["gbm"]["dr_loss"] == 0.0
    assert ctx["base_loss"].shape == (len(ctx["test_df"]),)
    assert {"adv_rmse", "adv_corr", "spread_ratio"} <= set(ctx["gbm"])


def test_a_config_runs_over_seeds_and_reports_its_ensemble(ctx):
    params = {**DEFAULT, "hidden": [32, 16], "head_hidden": 8, "max_epochs": 3, "patience": 2}
    out = run_config(5, params, [1, 2], ctx=ctx)
    row = out["row"]
    assert row["config"] == 5 and row["n_seeds"] == 2 and json.loads(row["params"]) == params
    assert np.isfinite([row["dr_loss"], row["dr_loss_seed_sd"], row["dr_loss_paired_se"],
                        row["ens_dr_loss"], row["adv_rmse"], row["ens_adv_rmse"]]).all()
    assert row["epochs"] <= 3
    assert out["loss"].shape == (len(ctx["test_df"]),)


def test_score_matches_its_loss(ctx):
    q = np.random.default_rng(0).normal(size=ctx["pi_b"].shape)
    sc, loss = score(q, ctx)
    assert np.isclose(sc["dr_loss"], float((loss - ctx["base_loss"]).mean()))


def test_the_selector_check_ranks_against_the_truth(tmp_path):
    import pandas as pd
    d = pd.DataFrame({"config": [-1, 0, 1, 2, 3], "dr_loss": [0.0, 0.0, -0.2, 0.1, 0.3],
                      "adv_rmse": [0.5, 0.02, 0.01, 0.03, 0.04]})     # -1: the GBM row
    p = tmp_path / "s.csv"
    d.to_csv(p, index=False)
    out = selector_check(str(p))
    assert out["spearman"] == pytest.approx(1.0) and out["n_configs"] == 4
    assert out["dr_pick_config"] == 1 and out["dr_pick_true_rank"] == 1
