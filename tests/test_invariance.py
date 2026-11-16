"""
Invariance across environments and transfer to unseen groups.

Anchored on `generate_synthetic_alignment(effect_heterogeneity=...)`: the
alignment SCM, optionally with each defense-season's alignment effect shifted
by its own draw. The test must stay quiet when the effect is shared and speak
when it is not; the transfer harness must refit without the held-out group and
score every group once.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest


from _config import CFG_INV, CFG_L1
from src.data.features import build_dataset, level1_components, time_aware_split
from src.data.load import generate_synthetic_alignment
from src.evaluation.invariance import (cluster_ids, component_policy, environment_ids,
                                       invariance_test)
from src.folds import game_folds
from src.stats import heterogeneity


# ── Pieces ───────────────────────────────────────────────────────────────────
def test_component_policy_keeps_the_mix_within_the_component():
    rng = np.random.default_rng(0)
    pi_b = rng.dirichlet(np.ones(12), 5)
    pkg = level1_components(np.arange(12), CFG_L1)["package"]
    p = component_policy(pi_b, pkg, 2)
    np.testing.assert_allclose(p.sum(1), 1.0)
    assert (p[:, pkg != 2] == 0).all()
    np.testing.assert_allclose(p[:, 8] / p[:, 9], pi_b[:, 8] / pi_b[:, 9])


def test_heterogeneity_is_zero_for_identical_estimates():
    h = heterogeneity(np.full(10, 0.03), np.full(10, 0.01))
    assert h["q"] == 0 and h["tau"] == 0 and h["pooled"] == pytest.approx(0.03)


def test_heterogeneity_recovers_a_known_spread():
    rng = np.random.default_rng(1)
    est = rng.normal(0, 0.10, 4000) + rng.normal(0, 0.05, 4000)
    h = heterogeneity(est, np.full(4000, 0.05))
    assert h["tau"] == pytest.approx(0.10, abs=0.01) and h["p"] < 1e-6


def test_environments_and_clusters():
    df = pd.DataFrame({"game_id": ["g1", "g1", "g2"], "defteam": ["A", "B", "A"],
                       "season": [2022, 2022, 2023]})
    assert environment_ids(df, CFG_INV).tolist() == ["A_2022", "B_2022", "A_2023"]
    assert len(set(cluster_ids(df))) == 3          # the two sides of g1 are apart
    with pytest.raises(ValueError, match="environment"):
        environment_ids(df, {"invariance": {"environment": "week"}})


def test_a_cluster_spanning_environments_is_rejected():
    q = np.zeros((4, 12)); pi_b = np.full((4, 12), 1 / 12)
    with pytest.raises(ValueError, match="cluster"):
        invariance_test(q, pi_b, np.array([0, 1, 4, 5]), np.zeros(4),
                        np.array(["e1", "e2", "e1", "e2"]), np.array([0, 0, 1, 1]), CFG_INV)


def test_game_folds_keep_games_whole():
    df = pd.DataFrame({"game_id": np.repeat(np.arange(40), 5)})
    fold = game_folds(df, 5, 0)
    assert (df.assign(f=fold).groupby("game_id")["f"].nunique() == 1).all()
    assert set(fold) == set(range(5))


# ── The test on a known SCM ──────────────────────────────────────────────────
def test_shared_effects_are_not_flagged(inv_null):
    t = inv_null["table"]
    assert not inv_null["needed"]
    assert (t["verdict"] == "invariant").all()
    assert (t["n_env"] >= 20).all()


def test_clustered_errors_understate_the_noise_and_the_permutation_says_so(inv_null):
    """phi > 1: under the permutation null, Cochran's Q exceeds its degrees of
    freedom — the clustered SEs are too small at ~8 games per environment."""
    assert (inv_null["table"]["phi"] > 1.2).all()


def test_environment_specific_effects_are_flagged(inv_varies):
    t = inv_varies["table"].set_index("contrast")
    assert inv_varies["needed"]
    assert t.loc["box_ge8 vs box_le5", "verdict"] == "varies"
    # the heaviest-vs-lightest box spans the most heaviness, so the most spread
    assert t.loc["box_ge8 vs box_le5", "unmodeled_tau"] == t["unmodeled_tau"].max()


def test_unmodeled_spread_grows_with_the_true_spread(inv_null, inv_varies):
    a = inv_null["table"].set_index("contrast")["unmodeled_tau"]
    b = inv_varies["table"].set_index("contrast")["unmodeled_tau"]
    assert b.mean() > a.mean() + 0.05


# ── Transfer harness ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def transfer_small():
    from src.evaluation.transfer import (crossfit_nuisances, summarize_transfer,
                                         transfer_evaluation)
    cfg = {**CFG_INV, "invariance": {**CFG_INV["invariance"], "crossfit_folds": 3}}
    pbp = generate_synthetic_alignment(6000, seed=5, seasons=cfg["data"]["seasons"])
    ds = build_dataset(pbp, cfg)
    train, _ = time_aware_split(ds, cfg["data"]["holdout_season"])
    nu = crossfit_nuisances(train, cfg, with_policy=True)
    tab = transfer_evaluation(train, cfg, nu)
    return dict(train=train, nu=nu, table=tab, summary=summarize_transfer(tab))


def test_every_group_is_held_out_once(transfer_small):
    df, tab = transfer_small["train"].df, transfer_small["table"]
    for split, col in [("leave_one_team_out", "defteam"), ("leave_one_season_out", "season")]:
        t = tab[tab["split"] == split]
        assert sorted(t["held_out"]) == sorted(df[col].astype(str).unique())
        assert t["n"].sum() == len(df)
    assert np.isfinite(tab[["lift", "lift_se", "gap", "gap_se",
                            "lift_anchored"]].to_numpy()).all()


def test_anchoring_touches_only_the_deferred_snaps(transfer_small):
    """The anchored policy differs from the transferred one only where it
    defers. The harness policy is deterministic — it departs on every snap —
    so the two lifts coincide; deferring everywhere would be the coordinator
    and score exactly zero."""
    t = transfer_small["table"]
    assert (t["departs"] == 1).all()
    np.testing.assert_allclose(t["lift_anchored"], t["lift"])


def test_in_domain_policy_covers_every_snap(transfer_small):
    p = transfer_small["nu"]["policy_probs"]
    np.testing.assert_allclose(p.sum(1), 1.0)
    assert set(transfer_small["summary"]["split"]) == {"leave_one_team_out",
                                                       "leave_one_season_out"}
