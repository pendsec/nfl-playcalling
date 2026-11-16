"""
Neural outcome models (PyTorch) — TARNet and DragonNet behind `fit_q_model`.

Anchored on `generate_synthetic_alignment` with the read's effect switched off
(`conf_strength=0`), where `true_alignment_q` gives E[R | s, do(a)] for every
call, so recovered advantages can be checked against the truth.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest


pytest.importorskip("torch")

from _config import CFG_L1
from src.data.load import true_alignment_q, true_alignment_value
from src.models.behavior.propensity import fit_behavior_model
from src.models.outcome.q_model import fit_q_ensemble, fit_q_model

FAST_NET = {"max_epochs": 12, "patience": 3, "hidden": [32, 16], "head_hidden": 8}


def _cfg(**outcome):
    c = copy.deepcopy(CFG_L1)
    c["outcome_model"].update(outcome)
    return c


def _advantages(q, pi):
    return q - (pi * q).sum(axis=1, keepdims=True)


@pytest.fixture(scope="module")
def league(synthetic_league):
    train, test = synthetic_league(12000, seed=3)
    pi_b = fit_behavior_model(train, CFG_L1, diagnose=False)[0].propensity(test.df)
    return dict(train=train, test=test, q_true=true_alignment_q(test.df), pi_b=pi_b)


def _supported_corr(est, league):
    """Correlation of estimated and true advantages over the coordinator's mix,
    on the calls the policy may choose — off support every model extrapolates."""
    pi = league["pi_b"]
    sup = pi >= 0.05
    return np.corrcoef(_advantages(est, pi)[sup], _advantages(league["q_true"], pi)[sup])[0, 1]


@pytest.fixture(scope="module")
def fitted(league):
    specs = {"tarnet": _cfg(type="tarnet", neural={**FAST_NET, "heads": "separate"}),
             "dragonnet": _cfg(type="dragonnet", neural=FAST_NET)}      # factored by default
    return {k: fit_q_model(league["train"], c) for k, c in specs.items()}


# ── Truth helper ─────────────────────────────────────────────────────────────
def test_true_q_matches_the_policy_value(league):
    df = league["test"].df
    a = df["action"].to_numpy()
    q = league["q_true"]
    assert q.shape == (len(df), 12)
    np.testing.assert_allclose(q[np.arange(len(df)), a].mean(), true_alignment_value(df, a))


# ── Through the one interface ────────────────────────────────────────────────
@pytest.mark.parametrize("kind", ["tarnet", "dragonnet"])
def test_every_network_scores_every_call(fitted, league, kind):
    q = fitted[kind]
    out = q.predict_all_actions(league["test"].df)
    assert out.shape == (len(league["test"].df), 12)
    assert np.isfinite(out).all() and q.q_scale > 0


def test_the_factored_head_is_the_default(fitted):
    from src.models.outcome.neural import FactoredHead, SeparateHeads
    assert isinstance(fitted["dragonnet"].net.q_head, FactoredHead)
    assert isinstance(fitted["tarnet"].net.q_head, SeparateHeads)


def test_the_adjustment_set_is_enforced(league):
    t = league["train"]
    bad = type(t)(df=t.df, numeric_state=t.numeric_state + ["_coach_read"],
                  categorical_state=t.categorical_state, action_col=t.action_col,
                  reward_col=t.reward_col, n_actions=t.n_actions, action_labels=t.action_labels)
    with pytest.raises(ValueError, match="adjustment set"):
        fit_q_model(bad, _cfg(type="dragonnet"))


def test_unknown_types_and_heads_are_rejected(league):
    with pytest.raises(ValueError, match="outcome_model.type"):
        fit_q_model(league["train"], _cfg(type="xgboost"))
    with pytest.raises(ValueError, match="experiments/outcome_models"):
        fit_q_model(league["train"], _cfg(type="dr_learner"))
    with pytest.raises(ValueError, match="neural.heads"):
        fit_q_model(league["train"], _cfg(type="tarnet", neural={"heads": "shared"}))
    with pytest.raises(ValueError, match="neural.early_stop"):
        fit_q_model(league["train"], _cfg(type="dragonnet", neural={"early_stop": "auc"}))


# ── Behaviour ────────────────────────────────────────────────────────────────
def test_networks_recover_the_direction_of_the_effects(fitted, league):
    for kind in ("tarnet", "dragonnet"):
        assert _supported_corr(fitted[kind].predict_all_actions(league["test"].df), league) > 0.1


def test_early_stopping_keeps_the_best_epoch(fitted):
    hist = pd.DataFrame(fitted["tarnet"].history)
    assert len(hist) >= 2 and hist["val_loss"].iloc[0] > hist["val_loss"].min()


def test_dragonnet_learns_the_propensity(fitted, league):
    q = fitted["dragonnet"]
    df = league["test"].df
    p = q.propensity(df)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)
    a = df["action"].to_numpy()
    prior = np.bincount(league["train"].df["action"], minlength=12) / len(league["train"].df)
    # Beats the marginal call frequencies on the held-out season's log-likelihood.
    assert np.log(p[np.arange(len(a)), a]).mean() > np.log(prior[a] + 1e-12).mean()


def test_unseen_categories_get_the_zero_embedding(fitted, league):
    q = fitted["tarnet"]
    df = league["test"].df.head(50).copy()
    df["defteam"] = "NEW_TEAM"
    _, cat = q.encoder.transform(df)
    j = q.encoder.categorical.index("defteam")
    assert (cat[:, j] == 0).all()
    import torch
    assert torch.all(q.net.phi.embed[j].weight[0] == 0)


def test_embedding_width_is_capped_by_embed_max(league):
    q = fit_q_model(league["train"], _cfg(type="tarnet",
                                          neural={**FAST_NET, "max_epochs": 1, "embed_max": 3}))
    widths = [e.embedding_dim for e in q.net.phi.embed]
    assert max(widths) == 3
    assert all(w == min(3, (k + 1) // 2) for w, k in zip(widths, q.encoder.cardinalities))


def test_outcome_early_stopping_ignores_the_auxiliary_terms(league):
    """Stopping on Q's error alone: the logged validation loss is the outcome MSE
    (standardized reward), smaller than the whole DragonNet loss, which adds the
    propensity cross-entropy and the targeted term."""
    n = {**FAST_NET, "max_epochs": 2}
    whole = fit_q_model(league["train"], _cfg(type="dragonnet", neural=n))
    q_only = fit_q_model(league["train"], _cfg(type="dragonnet",
                                               neural={**n, "early_stop": "outcome"}))
    assert q_only.history[0]["val_loss"] < 1.5 < whole.history[0]["val_loss"]


def test_training_is_reproducible(league):
    c = _cfg(type="dragonnet", neural={**FAST_NET, "max_epochs": 3})
    df = league["test"].df.head(200)
    a = fit_q_model(league["train"], c).predict_all_actions(df)
    b = fit_q_model(league["train"], c).predict_all_actions(df)
    np.testing.assert_allclose(a, b)


def test_the_bootstrap_ensemble_takes_a_network(league):
    c = _cfg(type="dragonnet", neural={**FAST_NET, "max_epochs": 2}, ensemble={"n_boot": 2})
    ens = fit_q_ensemble(league["train"], c)
    df = league["test"].df.head(100)
    sd = ens.advantage_sd(df, league["pi_b"][:100])
    assert len(ens.members) == 2 and sd.shape == (100, 12) and np.isfinite(sd).all()


def test_the_ensemble_refits_with_the_given_fit(league):
    calls = []
    def fit(ds, cfg):
        calls.append(len(ds.df))
        return fit_q_model(ds, cfg)
    fit_q_ensemble(league["train"], _cfg(ensemble={"n_boot": 3}), fit=fit)
    assert len(calls) == 3


def test_lightgbm_still_fits_after_torch_has_trained(fitted, league):
    """The OpenMP guard: a LightGBM fit after a network trained must not hang."""
    q = fit_q_model(league["train"], copy.deepcopy(CFG_L1))
    assert np.isfinite(q.predict_all_actions(league["test"].df.head(100))).all()


def test_snap_batches_match_the_dataloader_stream():
    """The fast batcher yields exactly DataLoader's shuffled batches, epoch after
    epoch, so swapping it in changes speed and nothing else."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.models.outcome.neural import SnapBatches
    x, y = torch.arange(103), torch.arange(103) * 2.0
    dl = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True,
                    generator=torch.Generator().manual_seed(7))
    sb = SnapBatches((x, y), 16, torch.Generator().manual_seed(7))
    for _ in range(4):
        for (a1, b1), (a2, b2) in zip(dl, sb, strict=True):
            assert torch.equal(a1, a2) and torch.equal(b1, b2)


def test_a_net_that_never_validates_fails_clearly(league):
    """No epoch with a finite validation loss leaves no best state to restore:
    say so, rather than crash inside `load_state_dict(None)`."""
    cfg = _cfg(type="dragonnet", neural={**FAST_NET, "max_epochs": 0})
    with pytest.raises(ValueError, match="never reached a finite validation loss"):
        fit_q_model(league["train"], cfg)
