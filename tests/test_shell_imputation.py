"""
Shell-imputation regression tests.

Per CLAUDE.md's validation discipline: simulate from a known SCM, run the
machinery, check what comes back against the truth. Here the "truth" is the
complete-data answer, and the masked labels stand in for the run plays whose
coverage shell NGS never charts.

These tests are written to document where the method WORKS and where it does
not. The attenuation test in particular asserts a bias rather than the absence
of one — that bias is a real property of imputing a treatment without
conditioning on the outcome, and a suite that quietly passed over it would be
advertising an unbiasedness this method does not have.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _config import CFG

from src.data.load import generate_synthetic, N_SHELLS
from src.data.features import build_dataset
from src.models.imputation import (add_team_shell_tendency,
                                   assert_imputer_features, build_imputation_frame,
                                   fit_shell_imputer, imputer_features, pool_rubin,
                                   tendency_columns)

NUM = ["down", "ydstogo", "yardline_100", "score_diff", "qtr",
       "n_defense_box", "is_motion", "n_offense_backfield"]
CAT = ["formation", "qb_location"]


@pytest.fixture(scope="module")
def frame():
    ds = build_dataset(generate_synthetic(9000, seed=7), CFG)
    df = ds.df.copy()
    df["shell"] = (df["action"] // 2).astype(int)
    df["charted"] = True
    df["defteam"] = "SYN"
    return df


# ── Rubin's rules ────────────────────────────────────────────────────────────
def test_pool_rubin_widens_the_interval_as_draws_disagree():
    """Disagreement between draws must inflate the interval, not average away.

    The failure this guards: pooling m estimates and quoting the mean's standard
    error, which gets NARROWER with more draws — reporting confidence that comes
    from sampling effort rather than from knowledge.
    """
    within = [0.01] * 8
    agree = pool_rubin([1.0] * 8, within)
    disagree = pool_rubin([0.6, 1.4, 0.8, 1.2, 0.7, 1.3, 0.9, 1.1], within)

    assert disagree.se > agree.se
    assert agree.fraction_missing_information == pytest.approx(0.0, abs=1e-9)
    assert disagree.fraction_missing_information > 0.5
    # The identical-draw case must collapse to the within-imputation variance.
    assert agree.se == pytest.approx(np.sqrt(0.01), rel=1e-6)


def test_pool_rubin_rejects_degenerate_input():
    with pytest.raises(ValueError):
        pool_rubin([1.0], [0.1])              # m = 1 is not multiple imputation
    with pytest.raises(ValueError):
        pool_rubin([1.0, 2.0], [0.1])         # mismatched lengths


# ── Sampling behaviour ───────────────────────────────────────────────────────
def test_draws_sample_the_posterior_rather_than_its_argmax(frame):
    """Weighted sampling, not point imputation — the distinction the data forces.

    On real data the posterior's argmax is no more accurate than writing the
    team's modal shell on every play, so an argmax imputation would look like
    data while carrying none. Draws must reproduce the posterior's frequencies
    and must differ from one another.
    """
    imp = fit_shell_imputer(frame, NUM, CAT, holdout_season=None)
    draws = imp.draw(frame, n_draws=40, seed=0)

    assert draws.shape == (40, len(frame))
    assert set(np.unique(draws)) <= set(range(N_SHELLS))
    assert not np.array_equal(draws[0], draws[1]), "draws are identical — not sampling"

    # Empirical draw frequencies should track the posterior's column means.
    posterior = imp.posterior(frame).mean(axis=0)
    empirical = np.bincount(draws.ravel(), minlength=N_SHELLS) / draws.size
    assert np.abs(empirical - posterior).max() < 0.02

    # And must be strictly more diverse than the argmax would be.
    assert len(np.unique(draws[0])) >= len(np.unique(imp.posterior(frame).argmax(axis=1)))


def test_imputer_refuses_outcome_and_postsnap_features(frame):
    """The imputed variable is the treatment; conditioning it on the outcome
    would write the very association being measured into the data."""
    for bad in ("reward", "epa", "is_play_action", "action"):
        with pytest.raises(ValueError, match="[Pp]ost-snap or outcome"):
            fit_shell_imputer(frame, NUM + [bad], CAT, holdout_season=None)


def test_team_shell_tendency_is_leak_free(frame):
    """A play's own shell must not inform its own tendency feature.

    This feature is what lifts entropy reduction from ~9% to ~15%, so it is
    worth having — and it is one `groupby.mean()` away from leaking the label.
    """
    df = frame.head(400).copy()
    df["defteam"] = ["A"] * 200 + ["B"] * 200
    out = add_team_shell_tendency(df, "shell", "charted", "defteam", N_SHELLS)
    cols = tendency_columns(N_SHELLS)

    first = out.groupby("defteam").head(1)
    for _, row in first.iterrows():
        assert np.allclose([row[c] for c in cols], 1.0 / N_SHELLS), \
            "first play per team must fall back to the uniform prior"

    # Flipping one play's shell must not change that same play's own feature.
    flipped = out.copy()
    idx = flipped.index[10]
    flipped.loc[idx, "shell"] = (int(flipped.loc[idx, "shell"]) + 1) % N_SHELLS
    re_out = add_team_shell_tendency(flipped, "shell", "charted", "defteam", N_SHELLS)
    assert np.allclose([out.loc[idx, c] for c in cols],
                       [re_out.loc[idx, c] for c in cols])


# ── The honest limit ─────────────────────────────────────────────────────────
def test_imputation_attenuates_effects_toward_the_marginal(frame):
    """Imputing a treatment WITHOUT the outcome shrinks its estimated effect.

    This is the method's central cost, and it is a bias Rubin's rules do not
    remove — they correct the variance, not this. Imputed labels are drawn from
    P(shell | pre-snap state), so they are independent of reward given the
    state; plays assigned shell k by the imputer are a mixture of true shells,
    and their mean reward is pulled toward the overall mean.

    The test pins the DIRECTION (toward the marginal, i.e. conservative) and
    requires that the interval widen to acknowledge it. Asserting unbiasedness
    here would be asserting something false.
    """
    df = frame.reset_index(drop=True)
    reward = df["reward"].to_numpy()
    true_shell = df["shell"].to_numpy()
    grand_mean = reward.mean()

    # The shell whose complete-data mean sits furthest from the grand mean, so
    # attenuation has something to attenuate.
    means = {k: reward[true_shell == k].mean() for k in range(N_SHELLS)}
    target = max(means, key=lambda k: abs(means[k] - grand_mean))
    truth = means[target]

    rng = np.random.default_rng(0)
    masked = rng.random(len(df)) < 0.40          # MCAR: the friendly case
    imp = fit_shell_imputer(df[~masked], NUM, CAT, holdout_season=None)
    draws = imp.draw(df[masked], n_draws=10, seed=1)

    ests, variances = [], []
    for d in draws:
        shell = true_shell.copy()
        shell[masked] = d
        sel = shell == target
        ests.append(reward[sel].mean())
        variances.append(reward[sel].var(ddof=1) / sel.sum())
    pooled = pool_rubin(ests, variances)

    complete_case = reward[(~masked) & (true_shell == target)]
    cc_se = complete_case.std(ddof=1) / np.sqrt(len(complete_case))

    # Attenuated: strictly between the truth and the grand mean.
    assert abs(pooled.estimate - grand_mean) < abs(truth - grand_mean), (
        f"expected attenuation toward {grand_mean:+.4f}; "
        f"truth={truth:+.4f} pooled={pooled.estimate:+.4f}"
    )
    # ...and never so severe that it flips sign relative to the grand mean.
    assert np.sign(pooled.estimate - grand_mean) == np.sign(truth - grand_mean)
    # Uncertainty must be acknowledged, not hidden.
    assert pooled.fraction_missing_information > 0.0
    assert pooled.se > 0 and cc_se > 0


def test_information_ratio_reports_how_little_is_known(frame):
    """The headline honesty metric must exist and stay in [0, 1]."""
    imp = fit_shell_imputer(frame, NUM, CAT, holdout_season=None)
    ratio = imp.information_ratio(frame)
    assert 0.0 <= ratio <= 1.0
    assert imp.diagnostics["selection_warning"]


def test_diagnostics_are_time_aware_when_a_holdout_is_given(frame):
    """Diagnostics must come from a held-out season, never in-sample."""
    imp = fit_shell_imputer(frame, NUM, CAT, holdout_season=2023)
    d = imp.diagnostics
    assert d["holdout_season"] == 2023 and d["n_holdout"] > 50
    for key in ("accuracy", "marginal_accuracy", "log_loss", "marginal_log_loss",
                "information_ratio", "confidence_ece", "classwise_ece", "brier",
                "temperature"):
        assert key in d
    # The uncalibrated baseline is kept beside the calibrated number so the
    # temperature's effect is visible rather than asserted.
    assert d["confidence_ece"] <= d["uncalibrated_confidence_ece"] + 1e-9

# ── Declared feature set ─────────────────────────────────────────────────────
def test_builder_produces_every_declared_feature():
    """The declaration and the builder must agree — the imputer's version of
    `test_scm_adjustment_matches_state`.

    Before this existed the feature list lived only in an analysis script, so
    two callers could train different models and nothing would notice. The
    declaration is now authoritative and this is what keeps it honest.
    """
    pbp = generate_synthetic(2000, seed=3)
    pbp["defense_coverage_type"] = pbp["defense_coverage_type"].astype(str)
    frame = build_imputation_frame(pbp, N_SHELLS)

    declared = assert_imputer_features(frame, N_SHELLS)
    numeric, categorical = imputer_features(N_SHELLS)
    assert declared == numeric + categorical
    assert len(declared) == 19, f"expected 19 declared features, got {len(declared)}"
    assert tendency_columns(N_SHELLS)[0] in numeric
    # shell / charted come along so callers can split labeled from unlabeled.
    assert {"shell", "charted"} <= set(frame.columns)
    assert frame["shell"].between(-1, N_SHELLS - 1).all()


def test_declared_features_exclude_everything_post_snap():
    """The declaration cannot name a column the guard would reject."""
    numeric, categorical = imputer_features(N_SHELLS)
    from src.models.imputation import FORBIDDEN_FEATURES
    assert not (set(numeric + categorical) & FORBIDDEN_FEATURES)


def test_missing_declared_feature_is_an_error_not_a_silent_refit():
    frame = build_imputation_frame(generate_synthetic(500, seed=4), N_SHELLS)
    with pytest.raises(ValueError, match="absent from the frame"):
        assert_imputer_features(frame.drop(columns=["n_defense_box"]), N_SHELLS)


def test_fit_defaults_to_the_declared_feature_set():
    """Calling without feature lists must train the repo's declared model."""
    pbp = generate_synthetic(3000, seed=5)
    frame = build_imputation_frame(pbp, N_SHELLS)
    labeled = frame[frame["charted"]].copy()

    imp = fit_shell_imputer(labeled, n_shells=N_SHELLS, holdout_season=None)
    numeric, categorical = imputer_features(N_SHELLS)
    assert imp.numeric_features == numeric
    assert imp.categorical_features == categorical
    # ...and the fitted model records what it used, so a saved run is auditable.
    assert imp.diagnostics["numeric_features"] == numeric
    assert imp.posterior(labeled).shape == (len(labeled), N_SHELLS)
