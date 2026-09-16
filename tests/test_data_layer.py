"""
Data layer: the (S, A, R) build, leak discipline, and the FTN merge.

Anchored on `generate_synthetic`, which simulates from a KNOWN structural causal
model so the assertions have a truth to check against: 12 actions across all
downs, a hidden `coach_read` confounding the behavior policy, and Cover-0 seeded
to occur only on 3rd/4th-and-short (a deliberate positivity hole).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scm import graph as scm
from src.scm import identify
from _config import CFG
from src.data.features import CARRIED_COLUMNS, assert_ftn_coverage, build_dataset
from src.data.load import generate_synthetic, merge_ftn


def test_dataset_12_actions(fitted):
    ds = fitted["ds"]
    assert ds.n_actions == 12
    assert ds.df[ds.action_col].nunique() == 12
    assert set(ds.df["down"].unique()) == {1, 2, 3, 4}
    assert len(ds.df) > 4000


def test_no_postsnap_leakage_in_state(fitted):
    ds = fitted["ds"]
    forbidden = {"epa", "reward", "action", "off_playcall",
                 "_coach_read", "_true_blitz_effect"}
    assert forbidden.isdisjoint(set(ds.state_cols))


def test_ftn_presnap_features_are_in_the_state_and_adjustment_set(fitted):
    """FTN's pre-snap offensive presentation is an ordinary confounder."""
    ds = fitted["ds"]
    for c in ("is_motion", "n_offense_backfield", "qb_location"):
        assert c in ds.state_cols, f"{c} should be adjusted for"
        assert c in scm.adjustment_columns()
    identify.assert_adjustment_consistency(ds)


def test_carried_columns_never_enter_the_state(fitted):
    """n_defense_box and the post-snap flags are carried, never modeled.

    Two different errors guarded by one test. Box count is pre-snap and looks
    like a fine confounder, but it is part of the defense's own decision, so
    adjusting for it blocks a slice of the effect. The is_* flags are post-snap
    reveals and would be outright leakage.
    """
    ds = fitted["ds"]
    assert CARRIED_COLUMNS, "nothing carried — did the FTN merge silently drop?"
    for c in CARRIED_COLUMNS:
        assert c in ds.df.columns, f"{c} should be carried through build_dataset"
        assert c not in ds.state_cols, f"{c} must not be a state feature"
        assert c not in scm.adjustment_columns()


def test_ftn_zero_sentinel_is_scrubbed_to_missing():
    """FTN writes 0 / "0" where it charts nothing; 0 defenders is not a front."""
    pbp = pd.DataFrame({
        "game_id": ["g1", "g1"], "play_id": [1, 2],
        "season": [2023, 2023], "play_type": ["pass", "pass"], "epa": [0.1, -0.2],
    })
    ftn = pd.DataFrame({
        "nflverse_game_id": ["g1", "g1"], "nflverse_play_id": [1, 2],
        "n_defense_box": [0, 6], "qb_location": ["0", "S"],
        "is_motion": [True, False], "n_offense_backfield": [1.0, 1.0],
        "is_play_action": [False, False], "is_rpo": [False, False],
        "is_screen_pass": [False, False],
    })
    out = merge_ftn(pbp, ftn)
    assert pd.isna(out.loc[0, "n_defense_box"])     # sentinel -> missing
    assert out.loc[1, "n_defense_box"] == 6         # real count preserved
    assert pd.isna(out.loc[0, "qb_location"])


def test_merge_ftn_join_is_robust_to_key_dtype_mismatch():
    """nflfastR stores play_id as float, FTN as int — the join must still land.

    A dtype mismatch here fails silently rather than loudly: pandas returns an
    all-NaN right side, and every FTN feature becomes missing without an error.
    The keys are cast explicitly so that cannot happen.
    """
    pbp = pd.DataFrame({
        "game_id": ["g1", "g1", "g2"], "play_id": [1.0, 2.0, np.nan],
        "season": [2023] * 3, "play_type": ["pass"] * 3, "epa": [0.1, -0.2, 0.3],
    })
    ftn = pd.DataFrame({
        "nflverse_game_id": ["g1", "g1"], "nflverse_play_id": np.array([1, 2], dtype="int32"),
        "n_defense_box": [6, 7], "qb_location": ["S", "U"],
        "is_motion": [True, False], "n_offense_backfield": [1.0, 2.0],
        "is_play_action": [False, False], "is_rpo": [False, False],
        "is_screen_pass": [False, False],
    })
    out = merge_ftn(pbp, ftn)

    assert len(out) == len(pbp), "left join must not add or drop rows"
    assert out.loc[0, "n_defense_box"] == 6 and out.loc[1, "n_defense_box"] == 7
    assert pd.isna(out.loc[2, "n_defense_box"])   # NaN play_id simply does not match
    assert not any(k in out.columns for k in ("nflverse_game_id", "nflverse_play_id"))


def test_use_ftn_excludes_plays_ftn_never_charted(fitted):
    """With FTN features declared, an uncharted play has none to condition on.

    Defaulting them would make `is_motion = 0` read as "no motion" rather than
    "unknown" — the silent degradation `assert_ftn_coverage` prevents at season
    scale, scattered across individual plays instead.
    """
    ds = fitted["ds"]
    assert ds.df["n_defense_box"].notna().all()

    pbp = generate_synthetic(1500, seed=11)
    pbp.loc[pbp.index[:200], "n_defense_box"] = np.nan

    on = build_dataset(pbp, {**CFG, "data": {**CFG["data"], "use_ftn": True}})
    off = build_dataset(pbp, {**CFG, "data": {**CFG["data"], "use_ftn": False}})

    assert len(on.df) < len(off.df), "use_ftn must exclude the uncharted plays"
    assert on.df["n_defense_box"].notna().all()
    # Switched off, the rows survive on defaults so a pre-FTN config still builds.
    assert off.df["n_defense_box"].isna().any()


def test_ftn_coverage_guard_rejects_a_pre_2022_season():
    """Training FTN features on a season FTN does not cover must fail loudly.

    Silently defaulting them would make "no motion" a property of 2021 rather
    than of the play — a feature confounded with season across half of training.
    """
    df = pd.DataFrame({
        "season": [2021] * 50 + [2023] * 50,
        "play_type": ["pass"] * 100, "epa": [0.0] * 100,
        "n_defense_box": [np.nan] * 50 + [6.0] * 50,
    })
    cfg = {"data": {"use_ftn": True, "holdout_season": 2023, "ftn_min_coverage": 0.8}}
    with pytest.raises(ValueError, match="2021"):
        assert_ftn_coverage(df, cfg)
    # ...and must stay quiet when the seasons are covered.
    assert_ftn_coverage(df[df.season == 2023], cfg)
    # ...and be a no-op when FTN is switched off entirely.
    assert_ftn_coverage(df, {"data": {"use_ftn": False, "holdout_season": 2023}})
