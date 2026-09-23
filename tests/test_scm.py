"""
Causal graph: identification, positivity, and the structural claims the DAG makes.

The guards here are the ones that stop a causal error from becoming a silent
modelling choice — a mediator, a selection variable, or the defense's own front
turning up among the state features.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scm import graph as scm
from src.scm import identify
from src.data.load import FTN_POSTSNAP, FTN_PRESNAP_DEFENSE
from src.schemas.dataset import Dataset


def test_scm_adjustment_matches_state(fitted):
    """The SCM adjustment set and the engineered state must be identical."""
    ds = fitted["ds"]
    assert set(scm.adjustment_columns()) == set(ds.state_cols)
    identify.assert_adjustment_consistency(ds)  # raises on drift / mediator leak


def test_positivity_flags_seeded_hole(fitted):
    """Cover-0 (actions 0,1) is seeded to never occur on 1st down."""
    ds = fitted["ds"]
    assert not identify.is_action_observed(ds, 0, down=1)
    assert not identify.is_action_observed(ds, 1, down=1)
    assert identify.is_action_observed(ds, 0, down=3)   # but present on 3rd-short
    pos = identify.check_positivity(fitted["train"], min_count=10)
    assert pos["n_absent_cells"] > 0
    assert pos["pct_cells_supported"] < 1.0


def test_dowhy_backdoor_identifies(fitted):
    model = identify.build_causal_model(fitted["train"])
    est = identify.identify_estimand(model)
    bd = est.get_backdoor_variables()
    assert len(bd) == len(scm.adjustment_columns())


def test_graph_fingerprint_tracks_structural_edits():
    """The fingerprint must change when the DAG's structural claims change.

    GRAPH_VERSION is hand-maintained and can go stale; the fingerprint is the
    guard that makes a silent edit visible on every stored estimate.
    """
    before = scm.fingerprint()
    assert before == scm.fingerprint()          # deterministic

    scm.EDGES.append((scm.PLAYER_PROXIES, scm.OFF_FORMATION))
    try:
        assert scm.fingerprint() != before
    finally:
        scm.EDGES.pop()
    assert scm.fingerprint() == before


def test_declared_mediator_is_actually_a_mediator():
    """A variable in MEDIATORS must be a descendant of the treatment.

    Through v2.0 `off_playcall` was declared a mediator but had no incoming edge
    from `def_playcall`, so nothing in the graph made it one — the label and the
    structure disagreed, and only the label was ever read.
    """
    children_of_treatment = {t for s_, t in scm.EDGES if s_ == scm.TREATMENT}
    for m in scm.MEDIATORS:
        assert m in children_of_treatment, (
            f"{m} is declared a mediator but the treatment has no edge into it"
        )


def test_selection_node_is_a_collider_on_a_treatment_outcome_path():
    """The selection the dataset performs must be visible in the graph.

    Every modelled row has coverage_charted = 1. That node takes both
    `off_playcall` (runs are ~3% charted) and `epa` (a sack leaves no shell) as
    parents, which is what makes selecting on it both block part of the total
    effect and open a non-causal path.
    """
    assert scm.SELECTION, "the dataset is selected on something; say what"
    for sel in scm.SELECTION:
        parents = {s_ for s_, t in scm.EDGES if t == sel}
        assert len(parents) >= 2, f"{sel} is not a collider: parents={parents}"
        assert scm.OUTCOME in parents
        # ...and reachable from the treatment, or selecting on it would be benign.
        assert parents & ({scm.TREATMENT} | scm.MEDIATORS)


def test_selection_variables_cannot_be_used_as_features(fitted):
    """Adjusting for a variable the sample is conditioned on is a contradiction."""
    ds = fitted["ds"]
    assert not (set(scm.SELECTION) & set(ds.state_cols))

    polluted = Dataset(
        df=ds.df.assign(**{c: 1 for c in scm.SELECTION}),
        numeric_state=ds.numeric_state + sorted(scm.SELECTION),
        categorical_state=ds.categorical_state, action_col=ds.action_col,
        reward_col=ds.reward_col, n_actions=ds.n_actions,
        action_labels=ds.action_labels,
    )
    with pytest.raises(ValueError, match="[Ss]election"):
        identify.assert_adjustment_consistency(polluted)


def test_graph_version_was_bumped_for_the_structural_change():
    """Structural edits must move GRAPH_VERSION, not just the fingerprint.

    The fingerprint catches an unversioned edit after the fact; the version is
    what a human reads on a stored estimate. v2.0 predates the mediator edge and
    the selection node, so an estimate stamped v2.0 assumed a different graph.
    """
    assert scm.GRAPH_VERSION >= "v2.1"
    assert scm.version_tag().startswith(scm.GRAPH_VERSION + "+")


def test_box_count_is_rejected_as_a_confounder(fitted):
    """The guard must actually fire, not just be documented."""
    ds = fitted["ds"]
    box = FTN_PRESNAP_DEFENSE[0]
    polluted = Dataset(
        df=ds.df, numeric_state=ds.numeric_state + [box],
        categorical_state=ds.categorical_state, action_col=ds.action_col,
        reward_col=ds.reward_col, n_actions=ds.n_actions,
        action_labels=ds.action_labels,
    )
    with pytest.raises(ValueError, match="[Dd]efensive-choice"):
        identify.assert_adjustment_consistency(polluted)


def test_postsnap_ftn_flags_are_rejected_as_state(fitted):
    """is_play_action / is_rpo / is_screen_pass are mediators, not features."""
    ds = fitted["ds"]
    polluted = Dataset(
        df=ds.df, numeric_state=ds.numeric_state + list(FTN_POSTSNAP),
        categorical_state=ds.categorical_state, action_col=ds.action_col,
        reward_col=ds.reward_col, n_actions=ds.n_actions,
        action_labels=ds.action_labels,
    )
    with pytest.raises(ValueError):
        identify.assert_adjustment_consistency(polluted)
