"""
Conditional-independence machinery, checked on graphs whose truth is known.

The tests on real data are only as good as the test itself, so it is held to
three standards here: it must not reject an independence that is true, it must
reject one that is false, and it must refuse to test a claim that a latent
variable makes unfalsifiable.
"""

from __future__ import annotations


import numpy as np
import pandas as pd


from src.scm import graph as scm
from src.scm.ci_tests import (check_graph, ci_test, effective_parents,
                               implied_independencies)
from src.scm.dag import CausalGraph


def _chain(n=20_000, direct=0.0, seed=0):
    """x -> z -> y, optionally with a direct x -> y the graph will not declare."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = 0.8 * x + rng.normal(size=n)
    y = 0.8 * z + direct * x + rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": z, "y": y})


def test_a_true_independence_is_not_rejected():
    r = ci_test(_chain(), ["x"], ["y"], ["z"])
    assert r["p_value"] > 0.01 and r["delta_r2"] < 1e-3, r


def test_a_false_independence_is_rejected():
    r = ci_test(_chain(direct=0.3), ["x"], ["y"], ["z"])
    assert r["p_value"] < 1e-6 and r["delta_r2"] > 5e-3, r


def test_conditioning_does_not_manufacture_dependence_through_binning():
    """Bins alone leave within-bin variation of z, which x tracks — so x would
    appear to predict y given z even though the chain makes them independent.
    The linear term is what prevents that; this pins it."""
    r = ci_test(_chain(n=60_000, seed=3), ["x"], ["y"], ["z"])
    assert r["delta_r2"] < 5e-4, r


def test_a_latent_collider_makes_a_claim_untestable():
    """x -> t <- u -> y with u unobserved and t -> y: blocking the causal path
    through t opens the path through u, so no observed set separates x and y."""
    testable, untestable = implied_independencies(
        [("x", "t"), ("u", "t"), ("u", "y"), ("t", "y")], latent={"u"})
    assert ("x", "y") in untestable or ("y", "x") in untestable
    assert not testable


def test_the_alignment_graphs_no_direct_edge_claim_is_untestable():
    """The one confounder with no direct edge into EPA is offensive tendency —
    and the coach's unrecorded read makes exactly that claim uncheckable."""
    _, untestable = implied_independencies(
        scm.ALIGNMENT_GRAPH.edges,
        latent=scm.ALIGNMENT_GRAPH.unobserved | {scm.DEF_COVERAGE})
    assert {frozenset(p) for p in untestable} == {frozenset({scm.EPA, scm.OFF_TENDENCIES})}


def test_check_graph_flags_a_missing_edge_and_grades_edges_into_the_outcome():
    """c confounds t -> y; d drives only t. A graph that omits c -> t is wrong
    in a testable way, and a declared d -> y edge has no support in the data."""
    rng = np.random.default_rng(1)
    n = 40_000
    c, d = rng.normal(size=n), rng.normal(size=n)
    t = 0.7 * c + 0.7 * d + rng.normal(size=n)
    y = 0.5 * t + 0.6 * c + rng.normal(size=n)
    df = pd.DataFrame({"c": c, "d": d, "t": t, "y": y})
    g = CausalGraph(name="toy", version="t", treatment="t", outcome="y",
                    edges=[("d", "t"), ("t", "y"), ("c", "y"), ("d", "y")],
                    observed_confounders=["c", "d"],
                    confounder_columns={"c": ["c"], "d": ["d"]},
                    observed={"c", "d", "t", "y"}, unobserved=set(), mediators=set())
    report = check_graph(df, g, {"c": ["c"], "d": ["d"], "t": ["t"], "y": ["y"]}, latent=set())
    verdict = dict(zip(report["claim"], report["verdict"]))
    assert verdict["c ⊥ d"] == "consistent"            # true: c and d are independent
    assert verdict["c ⊥ t"] == "violated"              # the omitted c -> t edge
    assert verdict["c -> y"] == "supported"
    assert verdict["d -> y"] == "unsupported"


def test_execution_adds_no_testable_claim():
    """An unobserved node with a single child marginalizes to direct edges:
    routing player quality and the call through `execution` must leave every
    observed independence claim exactly as it was with direct edges into EPA.
    The node explains; it does not constrain."""
    g = scm.ALIGNMENT_GRAPH
    via = [s for s, t in g.edges if t == scm.EXECUTION]
    direct = [e for e in g.edges if scm.EXECUTION not in e] + [(s, scm.EPA) for s in via]
    latent = {scm.COACH_READ, scm.OFF_INTENT, scm.DEF_COVERAGE}
    with_node = implied_independencies(g.edges, latent | {scm.EXECUTION})
    without = implied_independencies(direct, latent)
    as_sets = lambda r: ({(frozenset({i.x, i.y}), i.given) for i in r[0]},
                         {frozenset(p) for p in r[1]})
    assert as_sets(with_node) == as_sets(without)


def test_effective_parents_see_through_execution_but_not_the_read():
    """`execution` only feeds EPA, so its parents act on EPA through it. The
    coach's read also drives the call — a confounder, not a pass-through — so
    nothing is promoted through it."""
    g = scm.ALIGNMENT_GRAPH
    eff = effective_parents(g.edges, scm.EPA, g.unobserved | {scm.DEF_COVERAGE})
    assert eff[scm.GAME_STATE] == ""                       # its own edge
    assert eff[scm.PLAYER_PROXIES] == scm.EXECUTION
    assert eff[scm.TEAMS] == scm.EXECUTION
    assert scm.COACH_READ not in eff and scm.EXECUTION not in eff


def test_execution_is_an_unobserved_mediator_of_the_call():
    g = scm.ALIGNMENT_GRAPH
    assert scm.EXECUTION in g.unobserved and scm.EXECUTION in g.mediators
    assert (g.treatment, scm.EXECUTION) in g.edges
    assert [t for s, t in g.edges if s == scm.EXECUTION] == [scm.EPA]
    assert g.version.startswith("v3.2")


# ── The offense's plan ───────────────────────────────────────────────────────
def test_the_read_now_confounds_through_the_offenses_call():
    """The read is a read OF the offense's plan, so a backdoor path runs
    alignment <- read <- intent -> run/pass. Before the plan was in the graph,
    the read reached the outcome only through execution."""
    import networkx as nx
    g = scm.ALIGNMENT_GRAPH
    dag = nx.DiGraph(g.edges)
    assert scm.OFF_INTENT in g.unobserved
    assert nx.has_path(dag, scm.OFF_INTENT, scm.COACH_READ)
    assert (scm.OFF_INTENT, scm.OFF_PLAYCALL) in g.edges
    assert scm.OFF_PLAYCALL in g.mediators   # still never conditioned on


def test_the_plan_adds_nothing_to_adjust_for():
    """An unobserved node cannot enter the adjustment set, so no estimate moves."""
    g = scm.ALIGNMENT_GRAPH
    assert scm.OFF_INTENT not in g.adjustment_set()
    assert set(g.adjustment_columns()) == set(sum(g.confounder_columns.values(), []))


def test_the_plan_keeps_the_claim_the_data_confirms():
    """tendency -> intent is left out on purpose: it would make personnel and
    tendency adjacent and erase the one independence the data confirms on both
    the training seasons and the holdout."""
    g = scm.ALIGNMENT_GRAPH
    assert (scm.OFF_TENDENCIES, scm.OFF_INTENT) not in g.edges
    testable, untestable = implied_independencies(g.edges, g.unobserved | {scm.DEF_COVERAGE})
    pairs = {frozenset({i.x, i.y}) for i in testable}
    assert frozenset({scm.OFF_PERSONNEL, scm.OFF_TENDENCIES}) in pairs
    # ...and the claim about player quality and the run/pass call is gone: with
    # the read reaching the offense, conditioning on the alignment opens a path.
    assert frozenset({scm.OFF_PLAYCALL, scm.PLAYER_PROXIES}) not in pairs
