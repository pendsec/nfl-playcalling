"""
Conditional-independence tests: checking a hand-built graph against the data.

A DAG makes two kinds of claims, and they are checked differently.

  * A MISSING edge is the strong claim. It implies conditional independencies —
    X ⊥ Y | Z for some observed Z — and each is a falsifiable prediction. If
    the data shows X and Y still dependent given Z, the graph is missing an
    edge (or a path through an unmodeled variable). `implied_independencies`
    enumerates these from d-separation; `check_graph` tests every one.

  * A PRESENT edge only permits dependence, so d-separation cannot test it.
    For edges into the outcome — the ones that decide what is a confounder —
    `check_graph` instead measures how much each parent still explains about
    the outcome given the other observed parents ("edge relevance").

Latent variables limit both. A claim is testable only if some set of OBSERVED
variables d-separates the pair; with the coach's unrecorded read in the graph,
conditioning on the alignment opens `x -> alignment <- read -> epa`, so some
claims have no observed separator at all. Those are reported as untestable
rather than silently skipped — an unfalsifiable claim is worth knowing about.

The statistical test is regression-based: does adding X improve the prediction
of Y beyond Z? Numeric columns are quantile-binned and categoricals one-hot
encoded, so dependence may be nonlinear in each variable, but interactions are
not modeled — a test can miss dependence that only appears jointly. At the sizes
here (~10^5 plays) almost any dependence is "significant", so every result
carries an effect size (incremental R^2) and verdicts require both a small
p-value AND an effect above a floor.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from .dag import CausalGraph


@dataclass(frozen=True)
class Implication:
    x: str
    y: str
    given: tuple[str, ...]


def implied_independencies(
    edges: list[tuple[str, str]], latent: set[str]
) -> tuple[list[Implication], list[tuple[str, str]]]:
    """Every independence the graph implies among observed nodes.

    Returns `(testable, untestable)`: for each non-adjacent pair of observed
    nodes, either a minimal observed set that d-separates them, or the pair
    listed as untestable because no observed set does.
    """
    dag = nx.DiGraph(edges)
    observed = sorted(set(dag.nodes) - set(latent))
    testable, untestable = [], []
    for x, y in itertools.combinations(observed, 2):
        if _adjacent_after_projection(dag, set(latent), x, y):
            continue
        z = nx.find_minimal_d_separator(dag, {x}, {y},
                                        restricted=set(observed) - {x, y})
        if z is None:
            untestable.append((x, y))
        else:
            testable.append(Implication(x, y, tuple(sorted(z))))
    return testable, untestable


def _adjacent_after_projection(dag: nx.DiGraph, latent: set[str], x: str, y: str) -> bool:
    """Are x and y adjacent once the unobserved nodes are marginalized out?

    A path that runs entirely through latent nodes acts, as far as the data can
    tell, like an edge: `player_proxies -> execution -> epa` is observationally
    `player_proxies -> epa`, and a latent common cause of x and y links them
    the same way. Such pairs are not claims of independence, so they are neither
    tested nor reported as untestable. (The latent projection of a DAG.)
    """
    if dag.has_edge(x, y) or dag.has_edge(y, x):
        return True
    def through_latent(a: str, b: str) -> bool:
        sub = dag.subgraph(latent | {a, b})
        return a in sub and b in sub and nx.has_path(sub, a, b)
    if through_latent(x, y) or through_latent(y, x):
        return True
    return any(through_latent(l, x) and through_latent(l, y) for l in latent if l in dag)


# ── The statistical test ─────────────────────────────────────────────────────
def _encode(df: pd.DataFrame, cols: list[str], bins: int) -> np.ndarray:
    """One-hot categoricals and low-cardinality numerics; for the rest, a linear
    term PLUS quantile-bin dummies.

    The linear term matters for conditioning. Bins alone control for Z only up
    to its within-bin variation, and anything correlated with Z then appears to
    predict Y through that residual — a false "violation" manufactured by the
    encoding. Linear + bins controls exactly for linear dependence and
    approximately for nonlinear.
    """
    parts = []
    for c in cols:
        s = df[c]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if numeric and s.nunique() > 6:
            z = s.astype(float)
            parts.append(((z - z.mean()) / (z.std() or 1.0)).rename(f"{c}_lin").to_frame())
            s = pd.qcut(s.rank(method="first"), bins, labels=False)
        parts.append(pd.get_dummies(s.astype(str), prefix=c, drop_first=True, dtype=float))
    if not parts:
        return np.empty((len(df), 0))
    return pd.concat(parts, axis=1).to_numpy()


def _is_response(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _rss_rank(y: np.ndarray, X: np.ndarray) -> tuple[float, int]:
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid), int(np.linalg.matrix_rank(X))


def ci_test(df: pd.DataFrame, x_cols: list[str], y_cols: list[str],
            z_cols: list[str], bins: int = 5) -> dict:
    """Does X still predict Y once Z is known?

    Independence is symmetric, so the response is taken from whichever side
    has numeric columns (preferring Y). With several response columns each is
    tested and the strongest dependence reported, with a Bonferroni-corrected
    p-value.
    """
    if not any(_is_response(df[c]) for c in y_cols):
        x_cols, y_cols = y_cols, x_cols
    responses = [c for c in y_cols if _is_response(df[c])]
    if not responses:
        raise ValueError(f"no numeric column to test on either side: {x_cols} / {y_cols}")

    frame = df[list(dict.fromkeys(x_cols + y_cols + z_cols))].dropna()
    Z = _encode(frame, z_cols, bins)
    XZ = np.column_stack([Z, _encode(frame, x_cols, bins)])
    best = {"delta_r2": 0.0, "p_value": 1.0, "response": responses[0]}
    for col in responses:
        y = frame[col].to_numpy(float)
        tss = float(((y - y.mean()) ** 2).sum())
        if tss == 0:
            continue
        rss_r, rank_r = _rss_rank(y, Z)
        rss_f, rank_f = _rss_rank(y, XZ)
        df1, df2 = max(rank_f - rank_r, 1), max(len(y) - rank_f, 1)
        F = ((rss_r - rss_f) / df1) / (rss_f / df2) if rss_f > 0 else np.inf
        p = min(1.0, float(stats.f.sf(F, df1, df2)) * len(responses))
        delta = (rss_r - rss_f) / tss
        if delta > best["delta_r2"]:
            best = {"delta_r2": float(delta), "p_value": p, "response": col}
    best["n"] = len(frame)
    return best


# ── Checking a whole graph ───────────────────────────────────────────────────
def alignment_node_columns(graph: CausalGraph, action_col: str = "action",
                           reward_col: str = "reward") -> dict[str, list[str]]:
    """Abstract node -> concrete columns for the every-snap dataset."""
    from . import graph as scm

    cols = {node: list(c) for node, c in graph.confounder_columns.items()}
    cols[graph.treatment] = [action_col]
    cols[graph.outcome] = [reward_col]
    cols[scm.OFF_PLAYCALL] = ["play_type"]
    return cols


def check_graph(
    df: pd.DataFrame,
    graph: CausalGraph,
    columns: dict[str, list[str]],
    latent: set[str] | None = None,
    alpha: float = 1e-3,
    min_effect: float = 1e-3,
) -> pd.DataFrame:
    """Test every implied independence, and the relevance of every edge into
    the outcome. One row per check, with a verdict.

    Verdicts for implied independencies: `consistent` (not rejected, or
    dependence below `min_effect`), `violated` (rejected with a non-trivial
    effect — the graph is missing an edge), `untestable`. For edges into the
    outcome: `supported`, `weak` (significant but below the floor),
    `unsupported`. Edge relevance conditions on the treatment, which the latent
    read makes a collider — so it is an approximate measure, not a clean test.
    """
    # Anything the data has no columns for is latent — by definition, not by
    # declaration, so a node can never be "observed" without data behind it.
    nodes = {n for e in graph.edges for n in e}
    latent = set(latent) if latent is not None else nodes - set(columns)
    rows = []
    testable, untestable = implied_independencies(graph.edges, latent)
    for imp in testable:
        r = ci_test(df, columns[imp.x], columns[imp.y],
                    [c for n in imp.given for c in columns[n]])
        rejected = r["p_value"] < alpha and r["delta_r2"] >= min_effect
        rows.append({"check": "implied independence", "claim": f"{imp.x} ⊥ {imp.y}",
                     "given": ", ".join(imp.given) or "—", **r,
                     "verdict": "violated" if rejected else "consistent"})
    for x, y in untestable:
        rows.append({"check": "implied independence", "claim": f"{x} ⊥ {y}",
                     "given": "no observed separator", "delta_r2": np.nan,
                     "p_value": np.nan, "response": "", "n": 0, "verdict": "untestable"})

    parents = effective_parents(graph.edges, graph.outcome, latent)
    for x, via in parents.items():
        if x == graph.treatment:
            continue
        others = [c for n in parents if n != x for c in columns[n]]
        r = ci_test(df, columns[x], columns[graph.outcome], others)
        verdict = ("supported" if r["p_value"] < alpha and r["delta_r2"] >= min_effect
                   else "weak" if r["p_value"] < alpha else "unsupported")
        claim = f"{x} -> {graph.outcome}" + (f" (via {via})" if via else "")
        rows.append({"check": "edge into outcome", "claim": claim,
                     "given": "other observed parents", **r, "verdict": verdict})
    return pd.DataFrame(rows)


def effective_parents(edges: list[tuple[str, str]], outcome: str,
                      latent: set[str]) -> dict[str, str]:
    """Observed variables that act on the outcome directly or only through
    unobserved pass-through nodes.

    An unobserved node whose every child is the outcome (or another such node)
    marginalizes to direct edges from its parents: `x -> execution -> epa` is,
    from the data's point of view, `x -> epa`. Returns {observed parent: name of
    the latent it acts through, or ""}. A latent with any other child — like the
    coach's read, which also drives the call — is a confounder, not a
    pass-through, and its parents are not promoted.
    """
    dag = nx.DiGraph(edges)
    passthrough: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node in latent - passthrough:
            if node in dag and set(dag.successors(node)) <= {outcome} | passthrough:
                passthrough.add(node)
                changed = True
    # Direct observed parents first, so a variable with its own edge into the
    # outcome is never labeled as acting only through a latent.
    out: dict[str, str] = {p: "" for p in dag.predecessors(outcome) if p not in latent}
    frontier = [p for p in dag.predecessors(outcome) if p in passthrough]
    seen = set(frontier)
    while frontier:
        lat = frontier.pop(0)
        for p in dag.predecessors(lat):
            if p in passthrough and p not in seen:
                seen.add(p)
                frontier.append(p)
            elif p not in latent and p not in out:
                out[p] = lat
    return out
