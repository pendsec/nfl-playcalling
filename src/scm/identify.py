"""
Causal identification (Step 2) — DoWhy backdoor confirmation + positivity.

Two jobs:

  1. `identify_estimand` — register the treatment/outcome/adjustment set with
     DoWhy and run its identification algorithm, confirming the causal effect of
     def_playcall on EPA is backdoor-identified by the observed adjustment set.
     Every OPE estimate downstream ships with this identified estimand + the
     assumed graph version (CLAUDE.md convention).

  2. `check_positivity` — per-stratum coverage of the action space. Strata where
     some call is (near-)never observed violate positivity; the learned policy
     must not make confident recommendations there. This bounds how aggressive
     pi* may be, independent of DoWhy.

The adjustment set comes from `scm.graph` so the DAG stays the single source of
truth — `assert_adjustment_consistency` guards against drift between the declared
SCM and the engineered features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import graph as scm
from ..schemas.dataset import Dataset


def assert_adjustment_consistency(ds: Dataset) -> list[str]:
    """Confirm the SCM adjustment columns exist in the data and no mediator leaks.

    Returns the adjustment columns actually present. Raises if the SCM declares a
    confounder the data layer never built, or if a post-snap mediator appears in
    the state — either is a causal-discipline violation, not a silent mismatch.
    """
    declared = scm.adjustment_columns()
    missing = [c for c in declared if c not in ds.df.columns]
    if missing:
        raise ValueError(
            f"SCM declares adjustment columns absent from the data: {missing}. "
            "graph.CONFOUNDER_COLUMNS and the data-layer state features disagree."
        )
    leaked = [m for m in scm.MEDIATORS if m in ds.state_cols]
    if leaked:
        raise ValueError(f"Post-snap mediator(s) leaked into the state: {leaked}")
    # State and adjustment set must be the same thing by construction.
    extra = set(ds.state_cols) - set(declared)
    if extra:
        raise ValueError(
            f"State features not accounted for in the SCM adjustment set: {sorted(extra)}"
        )
    return declared


def build_causal_model(ds: Dataset, outcome_col: str = "reward"):
    """Build a DoWhy CausalModel for def_playcall -> outcome given the SCM."""
    from dowhy import CausalModel

    common_causes = assert_adjustment_consistency(ds)
    return CausalModel(
        data=ds.df,
        treatment=ds.action_col,
        outcome=outcome_col,
        common_causes=common_causes,
    )


def identify_estimand(model, verbose: bool = False):
    """Run DoWhy identification; return the identified estimand (backdoor)."""
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    if verbose:
        print(estimand)
    return estimand


def check_positivity(
    ds: Dataset, min_count: int = 10, use_score_bin: bool = False
) -> dict:
    """Per-stratum action coverage. Surfaces where positivity fails.

    Strata = down x distance-bucket (optionally x score-bucket). With 12 actions
    even a rich dataset leaves many (stratum, action) cells thin — so the useful
    signals are at the CELL level: the fraction of (stratum, action) cells with
    adequate support, and the *structurally absent* cells (zero observations),
    which are true positivity holes the policy must never exploit.
    """
    df = ds.df.copy()
    df["_down_bin"] = df["down"].astype(int).astype(str)
    df["_dist_bin"] = pd.cut(
        df["ydstogo"], bins=[0, 3, 7, 15, 100],
        labels=["short", "med", "long", "vlong"],
    ).astype(str)
    strata = ["_down_bin", "_dist_bin"]
    if use_score_bin:
        df["_score_bin"] = pd.cut(
            df["score_diff"], bins=[-100, -8, 0, 8, 100],
            labels=["down_big", "down", "up", "up_big"],
        ).astype(str)
        strata.append("_score_bin")

    counts = (
        df.groupby(strata + [ds.action_col]).size()
        .unstack(fill_value=0)
        .reindex(columns=range(ds.n_actions), fill_value=0)
    )

    n_cells = counts.size
    absent = counts == 0
    supported = counts >= min_count
    coverage = counts.copy()
    coverage["positivity_ok"] = supported.all(axis=1)
    coverage["n_absent_calls"] = absent.sum(axis=1)

    # Structurally-absent (stratum, action) pairs — the true holes.
    absent_cells = [
        (idx if isinstance(idx, tuple) else (idx,)) + (int(a),)
        for idx, row in absent.iterrows()
        for a in counts.columns[row.values]
    ]

    return {
        "coverage": coverage.reset_index(),
        "strata_cols": strata,
        "pct_cells_supported": float(supported.to_numpy().mean()),
        "n_absent_cells": int(absent.to_numpy().sum()),
        "n_cells": int(n_cells),
        "absent_cells": absent_cells,
        "pct_strata_fully_covered": float(supported.all(axis=1).mean()),
        "n_strata": int(len(counts)),
        "min_count": min_count,
    }


def is_action_observed(ds: Dataset, action: int, down: int) -> bool:
    """Was `action` ever observed on `down`? (helper for positivity-hole checks)."""
    m = ds.df["down"].astype(int) == down
    return bool((ds.df.loc[m, ds.action_col] == action).any())


def visualize_dag(save_path: str) -> None:
    """Draw the abstract SCM DAG with networkx (dev artifact)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    g = nx.DiGraph()
    g.add_edges_from(scm.EDGES)
    color = {n: ("#D0021B" if n == scm.TREATMENT else
                 "#2C3E50" if n == scm.OUTCOME else
                 "#9B59B6" if n in scm.UNOBSERVED else
                 "#E87040" if n in scm.MEDIATORS else "#4A90D9")
             for n in g.nodes()}
    fig, ax = plt.subplots(figsize=(11, 8))
    pos = nx.spring_layout(g, seed=1)
    nx.draw_networkx(
        g, pos=pos, ax=ax, node_color=[color[n] for n in g.nodes()],
        node_size=2600, font_size=8, font_color="white", font_weight="bold",
        edge_color="#555555", arrows=True, arrowsize=16,
    )
    ax.set_title("Defensive Playcall — Causal DAG (V2)", fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
