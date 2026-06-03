"""
NFL Defensive Playcall Causal Inference Pipeline
Step 2: DAG Registration & Causal Identification via DoWhy
"""

import pandas as pd
import numpy as np
import dowhy
from dowhy import CausalModel
import networkx as nx
import matplotlib.pyplot as plt


# ── 1. Define the DAG ────────────────────────────────────────────────────────
# Uses DoWhy's GML graph format.
# Nodes and edges encode the structural causal model from the problem proposal.

CAUSAL_GRAPH_GML = """
graph [
  directed 1

  node [ id "game_state"        label "game_state" ]
  node [ id "off_personnel"     label "off_personnel" ]
  node [ id "off_formation"     label "off_formation" ]
  node [ id "off_tendencies"    label "off_tendencies" ]
  node [ id "off_playcall"      label "off_playcall" ]
  node [ id "def_playcall"      label "def_playcall" ]
  node [ id "player_proxies"    label "player_proxies" ]
  node [ id "epa"               label "epa" ]

  edge [ source "game_state"     target "off_personnel" ]
  edge [ source "game_state"     target "off_formation" ]
  edge [ source "game_state"     target "off_tendencies" ]
  edge [ source "game_state"     target "def_playcall" ]
  edge [ source "game_state"     target "epa" ]

  edge [ source "off_personnel"  target "off_formation" ]
  edge [ source "off_personnel"  target "off_playcall" ]
  edge [ source "off_personnel"  target "def_playcall" ]
  edge [ source "off_personnel"  target "epa" ]

  edge [ source "off_formation"  target "off_playcall" ]
  edge [ source "off_formation"  target "def_playcall" ]
  edge [ source "off_formation"  target "epa" ]

  edge [ source "off_tendencies" target "def_playcall" ]
  edge [ source "off_tendencies" target "off_playcall" ]

  edge [ source "off_playcall"   target "epa" ]

  edge [ source "def_playcall"   target "epa" ]

  edge [ source "player_proxies" target "epa" ]
  edge [ source "player_proxies" target "def_playcall" ]
]
"""


def build_causal_model(df: pd.DataFrame) -> CausalModel:
    """
    Register the DAG with DoWhy and identify the causal effect.
    
    Treatment:  def_playcall
    Outcome:    epa
    Confounders (adjustment set): game_state, off_personnel, off_formation,
                                  off_tendencies, player_proxies
    """
    # Map abstract node names to actual dataframe columns
    # DoWhy needs concrete column references for estimation
    treatment_col = "def_playcall_enc"
    outcome_col = "epa"

    common_causes = [
        # Game state
        "down", "ydstogo", "score_diff", "seconds_remaining",
        "yardline_100", "is_two_minute_drill", "is_red_zone",
        "is_third_or_fourth", "scoring_opp",
        # Formation & personnel
        "num_rb", "num_te", "num_wr",
        # Tendencies
        "off_pass_tendency",
        # Player proxies
        "def_team_avg_epa", "qb_avg_epa",
    ]

    model = CausalModel(
        data=df,
        treatment=treatment_col,
        outcome=outcome_col,
        common_causes=common_causes,
        graph=CAUSAL_GRAPH_GML,
    )

    return model


def identify_estimand(model: CausalModel):
    """
    Use DoWhy's identification algorithm to confirm backdoor identification.
    
    Expected result: backdoor criterion satisfied via the adjustment set
    {game_state, off_personnel, off_formation, off_tendencies, player_proxies}
    """
    identified_estimand = model.identify_effect(
        proceed_when_unidentifiable=True  # warns but continues if unidentified
    )
    print("=" * 60)
    print("IDENTIFIED ESTIMAND")
    print("=" * 60)
    print(identified_estimand)
    return identified_estimand


def check_positivity(df: pd.DataFrame, treatment_col: str, confounder_cols: list) -> pd.DataFrame:
    """
    Positivity check: for each stratum of confounders, verify that
    each defensive playcall appears at least min_count times.
    
    Strata with missing calls violate positivity and should be flagged.
    The agent should not make confident recommendations in these regions.
    """
    from itertools import product

    # Coarsen confounders into bins for tractable strata check
    df = df.copy()
    df["down_bin"] = df["down"].astype(str)
    df["dist_bin"] = pd.cut(df["ydstogo"], bins=[0, 3, 7, 15, 100],
                             labels=["short", "med", "long", "vlong"]).astype(str)
    df["score_bin"] = pd.cut(df["score_diff"], bins=[-50, -8, 0, 8, 50],
                              labels=["losing_big", "losing", "winning", "winning_big"]).astype(str)

    strata_cols = ["down_bin", "dist_bin", "score_bin"]
    calls = df[treatment_col].unique()

    coverage = (
        df.groupby(strata_cols + [treatment_col])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Flag strata where any call has < 10 observations
    min_count = 10
    call_cols = [c for c in calls if c in coverage.columns]
    coverage["positivity_ok"] = (coverage[call_cols] >= min_count).all(axis=1)
    coverage["missing_calls"] = (coverage[call_cols] == 0).sum(axis=1)

    pct_ok = coverage["positivity_ok"].mean() * 100
    print(f"\nPositivity check: {pct_ok:.1f}% of strata have full coverage")
    print(f"Strata with missing calls: {(~coverage['positivity_ok']).sum()}")

    return coverage


def visualize_dag(save_path: str = None):
    """Draw the causal DAG using networkx."""
    G = nx.DiGraph()
    nodes = [
        "Game State", "Off Personnel", "Off Formation",
        "Off Tendencies", "Off Playcall", "Def Playcall\n(Treatment)",
        "Player Proxies", "EPA (Outcome)"
    ]
    edges = [
        ("Game State", "Off Personnel"),
        ("Game State", "Off Formation"),
        ("Game State", "Off Tendencies"),
        ("Game State", "Def Playcall\n(Treatment)"),
        ("Game State", "EPA (Outcome)"),
        ("Off Personnel", "Off Formation"),
        ("Off Personnel", "Off Playcall"),
        ("Off Personnel", "Def Playcall\n(Treatment)"),
        ("Off Personnel", "EPA (Outcome)"),
        ("Off Formation", "Off Playcall"),
        ("Off Formation", "Def Playcall\n(Treatment)"),
        ("Off Formation", "EPA (Outcome)"),
        ("Off Tendencies", "Def Playcall\n(Treatment)"),
        ("Off Tendencies", "Off Playcall"),
        ("Off Playcall", "EPA (Outcome)"),
        ("Def Playcall\n(Treatment)", "EPA (Outcome)"),
        ("Player Proxies", "EPA (Outcome)"),
        ("Player Proxies", "Def Playcall\n(Treatment)"),
    ]

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos = {
        "Game State":               (0, 2),
        "Off Personnel":            (-2, 1),
        "Off Formation":            (-1, 0),
        "Off Tendencies":           (1, 1),
        "Off Playcall":             (-2, -1),
        "Def Playcall\n(Treatment)":(0, -1),
        "Player Proxies":           (2, 0),
        "EPA (Outcome)":            (0, -2.5),
    }

    node_colors = {
        "Game State": "#4A90D9",
        "Off Personnel": "#7EC8A4",
        "Off Formation": "#7EC8A4",
        "Off Tendencies": "#F5A623",
        "Off Playcall": "#E87040",
        "Def Playcall\n(Treatment)": "#D0021B",
        "Player Proxies": "#9B59B6",
        "EPA (Outcome)": "#2C3E50",
    }

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    colors = [node_colors[n] for n in G.nodes()]
    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=colors,
        node_size=2800,
        font_size=8,
        font_color="white",
        font_weight="bold",
        edge_color="#555555",
        arrows=True,
        arrowsize=20,
        connectionstyle="arc3,rad=0.05",
    )
    ax.set_title("NFL Defensive Playcall — Causal DAG", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"DAG saved to {save_path}")


if __name__ == "__main__":
    print("Causal identification module loaded.")
    print("Call build_causal_model(df) → identify_estimand(model) to identify the effect.")
