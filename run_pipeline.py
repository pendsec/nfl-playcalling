"""
NFL Defensive Playcall Causal Inference Pipeline
Step 5: Main runner — ties all modules together
"""

import pandas as pd
import numpy as np

from pipeline_01_data import (
    load_pbp, filter_plays, build_game_state, build_offensive_context,
    build_player_proxies, build_offensive_tendencies, build_treatment, build_model_df
)
from pipeline_02_identification import (
    build_causal_model, identify_estimand, check_positivity, visualize_dag
)
from pipeline_03_estimation import run_estimation_pipeline
from pipeline_04_agent import DefensiveCoordinatorAgent, GameState


def main(seasons: list[int] = [2021, 2022, 2023]):

    # ── 1. Load & engineer features ──────────────────────────────────────────
    print("[1/5] Loading play-by-play data...")
    pbp = load_pbp(seasons)
    df = filter_plays(pbp)
    df = build_game_state(df)
    df = build_offensive_context(df)
    df = build_player_proxies(df, roster_df=None)  # pass roster_df if available
    df = build_offensive_tendencies(df)
    df, label_encoder = build_treatment(df)
    df = build_model_df(df)
    print(f"  → {len(df):,} plays ready for modeling")

    # ── 2. Causal identification ─────────────────────────────────────────────
    print("\n[2/5] Registering DAG & identifying estimand...")
    visualize_dag(save_path="dag.png")
    causal_model = build_causal_model(df)
    estimand = identify_estimand(causal_model)
    coverage_df = check_positivity(df, treatment_col="def_playcall_enc",
                                   confounder_cols=["down", "ydstogo", "score_diff"])

    # ── 3. Estimation ─────────────────────────────────────────────────────────
    print("\n[3/5] Running estimation pipeline...")
    results = run_estimation_pipeline(df, label_encoder)

    # ── 4. Build agent ────────────────────────────────────────────────────────
    print("\n[4/5] Building defensive coordinator agent...")
    agent = DefensiveCoordinatorAgent(
        causal_forest=results["causal_forest"],
        preprocessor=results["preprocessor"],
        label_encoder=label_encoder,
        ate_df=results["ate"],
        coverage_df=coverage_df,
        baseline_epa=df["epa"].mean(),
    )

    # ── 5. Example recommendations ───────────────────────────────────────────
    print("\n[5/5] Example playcall recommendations:\n")

    scenarios = [
        GameState(
            down=3, ydstogo=8, score_diff=-3, seconds_remaining=420,
            yardline_100=65, formation="SHOTGUN",
            num_rb=1, num_te=1, num_wr=3,
            off_pass_tendency=0.82, def_team_avg_epa=-0.04, qb_avg_epa=0.15,
        ),
        GameState(
            down=1, ydstogo=10, score_diff=7, seconds_remaining=1800,
            yardline_100=75, formation="SINGLEBACK",
            num_rb=2, num_te=1, num_wr=2,
            off_pass_tendency=0.48, def_team_avg_epa=-0.02, qb_avg_epa=0.05,
        ),
        GameState(
            down=2, ydstogo=3, score_diff=0, seconds_remaining=90,
            yardline_100=15, formation="I_FORM",
            num_rb=2, num_te=2, num_wr=1,
            off_pass_tendency=0.38, def_team_avg_epa=0.01, qb_avg_epa=0.10,
        ),
    ]

    for i, state in enumerate(scenarios, 1):
        print(f"\nScenario {i}:")
        rec = agent.recommend(state)
        rec.display()

    # ── Save outputs ──────────────────────────────────────────────────────────
    results["ate"].to_csv("ate_estimates.csv", index=False)
    results["sensitivity"].to_csv("sensitivity_analysis.csv", index=False)
    print("\nOutputs saved: ate_estimates.csv, sensitivity_analysis.csv, dag.png")

    return agent, results


if __name__ == "__main__":
    agent, results = main(seasons=[2021, 2022, 2023])
