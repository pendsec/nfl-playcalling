"""
NFL Defensive Playcall Causal Inference Pipeline
Step 4: Defensive Coordinator Agent
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from econml.dr import ForestDRLearner
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder


@dataclass
class GameState:
    """Encapsulates the observable game context at the moment of playcalling."""
    down: int                       # 1–4
    ydstogo: float                  # yards to first down
    score_diff: float               # posteam_score - defteam_score (defense perspective: negative = we're winning)
    seconds_remaining: float        # total game seconds remaining
    yardline_100: float             # yards from opponent end zone (100 = own goal line)
    formation: str                  # offensive formation string
    num_rb: int                     # RBs on field
    num_te: int                     # TEs on field
    num_wr: int                     # WRs on field
    off_pass_tendency: float        # historical pass rate in this context [0,1]
    def_team_avg_epa: float         # defensive unit quality proxy
    qb_avg_epa: float               # QB quality proxy

    # Derived
    personnel_group: str = field(init=False)
    is_two_minute_drill: int = field(init=False)
    is_red_zone: int = field(init=False)
    is_third_or_fourth: int = field(init=False)
    scoring_opp: int = field(init=False)

    def __post_init__(self):
        self.personnel_group = f"{self.num_rb}{self.num_te}"
        self.is_two_minute_drill = int(self.seconds_remaining <= 120)
        self.is_red_zone = int(self.yardline_100 <= 20)
        self.is_third_or_fourth = int(self.down in [3, 4])
        self.scoring_opp = int(abs(self.score_diff) <= 8 and self.seconds_remaining <= 300)

    def to_feature_dict(self) -> dict:
        return {
            "down": self.down,
            "ydstogo": self.ydstogo,
            "score_diff": self.score_diff,
            "seconds_remaining": self.seconds_remaining,
            "yardline_100": self.yardline_100,
            "is_two_minute_drill": self.is_two_minute_drill,
            "is_red_zone": self.is_red_zone,
            "is_third_or_fourth": self.is_third_or_fourth,
            "scoring_opp": self.scoring_opp,
            "num_rb": self.num_rb,
            "num_te": self.num_te,
            "num_wr": self.num_wr,
            "off_pass_tendency": self.off_pass_tendency,
            "def_team_avg_epa": self.def_team_avg_epa,
            "qb_avg_epa": self.qb_avg_epa,
            "formation": self.formation,
            "personnel_group": self.personnel_group,
        }


@dataclass
class PlaycallRecommendation:
    """Output of the defensive agent for a given game state."""
    game_state: GameState
    recommended_call: str
    expected_epa: float
    epa_ci_lo: float
    epa_ci_hi: float
    alternatives: list[dict]
    positivity_ok: bool
    confidence: str  # "high" | "medium" | "low"
    reasoning: str

    def display(self):
        print("\n" + "═"*55)
        print("  DEFENSIVE COORDINATOR AGENT — RECOMMENDATION")
        print("═"*55)
        print(f"  Situation:   {self.game_state.down} & {self.game_state.ydstogo:.0f}"
              f" | {self.game_state.formation} | Score diff: {self.game_state.score_diff:+.0f}")
        print(f"  Field:       {100 - self.game_state.yardline_100:.0f} yard line"
              f" | {self.game_state.seconds_remaining:.0f}s remaining")
        print(f"\n  ✅ CALL:      {self.recommended_call}")
        print(f"  EPA est:     {self.expected_epa:+.3f}  "
              f"[{self.epa_ci_lo:+.3f}, {self.epa_ci_hi:+.3f}]")
        print(f"  Confidence:  {self.confidence}")
        if not self.positivity_ok:
            print("  ⚠️  WARNING: Low historical coverage for this context.")
        print(f"\n  Reasoning:   {self.reasoning}")
        print("\n  Alternatives:")
        for alt in self.alternatives:
            print(f"    {alt['call']:<45} EPA: {alt['expected_epa']:+.3f}")
        print("═"*55)


class DefensiveCoordinatorAgent:
    """
    Causal inference-based defensive playcall agent.

    Decision rule:
        argmin_d  E[EPA | do(DefPlaycall=d), context]
        
    Marginalizes over the distribution of offensive playcalls
    (via off_pass_tendency) to produce robust recommendations.

    Parameters
    ----------
    causal_forest   : trained ForestDRLearner
    preprocessor    : fitted sklearn ColumnTransformer
    label_encoder   : fitted LabelEncoder for defensive calls
    ate_df          : AIPW ATE estimates (pd.DataFrame)
    coverage_df     : positivity check results (pd.DataFrame)
    baseline_epa    : mean EPA under baseline defensive call (float)
    """

    def __init__(
        self,
        causal_forest: ForestDRLearner,
        preprocessor: ColumnTransformer,
        label_encoder: LabelEncoder,
        ate_df: pd.DataFrame,
        coverage_df: pd.DataFrame,
        baseline_epa: float = 0.0,
    ):
        self.cf = causal_forest
        self.preprocessor = preprocessor
        self.le = label_encoder
        self.ate_df = ate_df
        self.coverage_df = coverage_df
        self.baseline_epa = baseline_epa
        self.n_calls = len(label_encoder.classes_)

    def _featurize(self, state: GameState) -> np.ndarray:
        """Convert GameState → preprocessed feature matrix (1 row)."""
        df = pd.DataFrame([state.to_feature_dict()])
        return self.preprocessor.transform(df)

    def _check_positivity(self, state: GameState) -> bool:
        """Check if this game state has adequate historical coverage."""
        down_bin = str(state.down)
        dist_bin = pd.cut(
            [state.ydstogo], bins=[0, 3, 7, 15, 100],
            labels=["short", "med", "long", "vlong"]
        )[0]
        score_bin = pd.cut(
            [state.score_diff], bins=[-50, -8, 0, 8, 50],
            labels=["losing_big", "losing", "winning", "winning_big"]
        )[0]

        match = self.coverage_df[
            (self.coverage_df["down_bin"] == down_bin) &
            (self.coverage_df["dist_bin"] == str(dist_bin)) &
            (self.coverage_df["score_bin"] == str(score_bin))
        ]
        if match.empty:
            return False
        return bool(match["positivity_ok"].iloc[0])

    def _expected_epa_per_call(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate expected EPA for each defensive call using the causal forest.
        
        CATE gives us τ_d(x) = E[Y | do(T=d)] - E[Y | do(T=baseline)]
        We recover E[Y | do(T=d)] = baseline_epa + τ_d(x)
        """
        cate = self.cf.effect(X)  # shape (1, n_calls-1)
        # Reconstruct full EPA array: [baseline, baseline+τ_1, ...]
        epa_per_call = np.zeros(self.n_calls)
        epa_per_call[0] = self.baseline_epa
        if cate.ndim == 1:
            cate = cate.reshape(1, -1)
        epa_per_call[1:] = self.baseline_epa + cate[0]
        return epa_per_call

    def _confidence_level(self, positivity_ok: bool, ci_width: float) -> str:
        if not positivity_ok:
            return "low"
        if ci_width < 0.3:
            return "high"
        if ci_width < 0.6:
            return "medium"
        return "low"

    def _build_reasoning(self, state: GameState, best_call: str, epa: float) -> str:
        reasons = []
        if state.is_red_zone:
            reasons.append("red zone reduces running lanes")
        if state.is_two_minute_drill:
            reasons.append("two-minute drill shifts to pass-heavy")
        if state.off_pass_tendency > 0.70:
            reasons.append(f"offense passes {state.off_pass_tendency*100:.0f}% here")
        if state.num_te >= 2:
            reasons.append("heavy TE set suggests run or short routes")
        if state.down == 3 and state.ydstogo >= 7:
            reasons.append("3rd & long favors pass rush")
        if not reasons:
            reasons.append("standard situation, call optimizes expected EPA")
        return "; ".join(reasons).capitalize() + "."

    def recommend(self, state: GameState) -> PlaycallRecommendation:
        """
        Main agent interface. Takes a GameState and returns a PlaycallRecommendation.
        """
        X = self._featurize(state)
        positivity_ok = self._check_positivity(state)
        epa_per_call = self._expected_epa_per_call(X)

        # Rank calls by expected EPA (ascending — lower EPA = better for defense)
        ranked_idx = np.argsort(epa_per_call)
        best_idx = ranked_idx[0]
        best_call = self.le.classes_[best_idx]
        best_epa = epa_per_call[best_idx]

        # CI from AIPW ate_df for the best call
        ate_row = self.ate_df[self.ate_df["def_playcall"] == best_call]
        if not ate_row.empty:
            ci_lo = best_epa + ate_row["ci_95_lo"].values[0]
            ci_hi = best_epa + ate_row["ci_95_hi"].values[0]
        else:
            ci_lo = best_epa - 0.15
            ci_hi = best_epa + 0.15

        ci_width = ci_hi - ci_lo
        confidence = self._confidence_level(positivity_ok, ci_width)

        # Top 3 alternatives
        alternatives = []
        for idx in ranked_idx[1:4]:
            alternatives.append({
                "call": self.le.classes_[idx],
                "expected_epa": float(epa_per_call[idx]),
            })

        reasoning = self._build_reasoning(state, best_call, best_epa)

        return PlaycallRecommendation(
            game_state=state,
            recommended_call=best_call,
            expected_epa=float(best_epa),
            epa_ci_lo=float(ci_lo),
            epa_ci_hi=float(ci_hi),
            alternatives=alternatives,
            positivity_ok=positivity_ok,
            confidence=confidence,
            reasoning=reasoning,
        )


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Agent module loaded.")
    print("""
Example usage:

    from 04_agent import DefensiveCoordinatorAgent, GameState

    agent = DefensiveCoordinatorAgent(
        causal_forest=results["causal_forest"],
        preprocessor=results["preprocessor"],
        label_encoder=label_encoder,
        ate_df=results["ate"],
        coverage_df=coverage_df,
        baseline_epa=df["epa"].mean(),
    )

    state = GameState(
        down=3, ydstogo=8, score_diff=-3,
        seconds_remaining=420, yardline_100=65,
        formation="SHOTGUN",
        num_rb=1, num_te=1, num_wr=3,
        off_pass_tendency=0.78,
        def_team_avg_epa=-0.05,
        qb_avg_epa=0.12,
    )

    rec = agent.recommend(state)
    rec.display()
    """)
