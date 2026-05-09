# NFL Defensive Playcall — Causal Inference Pipeline

## Structure

| File | Purpose |
|------|---------|
| `01_data_pipeline.py` | Load nflverse data, engineer features, build treatment variable |
| `02_causal_identification.py` | DAG registration (DoWhy), backdoor identification, positivity check |
| `03_estimation.py` | Propensity model, outcome model, AIPW ATE, Causal Forest CATE |
| `04_agent.py` | DefensiveCoordinatorAgent — takes GameState, returns PlaycallRecommendation |
| `05_run_pipeline.py` | Main runner, ties everything together |
| `requirements.txt` | Python dependencies |

## Causal Question
> What is the causal effect of a defensive playcall on EPA,
> given the game state, offensive personnel, and offensive formation?

## Quick Start
```bash
pip install -r requirements.txt
python 05_run_pipeline.py
```

## Player Proxy Strategy
Instead of proprietary PFF grades, we use:
1. **Team-season defensive EPA** — absorbs unit-level quality
2. **QB season EPA** — captures offensive skill
3. **Sensitivity analysis** — quantifies robustness to omitted confounders
