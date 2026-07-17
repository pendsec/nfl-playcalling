# nfl-playcalling

Evaluating NFL playcalling using causal inference techniques and building a
defensive-playcall recommendation agent using causal reinforcement learning.

See [CLAUDE.md](CLAUDE.md) for the full project design and phased build plan.

## V1 — Walking Skeleton (current)

An end-to-end pipeline that produces and evaluates a defensive policy. It is
deliberately simple and **wrong in places by design** — the point is a
debuggable skeleton, not a finished model (per the walking-skeleton-first rule).

**Scope:** 3rd downs only · one team's defense (SF) · 2021–2023 · 4 actions
= {man, zone} × {blitz, no-blitz}.

**Pipeline:**

| Step | Component | V1 model |
|------|-----------|----------|
| Data | `src/data/` | nflfastR loader + synthetic-SCM generator; (S, A, R) builder |
| SCM | `src/scm/graph.py` | hand-built DAG; declares adjustment set + mediator exclusion |
| πb | `src/models/behavior/` | multinomial logistic, cross-fitted, calibration tracked |
| Q | `src/models/outcome/` | ridge on reward(−EPA), state×action design |
| OPE | `src/ope/` | Direct Method + behavior-recovery smoke test |
| Policy | `src/policy/` | behavior-constrained greedy |

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
python run_v1.py                  # real SF nflfastR data (configs/v1.yaml)
python run_v1.py --source synthetic   # known-SCM data, no network
pytest tests/                     # synthetic-SCM regression tests
```

For a narrated, step-by-step walkthrough of the same pipeline (with plots and
explanations of each stage), open
[`notebooks/v1_pipeline_walkthrough.ipynb`](notebooks/v1_pipeline_walkthrough.ipynb).

All V1 scope lives in [`configs/v1.yaml`](configs/v1.yaml) (team, seasons,
action thresholds, model hyperparameters).

### Known V1 limitations (the deliberately-wrong parts → fixed in V2)

- **Direct-Method extrapolation.** Ridge Q extrapolates to implausible values
  for actions rarely taken in a given state, so the greedy policy's estimated
  lift is wildly optimistic. The behavior-recovery smoke test passes (DM recovers
  πb's mean reward) while off-support estimates blow up — exactly the pathology
  doubly-robust OPE + causal pessimism address in V2.
- **πb below majority baseline.** `class_weight="balanced"` on ~350 training
  plays trades majority accuracy for minority recall. Fine for the skeleton;
  V2 moves to a calibrated GBM.
- **Unobserved confounding** (`coach_read`) is unaddressed; V2 adds the
  sensitivity analysis that bounds it.
