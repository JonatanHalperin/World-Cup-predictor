# World Cup Predictor

This project explores models for predicting World Cup knockout-stage outcomes.
It has two main modelling approaches:

- A machine learning approach
- A statistical approach

Shared data collection and processing lives in `data_pipeline/`. Some outputs in
that folder are model-specific; files for the first statistical phase are marked
as Statistical V1 in `data_pipeline/README.md`.

## Statistical Approach: V1

The first statistical phase uses a GLM/Poisson-style setup. The goal is to keep
the baseline simple and reproducible while preparing match-level features that
can later be improved with regularization, time weighting, Bayesian uncertainty,
or other smoothing methods.

## Setup

Poetry:

```bash
poetry install
poetry run python your_script.py
```

Non-Poetry:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python your_script.py
```

Use the same pattern for project commands: replace `poetry run python ...` with
`python ...` after activating the virtual environment.
