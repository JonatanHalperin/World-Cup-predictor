# World Cup Predictor

This project explores models for predicting World Cup knockout-stage outcomes.
It has two main modelling approaches:

- A machine learning approach
- A statistical approach

The statistical approach is developed separately from the ML approach. The
current statistical work is V1, which uses a generalized linear model (GLM) and
a Poisson goal model.

## Statistical Approach: V1

Building a good football prediction model can become complex quickly. The aim
of this first statistical phase is to lay the groundwork for future improvements
while keeping the model simple, explainable, and easy to iterate on.

In this phase, the project will:

1. Find and prepare relevant match and team data.
2. Fit a GLM using a Poisson model for goals scored.
3. Present model results in a clear and reproducible way.

The V1 statistical model is intended to be a baseline rather than a final
predictor. It should make it easier to compare later statistical improvements
against a simple foundation.

## Planned Improvements

After the first phase, the goal is to evolve the model by adding more advanced
features such as:

- Regularization
- Time weighting
- Elo-type team ratings
- Bayesian uncertainty
- Variational smoothing or related smoothing methods

## Expected Outputs

The initial version should produce:

- Cleaned input data
- Estimated team or match-level model parameters
- Predicted goal expectations
- Match outcome probabilities
- A readable summary of results

## Setup

With Poetry:

```bash
poetry install
poetry run python your_script.py
```

Without Poetry:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
