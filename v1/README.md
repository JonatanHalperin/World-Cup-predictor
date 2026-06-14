# Statistical V1 Poisson GLM

This folder contains a simple Poisson generalized linear model for football
goals. Each row should represent one team's goal count in one match.

The model is:

```text
goals_i ~ Poisson(lambda_i)
log(lambda_i) = eta_i
eta_i = X_i beta
```

`statsmodels` estimates `beta` with the same Poisson GLM likelihood and an L2
ridge penalty by default. Prediction returns `eta = X beta` and
`lambda = exp(eta)`, where `lambda` is expected goals.

Prediction scripts use regular Elo plus tuned sample-weighting and ridge
strength from `v1/config/tuned_poisson_glm.json`. The targeted second tuning run
can select a Kyrre half-life or `no_weight`. The current tracked config selects
`no_weight` and ridge alpha `0.003`. If that config is unavailable, the fallback
is an 8-year Kyrre-weight half-life and ridge alpha `0.01`:

```text
w_i = exp(-gamma * age_i)
gamma = ln(2) / 8
```

The GLM uses all currently available leak-free, non-redundant pre-match signal
groups: Elo strength, home/neutral/host context, rest and congestion, recent
form, streaks, competition type, and 60-year head-to-head history. Exact
duplicate columns are intentionally excluded because they make GLM coefficient
estimates unstable.

## Example

From the project root:

```bash
poetry run python v1/example_predict_one_match.py
```

The example uses `data_pipeline/processed/poisson_training_long.csv`, whose
target column is `goals_for`.

## Future Fixture Report

Predict the Switzerland-Qatar fixture from `future_fixtures.csv` and write
tables plus PNG plots under `v1/reports/`:

```bash
poetry run python v1/predict_fixture.py --home Switzerland --away Qatar
```

The stored fixture order is `Qatar vs Switzerland`, fixture `6`, on
`2026-06-13` in Santa Clara, United States. The command accepts either team
order and reports the stored fixture order.

Fixture predictions train on the full completed dataset by default.
Training rows use the tuned sample-weighting choice and ridge alpha by default.
Pass `--ignore-tuned-config` to use fallback defaults, `--ridge-alpha` to
override regularization, `--kyrre-weight-half-life-years` to force a half-life,
or `--no-kyrre-weight` to force unweighted fitting.

Predict the next 10 unresolved 2026 World Cup fixtures and write a collective
report:

```bash
poetry run python v1/predict_next_fixtures.py --as-of-date 2026-06-14 --count 10
```

The collective report includes expected goals, home/draw/away probabilities,
top overall scorelines, and the most probable exact scoreline inside each
outcome bucket.

Single-fixture artifacts include:

- `prediction_summary.md`
- `prediction_summary.csv`
- `score_matrix.csv`
- `scoreline_heatmap.png`
- `outcome_probabilities.png`
- `top_scorelines.png`
- `expected_goals.png`
- `model_weights.csv`
- `model_weights.png`

Collective next-fixture artifacts include:

- `collective_report.md`
- `fixture_predictions.csv`
- `top_scorelines.csv`
- `outcome_probabilities.png`
- `expected_goals.png`
- `best_outcomes.png`
- `model_weights.csv`
- `model_weights.png`

Refresh the source data used by the current model with:

```bash
poetry run python data_pipeline/build_data.py --refresh --check
```

V1 uses football history and match-context features only. Weather, altitude,
lineups, injuries, suspensions, and squad selection are listed in the report as
missing or not yet modeled. Weather is intentionally not used until comparable
historical weather is backfilled for the training rows.

## Validation And Test

Tune Kyrre half-life/no-weight and ridge alpha with rolling expanding-window
validation:

```bash
poetry run python v1/tune_hyperparameters.py
```

The tuning report is written to `v1/reports/hyperparameter_tuning/`, and the
selected defaults are written to tracked config
`v1/config/tuned_poisson_glm.json`.
The tuner treats configs within `0.00005` mean log-loss of the best grid row as
effectively tied, then chooses by RPS and Poisson NLL. This avoids chasing tiny
noise-level differences in the fourth decimal.

Run the fixed historical validation setup:

- Train: 2010-2018
- Validation: 2019-2021
- Test: 2022 FIFA World Cup

```bash
poetry run python v1/validate_model.py
```

The report is written to
`v1/reports/validation_2010_2018__2019_2021__2022_world_cup/` with validation
and test summary metrics, row-level goal predictions, match-level outcome
predictions, fixed-width predicted-lambda calibration tables, and PNG plots.
The calibration table groups predicted expected goals into bins, then compares
average predicted goals against average actual goals. The default bin width is
`0.5`; tune it with `--calibration-bin-width`.
The report also includes ridge metadata, RPS, Poisson deviance, outcome ECE,
per-outcome precision/recall, reliability curves, subgroup/yearly breakdowns,
baseline comparisons, paired tests, scoreline confusion summaries, and
`model_weights.png` / `model_weights.csv`.

## API Usage

```python
from v1.poisson_glm import (
    fit_poisson_glm,
    outcome_probabilities,
    poisson_score_matrix,
    predict_expected_goals,
)

features = [
    "team_elo_pre",
    "opp_elo_pre",
    "is_home",
]

result = fit_poisson_glm(
    "data_pipeline/processed/poisson_training_long.csv",
    feature_columns=features,
    target_column="goals_for",
    id_columns=["match_id", "team", "opponent"],
    kyrre_weight_half_life_years=8,
    ridge_alpha=0.01,
)

predictions = predict_expected_goals(result, new_rows)
score_matrix = poisson_score_matrix(
    predictions.loc[home_index, "lambda"],
    predictions.loc[away_index, "lambda"],
)
outcomes = outcome_probabilities(score_matrix)
```

Feature columns must be supplied explicitly. This code does not invent features,
encode categoricals, scale columns, or create model inputs beyond adding the GLM
intercept. The lower-level Python fit defaults still use the fallback
Kyrre-weight half-life and ridge regularization; pass
`kyrre_weight_half_life_years=None` / `ridge_alpha=0` in Python or
`--no-kyrre-weight` / `--ridge-alpha 0` in the CLI to disable either part.

## CLI Fit

You can also fit from the module CLI:

```bash
poetry run python v1/poisson_glm.py \
  data_pipeline/processed/poisson_training_long.csv \
  --target goals_for \
  --features team_elo_pre opp_elo_pre is_home \
  --ridge-alpha 0.01
```
