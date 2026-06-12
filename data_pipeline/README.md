# Data Pipeline

This folder contains shared data management and processing for both modelling
tracks:

- Machine learning models
- Statistical models

Keep shared source data, reusable cleaning logic, and cross-model reference
tables here. When a script or output is specific to one model track, name and
document it clearly.

Current model-specific pieces:

- `build_data.py`: Statistical V1 feature builder for the GLM/Poisson baseline.
- `processed/poisson_training_long.csv`: Statistical V1 training table with
  pre-match Elo and match-derived features.
- `elo.py`: shared Elo implementation used by the pipeline and available to both
  modelling tracks.
- `make_baseline_dataset.py`: older Elo-only baseline utility.

## Refresh Data

Build or refresh the Statistical V1 data outputs from the project root:

Poetry:

```bash
poetry run python data_pipeline/build_data.py --refresh
```

Non-Poetry:

```bash
source .venv/bin/activate
python data_pipeline/build_data.py --refresh
```

Validate the generated files:

Poetry:

```bash
poetry run python data_pipeline/build_data.py --check
```

Non-Poetry:

```bash
source .venv/bin/activate
python data_pipeline/build_data.py --check
```

## History Window

The builder uses all completed martj42 matches as feature history, including
matches before 1960. Exported model rows still start at `1960-01-01`.

This means rest days, rolling form, streaks, and head-to-head records can use
older pre-match history where martj42 has it. Head-to-head features use a
60-year lookback.

Elo is computed separately from completed martj42 matches starting at
`1900-01-01`, with every team initialized at 1500. Only pre-match Elo values for
1960+ matches are exported into the model datasets.

## Folder Layout

- `raw/martj42/`: committed source CSVs from `martj42/international_results`.
- `cache/`: ignored downloaded external files from Transfermarkt and World Bank.
- `reference/`: committed mapping files and placeholder popularity labels.
- `processed/`: committed curated CSV outputs. Check each file description before
  using it as shared data, because some outputs are model-specific.

## Processed Outputs

Shared or broadly reusable:

- `matches_1960_completed.csv`: completed martj42 matches from 1960 onward,
  cleaned and tagged with tournament/context flags plus pre-match home/away Elo.
  This is the base match table.
- `future_fixtures.csv`: martj42 rows with missing scores or future dates. These
  are excluded from training and kept for later prediction use.
- `team_year_population.csv`: World Bank population joined to football team names
  by year. Name joins use `reference/country_name_mappings.csv`.
- `team_year_market_value_proxy.csv`: Transfermarkt player valuation snapshots
  aggregated by citizenship and year. This is a talent/value proxy, not an exact
  national squad snapshot.
- `country_development_inputs.csv`: population, market-value proxy, log
  transforms, and placeholder football-popularity inputs for later country
  development-efficiency features.
- `name_mapping_diagnostics.csv`: unmatched external names from World Bank or
  Transfermarkt. Use this when improving `reference/country_name_mappings.csv`.
- `model_features_long.csv`: model-ready team-perspective table with deterministic
  imputation, missingness flags, and team/opponent pre-match Elo. It still
  contains targets/result labels, so model code should choose predictor columns
  explicitly.

Statistical V1-specific:

- `team_match_features.csv`: two rows per match, one from each team's perspective,
  with rolling form, fatigue, context, 60-year head-to-head, team/opponent
  pre-match Elo, and scoreline targets. This keeps real `NaN` values where
  history does not exist.
- `poisson_training_long.csv`: GLM-ready long table with `goals_for` as the
  Poisson target. It is derived from `model_features_long.csv` and excludes
  result-only columns like points and win/loss labels.

## Missing Values

Raw martj42 features keep `NaN` when the information genuinely does not exist.
`model_features_long.csv` fills those values for modelling:

- Rest days: global median.
- Rolling averages: global mean.
- Previous shootout flag: `0`.
- Missing head-to-head averages: neutral values, `0` goal difference and `1.0`
  points per game.

Missingness flags such as `missing_days_since_last_match` and `missing_h2h_60y`
are preserved so both GLM and ML models can learn from missing-history cases.
External population/value features stay separate because missing country data has
a different meaning than missing match history.

Raw and reference files:

- `raw/martj42/results.csv`: source match results used to generate all match
  tables.
- `raw/martj42/shootouts.csv`: source shootout data used as a proxy for matches
  that likely went beyond normal time.
- `raw/martj42/goalscorers.csv`: source scorer data, committed for future player
  features but not used in Statistical V1 yet.
- `raw/martj42/former_names.csv`: source team-name history, committed for future
  name-cleaning work.
- `reference/country_name_mappings.csv`: manual mappings between football names,
  World Bank names, and Transfermarkt names.
- `reference/football_popularity.csv`: placeholder popularity labels used by
  `country_development_inputs.csv`.

## Implemented Sources

- martj42 match results: base source for scores, dates, teams, tournaments,
  cities, countries, and neutral flags.
- martj42 shootouts: used as a proxy for matches that likely went beyond normal
  time.
- Elo: derived from martj42 results with the shared `elo.py` formula, starting
  in 1900 and exported for 1960+ matches.
- Transfermarkt/dcaribou exports: used to build yearly country player-value
  proxies.
- World Bank population: used for yearly country population joins.

## Future Sources From Research

These are not implemented yet, but are documented as likely next sources:

- API-Football, Football-data.org, SportMonks, or BallDontLie for live fixtures,
  lineups, events, referee, venue, and richer World Cup schedule data.
- Open-Meteo geocoding, weather, and elevation APIs for match-day weather and
  altitude.
- API-Football injuries or manual/news sources for player availability,
  injuries, and suspensions.
- Transfermarkt/dcaribou squad/player tables for fuller demographics such as
  age, caps, and top-league share.

## Limitations

- Transfermarkt values are approximate country/player-value proxies, not verified
  historical World Cup squad snapshots.
- Population and market-value data are kept separate from the main training table
  until joins and leakage risks are validated.
- Player availability, suspensions, injuries, weather, altitude, and true
  historical squad rosters are not included in V1.
- Previous extra time is represented only by a shootout proxy where available.
