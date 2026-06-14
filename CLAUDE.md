# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Two models for predicting the knockout stages of the World Cup:
- **ML model** (`ML_V1/`) — small neural network (PoissonNet) that predicts goals from Elo features
- **Classical statistics model** — not yet implemented; will live on a separate branch

Python 3.12+, managed with Poetry. Core dependencies: numpy, pandas, matplotlib. The ML model additionally requires `torch`, which is **not in `pyproject.toml`** — install it separately (`pip install torch`).

## Setup

```bash
# With Poetry (preferred)
poetry install
pip install torch   # not yet in pyproject.toml

# Without Poetry
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch
```

## Running Scripts

All scripts must be run as **modules from the repo root** because of cross-package imports (`ML_V1/predict.py` imports from both `data_pipeline` and `ML_V1`):

```bash
poetry run python -m data_pipeline.elo
poetry run python -m data_pipeline.make_baseline_dataset
poetry run python -m ML_V1.train_baseline
poetry run python -m ML_V1.predict
poetry run python -m ML_V1.predict_fixtures
```

The `data_pipeline/build_data.py` is the comprehensive Statistical V1 builder and supports flags:

```bash
poetry run python data_pipeline/build_data.py --refresh   # download fresh sources + build
poetry run python data_pipeline/build_data.py --check     # validate existing processed files
```

## Architecture

### Data flow (ML baseline)

```
data_pipeline/elo.py
  -> ML_V1/data/matches_with_elo.csv

data_pipeline/make_baseline_dataset.py
  -> ML_V1/data/baseline_{train,val,test}.csv + ML_V1/data/scaling.json

ML_V1/train_baseline.py
  -> ML_V1/data/baseline_model.pt

ML_V1/predict.py          # predict a single named fixture
ML_V1/predict_fixtures.py # predict all unplayed World Cup games from live feed
  -> world_cup_predictions.csv (repo root, gitignored)
```

All files under `ML_V1/data/` are gitignored. The directory is created automatically on first run.

### Data flow (Statistical V1 / shared pipeline)

```
data_pipeline/build_data.py --refresh
  downloads: martj42 CSVs -> data_pipeline/raw/martj42/
             Transfermarkt CSVs -> data_pipeline/cache/transfermarkt/   (gitignored)
             World Bank JSON -> data_pipeline/cache/world_bank/          (gitignored)
  produces:  data_pipeline/processed/*.csv                               (gitignored)
```

`data_pipeline/processed/` is gitignored — run `build_data.py` to generate it. The raw martj42 source files in `data_pipeline/raw/` are committed as a pinned snapshot and are sufficient for an offline run (no `--refresh` needed).

Key processed outputs (see `data_pipeline/README.md` for full descriptions):
- `matches_1960_completed.csv` — base match table with Elo and context flags
- `model_features_long.csv` — model-ready, imputed, two-rows-per-match table
- `poisson_training_long.csv` — Statistical V1 Poisson target table

### Core design decisions

**Perspective rows**: Every match is stored as two rows — one from each team's point of view — so the model trains on "team vs opponent" rather than "home vs away". The `is_home` and `is_listed_home` flags capture venue.

**Elo**: Computed from 1900 using the eloratings.net formula (K weighted by tournament importance, goal-difference multiplier, +100 home advantage). Only pre-match ratings are exported — no leakage. `elo.py` is shared by both the ML and Statistical pipelines.

**Time-based train/val/test split**: Never random, to prevent leakage. Both rows of a match always land in the same split.
- train: 1960–2015
- val: 2015–2020
- test: 2020+

**Missing values**: Raw features keep real `NaN` where history doesn't exist. `model_features_long.csv` fills them deterministically (rest days → median, rolling averages → mean, H2H → neutral defaults) and adds `missing_*` indicator flags.

**Name normalization**: `normalize_name()` in `build_data.py` strips accents, lowercases, and collapses punctuation. Country name mapping between martj42, World Bank, and Transfermarkt is in `data_pipeline/reference/country_name_mappings.csv`. Unmatched names are written to `data_pipeline/processed/name_mapping_diagnostics.csv`.

## Branch Convention

`ML/init` is the branch for the ML model. Classical statistics work is expected on a separate branch.
