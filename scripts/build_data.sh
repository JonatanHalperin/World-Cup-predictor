#!/usr/bin/env bash
# Generate all gitignored data files from the committed raw sources.
#
# Usage:
#   bash scripts/build_data.sh           # use committed raw files (offline)
#   bash scripts/build_data.sh --refresh # re-download raw sources first
#
# What this builds:
#   data_pipeline/processed/   - Statistical V1 feature tables
#   ML_V1/data/                - Elo CSV, baseline splits, scaling stats
#
# The ML V1 model (ML_V1/data/baseline_model.pt) is NOT trained here
# because it takes time. Run it separately once the data is ready:
#   python -m ML_V1.train_baseline

set -euo pipefail
cd "$(dirname "$0")/.."

if command -v poetry &>/dev/null && [ -f pyproject.toml ]; then
    PYTHON="poetry run python"
else
    PYTHON="python"
fi

REFRESH=""
if [[ "${1:-}" == "--refresh" ]]; then
    REFRESH="--refresh"
fi

echo "==> Building Statistical V1 processed data..."
$PYTHON data_pipeline/build_data.py $REFRESH

echo ""
echo "==> Computing Elo ratings for ML V1..."
$PYTHON -m data_pipeline.elo

echo ""
echo "==> Building ML V1 baseline dataset (train/val/test splits)..."
$PYTHON -m data_pipeline.make_baseline_dataset

echo ""
echo "All data files are ready."
echo "Next step — train the ML V1 model:"
echo "  $PYTHON -m ML_V1.train_baseline"
