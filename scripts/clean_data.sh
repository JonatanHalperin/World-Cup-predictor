#!/usr/bin/env bash
# Remove all gitignored generated data files.
#
# Usage:
#   bash scripts/clean_data.sh           # removes processed data and ML V1 artifacts
#   bash scripts/clean_data.sh --cache   # also removes downloaded cache (requires
#                                        # internet + --refresh to rebuild)

set -euo pipefail
cd "$(dirname "$0")/.."

CLEAN_CACHE=false
if [[ "${1:-}" == "--cache" ]]; then
    CLEAN_CACHE=true
fi

echo "Removing data_pipeline/processed/ ..."
rm -rf data_pipeline/processed/

echo "Removing ML_V1/data/ contents ..."
find ML_V1/data/ -type f ! -name '.gitkeep' -delete

echo "Removing world_cup_predictions.csv ..."
rm -f world_cup_predictions.csv

if $CLEAN_CACHE; then
    echo "Removing data_pipeline/cache/ ..."
    rm -rf data_pipeline/cache/
fi

echo "Done. Run bash scripts/build_data.sh to regenerate."
