"""Example: fit the V1 Poisson GLM and predict one football match."""
from __future__ import annotations

from pathlib import Path

from poisson_glm import (
    DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS,
    fit_poisson_glm,
    outcome_probabilities,
    poisson_score_matrix,
    predict_expected_goals,
)
from predict_fixture import DEFAULT_FEATURE_COLUMNS
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data_pipeline" / "processed" / "poisson_training_long.csv"

FEATURE_COLUMNS = DEFAULT_FEATURE_COLUMNS


def main() -> None:
    result = fit_poisson_glm(
        DATA_PATH,
        feature_columns=FEATURE_COLUMNS,
        target_column="goals_for",
        id_columns=["match_id", "team", "opponent"],
        kyrre_weight_half_life_years=DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS,
    )

    data = pd.read_csv(DATA_PATH)

    # Pick the first match where both team-perspective rows are available. One
    # row gives the home team's expected goals, and the paired row gives the
    # away team's expected goals.
    match_rows = data.groupby("match_id", sort=False).filter(lambda rows: len(rows) == 2)
    one_match = match_rows.groupby("match_id", sort=False).head(2).head(2)
    if len(one_match) != 2:
        raise RuntimeError("Could not find a two-row match example.")

    predictions = predict_expected_goals(result, one_match)
    home_row = one_match[one_match["is_listed_home"] == 1].iloc[0]
    away_row = one_match[one_match["is_listed_home"] == 0].iloc[0]

    lambda_home = float(predictions.loc[home_row.name, "lambda"])
    lambda_away = float(predictions.loc[away_row.name, "lambda"])

    score_matrix = poisson_score_matrix(lambda_home, lambda_away, max_goals=10)
    outcomes = outcome_probabilities(score_matrix)

    print("\nOne-match prediction example")
    print(f"Match: {home_row['team']} vs {away_row['team']}")
    print(f"Expected goals: {home_row['team']} {lambda_home:.3f}, {away_row['team']} {lambda_away:.3f}")
    print("Outcome probabilities within 0..10 goals:")
    print(f"  Home win: {outcomes['home_win']:.3%}")
    print(f"  Draw:     {outcomes['draw']:.3%}")
    print(f"  Away win: {outcomes['away_win']:.3%}")
    print(f"  Matrix probability mass: {score_matrix.sum():.3%}")


if __name__ == "__main__":
    main()
