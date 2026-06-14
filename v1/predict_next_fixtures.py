"""Predict the next N World Cup fixtures and write a collective report."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .model_weights import model_weights_table, plot_model_weights
    from .poisson_glm import (
        DEFAULT_RIDGE_ALPHA,
        fit_poisson_glm_from_data,
        outcome_probabilities,
        poisson_score_matrix,
        predict_expected_goals,
    )
    from .predict_fixture import (
        DEFAULT_FEATURE_COLUMNS,
        DEFAULT_FIXTURES_CSV,
        DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS,
        DEFAULT_RAW_RESULTS_CSV,
        DEFAULT_TRAINING_CSV,
        build_future_feature_rows,
        elo_metadata,
        format_probability,
        markdown_table,
        missing_data_tables,
        model_diagnostics_table,
        slugify,
        top_scorelines,
        weighting_metadata,
        regularization_metadata,
    )
    from .tuning_config import DEFAULT_TUNED_CONFIG_PATH, load_tuned_config
except ImportError:
    from model_weights import model_weights_table, plot_model_weights
    from poisson_glm import (
        DEFAULT_RIDGE_ALPHA,
        fit_poisson_glm_from_data,
        outcome_probabilities,
        poisson_score_matrix,
        predict_expected_goals,
    )
    from predict_fixture import (
        DEFAULT_FEATURE_COLUMNS,
        DEFAULT_FIXTURES_CSV,
        DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS,
        DEFAULT_RAW_RESULTS_CSV,
        DEFAULT_TRAINING_CSV,
        build_future_feature_rows,
        elo_metadata,
        format_probability,
        markdown_table,
        missing_data_tables,
        model_diagnostics_table,
        slugify,
        top_scorelines,
        weighting_metadata,
        regularization_metadata,
    )
    from tuning_config import DEFAULT_TUNED_CONFIG_PATH, load_tuned_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "v1" / "reports"
DEFAULT_AS_OF_DATE = "2026-06-14"


def select_next_world_cup_fixtures(
    fixtures: pd.DataFrame,
    *,
    as_of_date: str,
    count: int,
) -> pd.DataFrame:
    if count <= 0:
        raise ValueError("count must be positive.")
    selected = fixtures.copy()
    selected["date"] = pd.to_datetime(selected["date"], errors="coerce")
    if selected["date"].isna().any():
        raise ValueError("future fixture file contains invalid dates.")
    selected = selected[
        selected["date"].ge(pd.Timestamp(as_of_date))
        & selected["is_world_cup"].astype(bool)
        & selected[["home_score", "away_score"]].isna().any(axis=1)
    ].copy()
    selected = selected.sort_values(["date", "fixture_id"]).head(count)
    if len(selected) < count:
        raise ValueError(f"Only found {len(selected)} future World Cup fixtures on or after {as_of_date}.")
    return selected.reset_index(drop=True)


def best_scorelines_by_outcome(
    score_matrix: np.ndarray,
    home_team: str,
    away_team: str,
) -> dict[str, object]:
    """Find the most probable exact scoreline inside each outcome bucket."""
    outcome_specs = {
        "home_win": (
            lambda home_goals, away_goals: home_goals > away_goals,
            f"{home_team} win",
        ),
        "draw": (
            lambda home_goals, away_goals: home_goals == away_goals,
            "Draw",
        ),
        "away_win": (
            lambda home_goals, away_goals: home_goals < away_goals,
            f"{away_team} win",
        ),
    }
    best: dict[str, object] = {}
    for outcome_key, (predicate, outcome_label) in outcome_specs.items():
        best_home_goals: int | None = None
        best_away_goals: int | None = None
        best_probability = -1.0
        for home_goals in range(score_matrix.shape[0]):
            for away_goals in range(score_matrix.shape[1]):
                if not predicate(home_goals, away_goals):
                    continue
                probability = float(score_matrix[home_goals, away_goals])
                if probability > best_probability:
                    best_home_goals = home_goals
                    best_away_goals = away_goals
                    best_probability = probability

        scoreline = (
            ""
            if best_home_goals is None or best_away_goals is None
            else f"{home_team} {best_home_goals}-{best_away_goals} {away_team}"
        )
        best[f"{outcome_key}_best_scoreline"] = scoreline
        best[f"{outcome_key}_best_scoreline_probability"] = (
            np.nan if best_probability < 0 else best_probability
        )
        best[f"{outcome_key}_outcome"] = outcome_label
    return best


def predict_fixture_row(
    *,
    model_result,
    fixture: pd.Series,
    training: pd.DataFrame,
    raw_results_csv: Path,
    max_goals: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    future_rows = build_future_feature_rows(
        fixture=fixture,
        training=training,
        raw_results_csv=raw_results_csv,
    )
    predictions = predict_expected_goals(model_result, future_rows)
    prediction_rows = future_rows[["team", "is_listed_home"]].join(predictions)
    prediction_rows["side"] = np.where(
        prediction_rows["is_listed_home"].eq(1),
        "listed_home",
        "listed_away",
    )
    expected_goals = prediction_rows.rename(columns={"lambda": "expected_goals"})[
        ["team", "side", "eta", "expected_goals"]
    ]

    home_team = str(fixture["home_team"])
    away_team = str(fixture["away_team"])
    home_lambda = float(expected_goals.loc[expected_goals["side"].eq("listed_home"), "expected_goals"].iloc[0])
    away_lambda = float(expected_goals.loc[expected_goals["side"].eq("listed_away"), "expected_goals"].iloc[0])
    score_matrix = poisson_score_matrix(home_lambda, away_lambda, max_goals=max_goals)
    outcomes = outcome_probabilities(score_matrix)
    top_scores = top_scorelines(score_matrix, home_team, away_team, limit=5)
    best_scorelines = best_scorelines_by_outcome(score_matrix, home_team, away_team)

    best_outcome_key = max(outcomes, key=outcomes.get)
    best_outcome_label = {
        "home_win": f"{home_team} win",
        "draw": "Draw",
        "away_win": f"{away_team} win",
    }[best_outcome_key]

    summary = {
        "fixture_id": int(fixture["fixture_id"]),
        "date": pd.Timestamp(fixture["date"]).date().isoformat(),
        "home_team": home_team,
        "away_team": away_team,
        "match": f"{home_team} vs {away_team}",
        "city": fixture["city"],
        "country": fixture["country"],
        "neutral": bool(fixture["neutral"]),
        "lambda_home": home_lambda,
        "lambda_away": away_lambda,
        "home_win_probability": outcomes["home_win"],
        "draw_probability": outcomes["draw"],
        "away_win_probability": outcomes["away_win"],
        "best_outcome": best_outcome_label,
        "best_outcome_probability": outcomes[best_outcome_key],
        "top_scoreline": top_scores.iloc[0]["scoreline"],
        "top_scoreline_probability": top_scores.iloc[0]["probability"],
        "score_matrix_probability_mass": float(score_matrix.sum()),
    }
    summary.update(best_scorelines)
    top_scores = top_scores.copy()
    top_scores.insert(0, "fixture_id", int(fixture["fixture_id"]))
    top_scores.insert(1, "match", f"{home_team} vs {away_team}")
    return summary, top_scores


def plot_outcome_probabilities(predictions: pd.DataFrame, path: Path) -> None:
    labels = predictions["match"].tolist()
    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, predictions["home_win_probability"], width, label="Home win", color="#276fbf")
    ax.bar(x, predictions["draw_probability"], width, label="Draw", color="#8a8f98")
    ax.bar(x + width, predictions["away_win_probability"], width, label="Away win", color="#c44536")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Probability")
    ax.set_title("Outcome probabilities for next 2026 World Cup fixtures")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_expected_goals(predictions: pd.DataFrame, path: Path) -> None:
    labels = predictions["match"].tolist()
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width / 2, predictions["lambda_home"], width, label="Home/listed first", color="#276fbf")
    ax.bar(x + width / 2, predictions["lambda_away"], width, label="Away/listed second", color="#c44536")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Expected goals")
    ax.set_title("Expected goals for next 2026 World Cup fixtures")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_best_outcomes(predictions: pd.DataFrame, path: Path) -> None:
    plot_data = predictions.sort_values("best_outcome_probability", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(plot_data["match"], plot_data["best_outcome_probability"], color="#2f7d62")
    ax.set_xlabel("Probability")
    ax.set_title("Most likely outcome confidence")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for index, row in enumerate(plot_data.itertuples(index=False)):
        ax.text(row.best_outcome_probability, index, f" {row.best_outcome}: {row.best_outcome_probability:.1%}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_collective_report(
    *,
    output_dir: Path,
    fixture_predictions: pd.DataFrame,
    top_scorelines_table: pd.DataFrame,
    model_training: pd.DataFrame,
    diagnostics: pd.DataFrame,
    model_weights: pd.DataFrame,
    model_result,
    artifacts: dict[str, Path],
    as_of_date: str,
    tuned_config_source: str,
) -> None:
    missing_from_source, available_not_modeled = missing_data_tables()
    feature_columns = list(getattr(model_result, "feature_columns", []))
    metadata = elo_metadata(feature_columns)
    weights = weighting_metadata(model_result)
    summary_display = fixture_predictions[
        [
            "fixture_id",
            "date",
            "match",
            "city",
            "lambda_home",
            "lambda_away",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "best_outcome",
            "top_scoreline",
            "top_scoreline_probability",
            "home_win_best_scoreline",
            "home_win_best_scoreline_probability",
            "draw_best_scoreline",
            "draw_best_scoreline_probability",
            "away_win_best_scoreline",
            "away_win_best_scoreline_probability",
        ]
    ].copy()
    best_by_outcome_display = fixture_predictions[
        [
            "fixture_id",
            "match",
            "home_win_best_scoreline",
            "home_win_best_scoreline_probability",
            "draw_best_scoreline",
            "draw_best_scoreline_probability",
            "away_win_best_scoreline",
            "away_win_best_scoreline_probability",
        ]
    ].copy()
    top_display = top_scorelines_table[
        ["fixture_id", "match", "scoreline", "outcome", "probability"]
    ].copy()
    training_summary = pd.DataFrame(
        [
            {
                "training_matches": int(model_training["match_id"].nunique()),
                "training_rows": int(len(model_training)),
                "training_start_date": pd.Timestamp(model_training["date"].min()).date().isoformat(),
                "training_end_date": pd.Timestamp(model_training["date"].max()).date().isoformat(),
                **metadata,
                **weights,
                **regularization_metadata(model_result),
                "tuned_config_source": tuned_config_source,
            }
        ]
    )
    fixture_count = len(fixture_predictions)

    lines = [
        f"# Next {fixture_count} 2026 World Cup Fixture Predictions",
        "",
        f"Fixtures selected from `future_fixtures.csv` on or after `{as_of_date}`. Since kickoff times are not available, all fixtures dated `{as_of_date}` are treated as upcoming.",
        "",
        "## Model Training",
        "",
        markdown_table(training_summary),
        "",
        "## Prediction Summary",
        "",
        markdown_table(
            summary_display,
            percent_columns={
                "home_win_probability",
                "draw_probability",
                "away_win_probability",
                "top_scoreline_probability",
                "home_win_best_scoreline_probability",
                "draw_best_scoreline_probability",
                "away_win_best_scoreline_probability",
            },
        ),
        "",
        "## Most Probable Scoreline for Each Outcome",
        "",
        markdown_table(
            best_by_outcome_display,
            percent_columns={
                "home_win_best_scoreline_probability",
                "draw_best_scoreline_probability",
                "away_win_best_scoreline_probability",
            },
        ),
        "",
        "## Top Scorelines",
        "",
        markdown_table(top_display, percent_columns={"probability"}),
        "",
        "## Model Weights",
        "",
        "The chart `model_weights.png` shows standardized GLM coefficients:",
        "`beta * training feature standard deviation`. Positive values increase",
        "expected goals on the log scale; negative values decrease expected goals.",
        "",
        markdown_table(
            model_weights[model_weights["term"].ne("const")].head(20)[
                [
                    "term",
                    "coefficient",
                    "std_error",
                    "p_value",
                    "standardized_coefficient",
                    "rate_ratio_per_1sd",
                ]
            ]
        ),
        "",
        "## Model Diagnostics",
        "",
        f"- Log-likelihood: {model_result.llf:.6f}",
        f"- AIC: {model_result.aic:.6f}",
        f"- Deviance: {model_result.deviance:.6f}",
        "",
        markdown_table(diagnostics),
        "",
        "## Missing or Not Yet Modeled Data",
        "",
        "### Missing from Current Source Data",
        "",
        markdown_table(missing_from_source),
        "",
        "### Available in Principle But Not Used by V1",
        "",
        markdown_table(available_not_modeled),
        "",
        "## Artifacts",
        "",
        markdown_table(pd.DataFrame({"artifact": artifacts.keys(), "path": [str(path) for path in artifacts.values()]})),
        "",
    ]
    artifacts["collective_report_md"].write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict the next N 2026 World Cup fixtures.")
    parser.add_argument("--as-of-date", default=DEFAULT_AS_OF_DATE)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--fixtures-csv", type=Path, default=DEFAULT_FIXTURES_CSV)
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING_CSV)
    parser.add_argument("--raw-results-csv", type=Path, default=DEFAULT_RAW_RESULTS_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument(
        "--kyrre-weight-half-life-years",
        type=float,
        default=None,
        help="Half-life in years for Kyrre training weights. Defaults to tuned config, then 8.",
    )
    parser.add_argument(
        "--kyrre-weight-gamma",
        type=float,
        default=None,
        help="Optional gamma for Kyrre weights exp(-gamma * match_age_years). Overrides half-life.",
    )
    parser.add_argument(
        "--no-kyrre-weight",
        action="store_true",
        help="Disable Kyrre-weighted likelihood fitting.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=None,
        help="L2 ridge penalty strength. Defaults to tuned config, then 0.01. Use 0 for unregularized GLM.",
    )
    parser.add_argument(
        "--tuned-config",
        type=Path,
        default=DEFAULT_TUNED_CONFIG_PATH,
        help="Path to tuned Poisson GLM hyperparameter config.",
    )
    parser.add_argument(
        "--ignore-tuned-config",
        action="store_true",
        help="Ignore tuned config and use fallback defaults unless explicit flags are supplied.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURE_COLUMNS,
        help="Feature columns to use exactly as they appear in poisson_training_long.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixtures = pd.read_csv(args.fixtures_csv)
    training = pd.read_csv(args.training_csv)
    training["date"] = pd.to_datetime(training["date"], errors="coerce")
    tuned_config = load_tuned_config(
        args.tuned_config,
        feature_columns=args.features,
        ignore=args.ignore_tuned_config,
    )
    ridge_alpha = (
        float(args.ridge_alpha)
        if args.ridge_alpha is not None
        else float(tuned_config.get("ridge_alpha", DEFAULT_RIDGE_ALPHA))
    )
    if args.no_kyrre_weight:
        kyrre_half_life = None
    elif args.kyrre_weight_half_life_years is not None:
        kyrre_half_life = float(args.kyrre_weight_half_life_years)
    else:
        tuned_half_life = tuned_config.get("kyrre_weight_half_life_years", DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS)
        kyrre_half_life = None if tuned_half_life is None else float(tuned_half_life)
    selected = select_next_world_cup_fixtures(
        fixtures,
        as_of_date=args.as_of_date,
        count=args.count,
    )

    model_result = fit_poisson_glm_from_data(
        training,
        feature_columns=args.features,
        target_column="goals_for",
        id_columns=["match_id", "team", "opponent"],
        date_column="date",
        kyrre_weight_gamma=None if args.no_kyrre_weight else args.kyrre_weight_gamma,
        kyrre_weight_half_life_years=kyrre_half_life,
        ridge_alpha=ridge_alpha,
    )

    prediction_rows = []
    top_scoreline_rows = []
    for fixture in selected.itertuples(index=False):
        fixture_series = pd.Series(fixture._asdict())
        prediction, top_scores = predict_fixture_row(
            model_result=model_result,
            fixture=fixture_series,
            training=training,
            raw_results_csv=args.raw_results_csv,
            max_goals=args.max_goals,
        )
        prediction_rows.append(prediction)
        top_scoreline_rows.append(top_scores)

    fixture_predictions = pd.DataFrame(prediction_rows)
    top_scorelines_table = pd.concat(top_scoreline_rows, ignore_index=True)

    first_date = fixture_predictions["date"].iloc[0]
    last_date = fixture_predictions["date"].iloc[-1]
    output_dir = args.output_root / (
        f"next-{args.count}-wc-fixtures-{slugify(first_date)}-to-{slugify(last_date)}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "collective_report_md": output_dir / "collective_report.md",
        "fixture_predictions_csv": output_dir / "fixture_predictions.csv",
        "top_scorelines_csv": output_dir / "top_scorelines.csv",
        "outcome_probabilities_png": output_dir / "outcome_probabilities.png",
        "expected_goals_png": output_dir / "expected_goals.png",
        "best_outcomes_png": output_dir / "best_outcomes.png",
        "model_weights_csv": output_dir / "model_weights.csv",
        "model_weights_png": output_dir / "model_weights.png",
    }

    weights = model_weights_table(model_result)
    fixture_predictions.to_csv(artifacts["fixture_predictions_csv"], index=False)
    top_scorelines_table.to_csv(artifacts["top_scorelines_csv"], index=False)
    weights.to_csv(artifacts["model_weights_csv"], index=False)
    plot_outcome_probabilities(fixture_predictions, artifacts["outcome_probabilities_png"])
    plot_expected_goals(fixture_predictions, artifacts["expected_goals_png"])
    plot_best_outcomes(fixture_predictions, artifacts["best_outcomes_png"])
    plot_model_weights(weights, artifacts["model_weights_png"])
    write_collective_report(
        output_dir=output_dir,
        fixture_predictions=fixture_predictions,
        top_scorelines_table=top_scorelines_table,
        model_training=training,
        diagnostics=model_diagnostics_table(model_result),
        model_weights=weights,
        model_result=model_result,
        artifacts=artifacts,
        as_of_date=args.as_of_date,
        tuned_config_source=str(tuned_config.get("source", "")),
    )

    print("\nNext World Cup fixture predictions")
    display = fixture_predictions[
        [
            "fixture_id",
            "date",
            "match",
            "lambda_home",
            "lambda_away",
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "best_outcome",
            "top_scoreline",
            "home_win_best_scoreline",
            "draw_best_scoreline",
            "away_win_best_scoreline",
        ]
    ].copy()
    for column in ["home_win_probability", "draw_probability", "away_win_probability"]:
        display[column] = display[column].map(format_probability)
    print(display.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nLargest standardized model weights")
    top_weights = weights[weights["term"].ne("const")].head(12)[
        ["term", "coefficient", "standardized_coefficient", "rate_ratio_per_1sd", "p_value"]
    ]
    print(top_weights.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nReport artifacts")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
