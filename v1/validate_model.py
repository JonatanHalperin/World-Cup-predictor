"""Validate the V1 Poisson GLM on fixed historical train/validation/test splits."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson

try:
    from .poisson_glm import (
        DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS,
        fit_poisson_glm_from_data,
        outcome_probabilities,
        poisson_score_matrix,
        predict_expected_goals,
    )
except ImportError:
    from poisson_glm import (
        DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS,
        fit_poisson_glm_from_data,
        outcome_probabilities,
        poisson_score_matrix,
        predict_expected_goals,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_CSV = PROJECT_ROOT / "data_pipeline" / "processed" / "poisson_training_long.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "v1" / "reports" / "validation_2010_2018__2019_2021__2022_world_cup"

DEFAULT_TRAIN_START = "2010-01-01"
DEFAULT_TRAIN_END = "2018-12-31"
DEFAULT_VALIDATION_START = "2019-01-01"
DEFAULT_VALIDATION_END = "2021-12-31"
DEFAULT_TEST_YEAR = 2022
DEFAULT_TEST_TOURNAMENT = "FIFA World Cup"

# Match the expanded leak-free, non-redundant feature list used by the V1
# fixture report.
DEFAULT_FEATURE_COLUMNS = [
    "team_elo_pre",
    "opp_elo_pre",
    "is_home",
    "is_neutral",
    "team_is_host",
    "opponent_is_host",
    "missing_days_since_last_match",
    "days_since_last_match",
    "prev_match_went_to_shootout",
    "matches_last_30d",
    "matches_last_60d",
    "matches_last_365d",
    "unbeaten_streak_pre",
    "winless_streak_pre",
    "is_friendly",
    "is_qualifier",
    "is_world_cup",
    "is_continental",
    "matches_available_last_5",
    "points_per_game_last_5",
    "goals_for_avg_last_5",
    "goals_against_avg_last_5",
    "matches_available_last_10",
    "points_per_game_last_10",
    "goals_for_avg_last_10",
    "goals_against_avg_last_10",
    "h2h_matches_60y",
    "h2h_goal_diff_avg_60y",
    "h2h_points_per_game_60y",
    "missing_h2h_60y",
]

OUTCOME_ORDER = ["home_win", "draw", "away_win"]
EPSILON = 1e-15


def format_probability(value: float) -> str:
    return f"{value:.3%}"


def markdown_table(df: pd.DataFrame, percent_columns: set[str] | None = None) -> str:
    percent_columns = percent_columns or set()
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if column in percent_columns:
                    values.append(format_probability(value))
                else:
                    values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_training_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "match_id",
        "date",
        "team",
        "opponent",
        "home_team",
        "away_team",
        "is_listed_home",
        "goals_for",
        "goals_against",
        "tournament",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Training data is missing required column(s): {', '.join(missing)}")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        raise ValueError("Training data contains invalid dates.")
    return data


def ensure_two_rows_per_match(data: pd.DataFrame, split_name: str) -> None:
    group_sizes = data.groupby("match_id").size()
    bad_matches = group_sizes[group_sizes.ne(2)]
    if not bad_matches.empty:
        examples = ", ".join(str(match_id) for match_id in bad_matches.index[:10])
        raise ValueError(f"{split_name} matches must have exactly two rows. Bad match_id values: {examples}")


def split_fixed_periods(
    data: pd.DataFrame,
    *,
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_year: int,
    test_tournament: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)
    validation_start_ts = pd.Timestamp(validation_start)
    validation_end_ts = pd.Timestamp(validation_end)

    train = data[data["date"].between(train_start_ts, train_end_ts, inclusive="both")].copy()
    validation = data[
        data["date"].between(validation_start_ts, validation_end_ts, inclusive="both")
    ].copy()
    test = data[
        data["date"].dt.year.eq(test_year)
        & data["tournament"].astype(str).str.lower().eq(test_tournament.lower())
    ].copy()

    for split_name, split in {
        "train": train,
        "validation": validation,
        "test": test,
    }.items():
        if split.empty:
            raise ValueError(f"{split_name} split is empty.")
        ensure_two_rows_per_match(split, split_name)

    train_ids = set(train["match_id"])
    validation_ids = set(validation["match_id"])
    test_ids = set(test["match_id"])
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("Train, validation, and test splits must not overlap.")

    sort_columns = ["date", "match_id", "is_listed_home"]
    train = train.sort_values(sort_columns, ascending=[True, True, False]).reset_index(drop=True)
    validation = validation.sort_values(sort_columns, ascending=[True, True, False]).reset_index(drop=True)
    test = test.sort_values(sort_columns, ascending=[True, True, False]).reset_index(drop=True)
    return train, validation, test


def add_row_predictions(model_result, evaluation: pd.DataFrame) -> pd.DataFrame:
    predictions = predict_expected_goals(model_result, evaluation)
    rows = evaluation.copy().join(predictions)
    rows["actual_goals"] = rows["goals_for"].astype(int)
    rows["predicted_goals"] = rows["lambda"]
    rows["poisson_nll"] = -poisson.logpmf(rows["actual_goals"], rows["predicted_goals"])
    rows["absolute_error"] = (rows["actual_goals"] - rows["predicted_goals"]).abs()
    rows["squared_error"] = (rows["actual_goals"] - rows["predicted_goals"]) ** 2
    return rows


def actual_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def top_scoreline(score_matrix: np.ndarray) -> tuple[int, int, float]:
    flat_index = int(np.argmax(score_matrix))
    home_goals, away_goals = np.unravel_index(flat_index, score_matrix.shape)
    return int(home_goals), int(away_goals), float(score_matrix[home_goals, away_goals])


def build_match_predictions(row_predictions: pd.DataFrame, max_goals: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for match_id, group in row_predictions.groupby("match_id", sort=False):
        if len(group) != 2:
            raise ValueError(f"match_id {match_id} does not have exactly two evaluation rows.")
        home = group[group["is_listed_home"].eq(1)]
        away = group[group["is_listed_home"].eq(0)]
        if len(home) != 1 or len(away) != 1:
            raise ValueError(f"match_id {match_id} must have one listed-home and one listed-away row.")
        home_row = home.iloc[0]
        away_row = away.iloc[0]

        home_lambda = float(home_row["lambda"])
        away_lambda = float(away_row["lambda"])
        score_matrix = poisson_score_matrix(home_lambda, away_lambda, max_goals=max_goals)
        outcomes = outcome_probabilities(score_matrix)
        probability_mass = float(score_matrix.sum())
        normalized_outcomes = {
            outcome: probability / probability_mass
            for outcome, probability in outcomes.items()
        }
        actual_home_goals = int(home_row["goals_for"])
        actual_away_goals = int(away_row["goals_for"])
        observed_outcome = actual_outcome(actual_home_goals, actual_away_goals)
        predicted_outcome = max(outcomes, key=outcomes.get)
        top_home_goals, top_away_goals, top_score_probability = top_scoreline(score_matrix)

        if actual_home_goals < score_matrix.shape[0] and actual_away_goals < score_matrix.shape[1]:
            exact_score_probability = float(score_matrix[actual_home_goals, actual_away_goals])
        else:
            exact_score_probability = 0.0

        actual_outcome_probability = normalized_outcomes[observed_outcome]
        one_hot = {outcome: float(outcome == observed_outcome) for outcome in OUTCOME_ORDER}
        brier = sum((normalized_outcomes[outcome] - one_hot[outcome]) ** 2 for outcome in OUTCOME_ORDER)

        rows.append(
            {
                "match_id": int(match_id),
                "date": home_row["date"],
                "home_team": home_row["home_team"],
                "away_team": home_row["away_team"],
                "actual_home_goals": actual_home_goals,
                "actual_away_goals": actual_away_goals,
                "predicted_home_goals": home_lambda,
                "predicted_away_goals": away_lambda,
                "home_win_probability": outcomes["home_win"],
                "draw_probability": outcomes["draw"],
                "away_win_probability": outcomes["away_win"],
                "score_matrix_probability_mass": probability_mass,
                "actual_outcome": observed_outcome,
                "predicted_outcome": predicted_outcome,
                "actual_outcome_probability": actual_outcome_probability,
                "outcome_log_loss": -np.log(max(actual_outcome_probability, EPSILON)),
                "outcome_brier_score": brier,
                "outcome_correct": int(predicted_outcome == observed_outcome),
                "top_scoreline": f"{top_home_goals}-{top_away_goals}",
                "top_scoreline_probability": top_score_probability,
                "actual_scoreline": f"{actual_home_goals}-{actual_away_goals}",
                "exact_score_probability": exact_score_probability,
                "exact_score_top1_correct": int(
                    top_home_goals == actual_home_goals and top_away_goals == actual_away_goals
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)


def build_goal_calibration(row_predictions: pd.DataFrame, bin_width: float = 0.5) -> pd.DataFrame:
    """Group predicted lambdas into fixed bins and compare averages.

    A Poisson lambda is an expected goal value, not a point prediction. The
    calibration question is: among rows where the model predicted roughly 1.0
    to 1.5 goals, did those teams actually score about 1.0 to 1.5 goals on
    average?
    """
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("calibration bin width must be finite and positive.")

    calibration = row_predictions[["predicted_goals", "actual_goals", "absolute_error"]].copy()
    max_prediction = float(calibration["predicted_goals"].max())
    max_edge = max(bin_width, np.ceil(max_prediction / bin_width) * bin_width)
    edges = np.arange(0.0, max_edge + bin_width, bin_width)
    if edges[-1] <= max_prediction:
        edges = np.append(edges, edges[-1] + bin_width)

    labels = [f"{edges[index]:.1f}-{edges[index + 1]:.1f}" for index in range(len(edges) - 1)]
    calibration["predicted_lambda_bin"] = pd.cut(
        calibration["predicted_goals"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    grouped = calibration.groupby("predicted_lambda_bin", observed=True)
    result = grouped.agg(
        rows=("actual_goals", "size"),
        mean_predicted_goals=("predicted_goals", "mean"),
        mean_actual_goals=("actual_goals", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
    ).reset_index()
    result["bin_lower"] = result["predicted_lambda_bin"].astype(str).str.split("-").str[0].astype(float)
    result["bin_upper"] = result["predicted_lambda_bin"].astype(str).str.split("-").str[1].astype(float)
    result["calibration_error"] = result["mean_actual_goals"] - result["mean_predicted_goals"]
    result["absolute_calibration_error"] = result["calibration_error"].abs()
    return result[
        [
            "predicted_lambda_bin",
            "bin_lower",
            "bin_upper",
            "rows",
            "mean_predicted_goals",
            "mean_actual_goals",
            "calibration_error",
            "absolute_calibration_error",
            "mean_absolute_error",
        ]
    ]


def summary_row(
    *,
    split_name: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    row_predictions: pd.DataFrame,
    match_predictions: pd.DataFrame,
    max_goals: int,
    model_result,
) -> dict[str, object]:
    weight_summary = getattr(model_result, "sample_weight_summary", {}) or {}
    return {
        "split": split_name,
        "max_goals": max_goals,
        "train_matches": int(train["match_id"].nunique()),
        "train_rows": int(len(train)),
        "train_start_date": train["date"].min().date().isoformat(),
        "train_end_date": train["date"].max().date().isoformat(),
        "evaluation_matches": int(evaluation["match_id"].nunique()),
        "evaluation_rows": int(len(evaluation)),
        "evaluation_start_date": evaluation["date"].min().date().isoformat(),
        "evaluation_end_date": evaluation["date"].max().date().isoformat(),
        "elo_system": "Elo",
        "elo_decay": "none",
        "sample_weighting": weight_summary.get("weighting", "none"),
        "kyrre_weight_gamma": weight_summary.get("kyrre_weight_gamma", ""),
        "kyrre_weight_half_life_years": weight_summary.get("kyrre_weight_half_life_years", ""),
        "kyrre_weight_reference_date": weight_summary.get("kyrre_weight_reference_date", ""),
        "weight_sum": weight_summary.get("weight_sum", ""),
        "mean_poisson_nll": float(row_predictions["poisson_nll"].mean()),
        "goal_mae": float(row_predictions["absolute_error"].mean()),
        "goal_rmse": float(np.sqrt(row_predictions["squared_error"].mean())),
        "mean_predicted_goals": float(row_predictions["predicted_goals"].mean()),
        "mean_actual_goals": float(row_predictions["actual_goals"].mean()),
        "outcome_accuracy": float(match_predictions["outcome_correct"].mean()),
        "outcome_brier_score": float(match_predictions["outcome_brier_score"].mean()),
        "outcome_log_loss": float(match_predictions["outcome_log_loss"].mean()),
        "avg_actual_outcome_probability": float(match_predictions["actual_outcome_probability"].mean()),
        "exact_score_top1_accuracy": float(match_predictions["exact_score_top1_correct"].mean()),
        "mean_exact_score_probability": float(match_predictions["exact_score_probability"].mean()),
        "mean_score_matrix_probability_mass": float(match_predictions["score_matrix_probability_mass"].mean()),
    }


def plot_predicted_vs_actual(row_predictions: pd.DataFrame, split_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(row_predictions["predicted_goals"], row_predictions["actual_goals"], alpha=0.65, color="#276fbf")
    upper = max(row_predictions["predicted_goals"].max(), row_predictions["actual_goals"].max()) + 0.5
    ax.plot([0, upper], [0, upper], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("Predicted goals")
    ax.set_ylabel("Actual goals")
    ax.set_title(f"Predicted vs actual goals: {split_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_goal_calibration(goal_calibration: pd.DataFrame, split_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        goal_calibration["mean_predicted_goals"],
        goal_calibration["mean_actual_goals"],
        marker="o",
        label="Observed bin average",
    )
    lower = min(goal_calibration["mean_predicted_goals"].min(), goal_calibration["mean_actual_goals"].min())
    upper = max(goal_calibration["mean_predicted_goals"].max(), goal_calibration["mean_actual_goals"].max())
    ax.plot([lower, upper], [lower, upper], color="#333333", linestyle="--", linewidth=1, label="Perfect")
    ax.set_xlabel("Mean predicted goals")
    ax.set_ylabel("Mean actual goals")
    ax.set_title(f"Goal calibration by prediction bin: {split_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_outcome_probability_bars(match_predictions: pd.DataFrame, split_name: str, path: Path) -> None:
    actual_counts = match_predictions["actual_outcome"].value_counts(normalize=True).reindex(OUTCOME_ORDER).fillna(0)
    predicted_means = pd.Series(
        {
            "home_win": match_predictions["home_win_probability"].mean(),
            "draw": match_predictions["draw_probability"].mean(),
            "away_win": match_predictions["away_win_probability"].mean(),
        }
    )
    labels = ["Home win", "Draw", "Away win"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, predicted_means[OUTCOME_ORDER], width, label="Mean predicted", color="#276fbf")
    ax.bar(x + width / 2, actual_counts[OUTCOME_ORDER], width, label="Actual share", color="#c44536")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share / probability")
    ax.set_title(f"Outcome probabilities vs actual outcomes: {split_name}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(match_predictions: pd.DataFrame, split_name: str, path: Path) -> None:
    confusion = pd.crosstab(
        match_predictions["actual_outcome"],
        match_predictions["predicted_outcome"],
    ).reindex(index=OUTCOME_ORDER, columns=OUTCOME_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion.values, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Matches")
    labels = ["Home win", "Draw", "Away win"]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("Actual outcome")
    ax.set_title(f"Outcome confusion matrix: {split_name}")
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            ax.text(column, row, str(int(confusion.iat[row, column])), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_worst_log_loss(match_predictions: pd.DataFrame, split_name: str, path: Path, limit: int = 12) -> None:
    worst = match_predictions.sort_values("outcome_log_loss", ascending=False).head(limit).copy()
    worst["label"] = worst["home_team"] + " " + worst["actual_scoreline"] + " " + worst["away_team"]
    worst = worst.sort_values("outcome_log_loss")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(worst["label"], worst["outcome_log_loss"], color="#8f2d56")
    ax.set_xlabel("Outcome log loss")
    ax.set_title(f"Worst matches by outcome log loss: {split_name}")
    for index, value in enumerate(worst["outcome_log_loss"]):
        ax.text(value, index, f" {value:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def display_summary(summary: pd.DataFrame) -> pd.DataFrame:
    percent_columns = {
        "outcome_accuracy",
        "avg_actual_outcome_probability",
        "exact_score_top1_accuracy",
        "mean_exact_score_probability",
        "mean_score_matrix_probability_mass",
    }
    display = summary.copy()
    for column in percent_columns & set(display.columns):
        display[column] = display[column].map(format_probability)
    return display


def write_markdown_summary(
    *,
    path: Path,
    summary: pd.DataFrame,
    outputs_by_split: dict[str, dict[str, pd.DataFrame]],
    artifacts: dict[str, Path],
) -> None:
    lines = [
        "# V1 Fixed-Split Validation",
        "",
        "The model is trained on matches from 2010 through 2018, evaluated on",
        "2019 through 2021 validation matches, and tested on the 2022 FIFA World Cup.",
        "Fixture prediction scripts train on the full completed dataset instead.",
        "",
        "The GLM uses regular Elo plus all leak-free, non-redundant pre-match",
        "football history signals currently available in `poisson_training_long.csv`.",
        "Training rows are weighted by the Kyrre weight:",
        "w_i = exp(-gamma * age_i).",
        "",
        "## Summary Metrics",
        "",
        markdown_table(display_summary(summary)),
        "",
        "## Goal Calibration By Predicted Lambda Bin",
        "",
        "Lambda is expected goals, not a point prediction. These tables group",
        "team-match rows by predicted lambda and compare average predicted goals",
        "against average actual goals. A calibrated model should have similar",
        "values in those two columns.",
        "",
    ]

    for split_name, outputs in outputs_by_split.items():
        goal_calibration = outputs["goal_calibration"][
            [
                "predicted_lambda_bin",
                "rows",
                "mean_predicted_goals",
                "mean_actual_goals",
                "calibration_error",
                "absolute_calibration_error",
            ]
        ]
        match_predictions = outputs["match_predictions"]
        worst = match_predictions.sort_values("outcome_log_loss", ascending=False).head(10)[
            [
                "date",
                "home_team",
                "away_team",
                "actual_scoreline",
                "predicted_outcome",
                "actual_outcome",
                "actual_outcome_probability",
                "outcome_log_loss",
            ]
        ].copy()
        lines.extend(
            [
                f"### {split_name}",
                "",
                markdown_table(goal_calibration),
                "",
                f"## Worst Outcome Log-Loss Matches: {split_name}",
                "",
                markdown_table(worst, percent_columns={"actual_outcome_probability"}),
                "",
            ]
        )

    lines.extend(
        [
            "## Artifacts",
            "",
            markdown_table(pd.DataFrame({"artifact": artifacts.keys(), "path": [str(path) for path in artifacts.values()]})),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    outputs_by_split: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "validation_summary_md": output_dir / "validation_summary.md",
        "validation_summary_csv": output_dir / "validation_summary.csv",
    }
    summary.to_csv(artifacts["validation_summary_csv"], index=False)

    for split_name, outputs in outputs_by_split.items():
        artifacts[f"{split_name}_match_predictions_csv"] = output_dir / f"{split_name}_match_predictions.csv"
        artifacts[f"{split_name}_row_predictions_csv"] = output_dir / f"{split_name}_row_predictions.csv"
        artifacts[f"{split_name}_goal_calibration_csv"] = output_dir / f"{split_name}_goal_calibration.csv"
        artifacts[f"{split_name}_predicted_vs_actual_goals_png"] = output_dir / f"{split_name}_predicted_vs_actual_goals.png"
        artifacts[f"{split_name}_goal_calibration_png"] = output_dir / f"{split_name}_goal_calibration.png"
        artifacts[f"{split_name}_outcome_probability_bars_png"] = output_dir / f"{split_name}_outcome_probability_bars.png"
        artifacts[f"{split_name}_outcome_confusion_matrix_png"] = output_dir / f"{split_name}_outcome_confusion_matrix.png"
        artifacts[f"{split_name}_worst_match_log_loss_png"] = output_dir / f"{split_name}_worst_match_log_loss.png"

        outputs["match_predictions"].to_csv(artifacts[f"{split_name}_match_predictions_csv"], index=False)
        outputs["row_predictions"].to_csv(artifacts[f"{split_name}_row_predictions_csv"], index=False)
        outputs["goal_calibration"].to_csv(artifacts[f"{split_name}_goal_calibration_csv"], index=False)
        plot_predicted_vs_actual(
            outputs["row_predictions"],
            split_name,
            artifacts[f"{split_name}_predicted_vs_actual_goals_png"],
        )
        plot_goal_calibration(
            outputs["goal_calibration"],
            split_name,
            artifacts[f"{split_name}_goal_calibration_png"],
        )
        plot_outcome_probability_bars(
            outputs["match_predictions"],
            split_name,
            artifacts[f"{split_name}_outcome_probability_bars_png"],
        )
        plot_confusion_matrix(
            outputs["match_predictions"],
            split_name,
            artifacts[f"{split_name}_outcome_confusion_matrix_png"],
        )
        plot_worst_log_loss(
            outputs["match_predictions"],
            split_name,
            artifacts[f"{split_name}_worst_match_log_loss_png"],
        )

    write_markdown_summary(
        path=artifacts["validation_summary_md"],
        summary=summary,
        outputs_by_split=outputs_by_split,
        artifacts=artifacts,
    )
    return artifacts


def print_console_report(summary: pd.DataFrame, outputs_by_split: dict[str, dict[str, pd.DataFrame]], artifacts: dict[str, Path]) -> None:
    print("\nValidation/test summary")
    print(display_summary(summary).to_string(index=False))
    for split_name, outputs in outputs_by_split.items():
        print(f"\nGoal calibration by predicted lambda bin: {split_name}")
        calibration_display = outputs["goal_calibration"][
            [
                "predicted_lambda_bin",
                "rows",
                "mean_predicted_goals",
                "mean_actual_goals",
                "calibration_error",
                "absolute_calibration_error",
            ]
        ].copy()
        print(calibration_display.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

        print(f"\nWorst outcome log-loss matches: {split_name}")
        worst = outputs["match_predictions"].sort_values("outcome_log_loss", ascending=False).head(10)[
            [
                "date",
                "home_team",
                "away_team",
                "actual_scoreline",
                "predicted_outcome",
                "actual_outcome",
                "actual_outcome_probability",
                "outcome_log_loss",
            ]
        ].copy()
        print(worst.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nReport artifacts")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


def validate_outputs(
    *,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    row_predictions: pd.DataFrame,
    match_predictions: pd.DataFrame,
    goal_calibration: pd.DataFrame,
    split_name: str,
    max_goals: int,
) -> None:
    if set(train["match_id"]).intersection(set(evaluation["match_id"])):
        raise ValueError(f"Train set overlaps with {split_name} set.")
    if len(evaluation) != evaluation["match_id"].nunique() * 2:
        raise ValueError(f"{split_name} set must contain exactly two rows per match.")
    lambdas = row_predictions["lambda"].to_numpy(dtype=float)
    if not np.isfinite(lambdas).all() or not (lambdas > 0).all():
        raise ValueError(f"{split_name} lambdas must be finite and positive.")
    for _, row in match_predictions.iterrows():
        probabilities = np.array(
            [row["home_win_probability"], row["draw_probability"], row["away_win_probability"]],
            dtype=float,
        )
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError(f"{split_name} outcome probabilities must be finite and nonnegative.")
        if not np.isclose(probabilities.sum(), row["score_matrix_probability_mass"]):
            raise ValueError(f"{split_name} outcome probabilities must sum to score-matrix mass.")
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative.")
    if goal_calibration.empty:
        raise ValueError(f"{split_name} goal calibration table cannot be empty.")
    if int(goal_calibration["rows"].sum()) != len(row_predictions):
        raise ValueError(f"{split_name} goal calibration rows must cover all row predictions.")
    calibration_values = goal_calibration[
        ["mean_predicted_goals", "mean_actual_goals", "calibration_error"]
    ].to_numpy(dtype=float)
    if not np.isfinite(calibration_values).all():
        raise ValueError(f"{split_name} goal calibration values must be finite.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the V1 Poisson GLM on fixed historical splits.")
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    parser.add_argument("--validation-start", default=DEFAULT_VALIDATION_START)
    parser.add_argument("--validation-end", default=DEFAULT_VALIDATION_END)
    parser.add_argument("--test-year", type=int, default=DEFAULT_TEST_YEAR)
    parser.add_argument("--test-tournament", default=DEFAULT_TEST_TOURNAMENT)
    parser.add_argument(
        "--calibration-bin-width",
        type=float,
        default=0.5,
        help="Width of fixed predicted-lambda bins for goal calibration tables. Defaults to 0.5.",
    )
    parser.add_argument(
        "--kyrre-weight-half-life-years",
        type=float,
        default=DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS,
        help="Half-life in years for Kyrre training weights. Defaults to 8.",
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
        "--features",
        nargs="+",
        default=DEFAULT_FEATURE_COLUMNS,
        help="Feature columns to use exactly as they appear in poisson_training_long.csv.",
    )
    return parser.parse_args()


def evaluate_split(
    model_result,
    split_name: str,
    data: pd.DataFrame,
    max_goals: int,
    calibration_bin_width: float,
) -> dict[str, pd.DataFrame]:
    row_predictions = add_row_predictions(model_result, data)
    match_predictions = build_match_predictions(row_predictions, max_goals=max_goals)
    goal_calibration = build_goal_calibration(row_predictions, bin_width=calibration_bin_width)
    return {
        "row_predictions": row_predictions,
        "match_predictions": match_predictions,
        "goal_calibration": goal_calibration,
    }


def main() -> None:
    args = parse_args()
    data = load_training_data(args.training_csv)
    train, validation, test = split_fixed_periods(
        data,
        train_start=args.train_start,
        train_end=args.train_end,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        test_year=args.test_year,
        test_tournament=args.test_tournament,
    )
    model_result = fit_poisson_glm_from_data(
        train,
        feature_columns=args.features,
        target_column="goals_for",
        id_columns=["match_id", "team", "opponent"],
        print_diagnostics=True,
        date_column="date",
        kyrre_weight_gamma=None if args.no_kyrre_weight else args.kyrre_weight_gamma,
        kyrre_weight_half_life_years=(
            None if args.no_kyrre_weight else args.kyrre_weight_half_life_years
        ),
    )

    outputs_by_split = {
        "validation_2019_2021": evaluate_split(
            model_result,
            "validation_2019_2021",
            validation,
            args.max_goals,
            args.calibration_bin_width,
        ),
        "test_2022_world_cup": evaluate_split(
            model_result,
            "test_2022_world_cup",
            test,
            args.max_goals,
            args.calibration_bin_width,
        ),
    }

    summary = pd.DataFrame(
        [
            summary_row(
                split_name="validation_2019_2021",
                train=train,
                evaluation=validation,
                row_predictions=outputs_by_split["validation_2019_2021"]["row_predictions"],
                match_predictions=outputs_by_split["validation_2019_2021"]["match_predictions"],
                max_goals=args.max_goals,
                model_result=model_result,
            ),
            summary_row(
                split_name="test_2022_world_cup",
                train=train,
                evaluation=test,
                row_predictions=outputs_by_split["test_2022_world_cup"]["row_predictions"],
                match_predictions=outputs_by_split["test_2022_world_cup"]["match_predictions"],
                max_goals=args.max_goals,
                model_result=model_result,
            ),
        ]
    )

    validate_outputs(
        train=train,
        evaluation=validation,
        row_predictions=outputs_by_split["validation_2019_2021"]["row_predictions"],
        match_predictions=outputs_by_split["validation_2019_2021"]["match_predictions"],
        goal_calibration=outputs_by_split["validation_2019_2021"]["goal_calibration"],
        split_name="validation_2019_2021",
        max_goals=args.max_goals,
    )
    validate_outputs(
        train=train,
        evaluation=test,
        row_predictions=outputs_by_split["test_2022_world_cup"]["row_predictions"],
        match_predictions=outputs_by_split["test_2022_world_cup"]["match_predictions"],
        goal_calibration=outputs_by_split["test_2022_world_cup"]["goal_calibration"],
        split_name="test_2022_world_cup",
        max_goals=args.max_goals,
    )

    artifacts = write_reports(
        output_dir=args.output_dir,
        summary=summary,
        outputs_by_split=outputs_by_split,
    )
    print_console_report(summary, outputs_by_split, artifacts)


if __name__ == "__main__":
    main()
