"""Validate the V1 Poisson GLM on fixed historical train/validation/test splits."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, poisson, ttest_rel, wilcoxon
import statsmodels.api as sm

try:
    from .model_weights import model_weights_table, plot_model_weights
    from .poisson_glm import (
        DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS,
        DEFAULT_RIDGE_ALPHA,
        fit_poisson_glm_from_data,
        outcome_probabilities,
        poisson_score_matrix,
        predict_expected_goals,
    )
    from .tuning_config import DEFAULT_TUNED_CONFIG_PATH, load_tuned_config
except ImportError:
    from model_weights import model_weights_table, plot_model_weights
    from poisson_glm import (
        DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS,
        DEFAULT_RIDGE_ALPHA,
        fit_poisson_glm_from_data,
        outcome_probabilities,
        poisson_score_matrix,
        predict_expected_goals,
    )
    from tuning_config import DEFAULT_TUNED_CONFIG_PATH, load_tuned_config


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


def safe_probability(value: float) -> float:
    return float(np.clip(value, EPSILON, 1.0))


def outcome_probability_column(outcome: str) -> str:
    return {
        "home_win": "home_win_probability",
        "draw": "draw_probability",
        "away_win": "away_win_probability",
    }[outcome]


def ranked_probability_score(probabilities: dict[str, float], actual: str) -> float:
    predicted = np.array([probabilities[outcome] for outcome in OUTCOME_ORDER], dtype=float)
    predicted = predicted / predicted.sum()
    observed = np.array([float(outcome == actual) for outcome in OUTCOME_ORDER], dtype=float)
    return float(np.mean((np.cumsum(predicted)[:-1] - np.cumsum(observed)[:-1]) ** 2))


def poisson_deviance(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> np.ndarray:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    terms = np.zeros_like(actual_values, dtype=float)
    positive = actual_values > 0
    terms[positive] = actual_values[positive] * np.log(actual_values[positive] / predicted_values[positive])
    return 2.0 * (terms - (actual_values - predicted_values))


def outcome_ece(match_predictions: pd.DataFrame, bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    reliability_rows = []
    ece_rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for outcome in OUTCOME_ORDER:
        probabilities = match_predictions[outcome_probability_column(outcome)].to_numpy(dtype=float)
        actual = match_predictions["actual_outcome"].eq(outcome).astype(float).to_numpy()
        class_ece = 0.0
        for index in range(bins):
            lower = edges[index]
            upper = edges[index + 1]
            if index == bins - 1:
                mask = (probabilities >= lower) & (probabilities <= upper)
            else:
                mask = (probabilities >= lower) & (probabilities < upper)
            count = int(mask.sum())
            if count == 0:
                continue
            mean_predicted = float(probabilities[mask].mean())
            empirical_frequency = float(actual[mask].mean())
            gap = abs(empirical_frequency - mean_predicted)
            class_ece += count / len(match_predictions) * gap
            reliability_rows.append(
                {
                    "outcome": outcome,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "matches": count,
                    "mean_predicted_probability": mean_predicted,
                    "empirical_frequency": empirical_frequency,
                    "calibration_gap": empirical_frequency - mean_predicted,
                    "absolute_calibration_gap": gap,
                }
            )
        ece_rows.append({"outcome": outcome, "ece": class_ece})
    ece = pd.DataFrame(ece_rows)
    ece = pd.concat(
        [
            ece,
            pd.DataFrame([{"outcome": "macro_average", "ece": float(ece["ece"].mean())}]),
        ],
        ignore_index=True,
    )
    return pd.DataFrame(reliability_rows), ece


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
    rows["poisson_deviance"] = poisson_deviance(rows["actual_goals"], rows["predicted_goals"])
    rows["absolute_error"] = (rows["actual_goals"] - rows["predicted_goals"]).abs()
    rows["squared_error"] = (rows["actual_goals"] - rows["predicted_goals"]) ** 2
    rows["within_one_goal"] = rows["absolute_error"].le(1.0).astype(int)
    rows["goal_interval_95_lower"] = poisson.ppf(0.025, rows["predicted_goals"]).astype(int)
    rows["goal_interval_95_upper"] = poisson.ppf(0.975, rows["predicted_goals"]).astype(int)
    rows["goal_interval_95_contains"] = (
        rows["actual_goals"].ge(rows["goal_interval_95_lower"])
        & rows["actual_goals"].le(rows["goal_interval_95_upper"])
    ).astype(int)
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
        rps = ranked_probability_score(normalized_outcomes, observed_outcome)
        scoreline_abs_error = abs(actual_home_goals - top_home_goals) + abs(actual_away_goals - top_away_goals)
        scoreline_within_one_goal = int(
            abs(actual_home_goals - top_home_goals) <= 1
            and abs(actual_away_goals - top_away_goals) <= 1
        )
        home_elo_diff = float(home_row["elo_diff"]) if "elo_diff" in home_row else np.nan
        favorite_bucket = "home_favorite" if home_elo_diff > 50 else "away_favorite" if home_elo_diff < -50 else "close"

        rows.append(
            {
                "match_id": int(match_id),
                "date": home_row["date"],
                "year": int(pd.Timestamp(home_row["date"]).year),
                "home_team": home_row["home_team"],
                "away_team": home_row["away_team"],
                "tournament": home_row.get("tournament", ""),
                "tournament_type": home_row.get("tournament_type", ""),
                "is_neutral": int(home_row.get("is_neutral", 0)),
                "is_world_cup": int(home_row.get("is_world_cup", 0)),
                "home_elo_diff": home_elo_diff,
                "favorite_bucket": favorite_bucket,
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
                "rps": rps,
                "outcome_correct": int(predicted_outcome == observed_outcome),
                "top_scoreline": f"{top_home_goals}-{top_away_goals}",
                "top_home_goals": top_home_goals,
                "top_away_goals": top_away_goals,
                "top_scoreline_probability": top_score_probability,
                "actual_scoreline": f"{actual_home_goals}-{actual_away_goals}",
                "exact_score_probability": exact_score_probability,
                "exact_score_top1_correct": int(
                    top_home_goals == actual_home_goals and top_away_goals == actual_away_goals
                ),
                "scoreline_abs_error": scoreline_abs_error,
                "scoreline_within_one_goal": scoreline_within_one_goal,
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


def build_match_context(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match_id, group in data.groupby("match_id", sort=False):
        home = group[group["is_listed_home"].eq(1)]
        away = group[group["is_listed_home"].eq(0)]
        if len(home) != 1 or len(away) != 1:
            raise ValueError(f"match_id {match_id} must have one listed-home and one listed-away row.")
        home_row = home.iloc[0]
        away_row = away.iloc[0]
        actual_home_goals = int(home_row["goals_for"])
        actual_away_goals = int(away_row["goals_for"])
        observed = actual_outcome(actual_home_goals, actual_away_goals)
        rows.append(
            {
                "match_id": int(match_id),
                "date": home_row["date"],
                "year": int(pd.Timestamp(home_row["date"]).year),
                "home_team": home_row["home_team"],
                "away_team": home_row["away_team"],
                "actual_home_goals": actual_home_goals,
                "actual_away_goals": actual_away_goals,
                "actual_outcome": observed,
                "actual_scoreline": f"{actual_home_goals}-{actual_away_goals}",
                "home_elo_diff": float(home_row["elo_diff"]) if "elo_diff" in home_row else 0.0,
                "is_neutral": int(home_row.get("is_neutral", 0)),
                "is_world_cup": int(home_row.get("is_world_cup", 0)),
                "tournament_type": home_row.get("tournament_type", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)


def finalize_baseline_predictions(
    baseline_name: str,
    context: pd.DataFrame,
    probabilities: pd.DataFrame,
) -> pd.DataFrame:
    predictions = context.copy()
    for outcome in OUTCOME_ORDER:
        predictions[outcome_probability_column(outcome)] = probabilities[outcome].to_numpy(dtype=float)
    probability_sum = predictions[[outcome_probability_column(outcome) for outcome in OUTCOME_ORDER]].sum(axis=1)
    for outcome in OUTCOME_ORDER:
        column = outcome_probability_column(outcome)
        predictions[column] = predictions[column] / probability_sum
    predictions["baseline"] = baseline_name
    predictions["predicted_outcome"] = predictions[
        [outcome_probability_column(outcome) for outcome in OUTCOME_ORDER]
    ].idxmax(axis=1).map(
        {
            "home_win_probability": "home_win",
            "draw_probability": "draw",
            "away_win_probability": "away_win",
        }
    )
    predictions["actual_outcome_probability"] = [
        row[outcome_probability_column(row["actual_outcome"])]
        for _, row in predictions.iterrows()
    ]
    predictions["outcome_log_loss"] = -np.log(predictions["actual_outcome_probability"].map(safe_probability))
    predictions["rps"] = [
        ranked_probability_score(
            {outcome: row[outcome_probability_column(outcome)] for outcome in OUTCOME_ORDER},
            row["actual_outcome"],
        )
        for _, row in predictions.iterrows()
    ]
    predictions["outcome_brier_score"] = [
        sum(
            (row[outcome_probability_column(outcome)] - float(outcome == row["actual_outcome"])) ** 2
            for outcome in OUTCOME_ORDER
        )
        for _, row in predictions.iterrows()
    ]
    predictions["outcome_correct"] = predictions["predicted_outcome"].eq(predictions["actual_outcome"]).astype(int)
    return predictions


def constant_outcome_baseline(train: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    train_context = build_match_context(train)
    evaluation_context = build_match_context(evaluation)
    counts = train_context["actual_outcome"].value_counts().reindex(OUTCOME_ORDER).fillna(0) + 1.0
    probabilities = counts / counts.sum()
    probability_frame = pd.DataFrame(
        [probabilities.to_dict()] * len(evaluation_context),
        columns=OUTCOME_ORDER,
    )
    return finalize_baseline_predictions("constant_outcome", evaluation_context, probability_frame)


def constant_goal_poisson_baseline(train: pd.DataFrame, evaluation: pd.DataFrame, max_goals: int) -> pd.DataFrame:
    train_home = train[train["is_listed_home"].eq(1)]
    train_away = train[train["is_listed_home"].eq(0)]
    lambda_home = float(train_home["goals_for"].mean())
    lambda_away = float(train_away["goals_for"].mean())
    context = build_match_context(evaluation)
    score_matrix = poisson_score_matrix(lambda_home, lambda_away, max_goals=max_goals)
    outcomes = outcome_probabilities(score_matrix)
    probability_frame = pd.DataFrame(
        [outcomes] * len(context),
        columns=OUTCOME_ORDER,
    )
    return finalize_baseline_predictions("constant_goal_poisson", context, probability_frame)


def elo_logistic_baseline(train: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    train_context = build_match_context(train)
    evaluation_context = build_match_context(evaluation)
    outcome_codes = {outcome: index for index, outcome in enumerate(OUTCOME_ORDER)}
    y = train_context["actual_outcome"].map(outcome_codes).astype(int)
    x_train = sm.add_constant(pd.DataFrame({"elo_diff_scaled": train_context["home_elo_diff"] / 400.0}), has_constant="add")
    x_eval = sm.add_constant(pd.DataFrame({"elo_diff_scaled": evaluation_context["home_elo_diff"] / 400.0}), has_constant="add")
    try:
        model = sm.MNLogit(y, x_train)
        result = model.fit_regularized(alpha=0.01, L1_wt=0.0, disp=False, maxiter=200)
        predicted = result.predict(x_eval)
        probability_frame = pd.DataFrame(np.asarray(predicted, dtype=float), columns=OUTCOME_ORDER)
        if not np.isfinite(probability_frame.to_numpy(dtype=float)).all():
            raise ValueError("non-finite Elo logistic probabilities")
    except Exception:
        return constant_outcome_baseline(train, evaluation).assign(baseline="elo_logistic_failed_constant")
    return finalize_baseline_predictions("elo_logistic", evaluation_context, probability_frame)


def build_baseline_predictions(train: pd.DataFrame, evaluation: pd.DataFrame, max_goals: int) -> pd.DataFrame:
    baselines = [
        constant_outcome_baseline(train, evaluation),
        constant_goal_poisson_baseline(train, evaluation, max_goals=max_goals),
        elo_logistic_baseline(train, evaluation),
    ]
    return pd.concat(baselines, ignore_index=True)


def aggregate_outcome_metrics(data: pd.DataFrame, label: str, split_name: str) -> dict[str, object]:
    return {
        "split": split_name,
        "model_or_baseline": label,
        "matches": int(len(data)),
        "outcome_accuracy": float(data["outcome_correct"].mean()),
        "outcome_log_loss": float(data["outcome_log_loss"].mean()),
        "outcome_brier_score": float(data["outcome_brier_score"].mean()),
        "rps": float(data["rps"].mean()),
        "avg_actual_outcome_probability": float(data["actual_outcome_probability"].mean()),
    }


def baseline_comparison(split_name: str, match_predictions: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows = [aggregate_outcome_metrics(match_predictions, "poisson_glm", split_name)]
    for baseline_name, group in baselines.groupby("baseline", sort=False):
        rows.append(aggregate_outcome_metrics(group, baseline_name, split_name))
    return pd.DataFrame(rows)


def paired_comparison_tests(split_name: str, match_predictions: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model = match_predictions.set_index("match_id")
    for baseline_name, baseline_group in baselines.groupby("baseline", sort=False):
        baseline = baseline_group.set_index("match_id").reindex(model.index)
        for metric in ["outcome_log_loss", "rps"]:
            model_values = model[metric].to_numpy(dtype=float)
            baseline_values = baseline[metric].to_numpy(dtype=float)
            diff = model_values - baseline_values
            t_p = float(ttest_rel(model_values, baseline_values, nan_policy="omit").pvalue)
            try:
                w_p = float(wilcoxon(diff).pvalue) if not np.allclose(diff, 0) else 1.0
            except ValueError:
                w_p = np.nan
            rows.append(
                {
                    "split": split_name,
                    "baseline": baseline_name,
                    "metric": metric,
                    "mean_model_minus_baseline": float(np.nanmean(diff)),
                    "paired_t_p_value": t_p,
                    "wilcoxon_p_value": w_p,
                }
            )
        model_correct = model["outcome_correct"].astype(bool)
        baseline_correct = baseline["outcome_correct"].astype(bool)
        model_only = int((model_correct & ~baseline_correct).sum())
        baseline_only = int((~model_correct & baseline_correct).sum())
        denominator = model_only + baseline_only
        statistic = 0.0 if denominator == 0 else (abs(model_only - baseline_only) - 1.0) ** 2 / denominator
        rows.append(
            {
                "split": split_name,
                "baseline": baseline_name,
                "metric": "outcome_accuracy_mcnemar",
                "mean_model_minus_baseline": float(model_correct.mean() - baseline_correct.mean()),
                "paired_t_p_value": float(chi2.sf(statistic, df=1)) if denominator else 1.0,
                "wilcoxon_p_value": np.nan,
            }
        )
    return pd.DataFrame(rows)


def per_outcome_metrics(split_name: str, match_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOME_ORDER:
        predicted = match_predictions["predicted_outcome"].eq(outcome)
        actual = match_predictions["actual_outcome"].eq(outcome)
        tp = int((predicted & actual).sum())
        fp = int((predicted & ~actual).sum())
        fn = int((~predicted & actual).sum())
        support = int(actual.sum())
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall else np.nan
        rows.append(
            {
                "split": split_name,
                "outcome": outcome,
                "support": support,
                "predicted_count": int(predicted.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_predicted_probability": float(match_predictions[outcome_probability_column(outcome)].mean()),
                "actual_frequency": float(actual.mean()),
            }
        )
    return pd.DataFrame(rows)


def grouped_metrics(split_name: str, match_predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for value, group in match_predictions.groupby(group_column, dropna=False):
        row = aggregate_outcome_metrics(group, "poisson_glm", split_name)
        row["group"] = group_column
        row["group_value"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def scoreline_confusion(split_name: str, match_predictions: pd.DataFrame, limit: int = 30) -> pd.DataFrame:
    confusion = (
        match_predictions.groupby(["top_scoreline", "actual_scoreline"], dropna=False)
        .size()
        .reset_index(name="matches")
        .sort_values("matches", ascending=False)
        .head(limit)
    )
    confusion.insert(0, "split", split_name)
    return confusion


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
    regularization = getattr(model_result, "regularization_summary", {}) or {}
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
        "fit_method": regularization.get("fit_method", ""),
        "ridge_alpha": regularization.get("ridge_alpha", ""),
        "tuned_config_source": getattr(model_result, "tuned_config_source", ""),
        "mean_poisson_nll": float(row_predictions["poisson_nll"].mean()),
        "mean_poisson_deviance": float(row_predictions["poisson_deviance"].mean()),
        "goal_mae": float(row_predictions["absolute_error"].mean()),
        "goal_rmse": float(np.sqrt(row_predictions["squared_error"].mean())),
        "within_one_goal_rate": float(row_predictions["within_one_goal"].mean()),
        "goal_interval_95_coverage": float(row_predictions["goal_interval_95_contains"].mean()),
        "mean_predicted_goals": float(row_predictions["predicted_goals"].mean()),
        "mean_actual_goals": float(row_predictions["actual_goals"].mean()),
        "outcome_accuracy": float(match_predictions["outcome_correct"].mean()),
        "outcome_brier_score": float(match_predictions["outcome_brier_score"].mean()),
        "outcome_log_loss": float(match_predictions["outcome_log_loss"].mean()),
        "rps": float(match_predictions["rps"].mean()),
        "avg_actual_outcome_probability": float(match_predictions["actual_outcome_probability"].mean()),
        "exact_score_top1_accuracy": float(match_predictions["exact_score_top1_correct"].mean()),
        "mean_exact_score_probability": float(match_predictions["exact_score_probability"].mean()),
        "scoreline_within_one_goal_rate": float(match_predictions["scoreline_within_one_goal"].mean()),
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


def plot_reliability(reliability: pd.DataFrame, split_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for outcome, group in reliability.groupby("outcome", sort=False):
        ax.plot(
            group["mean_predicted_probability"],
            group["empirical_frequency"],
            marker="o",
            label=outcome,
        )
    ax.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(f"Outcome reliability: {split_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_yearly_metrics(yearly_metrics: pd.DataFrame, split_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(yearly_metrics["group_value"], yearly_metrics["outcome_log_loss"], marker="o", label="Log-loss")
    ax.plot(yearly_metrics["group_value"], yearly_metrics["rps"], marker="o", label="RPS")
    ax.set_xlabel("Year")
    ax.set_ylabel("Metric")
    ax.set_title(f"Yearly probabilistic metrics: {split_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_subgroup_log_loss(subgroup_metrics: pd.DataFrame, split_name: str, path: Path) -> None:
    plot_data = subgroup_metrics.copy()
    plot_data["label"] = plot_data["group"] + "=" + plot_data["group_value"].astype(str)
    plot_data = plot_data.sort_values("outcome_log_loss")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot_data) + 1.5)))
    ax.barh(plot_data["label"], plot_data["outcome_log_loss"], color="#276fbf")
    ax.set_xlabel("Outcome log-loss")
    ax.set_title(f"Subgroup log-loss: {split_name}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def display_summary(summary: pd.DataFrame) -> pd.DataFrame:
    percent_columns = {
        "outcome_accuracy",
        "within_one_goal_rate",
        "goal_interval_95_coverage",
        "avg_actual_outcome_probability",
        "exact_score_top1_accuracy",
        "mean_exact_score_probability",
        "scoreline_within_one_goal_rate",
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
    model_weights: pd.DataFrame,
    outputs_by_split: dict[str, dict[str, pd.DataFrame]],
    artifacts: dict[str, Path],
) -> None:
    top_weights = model_weights[model_weights["term"].ne("const")].head(20)[
        [
            "term",
            "coefficient",
            "std_error",
            "p_value",
            "standardized_coefficient",
            "rate_ratio_per_1sd",
        ]
    ]
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
        "## Model Weights",
        "",
        "The chart `model_weights.png` shows standardized GLM coefficients:",
        "`beta * training feature standard deviation`. This makes features with",
        "different units easier to compare. Positive values increase expected",
        "goals on the log scale; negative values decrease expected goals.",
        "",
        markdown_table(top_weights),
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
        baseline_summary = outputs["baseline_summary"]
        paired_tests = outputs["paired_tests"]
        ece = outputs["ece"]
        per_outcome = outputs["per_outcome_metrics"]
        subgroup = outputs["subgroup_metrics"].sort_values("outcome_log_loss", ascending=False).head(20)
        scoreline = outputs["scoreline_confusion"].head(20)
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
                f"## Baseline Comparison: {split_name}",
                "",
                markdown_table(
                    baseline_summary,
                    percent_columns={"outcome_accuracy", "avg_actual_outcome_probability"},
                ),
                "",
                f"## Paired Baseline Tests: {split_name}",
                "",
                markdown_table(paired_tests),
                "",
                f"## Outcome Calibration And Per-Class Metrics: {split_name}",
                "",
                markdown_table(ece),
                "",
                markdown_table(per_outcome, percent_columns={"precision", "recall", "f1", "actual_frequency"}),
                "",
                f"## Highest-Loss Subgroups: {split_name}",
                "",
                markdown_table(subgroup, percent_columns={"outcome_accuracy", "avg_actual_outcome_probability"}),
                "",
                f"## Frequent Scoreline Confusions: {split_name}",
                "",
                markdown_table(scoreline),
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
    model_result,
    outputs_by_split: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "validation_summary_md": output_dir / "validation_summary.md",
        "validation_summary_csv": output_dir / "validation_summary.csv",
        "model_weights_csv": output_dir / "model_weights.csv",
        "model_weights_png": output_dir / "model_weights.png",
    }
    summary.to_csv(artifacts["validation_summary_csv"], index=False)
    weights = model_weights_table(model_result)
    weights.to_csv(artifacts["model_weights_csv"], index=False)
    plot_model_weights(weights, artifacts["model_weights_png"])

    for split_name, outputs in outputs_by_split.items():
        artifacts[f"{split_name}_match_predictions_csv"] = output_dir / f"{split_name}_match_predictions.csv"
        artifacts[f"{split_name}_row_predictions_csv"] = output_dir / f"{split_name}_row_predictions.csv"
        artifacts[f"{split_name}_goal_calibration_csv"] = output_dir / f"{split_name}_goal_calibration.csv"
        artifacts[f"{split_name}_baseline_predictions_csv"] = output_dir / f"{split_name}_baseline_predictions.csv"
        artifacts[f"{split_name}_baseline_summary_csv"] = output_dir / f"{split_name}_baseline_summary.csv"
        artifacts[f"{split_name}_paired_tests_csv"] = output_dir / f"{split_name}_paired_tests.csv"
        artifacts[f"{split_name}_per_outcome_metrics_csv"] = output_dir / f"{split_name}_per_outcome_metrics.csv"
        artifacts[f"{split_name}_reliability_csv"] = output_dir / f"{split_name}_reliability.csv"
        artifacts[f"{split_name}_ece_csv"] = output_dir / f"{split_name}_ece.csv"
        artifacts[f"{split_name}_yearly_metrics_csv"] = output_dir / f"{split_name}_yearly_metrics.csv"
        artifacts[f"{split_name}_subgroup_metrics_csv"] = output_dir / f"{split_name}_subgroup_metrics.csv"
        artifacts[f"{split_name}_scoreline_confusion_csv"] = output_dir / f"{split_name}_scoreline_confusion.csv"
        artifacts[f"{split_name}_predicted_vs_actual_goals_png"] = output_dir / f"{split_name}_predicted_vs_actual_goals.png"
        artifacts[f"{split_name}_goal_calibration_png"] = output_dir / f"{split_name}_goal_calibration.png"
        artifacts[f"{split_name}_outcome_probability_bars_png"] = output_dir / f"{split_name}_outcome_probability_bars.png"
        artifacts[f"{split_name}_outcome_confusion_matrix_png"] = output_dir / f"{split_name}_outcome_confusion_matrix.png"
        artifacts[f"{split_name}_worst_match_log_loss_png"] = output_dir / f"{split_name}_worst_match_log_loss.png"
        artifacts[f"{split_name}_reliability_png"] = output_dir / f"{split_name}_reliability.png"
        artifacts[f"{split_name}_yearly_metrics_png"] = output_dir / f"{split_name}_yearly_metrics.png"
        artifacts[f"{split_name}_subgroup_log_loss_png"] = output_dir / f"{split_name}_subgroup_log_loss.png"

        outputs["match_predictions"].to_csv(artifacts[f"{split_name}_match_predictions_csv"], index=False)
        outputs["row_predictions"].to_csv(artifacts[f"{split_name}_row_predictions_csv"], index=False)
        outputs["goal_calibration"].to_csv(artifacts[f"{split_name}_goal_calibration_csv"], index=False)
        outputs["baseline_predictions"].to_csv(artifacts[f"{split_name}_baseline_predictions_csv"], index=False)
        outputs["baseline_summary"].to_csv(artifacts[f"{split_name}_baseline_summary_csv"], index=False)
        outputs["paired_tests"].to_csv(artifacts[f"{split_name}_paired_tests_csv"], index=False)
        outputs["per_outcome_metrics"].to_csv(artifacts[f"{split_name}_per_outcome_metrics_csv"], index=False)
        outputs["reliability"].to_csv(artifacts[f"{split_name}_reliability_csv"], index=False)
        outputs["ece"].to_csv(artifacts[f"{split_name}_ece_csv"], index=False)
        outputs["yearly_metrics"].to_csv(artifacts[f"{split_name}_yearly_metrics_csv"], index=False)
        outputs["subgroup_metrics"].to_csv(artifacts[f"{split_name}_subgroup_metrics_csv"], index=False)
        outputs["scoreline_confusion"].to_csv(artifacts[f"{split_name}_scoreline_confusion_csv"], index=False)
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
        plot_reliability(
            outputs["reliability"],
            split_name,
            artifacts[f"{split_name}_reliability_png"],
        )
        plot_yearly_metrics(
            outputs["yearly_metrics"],
            split_name,
            artifacts[f"{split_name}_yearly_metrics_png"],
        )
        plot_subgroup_log_loss(
            outputs["subgroup_metrics"],
            split_name,
            artifacts[f"{split_name}_subgroup_log_loss_png"],
        )

    write_markdown_summary(
        path=artifacts["validation_summary_md"],
        summary=summary,
        model_weights=weights,
        outputs_by_split=outputs_by_split,
        artifacts=artifacts,
    )
    return artifacts


def print_console_report(summary: pd.DataFrame, outputs_by_split: dict[str, dict[str, pd.DataFrame]], artifacts: dict[str, Path]) -> None:
    print("\nValidation/test summary")
    print(display_summary(summary).to_string(index=False))
    weights = pd.read_csv(artifacts["model_weights_csv"])
    top_weights = weights[weights["term"].ne("const")].head(12)[
        ["term", "coefficient", "standardized_coefficient", "rate_ratio_per_1sd", "p_value"]
    ]
    print("\nLargest standardized model weights")
    print(top_weights.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    for split_name, outputs in outputs_by_split.items():
        print(f"\nBaseline comparison: {split_name}")
        print(display_summary(outputs["baseline_summary"]).to_string(index=False))

        print(f"\nOutcome ECE: {split_name}")
        print(outputs["ece"].to_string(index=False, float_format=lambda value: f"{value:.6f}"))

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


def evaluate_split(
    model_result,
    split_name: str,
    train: pd.DataFrame,
    data: pd.DataFrame,
    max_goals: int,
    calibration_bin_width: float,
) -> dict[str, pd.DataFrame]:
    row_predictions = add_row_predictions(model_result, data)
    match_predictions = build_match_predictions(row_predictions, max_goals=max_goals)
    goal_calibration = build_goal_calibration(row_predictions, bin_width=calibration_bin_width)
    baselines = build_baseline_predictions(train, data, max_goals=max_goals)
    baseline_summary = baseline_comparison(split_name, match_predictions, baselines)
    paired_tests = paired_comparison_tests(split_name, match_predictions, baselines)
    reliability, ece = outcome_ece(match_predictions)
    reliability.insert(0, "split", split_name)
    ece.insert(0, "split", split_name)
    per_outcome = per_outcome_metrics(split_name, match_predictions)
    yearly = grouped_metrics(split_name, match_predictions, "year")
    subgroups = pd.concat(
        [
            grouped_metrics(split_name, match_predictions, "is_neutral"),
            grouped_metrics(split_name, match_predictions, "is_world_cup"),
            grouped_metrics(split_name, match_predictions, "favorite_bucket"),
            grouped_metrics(split_name, match_predictions, "tournament_type"),
        ],
        ignore_index=True,
    )
    scoreline = scoreline_confusion(split_name, match_predictions)
    return {
        "row_predictions": row_predictions,
        "match_predictions": match_predictions,
        "goal_calibration": goal_calibration,
        "baseline_predictions": baselines,
        "baseline_summary": baseline_summary,
        "paired_tests": paired_tests,
        "reliability": reliability,
        "ece": ece,
        "per_outcome_metrics": per_outcome,
        "yearly_metrics": yearly,
        "subgroup_metrics": subgroups,
        "scoreline_confusion": scoreline,
    }


def main() -> None:
    args = parse_args()
    data = load_training_data(args.training_csv)
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
        kyrre_weight_half_life_years=kyrre_half_life,
        ridge_alpha=ridge_alpha,
    )
    model_result.tuned_config_source = str(tuned_config.get("source", ""))

    outputs_by_split = {
        "validation_2019_2021": evaluate_split(
            model_result,
            "validation_2019_2021",
            train,
            validation,
            args.max_goals,
            args.calibration_bin_width,
        ),
        "test_2022_world_cup": evaluate_split(
            model_result,
            "test_2022_world_cup",
            train,
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
        model_result=model_result,
        outputs_by_split=outputs_by_split,
    )
    print_console_report(summary, outputs_by_split, artifacts)


if __name__ == "__main__":
    main()
