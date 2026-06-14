"""Predict a future fixture and write useful V1 report artifacts."""
from __future__ import annotations

import argparse
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .model_weights import model_weights_table, plot_model_weights
    from .poisson_glm import (
        DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS,
        DEFAULT_RIDGE_ALPHA,
        fit_poisson_glm_from_data,
        model_diagnostics_frame,
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
        model_diagnostics_frame,
        outcome_probabilities,
        poisson_score_matrix,
        predict_expected_goals,
    )
    from tuning_config import DEFAULT_TUNED_CONFIG_PATH, load_tuned_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PIPELINE_DIR = PROJECT_ROOT / "data_pipeline"
if str(DATA_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PIPELINE_DIR))

from elo import compute_elo


DEFAULT_TRAINING_CSV = PROJECT_ROOT / "data_pipeline" / "processed" / "poisson_training_long.csv"
DEFAULT_FIXTURES_CSV = PROJECT_ROOT / "data_pipeline" / "processed" / "future_fixtures.csv"
DEFAULT_RAW_RESULTS_CSV = PROJECT_ROOT / "data_pipeline" / "raw" / "martj42" / "results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "v1" / "reports"
HOME_ADVANTAGE = 100.0
BASE_ELO = 1500.0
H2H_WINDOW_YEARS = 60

# These are the leak-free, non-redundant pre-match numeric signals in
# poisson_training_long.csv. Exact linear duplicates are intentionally excluded
# because they make GLM coefficient estimates unstable.
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


def elo_metadata(feature_columns: list[str]) -> dict[str, str]:
    """Describe the Elo feature family used by a supplied feature list."""
    columns = set(feature_columns)
    uses_regular = bool(columns & {"team_elo_pre", "opp_elo_pre", "elo_diff"})
    if uses_regular:
        return {
            "elo_system": "Elo",
            "elo_decay": "none",
        }
    return {
        "elo_system": "none",
        "elo_decay": "not used",
    }


def weighting_metadata(model_result) -> dict[str, object]:
    summary = getattr(model_result, "sample_weight_summary", None)
    if not summary:
        return {
            "sample_weighting": "none",
            "kyrre_weight_gamma": "",
            "kyrre_weight_half_life_years": "",
            "kyrre_weight_reference_date": "",
            "weight_sum": "",
        }
    return {
        "sample_weighting": summary.get("weighting", ""),
        "kyrre_weight_gamma": summary.get("kyrre_weight_gamma", ""),
        "kyrre_weight_half_life_years": summary.get("kyrre_weight_half_life_years", ""),
        "kyrre_weight_reference_date": summary.get("kyrre_weight_reference_date", ""),
        "weight_sum": summary.get("weight_sum", ""),
    }


def regularization_metadata(model_result) -> dict[str, object]:
    summary = getattr(model_result, "regularization_summary", None) or {}
    return {
        "fit_method": summary.get("fit_method", ""),
        "ridge_alpha": summary.get("ridge_alpha", ""),
        "intercept_penalized": summary.get("intercept_penalized", ""),
    }


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify(value: str) -> str:
    text = normalize_name(value)
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def bool_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def result_and_points(goals_for: float, goals_against: float) -> tuple[str, int]:
    if goals_for > goals_against:
        return "win", 3
    if goals_for < goals_against:
        return "loss", 0
    return "draw", 1


def mean_or_nan(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else math.nan


def prepare_training_history(training: pd.DataFrame, fixture_date: pd.Timestamp) -> pd.DataFrame:
    history = training.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history[history["date"].notna() & history["date"].lt(fixture_date)].copy()
    history["goals_for"] = pd.to_numeric(history["goals_for"], errors="coerce")
    history["goals_against"] = pd.to_numeric(history["goals_against"], errors="coerce")
    history = history.dropna(subset=["goals_for", "goals_against"])
    history["result"] = np.select(
        [
            history["goals_for"].gt(history["goals_against"]),
            history["goals_for"].lt(history["goals_against"]),
        ],
        ["win", "loss"],
        default="draw",
    )
    history["points"] = np.select(
        [
            history["result"].eq("win"),
            history["result"].eq("draw"),
        ],
        [3, 1],
        default=0,
    )
    sort_columns = [column for column in ["date", "match_id", "is_listed_home"] if column in history.columns]
    return history.sort_values(sort_columns).reset_index(drop=True)


def recent_summary(history: pd.DataFrame, window: int) -> dict[str, float]:
    recent = history.tail(window)
    results = recent["result"].tolist()
    return {
        f"matches_available_last_{window}": int(len(recent)),
        f"wins_last_{window}": int(results.count("win")),
        f"draws_last_{window}": int(results.count("draw")),
        f"losses_last_{window}": int(results.count("loss")),
        f"points_per_game_last_{window}": mean_or_nan(recent["points"].astype(float)),
        f"goals_for_avg_last_{window}": mean_or_nan(recent["goals_for"].astype(float)),
        f"goals_against_avg_last_{window}": mean_or_nan(recent["goals_against"].astype(float)),
        f"goal_diff_avg_last_{window}": mean_or_nan(
            recent["goals_for"].astype(float) - recent["goals_against"].astype(float)
        ),
    }


def count_recent(history: pd.DataFrame, fixture_date: pd.Timestamp, days: int) -> int:
    cutoff = fixture_date - pd.Timedelta(days=days)
    return int(history["date"].ge(cutoff).sum())


def streak_length(history: pd.DataFrame, streak_type: str) -> int:
    length = 0
    for result in reversed(history["result"].tolist()):
        if streak_type == "unbeaten" and result != "loss":
            length += 1
        elif streak_type == "winless" and result != "win":
            length += 1
        else:
            break
    return length


def h2h_summary(history: pd.DataFrame, opponent: str, fixture_date: pd.Timestamp) -> dict[str, float]:
    cutoff = fixture_date - pd.DateOffset(years=H2H_WINDOW_YEARS)
    meetings = history[history["opponent"].eq(opponent) & history["date"].ge(cutoff)].copy()
    results = meetings["result"].tolist()
    goal_diff = meetings["goals_for"].astype(float) - meetings["goals_against"].astype(float)
    return {
        "h2h_matches_60y": int(len(meetings)),
        "h2h_wins_60y": int(results.count("win")),
        "h2h_draws_60y": int(results.count("draw")),
        "h2h_losses_60y": int(results.count("loss")),
        "h2h_goal_diff_avg_60y": mean_or_nan(goal_diff),
        "h2h_points_per_game_60y": mean_or_nan(meetings["points"].astype(float)),
    }


def find_fixture(fixtures: pd.DataFrame, requested_home: str, requested_away: str) -> tuple[pd.Series, bool]:
    fixtures = fixtures.copy()
    fixtures["home_norm"] = fixtures["home_team"].map(normalize_name)
    fixtures["away_norm"] = fixtures["away_team"].map(normalize_name)
    home_norm = normalize_name(requested_home)
    away_norm = normalize_name(requested_away)

    same_order = fixtures["home_norm"].eq(home_norm) & fixtures["away_norm"].eq(away_norm)
    reverse_order = fixtures["home_norm"].eq(away_norm) & fixtures["away_norm"].eq(home_norm)
    candidates = fixtures[same_order | reverse_order].copy()
    if candidates.empty:
        teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))
        close = [team for team in teams if home_norm in normalize_name(team) or away_norm in normalize_name(team)]
        hint = f" Close teams: {', '.join(close[:10])}." if close else ""
        raise ValueError(f"No unresolved fixture found for {requested_home} vs {requested_away}.{hint}")

    candidates["date"] = pd.to_datetime(candidates["date"], errors="coerce")
    candidates = candidates.sort_values(["date", "fixture_id"])
    fixture = candidates.iloc[0]
    return fixture, bool(same_order.loc[fixture.name])


def load_current_elo(raw_results_csv: Path, fixture_date: pd.Timestamp) -> dict[str, float]:
    results = pd.read_csv(raw_results_csv)
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date", "home_score", "away_score"])
    results = results[results["date"].lt(fixture_date)].copy()
    if results.empty:
        return {}
    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)
    results["neutral"] = results["neutral"].map(bool_value)
    results = results.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    _, ratings = compute_elo(results, base_rating=BASE_ELO, home_advantage=HOME_ADVANTAGE)
    return ratings


def build_perspective_row(
    *,
    fixture: pd.Series,
    listed_home_side: bool,
    history: pd.DataFrame,
    ratings: dict[str, float],
) -> dict[str, object]:
    fixture_date = pd.Timestamp(fixture["date"])
    neutral = bool_value(fixture["neutral"])
    home_team = str(fixture["home_team"])
    away_team = str(fixture["away_team"])
    home_elo = round(float(ratings.get(home_team, BASE_ELO)), 2)
    away_elo = round(float(ratings.get(away_team, BASE_ELO)), 2)
    home_adv = 0.0 if neutral else HOME_ADVANTAGE
    listed_home_elo_diff = round((home_elo + home_adv) - away_elo, 2)

    if listed_home_side:
        team = home_team
        opponent = away_team
        team_is_host = bool_value(fixture["home_is_host"])
        opponent_is_host = bool_value(fixture["away_is_host"])
        team_elo_pre = home_elo
        opp_elo_pre = away_elo
        elo_diff = listed_home_elo_diff
    else:
        team = away_team
        opponent = home_team
        team_is_host = bool_value(fixture["away_is_host"])
        opponent_is_host = bool_value(fixture["home_is_host"])
        team_elo_pre = away_elo
        opp_elo_pre = home_elo
        elo_diff = -listed_home_elo_diff

    team_history = history[history["team"].eq(team)].copy()
    last_match = team_history.tail(1)

    row: dict[str, object] = {
        "fixture_id": int(fixture["fixture_id"]),
        "match_id": int(fixture["fixture_id"]),
        "date": fixture_date,
        "year": int(fixture_date.year),
        "tournament": fixture["tournament"],
        "tournament_type": fixture["tournament_type"],
        "city": fixture["city"],
        "country": fixture["country"],
        "team": team,
        "opponent": opponent,
        "team_elo_pre": team_elo_pre,
        "opp_elo_pre": opp_elo_pre,
        "elo_diff": elo_diff,
        "home_team": home_team,
        "away_team": away_team,
        "is_listed_home": int(listed_home_side),
        "is_home": int(listed_home_side and not neutral),
        "is_neutral": int(neutral),
        "team_is_host": int(team_is_host),
        "opponent_is_host": int(opponent_is_host),
        "went_to_shootout": math.nan,
        "has_prior_match": int(not last_match.empty),
        "missing_days_since_last_match": int(last_match.empty),
        "days_since_last_match": (
            int((fixture_date - last_match["date"].iloc[0]).days) if not last_match.empty else math.nan
        ),
        "prev_match_went_to_shootout": (
            int(last_match["went_to_shootout"].iloc[0]) if not last_match.empty else math.nan
        ),
        "matches_last_30d": count_recent(team_history, fixture_date, 30),
        "matches_last_60d": count_recent(team_history, fixture_date, 60),
        "matches_last_365d": count_recent(team_history, fixture_date, 365),
        "unbeaten_streak_pre": streak_length(team_history, "unbeaten"),
        "winless_streak_pre": streak_length(team_history, "winless"),
        "is_friendly": int(bool_value(fixture["is_friendly"])),
        "is_qualifier": int(bool_value(fixture["is_qualifier"])),
        "is_world_cup": int(bool_value(fixture["is_world_cup"])),
        "is_continental": int(bool_value(fixture["is_continental"])),
        "is_competitive": int(bool_value(fixture["is_competitive"])),
    }
    row.update(recent_summary(team_history, 5))
    row.update(recent_summary(team_history, 10))
    h2h = h2h_summary(team_history, opponent, fixture_date)
    row.update(h2h)
    row["missing_h2h_60y"] = int(h2h["h2h_matches_60y"] == 0)
    return row


def apply_future_imputation(future_rows: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    rows = future_rows.copy()
    if "days_since_last_match" in rows:
        rows["days_since_last_match"] = rows["days_since_last_match"].fillna(
            training["days_since_last_match"].median()
        )
    if "prev_match_went_to_shootout" in rows:
        rows["prev_match_went_to_shootout"] = rows["prev_match_went_to_shootout"].fillna(0)

    rolling_average_columns = [
        column
        for column in rows.columns
        if (
            column.startswith("points_per_game_last_")
            or column.startswith("goals_for_avg_last_")
            or column.startswith("goals_against_avg_last_")
            or column.startswith("goal_diff_avg_last_")
        )
    ]
    for column in rolling_average_columns:
        if column in training:
            rows[column] = rows[column].fillna(training[column].mean())

    if "h2h_goal_diff_avg_60y" in rows:
        rows["h2h_goal_diff_avg_60y"] = rows["h2h_goal_diff_avg_60y"].fillna(0)
    if "h2h_points_per_game_60y" in rows:
        rows["h2h_points_per_game_60y"] = rows["h2h_points_per_game_60y"].fillna(1.0)
    return rows


def build_future_feature_rows(
    *,
    fixture: pd.Series,
    training: pd.DataFrame,
    raw_results_csv: Path,
) -> pd.DataFrame:
    fixture_date = pd.Timestamp(fixture["date"])
    history = prepare_training_history(training, fixture_date)
    ratings = load_current_elo(raw_results_csv, fixture_date)
    rows = pd.DataFrame(
        [
            build_perspective_row(
                fixture=fixture,
                listed_home_side=True,
                history=history,
                ratings=ratings,
            ),
            build_perspective_row(
                fixture=fixture,
                listed_home_side=False,
                history=history,
                ratings=ratings,
            ),
        ]
    )
    return apply_future_imputation(rows, training)


def top_scorelines(
    score_matrix: np.ndarray,
    home_team: str,
    away_team: str,
    limit: int = 10,
) -> pd.DataFrame:
    rows = []
    for home_goals in range(score_matrix.shape[0]):
        for away_goals in range(score_matrix.shape[1]):
            if home_goals > away_goals:
                outcome = f"{home_team} win"
            elif home_goals < away_goals:
                outcome = f"{away_team} win"
            else:
                outcome = "Draw"
            rows.append(
                {
                    "scoreline": f"{home_team} {home_goals}-{away_goals} {away_team}",
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "outcome": outcome,
                    "probability": float(score_matrix[home_goals, away_goals]),
                }
            )
    return pd.DataFrame(rows).sort_values("probability", ascending=False).head(limit).reset_index(drop=True)


def model_diagnostics_table(model_result) -> pd.DataFrame:
    return model_diagnostics_frame(model_result)


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


def save_score_matrix(score_matrix: np.ndarray, home_team: str, away_team: str, path: Path) -> None:
    matrix = pd.DataFrame(
        score_matrix,
        index=[f"{home_team}_{goals}" for goals in range(score_matrix.shape[0])],
        columns=[f"{away_team}_{goals}" for goals in range(score_matrix.shape[1])],
    )
    matrix.to_csv(path)


def plot_score_heatmap(score_matrix: np.ndarray, home_team: str, away_team: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(score_matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Probability")
    ax.set_title(f"Scoreline probabilities: {home_team} vs {away_team}")
    ax.set_xlabel(f"{away_team} goals")
    ax.set_ylabel(f"{home_team} goals")
    goals = np.arange(score_matrix.shape[0])
    ax.set_xticks(goals)
    ax.set_yticks(goals)
    for i in goals:
        for j in goals:
            value = score_matrix[i, j]
            if value >= score_matrix.max() * 0.35:
                ax.text(j, i, f"{value:.1%}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_outcomes(outcomes: dict[str, float], home_team: str, away_team: str, path: Path) -> None:
    labels = [f"{home_team} win", "Draw", f"{away_team} win"]
    values = [outcomes["home_win"], outcomes["draw"], outcomes["away_win"]]
    colors = ["#276fbf", "#8a8f98", "#c44536"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("Probability")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title("Outcome probabilities")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_top_scorelines(top_scores: pd.DataFrame, path: Path) -> None:
    plot_data = top_scores.sort_values("probability", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_data["scoreline"], plot_data["probability"], color="#2f7d62")
    ax.set_xlabel("Probability")
    ax.set_title("Top scorelines")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for index, value in enumerate(plot_data["probability"]):
        ax.text(value, index, f" {value:.1%}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_expected_goals(expected_goals: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(expected_goals["team"], expected_goals["expected_goals"], color=["#276fbf", "#c44536"])
    ax.set_ylabel("Expected goals")
    ax.set_title("Expected goals comparison")
    ax.set_ylim(0, max(expected_goals["expected_goals"]) * 1.35)
    for bar, value in zip(bars, expected_goals["expected_goals"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def missing_data_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_from_source = pd.DataFrame(
        [
            {
                "item": "kickoff_time",
                "status": "missing from future_fixtures.csv",
                "impact": "Cannot align weather or rest effects to kickoff hour.",
            },
            {
                "item": "stadium",
                "status": "missing from future_fixtures.csv",
                "impact": "Venue-specific effects are not modeled.",
            },
            {
                "item": "stadium_coordinates",
                "status": "missing from future_fixtures.csv",
                "impact": "Weather and altitude cannot be joined precisely.",
            },
            {
                "item": "lineups",
                "status": "not available in current sources",
                "impact": "Player selection strength is not modeled.",
            },
            {
                "item": "injuries_suspensions",
                "status": "not available in current sources",
                "impact": "Player availability is not modeled.",
            },
            {
                "item": "squad_selection",
                "status": "not available in current sources",
                "impact": "Final tournament squad composition is not modeled.",
            },
        ]
    )
    available_not_modeled = pd.DataFrame(
        [
            {
                "item": "weather",
                "status": "available in principle, not used by V1",
                "impact": "Would require historical weather backfill before using as a GLM feature.",
            },
            {
                "item": "altitude",
                "status": "available in principle, not used by V1",
                "impact": "Would require stadium coordinates and historical training joins.",
            },
        ]
    )
    return missing_from_source, available_not_modeled


def write_markdown_summary(
    *,
    path: Path,
    fixture: pd.Series,
    training_summary: pd.DataFrame,
    requested_order_matches: bool,
    expected_goals: pd.DataFrame,
    outcomes_table: pd.DataFrame,
    top_scores: pd.DataFrame,
    diagnostics: pd.DataFrame,
    model_weights: pd.DataFrame,
    model_result,
    artifacts: dict[str, Path],
) -> None:
    missing_from_source, available_not_modeled = missing_data_tables()
    order_note = (
        "Requested order matched stored fixture order."
        if requested_order_matches
        else "Requested order was reversed; report uses stored fixture order from future_fixtures.csv."
    )
    lines = [
        f"# {fixture['home_team']} vs {fixture['away_team']} Prediction",
        "",
        "## Fixture",
        "",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "fixture_id": fixture["fixture_id"],
                        "date": fixture["date"],
                        "home_team": fixture["home_team"],
                        "away_team": fixture["away_team"],
                        "city": fixture["city"],
                        "country": fixture["country"],
                        "neutral": fixture["neutral"],
                        "tournament": fixture["tournament"],
                    }
                ]
            )
        ),
        "",
        order_note,
        "",
        "## Model Training",
        "",
        markdown_table(training_summary),
        "",
        "## Expected Goals",
        "",
        markdown_table(expected_goals),
        "",
        "## Outcome Probabilities",
        "",
        markdown_table(outcomes_table, percent_columns={"probability"}),
        "",
        "## Top Scorelines",
        "",
        markdown_table(top_scores[["scoreline", "outcome", "probability"]], percent_columns={"probability"}),
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
        "The current V1 model uses football history/context features only. Weather is intentionally not",
        "used until comparable historical weather is backfilled for the training rows. If added later,",
        "Open-Meteo forecast and historical APIs are plausible no-key sources:",
        "https://open-meteo.com/en/docs and https://open-meteo.com/en/docs/historical-weather-api.",
        "",
        "Refresh current model data with:",
        "",
        "```bash",
        "poetry run python data_pipeline/build_data.py --refresh --check",
        "```",
        "",
        "## Artifacts",
        "",
        markdown_table(pd.DataFrame({"artifact": artifacts.keys(), "path": [str(path) for path in artifacts.values()]})),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    *,
    output_dir: Path,
    fixture: pd.Series,
    model_training: pd.DataFrame,
    feature_columns: list[str],
    requested_order_matches: bool,
    expected_goals: pd.DataFrame,
    outcomes: dict[str, float],
    score_matrix: np.ndarray,
    top_scores: pd.DataFrame,
    model_result,
    tuned_config_source: str,
) -> dict[str, Path]:
    home_team = str(fixture["home_team"])
    away_team = str(fixture["away_team"])
    output_dir.mkdir(parents=True, exist_ok=True)

    outcomes_table = pd.DataFrame(
        [
            {"outcome": f"{home_team} win", "probability": outcomes["home_win"]},
            {"outcome": "Draw", "probability": outcomes["draw"]},
            {"outcome": f"{away_team} win", "probability": outcomes["away_win"]},
        ]
    )
    training_summary = pd.DataFrame(
        [
            {
                "training_matches": int(model_training["match_id"].nunique()),
                "training_rows": int(len(model_training)),
                "training_start_date": pd.Timestamp(model_training["date"].min()).date().isoformat(),
                "training_end_date": pd.Timestamp(model_training["date"].max()).date().isoformat(),
                **elo_metadata(feature_columns),
                **weighting_metadata(model_result),
                **regularization_metadata(model_result),
                "tuned_config_source": tuned_config_source,
                "feature_columns": ", ".join(feature_columns),
            }
        ]
    )
    top_scoreline = top_scores.iloc[0]
    summary = pd.DataFrame(
        [
            {
                "fixture_id": fixture["fixture_id"],
                "date": fixture["date"],
                "home_team": home_team,
                "away_team": away_team,
                "city": fixture["city"],
                "country": fixture["country"],
                "training_matches": training_summary["training_matches"].iloc[0],
                "training_start_date": training_summary["training_start_date"].iloc[0],
                "training_end_date": training_summary["training_end_date"].iloc[0],
                "elo_system": training_summary["elo_system"].iloc[0],
                "sample_weighting": training_summary["sample_weighting"].iloc[0],
                "kyrre_weight_gamma": training_summary["kyrre_weight_gamma"].iloc[0],
                "kyrre_weight_half_life_years": training_summary["kyrre_weight_half_life_years"].iloc[0],
                "kyrre_weight_reference_date": training_summary["kyrre_weight_reference_date"].iloc[0],
                "fit_method": training_summary["fit_method"].iloc[0],
                "ridge_alpha": training_summary["ridge_alpha"].iloc[0],
                "tuned_config_source": training_summary["tuned_config_source"].iloc[0],
                "lambda_home": expected_goals.loc[expected_goals["side"].eq("listed_home"), "expected_goals"].iloc[0],
                "lambda_away": expected_goals.loc[expected_goals["side"].eq("listed_away"), "expected_goals"].iloc[0],
                "home_win_probability": outcomes["home_win"],
                "draw_probability": outcomes["draw"],
                "away_win_probability": outcomes["away_win"],
                "top_scoreline": top_scoreline["scoreline"],
                "top_scoreline_probability": top_scoreline["probability"],
                "score_matrix_probability_mass": float(score_matrix.sum()),
            }
        ]
    )

    artifacts = {
        "prediction_summary_md": output_dir / "prediction_summary.md",
        "prediction_summary_csv": output_dir / "prediction_summary.csv",
        "score_matrix_csv": output_dir / "score_matrix.csv",
        "scoreline_heatmap_png": output_dir / "scoreline_heatmap.png",
        "outcome_probabilities_png": output_dir / "outcome_probabilities.png",
        "top_scorelines_png": output_dir / "top_scorelines.png",
        "expected_goals_png": output_dir / "expected_goals.png",
        "model_weights_csv": output_dir / "model_weights.csv",
        "model_weights_png": output_dir / "model_weights.png",
    }

    weights = model_weights_table(model_result)
    summary.to_csv(artifacts["prediction_summary_csv"], index=False)
    weights.to_csv(artifacts["model_weights_csv"], index=False)
    save_score_matrix(score_matrix, home_team, away_team, artifacts["score_matrix_csv"])
    plot_score_heatmap(score_matrix, home_team, away_team, artifacts["scoreline_heatmap_png"])
    plot_outcomes(outcomes, home_team, away_team, artifacts["outcome_probabilities_png"])
    plot_top_scorelines(top_scores, artifacts["top_scorelines_png"])
    plot_expected_goals(expected_goals, artifacts["expected_goals_png"])
    plot_model_weights(weights, artifacts["model_weights_png"])
    write_markdown_summary(
        path=artifacts["prediction_summary_md"],
        fixture=fixture,
        training_summary=training_summary,
        requested_order_matches=requested_order_matches,
        expected_goals=expected_goals,
        outcomes_table=outcomes_table,
        top_scores=top_scores,
        diagnostics=model_diagnostics_table(model_result),
        model_weights=weights,
        model_result=model_result,
        artifacts=artifacts,
    )
    return artifacts


def print_console_report(
    *,
    fixture: pd.Series,
    model_training: pd.DataFrame,
    feature_columns: list[str],
    model_result,
    expected_goals: pd.DataFrame,
    outcomes: dict[str, float],
    top_scores: pd.DataFrame,
    artifacts: dict[str, Path],
) -> None:
    home_team = str(fixture["home_team"])
    away_team = str(fixture["away_team"])
    metadata = elo_metadata(feature_columns)
    weight_metadata = weighting_metadata(model_result)
    regularization = regularization_metadata(model_result)
    print("\nModel training")
    print(
        pd.DataFrame(
            [
                {
                    "training_matches": int(model_training["match_id"].nunique()),
                    "training_rows": int(len(model_training)),
                    "training_start_date": pd.Timestamp(model_training["date"].min()).date().isoformat(),
                    "training_end_date": pd.Timestamp(model_training["date"].max()).date().isoformat(),
                    "elo_system": metadata["elo_system"],
                    "elo_decay": metadata["elo_decay"],
                    "sample_weighting": weight_metadata["sample_weighting"],
                    "kyrre_weight_half_life_years": weight_metadata["kyrre_weight_half_life_years"],
                    "fit_method": regularization["fit_method"],
                    "ridge_alpha": regularization["ridge_alpha"],
                }
            ]
        ).to_string(index=False)
    )
    print("\nFixture")
    print(
        pd.DataFrame(
            [
                {
                    "fixture_id": fixture["fixture_id"],
                    "date": fixture["date"],
                    "match": f"{home_team} vs {away_team}",
                    "city": fixture["city"],
                    "country": fixture["country"],
                    "neutral": fixture["neutral"],
                }
            ]
        ).to_string(index=False)
    )
    print("\nExpected goals")
    print(expected_goals[["team", "side", "expected_goals", "eta"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nOutcome probabilities")
    outcome_table = pd.DataFrame(
        [
            {"outcome": f"{home_team} win", "probability": format_probability(outcomes["home_win"])},
            {"outcome": "Draw", "probability": format_probability(outcomes["draw"])},
            {"outcome": f"{away_team} win", "probability": format_probability(outcomes["away_win"])},
        ]
    )
    print(outcome_table.to_string(index=False))
    print("\nTop scorelines")
    top_print = top_scores[["scoreline", "outcome", "probability"]].copy()
    top_print["probability"] = top_print["probability"].map(format_probability)
    print(top_print.to_string(index=False))
    print("\nLargest standardized model weights")
    weights = model_weights_table(model_result)
    top_weights = weights[weights["term"].ne("const")].head(12)[
        ["term", "coefficient", "standardized_coefficient", "rate_ratio_per_1sd", "p_value"]
    ]
    print(top_weights.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nMissing or not yet modeled data")
    missing_from_source, available_not_modeled = missing_data_tables()
    print("Missing from source data:")
    print(missing_from_source.to_string(index=False))
    print("\nAvailable in principle but not used by V1:")
    print(available_not_modeled.to_string(index=False))
    print("\nReport artifacts")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


def validate_prediction_outputs(
    expected_goals: pd.DataFrame,
    score_matrix: np.ndarray,
    outcomes: dict[str, float],
    max_goals: int,
) -> None:
    lambdas = expected_goals["expected_goals"].to_numpy(dtype=float)
    if not np.isfinite(lambdas).all() or not (lambdas > 0).all():
        raise ValueError("Both predicted lambdas must be finite and positive.")
    expected_shape = (max_goals + 1, max_goals + 1)
    if score_matrix.shape != expected_shape:
        raise ValueError(f"Score matrix shape must be {expected_shape}; got {score_matrix.shape}.")
    outcome_values = np.array(list(outcomes.values()), dtype=float)
    if not np.isfinite(outcome_values).all() or (outcome_values < 0).any():
        raise ValueError("Outcome probabilities must be finite and nonnegative.")
    if not np.isclose(outcome_values.sum(), score_matrix.sum()):
        raise ValueError("Outcome probabilities must sum to the score matrix probability mass.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a future fixture with the V1 Poisson GLM.")
    parser.add_argument("--home", default="Switzerland", help="Requested home/listed first team.")
    parser.add_argument("--away", default="Qatar", help="Requested away/listed second team.")
    parser.add_argument("--fixtures-csv", type=Path, default=DEFAULT_FIXTURES_CSV)
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING_CSV)
    parser.add_argument("--raw-results-csv", type=Path, default=DEFAULT_RAW_RESULTS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    model_training = training.copy()
    model_training["date"] = pd.to_datetime(model_training["date"], errors="coerce")
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
    fixture, requested_order_matches = find_fixture(fixtures, args.home, args.away)
    fixture_slug = (
        f"{pd.Timestamp(fixture['date']).date()}-"
        f"{slugify(str(fixture['home_team']))}-vs-{slugify(str(fixture['away_team']))}"
    )
    report_dir = args.output_dir / fixture_slug

    model_result = fit_poisson_glm_from_data(
        model_training,
        feature_columns=args.features,
        target_column="goals_for",
        id_columns=["match_id", "team", "opponent"],
        date_column="date",
        kyrre_weight_gamma=None if args.no_kyrre_weight else args.kyrre_weight_gamma,
        kyrre_weight_half_life_years=kyrre_half_life,
        ridge_alpha=ridge_alpha,
    )
    future_rows = build_future_feature_rows(
        fixture=fixture,
        training=training,
        raw_results_csv=args.raw_results_csv,
    )
    predictions = predict_expected_goals(model_result, future_rows)
    prediction_rows = future_rows[["team", "is_listed_home"]].join(predictions)
    prediction_rows["side"] = np.where(prediction_rows["is_listed_home"].eq(1), "listed_home", "listed_away")
    expected_goals = prediction_rows.rename(columns={"lambda": "expected_goals"})[
        ["team", "side", "eta", "expected_goals"]
    ]

    home_lambda = float(expected_goals.loc[expected_goals["side"].eq("listed_home"), "expected_goals"].iloc[0])
    away_lambda = float(expected_goals.loc[expected_goals["side"].eq("listed_away"), "expected_goals"].iloc[0])
    score_matrix = poisson_score_matrix(home_lambda, away_lambda, max_goals=args.max_goals)
    outcomes = outcome_probabilities(score_matrix)
    top_scores = top_scorelines(score_matrix, str(fixture["home_team"]), str(fixture["away_team"]))
    validate_prediction_outputs(expected_goals, score_matrix, outcomes, args.max_goals)

    artifacts = write_reports(
        output_dir=report_dir,
        fixture=fixture,
        model_training=model_training,
        feature_columns=args.features,
        requested_order_matches=requested_order_matches,
        expected_goals=expected_goals,
        outcomes=outcomes,
        score_matrix=score_matrix,
        top_scores=top_scores,
        model_result=model_result,
        tuned_config_source=str(tuned_config.get("source", "")),
    )
    print_console_report(
        fixture=fixture,
        model_training=model_training,
        feature_columns=args.features,
        model_result=model_result,
        expected_goals=expected_goals,
        outcomes=outcomes,
        top_scores=top_scores,
        artifacts=artifacts,
    )


if __name__ == "__main__":
    main()
