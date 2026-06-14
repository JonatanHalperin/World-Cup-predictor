"""Poisson GLM utilities for football goal prediction.

Each training row is one team's goal count in one match. The model assumes

    goals_i ~ Poisson(lambda_i)
    log(lambda_i) = eta_i = X_i beta

Statsmodels estimates beta by weighted maximum likelihood. The default
time-decay sample weights are called Kyrre weights. Predictions exponentiate
the linear predictor eta to return lambda, the expected goals for that row.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import poisson
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLMResultsWrapper


DAYS_PER_YEAR = 365.25
DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS = 8.0


def _as_list(columns: Iterable[str] | None) -> list[str]:
    return list(columns or [])


def _validate_columns(
    data: pd.DataFrame,
    *,
    target_column: str | None,
    feature_columns: list[str],
    id_columns: list[str] | None = None,
) -> None:
    """Validate required columns and missing values before modeling."""
    id_columns = _as_list(id_columns)
    required_columns = feature_columns + id_columns
    if target_column is not None:
        required_columns = [target_column, *required_columns]

    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {', '.join(missing_columns)}")

    used_columns = [column for column in required_columns if column in data.columns]
    columns_with_missing = data[used_columns].columns[data[used_columns].isna().any()].tolist()
    if columns_with_missing:
        raise ValueError(f"Missing values found in column(s): {', '.join(columns_with_missing)}")


def _numeric_frame(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a float design matrix, failing clearly for non-numeric columns."""
    try:
        return data.loc[:, columns].astype(float)
    except ValueError as exc:
        raise ValueError(
            "Feature columns must be numeric before fitting. "
            "Create any categorical encodings before calling this module."
        ) from exc


def _design_matrix(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Build X and add the intercept term used by eta = X beta."""
    features = _numeric_frame(data, feature_columns)
    return sm.add_constant(features, has_constant="add")


def gamma_from_half_life(half_life_years: float) -> float:
    """Convert a Kyrre-weight half-life in years to gamma for exp(-gamma * age)."""
    if not np.isfinite(half_life_years) or half_life_years <= 0:
        raise ValueError("half_life_years must be finite and positive.")
    return float(np.log(2.0) / half_life_years)


def compute_kyrre_weights(
    data: pd.DataFrame,
    *,
    date_column: str = "date",
    gamma: float | None = None,
    half_life_years: float = DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS,
    reference_date: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Return Kyrre weights w_i = exp(-gamma * age_i) for dated training rows.

    Age is measured in years relative to the latest training date unless an
    explicit reference date is supplied. The most recent rows have weight 1.
    """
    if date_column not in data.columns:
        raise ValueError(f"Missing date column required for Kyrre weights: {date_column}")
    if gamma is None:
        gamma = gamma_from_half_life(half_life_years)
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("Kyrre-weight gamma must be finite and non-negative.")

    dates = pd.to_datetime(data[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{date_column} contains invalid dates.")
    reference = pd.Timestamp(reference_date) if reference_date is not None else dates.max()
    ages = (reference - dates).dt.days.astype(float) / DAYS_PER_YEAR
    if (ages < -1e-9).any():
        raise ValueError("Kyrre-weight reference date cannot be earlier than training rows.")
    ages = ages.clip(lower=0)
    weights = np.exp(-gamma * ages)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Kyrre weights must be finite and positive.")
    return pd.Series(weights, index=data.index, name="kyrre_weight")


def _validated_weights(sample_weights: Iterable[float] | pd.Series, index: pd.Index) -> pd.Series:
    weights = pd.Series(sample_weights, index=index, dtype=float, name="sample_weight")
    if weights.isna().any():
        raise ValueError("Sample weights contain missing values.")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Sample weights must be finite and positive.")
    return weights


def load_model_data(
    csv_path: str | Path,
    feature_columns: Iterable[str],
    target_column: str = "goals",
    id_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load a CSV and validate the columns used by the Poisson GLM."""
    feature_columns = _as_list(feature_columns)
    if not feature_columns:
        raise ValueError("At least one feature column is required.")

    data = pd.read_csv(csv_path)
    _validate_columns(
        data,
        target_column=target_column,
        feature_columns=feature_columns,
        id_columns=_as_list(id_columns),
    )
    return data


def print_model_diagnostics(model_result: GLMResultsWrapper) -> None:
    """Print the key maximum-likelihood diagnostics for the fitted GLM."""
    diagnostics = pd.DataFrame(
        {
            "coefficient": model_result.params,
            "std_error": model_result.bse,
            "p_value": model_result.pvalues,
        }
    )

    print("\nCoefficient estimates")
    print(diagnostics.to_string(float_format=lambda value: f"{value: .6f}"))
    print(f"\nLog-likelihood: {model_result.llf: .6f}")
    print(f"AIC:            {model_result.aic: .6f}")
    print(f"Deviance:       {model_result.deviance: .6f}")
    weight_summary = getattr(model_result, "sample_weight_summary", None)
    if weight_summary:
        print("\nSample weights")
        print(
            pd.DataFrame([weight_summary]).to_string(
                index=False,
                float_format=lambda value: f"{value: .6f}",
            )
        )


def fit_poisson_glm(
    csv_path: str | Path,
    feature_columns: Iterable[str],
    target_column: str = "goals",
    id_columns: Iterable[str] | None = None,
    print_diagnostics: bool = True,
    sample_weights: Iterable[float] | None = None,
    date_column: str = "date",
    kyrre_weight_gamma: float | None = None,
    kyrre_weight_half_life_years: float | None = DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS,
    kyrre_weight_reference_date: str | pd.Timestamp | None = None,
) -> GLMResultsWrapper:
    """Load a CSV and fit a Poisson GLM with a log link."""
    feature_columns = _as_list(feature_columns)
    id_columns = _as_list(id_columns)
    data = load_model_data(csv_path, feature_columns, target_column, id_columns)
    return fit_poisson_glm_from_data(
        data,
        feature_columns=feature_columns,
        target_column=target_column,
        id_columns=id_columns,
        print_diagnostics=print_diagnostics,
        sample_weights=sample_weights,
        date_column=date_column,
        kyrre_weight_gamma=kyrre_weight_gamma,
        kyrre_weight_half_life_years=kyrre_weight_half_life_years,
        kyrre_weight_reference_date=kyrre_weight_reference_date,
    )


def fit_poisson_glm_from_data(
    data: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str = "goals",
    id_columns: Iterable[str] | None = None,
    print_diagnostics: bool = True,
    sample_weights: Iterable[float] | None = None,
    date_column: str = "date",
    kyrre_weight_gamma: float | None = None,
    kyrre_weight_half_life_years: float | None = DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS,
    kyrre_weight_reference_date: str | pd.Timestamp | None = None,
) -> GLMResultsWrapper:
    """Fit a Poisson GLM with a log link using maximum likelihood.

    The caller supplies the feature columns. This function does not invent,
    encode, scale, or transform features beyond adding an intercept column.
    When Kyrre weighting is enabled, statsmodels maximizes
    sum_i w_i log P(goals_i | theta), where w_i = exp(-gamma * age_i).
    """
    feature_columns = _as_list(feature_columns)
    id_columns = _as_list(id_columns)
    if not feature_columns:
        raise ValueError("At least one feature column is required.")
    _validate_columns(
        data,
        target_column=target_column,
        feature_columns=feature_columns,
        id_columns=id_columns,
    )

    y = data[target_column].astype(float)
    if (y < 0).any():
        raise ValueError("Poisson targets must be non-negative goal counts.")

    # X beta is the linear predictor eta. The Poisson log link maps eta to
    # lambda with exp(eta), guaranteeing positive expected goals.
    x = _design_matrix(data, feature_columns)
    weights = None
    weight_metadata: dict[str, object] = {}
    if sample_weights is not None:
        if kyrre_weight_gamma is not None:
            raise ValueError("Pass either explicit sample_weights or Kyrre-weight gamma, not both.")
        weights = _validated_weights(sample_weights, data.index)
        weight_metadata = {"weighting": "explicit_sample_weights"}
    elif kyrre_weight_gamma is not None or kyrre_weight_half_life_years is not None:
        half_life = (
            DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS
            if kyrre_weight_half_life_years is None
            else float(kyrre_weight_half_life_years)
        )
        gamma = kyrre_weight_gamma if kyrre_weight_gamma is not None else gamma_from_half_life(half_life)
        weights = compute_kyrre_weights(
            data,
            date_column=date_column,
            gamma=gamma,
            half_life_years=half_life,
            reference_date=kyrre_weight_reference_date,
        )
        weight_metadata = {
            "weighting": "Kyrre_weight",
            "date_column": date_column,
            "kyrre_weight_gamma": float(gamma),
            "kyrre_weight_half_life_years": half_life,
            "kyrre_weight_reference_date": pd.Timestamp(
                kyrre_weight_reference_date
                if kyrre_weight_reference_date is not None
                else pd.to_datetime(data[date_column]).max()
            ).date().isoformat(),
        }

    model = sm.GLM(y, x, family=sm.families.Poisson(), freq_weights=weights)
    result = model.fit()

    # Store the training schema on the result so prediction can rebuild X in the
    # same column order, including the intercept.
    result.feature_columns = feature_columns
    result.design_columns = x.columns.tolist()
    result.target_column = target_column
    result.id_columns = id_columns
    if weights is not None:
        result.sample_weights = weights.copy()
        result.sample_weight_summary = {
            **weight_metadata,
            "weight_min": float(weights.min()),
            "weight_mean": float(weights.mean()),
            "weight_max": float(weights.max()),
            "weight_sum": float(weights.sum()),
        }
    if id_columns:
        result.training_identifiers = data.loc[:, id_columns].copy()

    if print_diagnostics:
        print_model_diagnostics(result)
    return result


def predict_expected_goals(model_result: GLMResultsWrapper, new_data: pd.DataFrame) -> pd.DataFrame:
    """Predict eta = X beta and lambda = exp(eta) for new team-match rows."""
    feature_columns = getattr(model_result, "feature_columns", None)
    design_columns = getattr(model_result, "design_columns", None)
    if feature_columns is None or design_columns is None:
        raise ValueError("Model result is missing stored feature metadata.")

    _validate_columns(
        new_data,
        target_column=None,
        feature_columns=feature_columns,
        id_columns=None,
    )

    x = _design_matrix(new_data, feature_columns)
    x = x.reindex(columns=design_columns)
    if x.isna().any().any():
        raise ValueError("Prediction design matrix does not match the fitted model columns.")

    # eta is on the log-goal scale; lambda is expected goals on the original
    # goal-count scale.
    eta = x.dot(model_result.params)
    lambdas = np.exp(eta)
    if not np.isfinite(lambdas).all() or not (lambdas > 0).all():
        raise ValueError("Predicted lambdas must be finite and positive.")

    return pd.DataFrame({"eta": eta, "lambda": lambdas}, index=new_data.index)


def poisson_score_matrix(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = 10,
) -> np.ndarray:
    """Return P(home goals = i, away goals = j) for i,j in 0..max_goals."""
    if not isinstance(max_goals, int):
        raise TypeError("max_goals must be an integer.")
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative.")
    if not np.isfinite(lambda_home) or not np.isfinite(lambda_away):
        raise ValueError("Lambdas must be finite.")
    if lambda_home <= 0 or lambda_away <= 0:
        raise ValueError("Lambdas must be positive.")

    goals = np.arange(max_goals + 1)
    home_goal_probs = poisson.pmf(goals, lambda_home)
    away_goal_probs = poisson.pmf(goals, lambda_away)

    # Assuming the two teams' goal counts are independent conditional on their
    # lambdas, the joint scoreline probability is the outer product.
    return np.outer(home_goal_probs, away_goal_probs)


def outcome_probabilities(score_matrix: np.ndarray) -> dict[str, float]:
    """Aggregate a scoreline matrix into home win, draw, and away win odds."""
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("score_matrix must be a square 2D matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("score_matrix must contain finite probabilities.")
    if (matrix < 0).any():
        raise ValueError("score_matrix cannot contain negative probabilities.")

    return {
        "home_win": float(np.tril(matrix, k=-1).sum()),
        "draw": float(np.trace(matrix)),
        "away_win": float(np.triu(matrix, k=1).sum()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a Poisson GLM for football goals.")
    parser.add_argument("csv_path", type=Path, help="Path to a long-format team-match CSV.")
    parser.add_argument(
        "--target",
        default="goals",
        help='Target column name. Defaults to "goals".',
    )
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="Feature columns to use exactly as they appear in the CSV.",
    )
    parser.add_argument(
        "--id-columns",
        nargs="*",
        default=None,
        help="Optional identifying columns to validate and keep on the fitted result.",
    )
    parser.add_argument(
        "--kyrre-weight-half-life-years",
        type=float,
        default=DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS,
        help="Half-life in years for Kyrre weights. Defaults to 8.",
    )
    parser.add_argument(
        "--kyrre-weight-gamma",
        type=float,
        default=None,
        help="Enable Kyrre weights with gamma in exp(-gamma * age_years).",
    )
    parser.add_argument(
        "--date-column",
        default="date",
        help='Date column for Kyrre weighting. Defaults to "date".',
    )
    parser.add_argument(
        "--no-kyrre-weight",
        action="store_true",
        help="Disable Kyrre-weighted likelihood fitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fit_poisson_glm(
        args.csv_path,
        feature_columns=args.features,
        target_column=args.target,
        id_columns=args.id_columns,
        date_column=args.date_column,
        kyrre_weight_gamma=args.kyrre_weight_gamma,
        kyrre_weight_half_life_years=(
            None if args.no_kyrre_weight else args.kyrre_weight_half_life_years
        ),
    )


if __name__ == "__main__":
    main()
