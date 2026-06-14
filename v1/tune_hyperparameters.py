"""Tune V1 Poisson GLM Kyrre half-life and ridge alpha with rolling validation."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .poisson_glm import fit_poisson_glm_from_data, gamma_from_half_life
    from .tuning_config import DEFAULT_TUNED_CONFIG_PATH, feature_list_hash, write_tuned_config
    from .validate_model import (
        DEFAULT_FEATURE_COLUMNS,
        DEFAULT_TRAINING_CSV,
        add_row_predictions,
        build_match_predictions,
        load_training_data,
        markdown_table,
    )
except ImportError:
    from poisson_glm import fit_poisson_glm_from_data, gamma_from_half_life
    from tuning_config import DEFAULT_TUNED_CONFIG_PATH, feature_list_hash, write_tuned_config
    from validate_model import (
        DEFAULT_FEATURE_COLUMNS,
        DEFAULT_TRAINING_CSV,
        add_row_predictions,
        build_match_predictions,
        load_training_data,
        markdown_table,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "v1" / "reports" / "hyperparameter_tuning"
DEFAULT_HALF_LIVES = ["12", "16", "20", "24", "32", "48", "none"]
DEFAULT_RIDGE_ALPHAS = [0.0003, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.02]
DEFAULT_VALIDATION_YEARS = [2016, 2017, 2018, 2019, 2020, 2021]
DEFAULT_SELECTION_TOLERANCE = 0.00005


def parse_half_life_candidate(value: str | float | int | None) -> float | None:
    """Parse a candidate half-life, allowing no Kyrre weighting as a candidate."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "no-weight", "no_weight", "off"}:
        return None
    half_life = float(value)
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError("Half-life candidates must be positive years or 'none'.")
    return half_life


def half_life_label(half_life: float | None) -> str:
    if half_life is None:
        return "no_weight"
    return f"{half_life:g}y"


def half_life_sort_value(half_life: float | None) -> float:
    if half_life is None:
        return 1_000_000.0
    return float(half_life)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune V1 Poisson GLM hyperparameters with rolling validation.")
    parser.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tuned-config", type=Path, default=DEFAULT_TUNED_CONFIG_PATH)
    parser.add_argument("--train-start-year", type=int, default=2010)
    parser.add_argument("--validation-years", nargs="+", type=int, default=DEFAULT_VALIDATION_YEARS)
    parser.add_argument(
        "--half-lives",
        nargs="+",
        default=DEFAULT_HALF_LIVES,
        help="Half-life candidates in years. Use 'none' or 'no-weight' to test unweighted fitting.",
    )
    parser.add_argument("--ridge-alphas", nargs="+", type=float, default=DEFAULT_RIDGE_ALPHAS)
    parser.add_argument(
        "--selection-tolerance",
        type=float,
        default=DEFAULT_SELECTION_TOLERANCE,
        help=(
            "Treat configs within this mean log-loss distance of the best as tied, "
            "then select by RPS and Poisson NLL. Defaults to 0.00005."
        ),
    )
    parser.add_argument("--max-goals", type=int, default=10)
    parser.add_argument(
        "--features",
        nargs="+",
        default=DEFAULT_FEATURE_COLUMNS,
        help="Feature columns to use exactly as they appear in poisson_training_long.csv.",
    )
    return parser.parse_args()


def evaluate_fold(
    *,
    data: pd.DataFrame,
    train_start_year: int,
    validation_year: int,
    half_life: float | None,
    ridge_alpha: float,
    feature_columns: list[str],
    max_goals: int,
) -> dict[str, object]:
    train = data[
        data["date"].dt.year.ge(train_start_year)
        & data["date"].dt.year.lt(validation_year)
    ].copy()
    validation = data[data["date"].dt.year.eq(validation_year)].copy()
    if train.empty or validation.empty:
        raise ValueError(f"Empty train/validation split for validation year {validation_year}.")

    model_result = fit_poisson_glm_from_data(
        train,
        feature_columns=feature_columns,
        target_column="goals_for",
        id_columns=["match_id", "team", "opponent"],
        print_diagnostics=False,
        date_column="date",
        kyrre_weight_half_life_years=half_life,
        ridge_alpha=ridge_alpha,
    )
    row_predictions = add_row_predictions(model_result, validation)
    match_predictions = build_match_predictions(row_predictions, max_goals=max_goals)
    return {
        "validation_year": validation_year,
        "kyrre_weight_label": half_life_label(half_life),
        "kyrre_weight_sort": half_life_sort_value(half_life),
        "kyrre_weight_half_life_years": half_life,
        "kyrre_weight_gamma": None if half_life is None else gamma_from_half_life(half_life),
        "ridge_alpha": ridge_alpha,
        "train_matches": int(train["match_id"].nunique()),
        "validation_matches": int(validation["match_id"].nunique()),
        "mean_outcome_log_loss": float(match_predictions["outcome_log_loss"].mean()),
        "mean_rps": float(match_predictions["rps"].mean()),
        "mean_poisson_nll": float(row_predictions["poisson_nll"].mean()),
        "mean_poisson_deviance": float(row_predictions["poisson_deviance"].mean()),
        "outcome_accuracy": float(match_predictions["outcome_correct"].mean()),
        "goal_rmse": float(np.sqrt(row_predictions["squared_error"].mean())),
    }


def aggregate_grid(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = fold_metrics.groupby(["kyrre_weight_label", "ridge_alpha"], as_index=False)
    result = grouped.agg(
        kyrre_weight_half_life_years=("kyrre_weight_half_life_years", "first"),
        kyrre_weight_gamma=("kyrre_weight_gamma", "first"),
        kyrre_weight_sort=("kyrre_weight_sort", "first"),
        folds=("validation_year", "nunique"),
        train_matches_mean=("train_matches", "mean"),
        validation_matches_sum=("validation_matches", "sum"),
        mean_outcome_log_loss=("mean_outcome_log_loss", "mean"),
        std_outcome_log_loss=("mean_outcome_log_loss", "std"),
        mean_rps=("mean_rps", "mean"),
        mean_poisson_nll=("mean_poisson_nll", "mean"),
        mean_poisson_deviance=("mean_poisson_deviance", "mean"),
        outcome_accuracy=("outcome_accuracy", "mean"),
        goal_rmse=("goal_rmse", "mean"),
    )
    return result.sort_values(
        ["mean_outcome_log_loss", "mean_rps", "mean_poisson_nll"],
        ascending=True,
    ).reset_index(drop=True)


def select_stable_row(grid: pd.DataFrame, selection_tolerance: float) -> pd.Series:
    """Pick from the statistically tiny top region instead of chasing noise."""
    if selection_tolerance < 0 or not np.isfinite(selection_tolerance):
        raise ValueError("selection_tolerance must be finite and non-negative.")
    best_log_loss = float(grid["mean_outcome_log_loss"].min())
    tied = grid[grid["mean_outcome_log_loss"].le(best_log_loss + selection_tolerance)].copy()
    return tied.sort_values(
        ["mean_rps", "mean_poisson_nll", "mean_outcome_log_loss"],
        ascending=True,
    ).iloc[0]


def selected_config(
    *,
    selected: pd.Series,
    feature_columns: list[str],
    validation_years: list[int],
    selection_tolerance: float,
) -> dict[str, object]:
    half_life_value = selected["kyrre_weight_half_life_years"]
    half_life = None if pd.isna(half_life_value) else float(half_life_value)
    ridge_alpha = float(selected["ridge_alpha"])
    return {
        "model": "poisson_glm",
        "fit_method": "ridge_regularized_poisson_glm",
        "selection_metric": "mean_outcome_log_loss",
        "selection_policy": "within_log_loss_tolerance_then_rps_then_poisson_nll",
        "selection_tolerance": float(selection_tolerance),
        "selection_tiebreakers": ["mean_rps", "mean_poisson_nll"],
        "kyrre_weight_label": half_life_label(half_life),
        "kyrre_weight_half_life_years": half_life,
        "kyrre_weight_gamma": None if half_life is None else gamma_from_half_life(half_life),
        "ridge_alpha": ridge_alpha,
        "feature_list_hash": feature_list_hash(feature_columns),
        "feature_columns": feature_columns,
        "validation_years": validation_years,
        "selected_metrics": {
            "mean_outcome_log_loss": float(selected["mean_outcome_log_loss"]),
            "mean_rps": float(selected["mean_rps"]),
            "mean_poisson_nll": float(selected["mean_poisson_nll"]),
            "outcome_accuracy": float(selected["outcome_accuracy"]),
        },
        "tuning_date": date.today().isoformat(),
    }


def plot_heatmap(grid: pd.DataFrame, value_column: str, title: str, path: Path) -> None:
    ordered = grid.sort_values("kyrre_weight_sort")
    pivot = ordered.pivot(index="kyrre_weight_label", columns="ridge_alpha", values=value_column)
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    fig.colorbar(image, ax=ax, label=value_column)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:g}" for value in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel("Ridge alpha")
    ax.set_ylabel("Kyrre half-life")
    ax.set_title(title)
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            ax.text(column, row, f"{pivot.iat[row, column]:.3f}", ha="center", va="center", fontsize=7, color="white")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_selected_rolling(fold_metrics: pd.DataFrame, selected: pd.Series, path: Path) -> None:
    selected_rows = fold_metrics[
        fold_metrics["kyrre_weight_label"].eq(selected["kyrre_weight_label"])
        & fold_metrics["ridge_alpha"].eq(selected["ridge_alpha"])
    ].sort_values("validation_year")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(selected_rows["validation_year"], selected_rows["mean_outcome_log_loss"], marker="o", label="Log-loss")
    ax.plot(selected_rows["validation_year"], selected_rows["mean_rps"], marker="o", label="RPS")
    ax.set_xlabel("Validation year")
    ax.set_ylabel("Metric")
    ax.set_title("Selected hyperparameters: rolling validation metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary(path: Path, grid: pd.DataFrame, fold_metrics: pd.DataFrame, config: dict[str, object], artifacts: dict[str, Path]) -> None:
    top = grid.head(15)
    selected_folds = fold_metrics[
        fold_metrics["kyrre_weight_label"].eq(config["kyrre_weight_label"])
        & fold_metrics["ridge_alpha"].eq(config["ridge_alpha"])
    ].sort_values("validation_year")
    lines = [
        "# V1 Hyperparameter Tuning",
        "",
        "Rolling expanding-window validation chooses Kyrre half-life and L2 ridge alpha.",
        "The base model remains a Poisson GLM with log link.",
        "",
        "## Selected Hyperparameters",
        "",
        markdown_table(pd.DataFrame([config])),
        "",
        (
            "Selection treats configs within "
            f"{config.get('selection_tolerance')} mean log-loss of the best row as tied, "
            "then uses RPS and Poisson NLL as tie-breakers."
        ),
        "",
        "## Top Grid Results",
        "",
        markdown_table(top),
        "",
        "## Selected Rolling Fold Metrics",
        "",
        markdown_table(selected_folds),
        "",
        "## Artifacts",
        "",
        markdown_table(pd.DataFrame({"artifact": artifacts.keys(), "path": [str(value) for value in artifacts.values()]})),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = load_training_data(args.training_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    half_lives = [parse_half_life_candidate(value) for value in args.half_lives]
    fold_rows = []
    for half_life in half_lives:
        for ridge_alpha in args.ridge_alphas:
            for validation_year in args.validation_years:
                fold_rows.append(
                    evaluate_fold(
                        data=data,
                        train_start_year=args.train_start_year,
                        validation_year=validation_year,
                        half_life=half_life,
                        ridge_alpha=ridge_alpha,
                        feature_columns=list(args.features),
                        max_goals=args.max_goals,
                    )
                )
                print(
                    f"evaluated half_life={half_life_label(half_life)}, ridge_alpha={ridge_alpha:g}, "
                    f"validation_year={validation_year}"
                )

    fold_metrics = pd.DataFrame(fold_rows)
    grid = aggregate_grid(fold_metrics)
    selected = select_stable_row(grid, args.selection_tolerance)
    config = selected_config(
        selected=selected,
        feature_columns=list(args.features),
        validation_years=list(args.validation_years),
        selection_tolerance=args.selection_tolerance,
    )
    config_path = write_tuned_config(args.tuned_config, config)

    artifacts = {
        "tuning_summary_md": args.output_dir / "tuning_summary.md",
        "tuning_grid_results_csv": args.output_dir / "tuning_grid_results.csv",
        "rolling_fold_metrics_csv": args.output_dir / "rolling_fold_metrics.csv",
        "selected_hyperparameters_json": args.output_dir / "selected_hyperparameters.json",
        "log_loss_heatmap_png": args.output_dir / "log_loss_heatmap.png",
        "rps_heatmap_png": args.output_dir / "rps_heatmap.png",
        "selected_rolling_metrics_png": args.output_dir / "selected_rolling_metrics.png",
    }
    grid.to_csv(artifacts["tuning_grid_results_csv"], index=False)
    fold_metrics.to_csv(artifacts["rolling_fold_metrics_csv"], index=False)
    artifacts["selected_hyperparameters_json"].write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_heatmap(grid, "mean_outcome_log_loss", "Rolling validation log-loss", artifacts["log_loss_heatmap_png"])
    plot_heatmap(grid, "mean_rps", "Rolling validation RPS", artifacts["rps_heatmap_png"])
    plot_selected_rolling(fold_metrics, selected, artifacts["selected_rolling_metrics_png"])
    write_summary(artifacts["tuning_summary_md"], grid, fold_metrics, config, artifacts)

    print("\nSelected hyperparameters")
    print(pd.DataFrame([config]).to_string(index=False))
    print(f"\nUpdated tuned config: {config_path}")
    print("\nReport artifacts")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
