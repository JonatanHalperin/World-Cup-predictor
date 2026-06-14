"""Model weight tables and plots for fitted V1 Poisson GLMs."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def model_weights_table(model_result) -> pd.DataFrame:
    """Return GLM coefficients and comparable one-standard-deviation effects.

    Coefficients are beta weights on the log-goal scale. Unregularized fits use
    raw feature units. Ridge fits standardize features before estimation, so
    their fitted coefficients are already one-training-standard-deviation
    effects for non-constant terms.
    """
    terms = list(model_result.params.index)
    exog = np.asarray(model_result.model.exog, dtype=float)
    if exog.shape[1] != len(terms):
        raise ValueError("Model design matrix does not match fitted coefficient terms.")

    features_standardized = bool(getattr(model_result, "standardize_features", False))
    feature_std = pd.Series(np.std(exog, axis=0, ddof=0), index=terms)
    table = pd.DataFrame(
        {
            "term": terms,
            "coefficient": model_result.params.values,
            "std_error": model_result.bse.values,
            "p_value": model_result.pvalues.values,
            "feature_std": feature_std.reindex(terms).values,
        }
    )
    table["coefficient_scale"] = np.where(
        table["term"].eq("const"),
        "intercept",
        "standardized_feature" if features_standardized else "raw_feature",
    )
    table["ci_lower"] = table["coefficient"] - 1.96 * table["std_error"]
    table["ci_upper"] = table["coefficient"] + 1.96 * table["std_error"]
    table["standardized_coefficient"] = table["coefficient"] * table["feature_std"]
    table["standardized_std_error"] = table["std_error"] * table["feature_std"]
    table["standardized_ci_lower"] = table["standardized_coefficient"] - 1.96 * table["standardized_std_error"]
    table["standardized_ci_upper"] = table["standardized_coefficient"] + 1.96 * table["standardized_std_error"]
    table["abs_standardized_coefficient"] = table["standardized_coefficient"].abs()
    table["rate_ratio_per_model_unit"] = np.exp(table["coefficient"].clip(lower=-50, upper=50))
    table["rate_ratio_per_unit"] = table["rate_ratio_per_model_unit"]
    if features_standardized:
        feature_terms = table["term"].ne("const")
        table.loc[feature_terms, "rate_ratio_per_unit"] = np.nan
    table["rate_ratio_per_1sd"] = np.exp(table["standardized_coefficient"].clip(lower=-50, upper=50))
    return table.sort_values("abs_standardized_coefficient", ascending=False).reset_index(drop=True)


def plot_model_weights(weights: pd.DataFrame, path: Path, *, limit: int = 30) -> None:
    """Save a horizontal bar chart of the largest standardized coefficients."""
    plot_data = weights[weights["term"].ne("const")].copy()
    plot_data = plot_data[plot_data["feature_std"].gt(0)]
    plot_data = plot_data.sort_values("abs_standardized_coefficient", ascending=False).head(limit)
    plot_data = plot_data.sort_values("standardized_coefficient")
    if plot_data.empty:
        raise ValueError("No non-constant model weights are available to plot.")

    colors = np.where(plot_data["standardized_coefficient"].ge(0), "#2f7d62", "#b23a48")
    xerr = 1.96 * plot_data["standardized_std_error"]
    height = max(6.0, 0.34 * len(plot_data) + 1.8)
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(
        plot_data["term"],
        plot_data["standardized_coefficient"],
        xerr=xerr,
        color=colors,
        alpha=0.9,
        error_kw={"ecolor": "#333333", "elinewidth": 0.8, "capsize": 2},
    )
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_xlabel("Standardized coefficient: beta * training feature std")
    ax.set_ylabel("Feature")
    ax.set_title("Poisson GLM model weights")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
