"""Tracked hyperparameter defaults for the V1 Poisson GLM."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Iterable

try:
    from .poisson_glm import DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS, DEFAULT_RIDGE_ALPHA, gamma_from_half_life
except ImportError:
    from poisson_glm import DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS, DEFAULT_RIDGE_ALPHA, gamma_from_half_life


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TUNED_CONFIG_PATH = PROJECT_ROOT / "v1" / "config" / "tuned_poisson_glm.json"


def feature_list_hash(feature_columns: Iterable[str]) -> str:
    payload = json.dumps(list(feature_columns), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def fallback_config(feature_columns: Iterable[str]) -> dict[str, object]:
    half_life = DEFAULT_KYRRE_WEIGHT_HALF_LIFE_YEARS
    return {
        "model": "poisson_glm",
        "fit_method": "ridge_regularized_poisson_glm",
        "selection_metric": "fallback_default",
        "kyrre_weight_half_life_years": half_life,
        "kyrre_weight_gamma": gamma_from_half_life(half_life),
        "ridge_alpha": DEFAULT_RIDGE_ALPHA,
        "feature_list_hash": feature_list_hash(feature_columns),
        "feature_columns": list(feature_columns),
        "tuning_date": date.today().isoformat(),
        "source": "fallback",
    }


def load_tuned_config(
    path: Path | str | None,
    *,
    feature_columns: Iterable[str],
    ignore: bool = False,
) -> dict[str, object]:
    if ignore:
        return fallback_config(feature_columns)
    config_path = Path(path) if path is not None else DEFAULT_TUNED_CONFIG_PATH
    if not config_path.exists():
        return fallback_config(feature_columns)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    expected_hash = feature_list_hash(feature_columns)
    if config.get("feature_list_hash") != expected_hash:
        fallback = fallback_config(feature_columns)
        fallback["source"] = f"fallback_feature_hash_mismatch:{config_path}"
        return fallback
    required = ["kyrre_weight_half_life_years", "ridge_alpha"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Tuned config is missing required key(s): {', '.join(missing)}")
    config = dict(config)
    config["source"] = str(config_path)
    return config


def write_tuned_config(path: Path | str | None, config: dict[str, object]) -> Path:
    config_path = Path(path) if path is not None else DEFAULT_TUNED_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return config_path
