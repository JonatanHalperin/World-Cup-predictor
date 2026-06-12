"""Build shared and Statistical V1 data tables."""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from elo import compute_elo


DATA_DIR = Path(__file__).resolve().parent
RAW_MARTJ42_DIR = DATA_DIR / "raw" / "martj42"
CACHE_DIR = DATA_DIR / "cache"
TRANSFERMARKT_CACHE_DIR = CACHE_DIR / "transfermarkt"
WORLD_BANK_CACHE_DIR = CACHE_DIR / "world_bank"
REFERENCE_DIR = DATA_DIR / "reference"
PROCESSED_DIR = DATA_DIR / "processed"

MIN_DATE = pd.Timestamp("1960-01-01")
ELO_START_DATE = pd.Timestamp("1900-01-01")
H2H_WINDOW_YEARS = 60
HOME_ADVANTAGE = 100.0
USER_AGENT = "World-Cup-predictor data pipeline"

MARTJ42_BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
MARTJ42_FILES = {
    "results.csv": f"{MARTJ42_BASE_URL}/results.csv",
    "shootouts.csv": f"{MARTJ42_BASE_URL}/shootouts.csv",
    "goalscorers.csv": f"{MARTJ42_BASE_URL}/goalscorers.csv",
    "former_names.csv": f"{MARTJ42_BASE_URL}/former_names.csv",
}

TRANSFERMARKT_BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TRANSFERMARKT_FILES = {
    "players.csv.gz": f"{TRANSFERMARKT_BASE_URL}/players.csv.gz",
    "player_valuations.csv.gz": f"{TRANSFERMARKT_BASE_URL}/player_valuations.csv.gz",
    "countries.csv.gz": f"{TRANSFERMARKT_BASE_URL}/countries.csv.gz",
    "national_teams.csv.gz": f"{TRANSFERMARKT_BASE_URL}/national_teams.csv.gz",
}

WORLD_BANK_POPULATION_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL"
    "?format=json&per_page=20000"
)
WORLD_BANK_COUNTRIES_URL = "https://api.worldbank.org/v2/country?format=json&per_page=400"


def ensure_directories() -> None:
    for path in (
        RAW_MARTJ42_DIR,
        TRANSFERMARKT_CACHE_DIR,
        WORLD_BANK_CACHE_DIR,
        REFERENCE_DIR,
        PROCESSED_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def download(url: str, destination: Path, refresh: bool = False) -> None:
    if destination.exists() and not refresh:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=120) as response, tmp_path.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp_path.replace(destination)


def download_json(url: str, destination: Path, refresh: bool = False) -> None:
    if destination.exists() and not refresh:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=120) as response:
        payload = json.load(response)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def refresh_sources(refresh: bool) -> None:
    ensure_directories()
    for filename, url in MARTJ42_FILES.items():
        download(url, RAW_MARTJ42_DIR / filename, refresh=refresh)
    for filename, url in TRANSFERMARKT_FILES.items():
        download(url, TRANSFERMARKT_CACHE_DIR / filename, refresh=refresh)
    download_json(
        WORLD_BANK_POPULATION_URL,
        WORLD_BANK_CACHE_DIR / "population_total.json",
        refresh=refresh,
    )
    download_json(
        WORLD_BANK_COUNTRIES_URL,
        WORLD_BANK_CACHE_DIR / "countries.json",
        refresh=refresh,
    )


def read_reference_mappings() -> pd.DataFrame:
    path = REFERENCE_DIR / "country_name_mappings.csv"
    if not path.exists():
        return pd.DataFrame(columns=["football_name", "world_bank_name", "transfermarkt_country_name", "notes"])
    return pd.read_csv(path).fillna("")


def mapping_lookup(
    mappings: pd.DataFrame,
    source_column: str,
    target_column: str,
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in mappings.itertuples(index=False):
        source = getattr(row, source_column, "")
        target = getattr(row, target_column, "")
        if source and target:
            lookup[normalize_name(source)] = str(target)
    return lookup


def bool_from_csv(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().eq("TRUE")


def match_key(date_value: object, home_team: object, away_team: object) -> tuple[str, str, str]:
    return (pd.Timestamp(date_value).strftime("%Y-%m-%d"), str(home_team), str(away_team))


def load_martj42_results(as_of_date: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(RAW_MARTJ42_DIR / "results.csv")
    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results["neutral"] = bool_from_csv(results["neutral"])
    results = results.dropna(subset=["date"])

    score_columns = ["home_score", "away_score"]
    played = results[score_columns].notna().all(axis=1)
    not_future = results["date"].dt.date <= as_of_date
    after_min_date = results["date"] >= MIN_DATE

    completed_history = results[played & not_future].copy()
    completed_history["home_score"] = completed_history["home_score"].astype(int)
    completed_history["away_score"] = completed_history["away_score"].astype(int)
    completed_history = completed_history.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    completed_history.insert(0, "source_match_id", np.arange(1, len(completed_history) + 1))

    completed_export = completed_history[completed_history["date"] >= MIN_DATE].copy()
    completed_export = completed_export.reset_index(drop=True)
    completed_export.insert(0, "match_id", np.arange(1, len(completed_export) + 1))

    match_id_by_source = dict(zip(completed_export["source_match_id"], completed_export["match_id"]))
    completed_history["match_id"] = completed_history["source_match_id"].map(match_id_by_source)

    future = results[after_min_date & (~played | ~not_future)].copy()
    future = future.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    future.insert(0, "fixture_id", np.arange(1, len(future) + 1))

    return completed_history, completed_export, future


def add_elo_features(completed_history: pd.DataFrame) -> pd.DataFrame:
    elo_input = completed_history[completed_history["date"] >= ELO_START_DATE].copy()
    elo_table, _ = compute_elo(elo_input, home_advantage=HOME_ADVANTAGE)
    elo_columns = ["source_match_id", "home_elo_pre", "away_elo_pre", "elo_diff"]
    return completed_history.merge(
        elo_table[elo_columns],
        on="source_match_id",
        how="left",
    )


def add_match_context(matches: pd.DataFrame) -> pd.DataFrame:
    matches = matches.copy()
    tournament = matches["tournament"].fillna("").astype(str).str.lower()
    matches["is_friendly"] = tournament.str.contains("friendly", regex=False)
    matches["is_qualifier"] = tournament.str.contains("qualif", regex=False)
    matches["is_world_cup"] = tournament.str.contains("world cup", regex=False) & ~matches["is_qualifier"]
    continental_terms = (
        "uefa euro",
        "copa am",
        "african cup",
        "afc asian cup",
        "gold cup",
        "concacaf championship",
        "confederations",
        "nations league",
        "ofc nations",
    )
    matches["is_continental"] = pd.Series(False, index=matches.index)
    for term in continental_terms:
        matches["is_continental"] = matches["is_continental"] | tournament.str.contains(term, regex=False)
    matches["is_competitive"] = ~matches["is_friendly"]

    context = np.select(
        [
            matches["is_friendly"],
            matches["is_world_cup"],
            matches["is_qualifier"],
            matches["is_continental"],
        ],
        ["friendly", "world_cup", "qualifier", "continental"],
        default="other",
    )
    matches["tournament_type"] = context
    matches["home_is_host"] = [
        normalize_name(team) == normalize_name(country)
        for team, country in zip(matches["home_team"], matches["country"])
    ]
    matches["away_is_host"] = [
        normalize_name(team) == normalize_name(country)
        for team, country in zip(matches["away_team"], matches["country"])
    ]
    return matches


def load_shootout_keys() -> set[tuple[str, str, str]]:
    path = RAW_MARTJ42_DIR / "shootouts.csv"
    if not path.exists():
        return set()
    shootouts = pd.read_csv(path)
    if shootouts.empty:
        return set()
    shootouts["date"] = pd.to_datetime(shootouts["date"], errors="coerce")
    shootouts = shootouts.dropna(subset=["date"])
    return {
        match_key(row.date, row.home_team, row.away_team)
        for row in shootouts.itertuples(index=False)
    }


def attach_shootout_flag(matches: pd.DataFrame) -> pd.DataFrame:
    shootout_keys = load_shootout_keys()
    matches = matches.copy()
    matches["went_to_shootout"] = [
        match_key(row.date, row.home_team, row.away_team) in shootout_keys
        for row in matches.itertuples(index=False)
    ]
    return matches


def result_and_points(goals_for: int, goals_against: int) -> tuple[str, int]:
    if goals_for > goals_against:
        return "win", 3
    if goals_for < goals_against:
        return "loss", 0
    return "draw", 1


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def count_recent(history: list[dict[str, object]], current_date: pd.Timestamp, days: int) -> int:
    cutoff = current_date - pd.Timedelta(days=days)
    return sum(item["date"] >= cutoff for item in history)


def streak_length(history: list[dict[str, object]], streak_type: str) -> int:
    length = 0
    for item in reversed(history):
        result = item["result"]
        if streak_type == "unbeaten" and result != "loss":
            length += 1
        elif streak_type == "winless" and result != "win":
            length += 1
        else:
            break
    return length


def recent_summary(history: list[dict[str, object]], window: int) -> dict[str, float]:
    recent = history[-window:]
    points = [float(item["points"]) for item in recent]
    goals_for = [float(item["goals_for"]) for item in recent]
    goals_against = [float(item["goals_against"]) for item in recent]
    results = [item["result"] for item in recent]
    return {
        f"matches_available_last_{window}": len(recent),
        f"wins_last_{window}": results.count("win"),
        f"draws_last_{window}": results.count("draw"),
        f"losses_last_{window}": results.count("loss"),
        f"points_per_game_last_{window}": mean_or_nan(points),
        f"goals_for_avg_last_{window}": mean_or_nan(goals_for),
        f"goals_against_avg_last_{window}": mean_or_nan(goals_against),
        f"goal_diff_avg_last_{window}": mean_or_nan(
            [gf - ga for gf, ga in zip(goals_for, goals_against)]
        ),
    }


def h2h_summary(
    history: list[dict[str, object]],
    opponent: str,
    current_date: pd.Timestamp,
) -> dict[str, float]:
    cutoff = current_date - pd.DateOffset(years=H2H_WINDOW_YEARS)
    meetings = [
        item
        for item in history
        if item["opponent"] == opponent and item["date"] >= cutoff
    ]
    results = [item["result"] for item in meetings]
    points = [float(item["points"]) for item in meetings]
    goal_diff = [float(item["goals_for"]) - float(item["goals_against"]) for item in meetings]
    return {
        "h2h_matches_60y": len(meetings),
        "h2h_wins_60y": results.count("win"),
        "h2h_draws_60y": results.count("draw"),
        "h2h_losses_60y": results.count("loss"),
        "h2h_goal_diff_avg_60y": mean_or_nan(goal_diff),
        "h2h_points_per_game_60y": mean_or_nan(points),
    }


def perspective_row(
    match: object,
    listed_home_side: bool,
    history: list[dict[str, object]],
) -> dict[str, object]:
    if listed_home_side:
        team = match.home_team
        opponent = match.away_team
        goals_for = int(match.home_score)
        goals_against = int(match.away_score)
        team_is_host = bool(match.home_is_host)
        opponent_is_host = bool(match.away_is_host)
        team_elo_pre = match.home_elo_pre
        opp_elo_pre = match.away_elo_pre
        perspective_elo_diff = match.elo_diff
    else:
        team = match.away_team
        opponent = match.home_team
        goals_for = int(match.away_score)
        goals_against = int(match.home_score)
        team_is_host = bool(match.away_is_host)
        opponent_is_host = bool(match.home_is_host)
        team_elo_pre = match.away_elo_pre
        opp_elo_pre = match.home_elo_pre
        perspective_elo_diff = -match.elo_diff

    result, points = result_and_points(goals_for, goals_against)
    last_match = history[-1] if history else None

    # Every history lookup is made before this match is appended.
    row: dict[str, object] = {
        "match_id": int(match.match_id),
        "date": match.date,
        "year": int(match.date.year),
        "tournament": match.tournament,
        "tournament_type": match.tournament_type,
        "city": match.city,
        "country": match.country,
        "team": team,
        "opponent": opponent,
        "team_elo_pre": team_elo_pre,
        "opp_elo_pre": opp_elo_pre,
        "elo_diff": perspective_elo_diff,
        "home_team": match.home_team,
        "away_team": match.away_team,
        "is_listed_home": int(listed_home_side),
        "is_home": int(listed_home_side and not match.neutral),
        "is_neutral": int(match.neutral),
        "team_is_host": int(team_is_host),
        "opponent_is_host": int(opponent_is_host),
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_diff": goals_for - goals_against,
        "result": result,
        "points": points,
        "went_to_shootout": int(match.went_to_shootout),
        "has_prior_match": int(last_match is not None),
        "missing_days_since_last_match": int(last_match is None),
        "days_since_last_match": (
            int((match.date - last_match["date"]).days) if last_match is not None else math.nan
        ),
        "prev_match_went_to_shootout": (
            int(last_match["went_to_shootout"]) if last_match is not None else math.nan
        ),
        "matches_last_30d": count_recent(history, match.date, 30),
        "matches_last_60d": count_recent(history, match.date, 60),
        "matches_last_365d": count_recent(history, match.date, 365),
        "unbeaten_streak_pre": streak_length(history, "unbeaten"),
        "winless_streak_pre": streak_length(history, "winless"),
        "is_friendly": int(match.is_friendly),
        "is_qualifier": int(match.is_qualifier),
        "is_world_cup": int(match.is_world_cup),
        "is_continental": int(match.is_continental),
        "is_competitive": int(match.is_competitive),
    }
    row.update(recent_summary(history, 5))
    row.update(recent_summary(history, 10))
    h2h = h2h_summary(history, opponent, match.date)
    row.update(h2h)
    row["missing_h2h_60y"] = int(h2h["h2h_matches_60y"] == 0)
    return row


def history_record(match: object, listed_home_side: bool) -> tuple[str, dict[str, object]]:
    if listed_home_side:
        team = match.home_team
        opponent = match.away_team
        goals_for = int(match.home_score)
        goals_against = int(match.away_score)
    else:
        team = match.away_team
        opponent = match.home_team
        goals_for = int(match.away_score)
        goals_against = int(match.home_score)
    result, points = result_and_points(goals_for, goals_against)
    return team, {
        "date": match.date,
        "opponent": opponent,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "result": result,
        "points": points,
        "went_to_shootout": int(match.went_to_shootout),
    }


def build_team_match_features(matches: pd.DataFrame, export_start: pd.Timestamp = MIN_DATE) -> pd.DataFrame:
    histories: dict[str, list[dict[str, object]]] = defaultdict(list)
    rows: list[dict[str, object]] = []

    # Process by date so same-day matches never become prior information.
    for _, group in matches.groupby("date", sort=True):
        pending_updates: list[tuple[str, dict[str, object]]] = []
        for match in group.sort_values("source_match_id").itertuples(index=False):
            if match.date >= export_start:
                rows.append(perspective_row(match, True, histories[match.home_team]))
                rows.append(perspective_row(match, False, histories[match.away_team]))
            pending_updates.append(history_record(match, True))
            pending_updates.append(history_record(match, False))
        for team, record in pending_updates:
            histories[team].append(record)

    features = pd.DataFrame(rows).sort_values(["date", "match_id", "is_listed_home"], ascending=[True, True, False])
    return features.reset_index(drop=True)


def fill_with_statistic(df: pd.DataFrame, column: str, statistic: str) -> None:
    value = df[column].median() if statistic == "median" else df[column].mean()
    if pd.isna(value):
        value = 0
    df[column] = df[column].fillna(value)


def build_model_features_table(team_features: pd.DataFrame) -> pd.DataFrame:
    model = team_features.copy()
    model["missing_days_since_last_match"] = model["days_since_last_match"].isna().astype(int)
    model["missing_h2h_60y"] = model["h2h_matches_60y"].eq(0).astype(int)

    fill_with_statistic(model, "days_since_last_match", "median")
    model["prev_match_went_to_shootout"] = model["prev_match_went_to_shootout"].fillna(0)

    rolling_average_columns = [
        column
        for column in model.columns
        if (
            column.startswith("points_per_game_last_")
            or column.startswith("goals_for_avg_last_")
            or column.startswith("goals_against_avg_last_")
            or column.startswith("goal_diff_avg_last_")
        )
    ]
    for column in rolling_average_columns:
        fill_with_statistic(model, column, "mean")

    model["h2h_goal_diff_avg_60y"] = model["h2h_goal_diff_avg_60y"].fillna(0)
    model["h2h_points_per_game_60y"] = model["h2h_points_per_game_60y"].fillna(1.0)

    # Final guard for numeric columns newly added to the shared model-ready table.
    numeric_columns = model.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        if model[column].isna().any():
            fill_with_statistic(model, column, "mean")
    return model


def build_poisson_training_table(team_features: pd.DataFrame) -> pd.DataFrame:
    leakage_columns = {"result", "points", "goal_diff"}
    columns = [column for column in team_features.columns if column not in leakage_columns]
    table = team_features[columns].copy()

    target_columns = ["goals_for", "goals_against"]
    leading_columns = ["match_id", "date", "team", "opponent", *target_columns]
    ordered = leading_columns + [column for column in table.columns if column not in leading_columns]
    return table[ordered]


def save_matches(completed: pd.DataFrame, future: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = attach_shootout_flag(add_match_context(completed))
    future = add_match_context(future)
    completed.to_csv(PROCESSED_DIR / "matches_1960_completed.csv", index=False)
    future.to_csv(PROCESSED_DIR / "future_fixtures.csv", index=False)
    return completed, future


def load_world_bank_population() -> tuple[pd.DataFrame, dict[str, str], list[dict[str, str]]]:
    population_payload = json.loads((WORLD_BANK_CACHE_DIR / "population_total.json").read_text(encoding="utf-8"))
    countries_payload = json.loads((WORLD_BANK_CACHE_DIR / "countries.json").read_text(encoding="utf-8"))
    population_rows = population_payload[1] if len(population_payload) > 1 else []
    country_rows = countries_payload[1] if len(countries_payload) > 1 else []

    country_meta = pd.DataFrame(
        {
            "countryiso3code": row.get("id", ""),
            "world_bank_name": row.get("name", ""),
            "region": (row.get("region") or {}).get("value", ""),
        }
        for row in country_rows
    )
    country_meta = country_meta[country_meta["region"].ne("Aggregates")]
    valid_iso3 = set(country_meta["countryiso3code"])
    wb_name_by_norm = {
        normalize_name(row.world_bank_name): row.world_bank_name
        for row in country_meta.itertuples(index=False)
    }

    population = pd.DataFrame(
        {
            "world_bank_name": (row.get("country") or {}).get("value", ""),
            "countryiso3code": row.get("countryiso3code", ""),
            "year": pd.to_numeric(row.get("date"), errors="coerce"),
            "population": row.get("value"),
        }
        for row in population_rows
    )
    population = population[population["countryiso3code"].isin(valid_iso3)].copy()
    population["year"] = population["year"].astype("Int64")
    population["population"] = pd.to_numeric(population["population"], errors="coerce")
    population = population.dropna(subset=["year", "population"])
    population["year"] = population["year"].astype(int)
    return population, wb_name_by_norm, []


def build_team_year_population(
    team_names: list[str],
    mappings: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    population, wb_name_by_norm, diagnostics = load_world_bank_population()
    football_to_wb = mapping_lookup(mappings, "football_name", "world_bank_name")
    records: list[pd.DataFrame] = []

    for team in team_names:
        requested_wb_name = football_to_wb.get(normalize_name(team), team)
        wb_name = wb_name_by_norm.get(normalize_name(requested_wb_name))
        if not wb_name:
            diagnostics.append(
                {
                    "source": "world_bank",
                    "source_name": team,
                    "issue": "no_world_bank_mapping",
                    "suggested_action": "Add or fix data_pipeline/reference/country_name_mappings.csv.",
                }
            )
            continue
        team_population = population[population["world_bank_name"].eq(wb_name)].copy()
        team_population["football_team"] = team
        records.append(team_population)

    if not records:
        return pd.DataFrame(), diagnostics

    result = pd.concat(records, ignore_index=True)
    result = result[result["year"] >= MIN_DATE.year].copy()
    result = result[
        ["football_team", "world_bank_name", "countryiso3code", "year", "population"]
    ].sort_values(["football_team", "year"])
    return result.reset_index(drop=True), diagnostics


def position_group(position: object) -> str:
    text = normalize_name(position)
    if "goalkeeper" in text:
        return "goalkeeper"
    if "attack" in text:
        return "attack"
    if "midfield" in text:
        return "midfield"
    if "defender" in text or "defence" in text or "defense" in text:
        return "defense"
    return "other"


def build_team_year_market_value_proxy(
    team_names: list[str],
    mappings: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    diagnostics: list[dict[str, str]] = []
    team_by_norm = {normalize_name(team): team for team in team_names}
    tm_to_football = mapping_lookup(mappings, "transfermarkt_country_name", "football_name")

    players = pd.read_csv(
        TRANSFERMARKT_CACHE_DIR / "players.csv.gz",
        usecols=["player_id", "country_of_citizenship", "position"],
    )
    valuations = pd.read_csv(
        TRANSFERMARKT_CACHE_DIR / "player_valuations.csv.gz",
        usecols=["player_id", "date", "market_value_in_eur"],
    )
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    valuations["market_value_in_eur"] = pd.to_numeric(
        valuations["market_value_in_eur"],
        errors="coerce",
    )
    valuations = valuations.dropna(subset=["date", "market_value_in_eur"])
    valuations = valuations.merge(players, on="player_id", how="left")
    valuations = valuations.dropna(subset=["country_of_citizenship"])
    valuations["position_group"] = valuations["position"].map(position_group)

    def resolve_team(country_name: str) -> str | None:
        normalized = normalize_name(country_name)
        if normalized in tm_to_football:
            return tm_to_football[normalized]
        return team_by_norm.get(normalized)

    valuations["football_team"] = valuations["country_of_citizenship"].map(resolve_team)
    unmatched = (
        valuations[valuations["football_team"].isna()]["country_of_citizenship"]
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    for name in unmatched:
        diagnostics.append(
            {
                "source": "transfermarkt",
                "source_name": str(name),
                "issue": "no_football_team_mapping",
                "suggested_action": "Add or fix data_pipeline/reference/country_name_mappings.csv if this country is needed.",
            }
        )

    valuations = valuations.dropna(subset=["football_team"]).sort_values(["player_id", "date"])
    if valuations.empty:
        return pd.DataFrame(), diagnostics

    min_year = max(MIN_DATE.year, int(valuations["date"].dt.year.min()))
    max_year = int(valuations["date"].dt.year.max())
    rows: list[pd.DataFrame] = []
    for year in range(min_year, max_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        snapshot = valuations[valuations["date"] <= cutoff].drop_duplicates("player_id", keep="last")
        if snapshot.empty:
            continue

        grouped = snapshot.groupby("football_team", as_index=False).agg(
            player_count=("player_id", "nunique"),
            total_market_value_eur=("market_value_in_eur", "sum"),
            avg_market_value_eur=("market_value_in_eur", "mean"),
        )
        by_position = snapshot.pivot_table(
            index="football_team",
            columns="position_group",
            values="market_value_in_eur",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        for column in ("attack", "midfield", "defense", "goalkeeper", "other"):
            if column not in by_position:
                by_position[column] = 0
        by_position = by_position.rename(
            columns={
                "attack": "attack_market_value_eur",
                "midfield": "midfield_market_value_eur",
                "defense": "defense_market_value_eur",
                "goalkeeper": "goalkeeper_market_value_eur",
                "other": "other_market_value_eur",
            }
        )
        year_table = grouped.merge(by_position, on="football_team", how="left")
        year_table["year"] = year
        rows.append(year_table)

    result = pd.concat(rows, ignore_index=True)
    result["log_total_market_value"] = np.log(result["total_market_value_eur"].where(result["total_market_value_eur"] > 0))
    result = result.sort_values(["football_team", "year"]).reset_index(drop=True)
    return result, diagnostics


def build_country_development_inputs(
    population: pd.DataFrame,
    market_values: pd.DataFrame,
) -> pd.DataFrame:
    if population.empty or market_values.empty:
        return pd.DataFrame()

    popularity_path = REFERENCE_DIR / "football_popularity.csv"
    popularity = pd.read_csv(popularity_path) if popularity_path.exists() else pd.DataFrame()
    defaults = popularity[popularity["football_name"].eq("__default__")]
    default_class = "unknown"
    default_score = 0
    if not defaults.empty:
        default_class = str(defaults.iloc[0]["popularity_class"])
        default_score = float(defaults.iloc[0]["popularity_score"])
    popularity = popularity[~popularity["football_name"].eq("__default__")].copy()

    inputs = market_values.merge(
        population[["football_team", "year", "population"]],
        on=["football_team", "year"],
        how="inner",
    )
    inputs = inputs[inputs["population"].gt(0) & inputs["total_market_value_eur"].gt(0)].copy()
    if inputs.empty:
        return inputs

    inputs["log_population"] = np.log(inputs["population"])
    inputs["log_talent_output"] = np.log(inputs["total_market_value_eur"])
    inputs = inputs.merge(
        popularity[["football_name", "popularity_class", "popularity_score"]],
        left_on="football_team",
        right_on="football_name",
        how="left",
    )
    inputs["popularity_class"] = inputs["popularity_class"].fillna(default_class)
    inputs["popularity_score"] = inputs["popularity_score"].fillna(default_score)
    inputs = inputs.drop(columns=["football_name"], errors="ignore")
    return inputs.sort_values(["football_team", "year"]).reset_index(drop=True)


def write_diagnostics(rows: list[dict[str, str]]) -> None:
    diagnostics = pd.DataFrame(rows)
    if diagnostics.empty:
        diagnostics = pd.DataFrame(columns=["source", "source_name", "issue", "suggested_action"])
    diagnostics.to_csv(PROCESSED_DIR / "name_mapping_diagnostics.csv", index=False)


def write_processed_outputs(as_of_date: date) -> None:
    completed_history, completed, future = load_martj42_results(as_of_date)
    completed_history = add_elo_features(completed_history)
    completed = completed.merge(
        completed_history[["source_match_id", "home_elo_pre", "away_elo_pre", "elo_diff"]],
        on="source_match_id",
        how="left",
    )
    completed_history = attach_shootout_flag(add_match_context(completed_history))
    completed, _ = save_matches(completed, future)

    team_features = build_team_match_features(completed_history)
    team_features.to_csv(PROCESSED_DIR / "team_match_features.csv", index=False)

    model_features = build_model_features_table(team_features)
    model_features.to_csv(PROCESSED_DIR / "model_features_long.csv", index=False)

    poisson_training = build_poisson_training_table(model_features)
    poisson_training.to_csv(PROCESSED_DIR / "poisson_training_long.csv", index=False)

    team_names = sorted(set(completed["home_team"]) | set(completed["away_team"]))
    mappings = read_reference_mappings()
    diagnostics: list[dict[str, str]] = []

    population, population_diagnostics = build_team_year_population(team_names, mappings)
    diagnostics.extend(population_diagnostics)
    population.to_csv(PROCESSED_DIR / "team_year_population.csv", index=False)

    market_values, market_diagnostics = build_team_year_market_value_proxy(team_names, mappings)
    diagnostics.extend(market_diagnostics)
    market_values.to_csv(PROCESSED_DIR / "team_year_market_value_proxy.csv", index=False)

    development_inputs = build_country_development_inputs(population, market_values)
    development_inputs.to_csv(PROCESSED_DIR / "country_development_inputs.csv", index=False)

    write_diagnostics(diagnostics)


def check_processed_outputs() -> None:
    required_files = [
        "matches_1960_completed.csv",
        "future_fixtures.csv",
        "team_match_features.csv",
        "model_features_long.csv",
        "poisson_training_long.csv",
        "team_year_population.csv",
        "team_year_market_value_proxy.csv",
        "country_development_inputs.csv",
        "name_mapping_diagnostics.csv",
    ]
    missing = [name for name in required_files if not (PROCESSED_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed outputs: {', '.join(missing)}")

    loaded = {
        name: pd.read_csv(PROCESSED_DIR / name)
        for name in required_files
    }
    matches = loaded["matches_1960_completed.csv"]
    team_features = loaded["team_match_features.csv"]
    model_features = loaded["model_features_long.csv"]
    poisson = loaded["poisson_training_long.csv"]
    future = loaded["future_fixtures.csv"]

    if pd.to_datetime(matches["date"]).min() < MIN_DATE:
        raise ValueError("Training matches include rows before 1960.")
    if matches[["home_score", "away_score"]].isna().any().any():
        raise ValueError("Completed training matches include missing scores.")
    if len(team_features) != len(matches) * 2:
        raise ValueError("Each completed match must produce exactly two team rows.")
    if len(model_features) != len(team_features):
        raise ValueError("Model-ready rows must match team feature rows.")
    if "goals_for" not in poisson:
        raise ValueError("Poisson training table is missing the goals_for target.")
    required_match_elo = {"home_elo_pre", "away_elo_pre", "elo_diff"}
    if not required_match_elo.issubset(matches.columns):
        missing_elo = ", ".join(sorted(required_match_elo - set(matches.columns)))
        raise ValueError(f"Completed matches are missing Elo columns: {missing_elo}")
    if matches[list(required_match_elo)].isna().any().any():
        raise ValueError("Completed match Elo columns contain missing values.")
    feature_tables = {
        "team_match_features": team_features,
        "model_features_long": model_features,
        "poisson_training_long": poisson,
    }
    for name, table in feature_tables.items():
        if pd.to_datetime(table["date"]).min() < MIN_DATE:
            raise ValueError(f"{name} includes rows before 1960.")
        required_elo = {"team_elo_pre", "opp_elo_pre", "elo_diff"}
        if not required_elo.issubset(table.columns):
            missing_elo = ", ".join(sorted(required_elo - set(table.columns)))
            raise ValueError(f"{name} is missing Elo columns: {missing_elo}")
        if table[list(required_elo)].isna().any().any():
            raise ValueError(f"{name} Elo columns contain missing values.")
        if any(column.startswith("h2h_") and column.endswith("_20y") for column in table.columns):
            raise ValueError(f"{name} still contains old 20-year head-to-head columns.")
        required_h2h = {"h2h_matches_60y", "h2h_goal_diff_avg_60y", "h2h_points_per_game_60y"}
        if not required_h2h.issubset(table.columns):
            missing_h2h = ", ".join(sorted(required_h2h - set(table.columns)))
            raise ValueError(f"{name} is missing 60-year head-to-head columns: {missing_h2h}")
    known_prior = team_features["days_since_last_match"].dropna()
    if (known_prior <= 0).any():
        raise ValueError("History features include same-day or future matches.")
    numeric_model = model_features.select_dtypes(include=["number"])
    if numeric_model.isna().any().any():
        missing_columns = numeric_model.columns[numeric_model.isna().any()].tolist()
        raise ValueError(f"Model-ready numeric columns contain missing values: {missing_columns}")
    numeric_poisson = poisson.select_dtypes(include=["number"])
    if numeric_poisson.isna().any().any():
        missing_columns = numeric_poisson.columns[numeric_poisson.isna().any()].tolist()
        raise ValueError(f"Poisson numeric columns contain missing values: {missing_columns}")
    if not future.empty:
        future_dates = pd.to_datetime(future["date"], errors="coerce")
        has_missing_score = future[["home_score", "away_score"]].isna().any(axis=1)
        is_future_date = future_dates.dt.date > date.today()
        if not (has_missing_score | is_future_date).all():
            raise ValueError("Future fixture file contains completed non-future matches.")

    print("All processed data checks passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build statistical V1 data CSVs.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download fresh source data before building processed outputs.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate existing processed outputs and exit.",
    )
    parser.add_argument(
        "--as-of-date",
        default=date.today().isoformat(),
        help="Latest date allowed in training data, default: today.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    as_of_date = date.fromisoformat(args.as_of_date)

    if args.check and not args.refresh:
        check_processed_outputs()
        return

    refresh_sources(refresh=args.refresh)
    write_processed_outputs(as_of_date=as_of_date)

    if args.check:
        check_processed_outputs()
    else:
        print(f"Processed data written to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
