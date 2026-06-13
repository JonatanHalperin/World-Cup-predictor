"""Compute rolling pre-match Elo ratings for international football.

Implements the eloratings.net formula (Lange, CC BY-SA 4.0) and produces, for
every match, the rating each team carried *into* the game -- which is exactly
the leak-free feature you want for a match-prediction model.

Data source (updated ~daily, no auth needed):
    https://raw.githubusercontent.com/martj42/international_results/master/results.csv

Typical use:
    from elo import load_results, compute_elo
    matches = load_results()                  # pulls the latest CSV
    feature_table, ratings = compute_elo(matches)
    feature_table.to_csv("matches_with_elo.csv", index=False)
    # `ratings` is a dict {team: current_elo} after the most recent played match
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)


def load_results(source: str = RESULTS_URL) -> pd.DataFrame:
    """Load match results from a URL or local path and keep only played games.

    Scheduled fixtures (NA scores) are dropped here because they can't update
    ratings -- but keep the raw file around separately if you want them as a
    list of games still to predict.
    """
    df = pd.read_csv(source)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    # the CSV stores the neutral flag as the strings "TRUE"/"FALSE"
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    return df.sort_values("date").reset_index(drop=True)


def tournament_weight(tournament: str) -> float:
    """eloratings.net K-factor by competition importance.

    This keyword mapping is a pragmatic approximation of eloratings.net's
    categories -- tune it against their exact definitions if you want a 1:1
    match. Higher K = a single result moves the rating more.
    """
    t = tournament.lower()
    if "friendly" in t:
        return 20.0
    is_qualifier = "qualif" in t
    if "world cup" in t:
        return 40.0 if is_qualifier else 60.0
    # continental championships (final tournaments) and the Confederations Cup
    continental = (
        "uefa euro", "copa am", "african cup", "afc asian cup",
        "gold cup", "concacaf championship", "confederations",
        "nations league", "ofc nations",
    )
    if any(c in t for c in continental):
        return 40.0 if is_qualifier else 50.0
    # everything else: minor cups, regional tournaments, etc.
    return 30.0


def goal_diff_multiplier(margin: int) -> float:
    """eloratings.net G: bigger wins count for more, with diminishing returns."""
    n = abs(margin)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11 + n) / 8.0


def expected_score(rating_for: float, rating_against: float) -> float:
    """Win expectancy We for `rating_for`, after any home-advantage is folded in."""
    return 1.0 / (1.0 + 10.0 ** (-(rating_for - rating_against) / 400.0))


def compute_elo(
    matches: pd.DataFrame,
    base_rating: float = 1500.0,
    home_advantage: float = 100.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Roll the Elo recurrence forward over every match in date order.

    Returns
    -------
    feature_table : DataFrame
        One row per match with the PRE-match rating of each side plus the
        signed Elo difference (home-advantage included). Drop-in model features.
    ratings : dict
        Each team's rating after the final match -- the "current" table.
    """
    ratings: dict[str, float] = defaultdict(lambda: base_rating)
    rows = []

    for m in matches.itertuples(index=False):
        r_home = ratings[m.home_team]
        r_away = ratings[m.away_team]
        adv = 0.0 if m.neutral else home_advantage

        # expected and actual result, from the home team's perspective
        we_home = expected_score(r_home + adv, r_away)
        if m.home_score > m.away_score:
            w_home = 1.0
        elif m.home_score < m.away_score:
            w_home = 0.0
        else:
            w_home = 0.5

        k = tournament_weight(m.tournament)
        g = goal_diff_multiplier(m.home_score - m.away_score)
        delta = k * g * (w_home - we_home)

        row = {
            "date": m.date,
            "tournament": m.tournament,
            "neutral": m.neutral,
            "home_team": m.home_team,
            "away_team": m.away_team,
            "home_score": m.home_score,
            "away_score": m.away_score,
            "home_elo_pre": round(r_home, 2),
            "away_elo_pre": round(r_away, 2),
            # signed strength gap going into the match (your headline feature)
            "elo_diff": round((r_home + adv) - r_away, 2),
        }
        for identifier in ("source_match_id", "match_id"):
            if hasattr(m, identifier):
                row[identifier] = getattr(m, identifier)
        rows.append(row)

        # zero-sum update: home gains exactly what away loses
        ratings[m.home_team] = r_home + delta
        ratings[m.away_team] = r_away - delta

    return pd.DataFrame(rows), dict(ratings)


def current_ratings_table(ratings: dict[str, float]) -> pd.DataFrame:
    """Sort the final ratings into a readable leaderboard."""
    return (
        pd.DataFrame(ratings.items(), columns=["team", "elo"])
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    matches = load_results()
    feature_table, ratings = compute_elo(matches)
    feature_table.to_csv("matches_with_elo.csv", index=False)
    print(f"Computed Elo over {len(feature_table):,} played matches")
    print(f"Latest match: {feature_table['date'].max().date()}\n")
    print("Top 15 teams by current Elo:")
    print(current_ratings_table(ratings).head(15).to_string(index=False))
