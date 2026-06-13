"""Build the Elo-only baseline training dataset for the World Cup predictor.

Takes matches_with_elo.csv (produced by elo.py) and turns it into the
per-team-perspective table the neural network trains on:

  * two rows per match (one from each side's perspective)
  * target = goals scored by that side (capped, for Poisson stability)
  * inputs = team_elo, opp_elo, elo_diff (home-adjusted), is_home
  * scaled copies of inputs, using TRAINING-set statistics only
  * time-based train / val / test split (never random -- avoids leakage,
    and guarantees both rows of a match land in the same split)

The same row format will carry the full feature set later: adding features
means adding columns, nothing else changes.

Usage:
    python make_baseline_dataset.py            # expects matches_with_elo.csv
Outputs:
    baseline_train.csv, baseline_val.csv, baseline_test.csv, scaling.json
"""
from __future__ import annotations

import json

import pandas as pd

# ---- configuration -------------------------------------------------------
MIN_DATE = "1960-01-01"      # drop the Elo burn-in era
VAL_START = "2015-01-01"     # train:  MIN_DATE .. VAL_START
TEST_START = "2020-01-01"    # val:    VAL_START .. TEST_START, test: after
GOAL_CAP = 8                 # cap freak scorelines (31-0 etc.) for stability
HOME_ADVANTAGE = 100.0       # must match the value used in elo.py
SCALED_COLS = ["team_elo", "opp_elo", "elo_diff"]  # is_home is already 0/1
# ---------------------------------------------------------------------------


def explode_to_perspectives(matches: pd.DataFrame) -> pd.DataFrame:
    """Turn one match row into two rows, one per team's perspective.

    elo_diff includes the home-advantage offset for whichever side is at
    home, mirroring how the Elo update itself treats venue.
    """
    home_adv = (~matches["neutral"]).astype(float) * HOME_ADVANTAGE

    home_view = pd.DataFrame(
        {
            "date": matches["date"],
            "tournament": matches["tournament"],
            "team": matches["home_team"],
            "opponent": matches["away_team"],
            "is_home": (~matches["neutral"]).astype(int),
            "team_elo": matches["home_elo_pre"],
            "opp_elo": matches["away_elo_pre"],
            "elo_diff": matches["home_elo_pre"] + home_adv - matches["away_elo_pre"],
            "team_goals": matches["home_score"],
            "opp_goals": matches["away_score"],
        }
    )
    away_view = pd.DataFrame(
        {
            "date": matches["date"],
            "tournament": matches["tournament"],
            "team": matches["away_team"],
            "opponent": matches["home_team"],
            "is_home": 0,  # the away side is never "at home"
            "team_elo": matches["away_elo_pre"],
            "opp_elo": matches["home_elo_pre"],
            "elo_diff": matches["away_elo_pre"] - (matches["home_elo_pre"] + home_adv),
            "team_goals": matches["away_score"],
            "opp_goals": matches["home_score"],
        }
    )
    return (
        pd.concat([home_view, away_view], ignore_index=True)
        .sort_values("date")
        .reset_index(drop=True)
    )


def build(matches_path: str = "matches_with_elo.csv") -> None:
    matches = pd.read_csv(matches_path, parse_dates=["date"])
    matches = matches[matches["date"] >= MIN_DATE]

    rows = explode_to_perspectives(matches)
    rows["team_goals"] = rows["team_goals"].clip(upper=GOAL_CAP)
    rows["opp_goals"] = rows["opp_goals"].clip(upper=GOAL_CAP)

    train = rows[rows["date"] < VAL_START]
    val = rows[(rows["date"] >= VAL_START) & (rows["date"] < TEST_START)]
    test = rows[rows["date"] >= TEST_START]

    # scale with TRAINING statistics only -- val/test must not influence them
    stats = {c: {"mean": train[c].mean(), "std": train[c].std()} for c in SCALED_COLS}
    for split in (train, val, test):
        for c in SCALED_COLS:
            split[f"{c}_scaled"] = (split[c] - stats[c]["mean"]) / stats[c]["std"]

    train.to_csv("baseline_train.csv", index=False)
    val.to_csv("baseline_val.csv", index=False)
    test.to_csv("baseline_test.csv", index=False)
    with open("scaling.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"train: {len(train):>7,} rows  ({train['date'].min().date()} .. {train['date'].max().date()})")
    print(f"val:   {len(val):>7,} rows  ({val['date'].min().date()} .. {val['date'].max().date()})")
    print(f"test:  {len(test):>7,} rows  ({test['date'].min().date()} .. {test['date'].max().date()})")
    print(f"\nmean goals per row (train): {train['team_goals'].mean():.3f}")
    print("scaling stats written to scaling.json")
    print("\nmodel inputs:  team_elo_scaled, opp_elo_scaled, elo_diff_scaled, is_home")
    print("target:        team_goals  (Poisson NLL)")


if __name__ == "__main__":
    build()
