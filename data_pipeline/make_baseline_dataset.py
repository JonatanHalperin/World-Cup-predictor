"""Build the ML training dataset from the shared processed features.

Reads model_features_long.csv (produced by build_data.py), which already has
one row per team-perspective and pre-imputed feature columns.  Adds opponent
form and rest-day columns via a self-join on match_id, z-score-scales all
continuous features using training-set statistics only, and writes:
    ML_V1/data/baseline_{train,val,test}.csv
    ML_V1/data/scaling.json
    ML_V1/data/team_form.json   -- latest form values + last_game_date per team
    ML_V1/data/h2h_table.json   -- latest H2H goal-diff per team pair

Usage:
    python -m data_pipeline.make_baseline_dataset
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parent.parent / "ML_V1" / "data"
_PIPELINE_DIR = Path(__file__).resolve().parent

# ---- configuration -------------------------------------------------------
MIN_DATE = "1960-01-01"
VAL_START = "2015-01-01"    # train: MIN_DATE .. VAL_START
TEST_START = "2020-01-01"   # val:   VAL_START .. TEST_START, test: after
GOAL_CAP = 8                # cap freak scorelines for Poisson stability

# Continuous features that get z-score normalised (training statistics only)
SCALED_COLS = [
    "team_elo", "opp_elo", "elo_diff",
    "goals_for_avg_last_5", "goals_against_avg_last_5",
    "opp_goals_for_avg_last_5", "opp_goals_against_avg_last_5",
    "h2h_goal_diff_avg_60y",
    "points_per_game_last_5", "opp_points_per_game_last_5",
]
# Binary features: is_home, is_competitive — already 0/1, no scaling needed
# ---------------------------------------------------------------------------


def build(features_path: Path | None = None) -> None:
    if features_path is None:
        features_path = _PIPELINE_DIR / "processed" / "model_features_long.csv"

    df = pd.read_csv(features_path, parse_dates=["date"])
    df = df[df["date"] >= MIN_DATE].copy()

    # Align to internal column convention used elsewhere in the ML pipeline
    df = df.rename(columns={
        "team_elo_pre": "team_elo",
        "opp_elo_pre":  "opp_elo",
        "goals_for":    "team_goals",
        "goals_against":"opp_goals",
    })

    # Self-join on match_id+opponent to bring the opponent's recent form into
    # the same row (needed so the model sees both attack/defence signals)
    opp_cols = (
        df[["match_id", "team", "goals_for_avg_last_5", "goals_against_avg_last_5",
            "points_per_game_last_5"]]
        .rename(columns={
            "team": "opponent",
            "goals_for_avg_last_5":    "opp_goals_for_avg_last_5",
            "goals_against_avg_last_5":"opp_goals_against_avg_last_5",
            "points_per_game_last_5":  "opp_points_per_game_last_5",
        })
    )
    df = df.merge(opp_cols, on=["match_id", "opponent"])

    df["team_goals"] = df["team_goals"].clip(upper=GOAL_CAP)
    df["opp_goals"]  = df["opp_goals"].clip(upper=GOAL_CAP)

    keep = [
        "date", "tournament", "team", "opponent",
        "team_elo", "opp_elo", "elo_diff", "is_home", "is_competitive", "is_world_cup",
        "goals_for_avg_last_5", "goals_against_avg_last_5",
        "opp_goals_for_avg_last_5", "opp_goals_against_avg_last_5",
        "h2h_goal_diff_avg_60y",
        "points_per_game_last_5", "opp_points_per_game_last_5",
        "team_goals", "opp_goals",
    ]
    rows = df[keep].sort_values("date").reset_index(drop=True)

    train = rows[rows["date"] < VAL_START]
    val   = rows[(rows["date"] >= VAL_START) & (rows["date"] < TEST_START)]
    test  = rows[rows["date"] >= TEST_START]

    # Compute scaling stats on training set only; apply to all splits
    stats = {
        c: {"mean": float(train[c].mean()), "std": float(train[c].std())}
        for c in SCALED_COLS
    }
    for split in (train, val, test):
        for c in SCALED_COLS:
            split[f"{c}_scaled"] = (split[c] - stats[c]["mean"]) / stats[c]["std"]

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(_DATA_DIR / "baseline_train.csv", index=False)
    val.to_csv(  _DATA_DIR / "baseline_val.csv",   index=False)
    test.to_csv( _DATA_DIR / "baseline_test.csv",  index=False)
    with open(_DATA_DIR / "scaling.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Persist latest form per team for use by predict.py at inference time
    latest_by_team = rows.sort_values("date").groupby("team").last()
    team_form = {
        team: {
            "goals_for_avg_last_5":    float(row["goals_for_avg_last_5"]),
            "goals_against_avg_last_5":float(row["goals_against_avg_last_5"]),
            "points_per_game_last_5":  float(row["points_per_game_last_5"]),
            "last_game_date":          str(row["date"].date()),
        }
        for team, row in latest_by_team.iterrows()
    }
    with open(_DATA_DIR / "team_form.json", "w") as f:
        json.dump(team_form, f, indent=2)

    # Persist H2H goal-diff per (team, opponent) pair for inference
    h2h_df = (
        df[["date", "team", "opponent", "h2h_goal_diff_avg_60y"]]
        .sort_values("date")
        .groupby(["team", "opponent"])
        .last()
        .reset_index()
    )
    h2h_table = {
        f"{row['team']}|{row['opponent']}": float(row["h2h_goal_diff_avg_60y"])
        for _, row in h2h_df.iterrows()
    }
    with open(_DATA_DIR / "h2h_table.json", "w") as f:
        json.dump(h2h_table, f, indent=2)

    print(f"train: {len(train):>7,} rows  ({train['date'].min().date()} .. {train['date'].max().date()})")
    print(f"val:   {len(val):>7,} rows  ({val['date'].min().date()} .. {val['date'].max().date()})")
    print(f"test:  {len(test):>7,} rows  ({test['date'].min().date()} .. {test['date'].max().date()})")
    print(f"\nmean goals per row (train): {train['team_goals'].mean():.3f}")
    print(f"scaling stats   -> scaling.json")
    print(f"team form       -> team_form.json  ({len(team_form)} teams, includes last_game_date)")
    print(f"H2H table       -> h2h_table.json  ({len(h2h_table)} pairs)")
    feature_cols = [f"{c}_scaled" for c in SCALED_COLS] + ["is_home", "is_competitive"]
    print(f"\nmodel inputs ({len(feature_cols)}): {', '.join(feature_cols)}")
    print("target:        team_goals  (Poisson NLL)")


if __name__ == "__main__":
    build()
