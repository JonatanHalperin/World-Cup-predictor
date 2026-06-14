"""Predict every scheduled World Cup fixture with the trained Elo baseline.

The results feed already contains upcoming World Cup games as rows with blank
(NA) scores. This script pulls those, predicts each with the trained model,
prints a readable slate, and saves world_cup_predictions.csv.

Prerequisites (run once, in order, from the repo root):
    python -m data_pipeline.elo
    python -m data_pipeline.make_baseline_dataset
    python -m ML_V1.train_baseline
Then:
    python -m ML_V1.predict_fixtures

Respects the feed's venue info: host-nation games (neutral = FALSE) get the
home-advantage bump; everything else is treated as neutral, as World Cup
games should be.
"""
from __future__ import annotations

import pandas as pd

from data_pipeline.elo import RESULTS_URL, compute_elo, load_results
from ML_V1.predict import load_ensemble, load_scaling, load_team_form, load_h2h_table, load_dc_rho, predict_match


def load_scheduled_world_cup(source: str = RESULTS_URL) -> pd.DataFrame:
    """Rows for unplayed World Cup finals matches (NA scores)."""
    df = pd.read_csv(source)
    df["date"] = pd.to_datetime(df["date"])
    unplayed = df["home_score"].isna() | df["away_score"].isna()
    t = df["tournament"].str.lower()
    is_world_cup = t.str.contains("fifa world cup") & ~t.str.contains("qualif")
    fix = df[unplayed & is_world_cup].copy()
    fix["neutral"] = fix["neutral"].astype(str).str.upper().eq("TRUE")
    return fix.sort_values("date").reset_index(drop=True)


def main() -> None:
    print("Loading latest Elo ratings...")
    _, ratings = compute_elo(load_results())
    models    = load_ensemble()
    scaling   = load_scaling()
    team_form = load_team_form()
    h2h_table = load_h2h_table()
    rho       = load_dc_rho()

    fixtures = load_scheduled_world_cup()
    if fixtures.empty:
        print("No scheduled World Cup fixtures found in the feed right now.")
        return

    print(f"Predicting {len(fixtures)} scheduled World Cup fixtures...\n")
    rows, current_date, skipped = [], None, 0

    for f in fixtures.itertuples(index=False):
        home_team = None if f.neutral else f.home_team
        try:
            r = predict_match(
                models, ratings, scaling,
                f.home_team, f.away_team, neutral=f.neutral, home=home_team,
                is_competitive=True, team_form=team_form, h2h_table=h2h_table, rho=rho,
            )
        except KeyError:
            skipped += 1  # placeholder team (e.g. "Winner Group A") not yet known
            continue

        # print a date header whenever the day changes
        day = f.date.date()
        if day != current_date:
            current_date = day
            print(f"\n{day:%A %d %B %Y}")

        a, b = r["team_a"], r["team_b"]
        s = r["most_likely_score"]
        fav = a if r["p_a_win"] >= r["p_b_win"] else b
        fav_pct = max(r["p_a_win"], r["p_b_win"]) * 100
        venue = "" if f.neutral else f"  (host: {f.home_team})"
        print(f"  {a:>16} {s[0]}-{s[1]} {b:<16}{venue}")
        print(f"  {'':>16}   {a} {r['p_a_win']*100:4.1f}% / draw "
              f"{r['p_draw']*100:4.1f}% / {b} {r['p_b_win']*100:4.1f}%   "
              f"-> {fav} favoured ({fav_pct:.0f}%)")

        rows.append(
            {
                "date": day,
                "team_a": a, "team_b": b,
                "exp_goals_a": round(r["lambda_a"], 2),
                "exp_goals_b": round(r["lambda_b"], 2),
                "p_a_win": round(r["p_a_win"], 3),
                "p_draw": round(r["p_draw"], 3),
                "p_b_win": round(r["p_b_win"], 3),
                "p_a_advance": round(r["p_a_advance"], 3),
                "most_likely_score": f"{s[0]}-{s[1]}",
                "favourite": fav,
            }
        )

    pd.DataFrame(rows).to_csv("world_cup_predictions.csv", index=False)
    print(f"\nSaved {len(rows)} predictions to world_cup_predictions.csv")
    if skipped:
        print(f"({skipped} fixtures skipped - teams not yet decided in the feed)")


if __name__ == "__main__":
    main()