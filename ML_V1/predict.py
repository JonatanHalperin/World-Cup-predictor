"""Predict a specific (future) match with the trained Elo baseline.

Pulls the latest Elo ratings (so it always uses current team strength),
loads the trained model and the training-set scaling, and turns a named
fixture into:
    * each team's expected goals (lambda)
    * win / draw / loss probabilities
    * the most likely exact scoreline and the top few alternatives
    * a rough knockout "advances" probability (splits the draw 50/50)

Prerequisites (run once, in order, from the repo root):
    python -m data_pipeline.elo
    python -m data_pipeline.make_baseline_dataset
    python -m ML_V1.train_baseline      # now also writes baseline_model.pt

Then either run this file (python -m ML_V1.predict) for the built-in examples,
or import predict_match and call it on your own fixtures.

Team names must match the results CSV exactly (e.g. "South Korea", not
"Korea"); a wrong name lists close matches to help you fix it.
"""
from __future__ import annotations

import difflib
import json

import numpy as np
import torch

from data_pipeline.elo import load_results, compute_elo
from ML_V1.train_baseline import PoissonNet, poisson_pmf_grid, GRID

HOME_ADVANTAGE = 100.0  # must match elo.py / make_baseline_dataset.py


def load_model(path: str = "baseline_model.pt") -> PoissonNet:
    model = PoissonNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def load_scaling(path: str = "scaling.json") -> dict:
    with open(path) as f:
        return json.load(f)


def _scale(value: float, stat: dict) -> float:
    return (value - stat["mean"]) / stat["std"]


def _row_features(team_elo: float, opp_elo: float, is_home: bool, scaling: dict) -> list[float]:
    """Build the 4 model inputs for one team's perspective, scaled like training."""
    home_adv = HOME_ADVANTAGE if is_home else 0.0
    elo_diff = team_elo + home_adv - opp_elo
    return [
        _scale(team_elo, scaling["team_elo"]),
        _scale(opp_elo, scaling["opp_elo"]),
        _scale(elo_diff, scaling["elo_diff"]),
        1.0 if is_home else 0.0,
    ]


def _check_name(name: str, ratings: dict) -> None:
    if name not in ratings:
        near = difflib.get_close_matches(name, ratings.keys(), n=5)
        hint = f" Did you mean: {', '.join(near)}?" if near else ""
        raise KeyError(f"'{name}' is not a known team.{hint}")


def predict_match(
    model: PoissonNet,
    ratings: dict,
    scaling: dict,
    team_a: str,
    team_b: str,
    neutral: bool = True,
    home: str | None = None,
) -> dict:
    """Predict one fixture. Set neutral=False and home="<team>" for a host game."""
    _check_name(team_a, ratings)
    _check_name(team_b, ratings)

    a_home = (not neutral) and home == team_a
    b_home = (not neutral) and home == team_b
    x = torch.tensor(
        [
            _row_features(ratings[team_a], ratings[team_b], a_home, scaling),
            _row_features(ratings[team_b], ratings[team_a], b_home, scaling),
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        lam = model(x).numpy()
    lam_a, lam_b = float(lam[0]), float(lam[1])

    # full scoreline grid: joint[i, j] = P(team_a scores i, team_b scores j)
    pa = poisson_pmf_grid(np.array([lam_a]))[0]
    pb = poisson_pmf_grid(np.array([lam_b]))[0]
    joint = np.outer(pa, pb)

    p_a_win = float(np.tril(joint, -1).sum())   # i > j
    p_draw = float(np.trace(joint))             # i == j
    p_b_win = float(np.triu(joint, 1).sum())    # i < j

    i, j = np.unravel_index(joint.argmax(), joint.shape)
    flat = joint.flatten()
    top = [
        (int(r), int(c), float(flat[k]))
        for k in flat.argsort()[::-1][:5]
        for r, c in [np.unravel_index(k, joint.shape)]
    ]

    return {
        "team_a": team_a, "team_b": team_b,
        "lambda_a": lam_a, "lambda_b": lam_b,
        "p_a_win": p_a_win, "p_draw": p_draw, "p_b_win": p_b_win,
        # knockout: no draws -> split the draw mass evenly (rough first pass)
        "p_a_advance": p_a_win + 0.5 * p_draw,
        "most_likely_score": (int(i), int(j)),
        "top_scores": top,
    }


def print_prediction(r: dict) -> None:
    a, b = r["team_a"], r["team_b"]
    print(f"\n{a} vs {b}")
    print(f"  expected goals:  {a} {r['lambda_a']:.2f} - {r['lambda_b']:.2f} {b}")
    print(f"  outcome (90 min): {a} win {r['p_a_win']*100:4.1f}%  |  "
          f"draw {r['p_draw']*100:4.1f}%  |  {b} win {r['p_b_win']*100:4.1f}%")
    print(f"  knockout advance: {a} {r['p_a_advance']*100:4.1f}%  |  "
          f"{b} {(1-r['p_a_advance'])*100:4.1f}%")
    s = r["most_likely_score"]
    print(f"  most likely score: {a} {s[0]}-{s[1]} {b}")
    print("  top scorelines:")
    for ga, gb, p in r["top_scores"]:
        print(f"     {ga}-{gb}   {p*100:4.1f}%")


if __name__ == "__main__":
    print("Loading latest Elo ratings...")
    _, ratings = compute_elo(load_results())
    model = load_model()
    scaling = load_scaling()

    # --- edit these or import predict_match and call your own fixtures ---
    print_prediction(predict_match(model, ratings, scaling, "Brazil", "Argentina"))
    print_prediction(predict_match(model, ratings, scaling, "Spain", "France"))
    # host nation example: USA at home, not a neutral venue
    print_prediction(
        predict_match(model, ratings, scaling, "United States", "Mexico",
                      neutral=False, home="United States")
    )