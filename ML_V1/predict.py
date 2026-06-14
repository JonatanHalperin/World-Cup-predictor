"""Predict a specific (future) match with the trained Elo baseline.

Pulls the latest Elo ratings (so it always uses current team strength),
loads the trained model and the training-set scaling, and turns a named
fixture into:
    * each team's expected goals (lambda)
    * win / draw / loss probabilities
    * the most likely exact scoreline and the top few alternatives
    * a rough knockout "advances" probability (splits the draw 50/50)

Prerequisites (run once, in order, from the repo root):
    python -m data_pipeline.build_data
    python -m data_pipeline.make_baseline_dataset
    python -m ML_V1.train_baseline      # writes baseline_model.pt

Then either run this file (python -m ML_V1.predict) for the built-in examples,
or import predict_match and call it on your own fixtures.

Team names must match the results CSV exactly (e.g. "South Korea", not
"Korea"); a wrong name lists close matches to help you fix it.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path

import numpy as np
import torch

from data_pipeline.elo import load_results, compute_elo
from ML_V1.train_baseline import PoissonNet, poisson_pmf_grid, dc_apply, GRID, FEATURES

_DATA_DIR = Path(__file__).resolve().parent / "data"

HOME_ADVANTAGE = 100.0  # must match elo.py / make_baseline_dataset.py


def load_model(path: Path = _DATA_DIR / "baseline_model.pt") -> PoissonNet:
    model = PoissonNet(n_in=len(FEATURES))
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def load_ensemble(data_dir: Path = _DATA_DIR) -> list[PoissonNet]:
    """Load all ensemble models if available; otherwise return the single model."""
    info_path = data_dir / "ensemble_info.json"
    if info_path.exists():
        with open(info_path) as f:
            n = json.load(f)["n_models"]
        return [load_model(data_dir / f"baseline_model_{i}.pt") for i in range(n)]
    return [load_model()]


def load_scaling(path: Path = _DATA_DIR / "scaling.json") -> dict:
    with open(path) as f:
        return json.load(f)


def load_team_form(path: Path = _DATA_DIR / "team_form.json") -> dict:
    with open(path) as f:
        return json.load(f)


def load_h2h_table(path: Path = _DATA_DIR / "h2h_table.json") -> dict:
    with open(path) as f:
        return json.load(f)


def load_dc_rho(path: Path = _DATA_DIR / "dc_rho.json") -> float:
    with open(path) as f:
        return json.load(f)["rho"]


def _scale(value: float, stat: dict) -> float:
    return (value - stat["mean"]) / stat["std"]


def _row_features(
    team: str,
    opponent: str,
    team_elo: float,
    opp_elo: float,
    is_home: bool,
    is_competitive: bool,
    scaling: dict,
    team_form: dict,
    h2h_table: dict,
) -> list[float]:
    """Build the model inputs for one team's perspective, scaled like training."""
    home_adv = HOME_ADVANTAGE if is_home else 0.0
    elo_diff = team_elo + home_adv - opp_elo

    # Form: use latest known values; fall back to training mean (scales to 0)
    t_form  = team_form.get(team, {})
    op_form = team_form.get(opponent, {})
    gf5  = t_form.get("goals_for_avg_last_5",    scaling["goals_for_avg_last_5"]["mean"])
    ga5  = t_form.get("goals_against_avg_last_5", scaling["goals_against_avg_last_5"]["mean"])
    ogf5 = op_form.get("goals_for_avg_last_5",    scaling["opp_goals_for_avg_last_5"]["mean"])
    oga5 = op_form.get("goals_against_avg_last_5", scaling["opp_goals_against_avg_last_5"]["mean"])
    ppg5  = t_form.get("points_per_game_last_5",   scaling["points_per_game_last_5"]["mean"])
    oppg5 = op_form.get("points_per_game_last_5",  scaling["opp_points_per_game_last_5"]["mean"])

    # H2H: directional goal-diff (team perspective); 0 = neutral history
    h2h = h2h_table.get(f"{team}|{opponent}", 0.0)

    return [
        _scale(team_elo, scaling["team_elo"]),
        _scale(opp_elo,  scaling["opp_elo"]),
        _scale(elo_diff, scaling["elo_diff"]),
        1.0 if is_home else 0.0,
        _scale(gf5,  scaling["goals_for_avg_last_5"]),
        _scale(ga5,  scaling["goals_against_avg_last_5"]),
        _scale(ogf5, scaling["opp_goals_for_avg_last_5"]),
        _scale(oga5, scaling["opp_goals_against_avg_last_5"]),
        _scale(h2h,  scaling["h2h_goal_diff_avg_60y"]),
        1.0 if is_competitive else 0.0,
        _scale(ppg5,  scaling["points_per_game_last_5"]),
        _scale(oppg5, scaling["opp_points_per_game_last_5"]),
    ]


def _check_name(name: str, ratings: dict) -> None:
    if name not in ratings:
        near = difflib.get_close_matches(name, ratings.keys(), n=5)
        hint = f" Did you mean: {', '.join(near)}?" if near else ""
        raise KeyError(f"'{name}' is not a known team.{hint}")


def predict_match(
    model: PoissonNet | list[PoissonNet],
    ratings: dict,
    scaling: dict,
    team_a: str,
    team_b: str,
    neutral: bool = True,
    home: str | None = None,
    is_competitive: bool = True,
    team_form: dict | None = None,
    h2h_table: dict | None = None,
    rho: float = 0.0,
) -> dict:
    """Predict one fixture. Set neutral=False and home="<team>" for a host game.

    model may be a single PoissonNet or a list (ensemble); when a list is given
    the predicted lambdas are averaged across all members before computing probs.
    """
    _check_name(team_a, ratings)
    _check_name(team_b, ratings)
    if team_form is None:
        team_form = {}
    if h2h_table is None:
        h2h_table = {}

    a_home = (not neutral) and home == team_a
    b_home = (not neutral) and home == team_b

    x = torch.tensor(
        [
            _row_features(team_a, team_b, ratings[team_a], ratings[team_b],
                          a_home, is_competitive, scaling, team_form, h2h_table),
            _row_features(team_b, team_a, ratings[team_b], ratings[team_a],
                          b_home, is_competitive, scaling, team_form, h2h_table),
        ],
        dtype=torch.float32,
    )

    models = [model] if isinstance(model, PoissonNet) else model
    lam_a_all, lam_b_all = [], []
    for m in models:
        with torch.no_grad():
            lam = m(x).numpy()
        lam_a_all.append(float(lam[0]))
        lam_b_all.append(float(lam[1]))
    lam_a = float(np.mean(lam_a_all))
    lam_b = float(np.mean(lam_b_all))

    # full scoreline grid: joint[i, j] = P(team_a scores i, team_b scores j)
    pa = poisson_pmf_grid(np.array([lam_a]))[0]
    pb = poisson_pmf_grid(np.array([lam_b]))[0]
    joint = np.outer(pa, pb)
    if rho != 0.0:
        joint = dc_apply(joint, lam_a, lam_b, rho)

    p_a_win = float(np.tril(joint, -1).sum())   # i > j
    p_draw  = float(np.trace(joint))             # i == j
    p_b_win = float(np.triu(joint, 1).sum())     # i < j

    i, j = np.unravel_index(joint.argmax(), joint.shape)
    flat = joint.flatten()
    top = [
        (int(r), int(c), float(flat[k]))
        for k in flat.argsort()[::-1][:5]
        for r, c in [np.unravel_index(k, joint.shape)]
    ]

    # Knockout shootout probability: calibrated from 677 historical shootouts.
    # Higher-Elo team wins at ~53.2%; home team adds ~2pp on top.
    elo_diff = ratings[team_a] - ratings[team_b]   # neutral (no home boost)
    p_a_shoot = 0.5 + 0.032 * float(np.tanh(elo_diff / 400))
    if not neutral:
        if home == team_a:
            p_a_shoot += 0.02
        elif home == team_b:
            p_a_shoot -= 0.02
    p_a_shoot = float(np.clip(p_a_shoot, 0.0, 1.0))

    return {
        "team_a": team_a, "team_b": team_b,
        "lambda_a": lam_a, "lambda_b": lam_b,
        "p_a_win": p_a_win, "p_draw": p_draw, "p_b_win": p_b_win,
        "p_a_shoot": p_a_shoot,
        "p_a_advance": p_a_win + p_draw * p_a_shoot,
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
          f"{b} {(1-r['p_a_advance'])*100:4.1f}%"
          f"  (shootout: {a} {r['p_a_shoot']*100:.0f}%)")
    s = r["most_likely_score"]
    print(f"  most likely score: {a} {s[0]}-{s[1]} {b}")
    print("  top scorelines:")
    for ga, gb, p in r["top_scores"]:
        print(f"     {ga}-{gb}   {p*100:4.1f}%")


if __name__ == "__main__":
    print("Loading latest Elo ratings...")
    _, ratings = compute_elo(load_results())
    models   = load_ensemble()
    scaling  = load_scaling()
    tf       = load_team_form()
    h2h      = load_h2h_table()
    rho      = load_dc_rho()

    kw = dict(team_form=tf, h2h_table=h2h, rho=rho)
    print_prediction(predict_match(models, ratings, scaling, "Brazil", "Argentina", **kw))
    print_prediction(predict_match(models, ratings, scaling, "Spain", "France", **kw))
    print_prediction(
        predict_match(models, ratings, scaling, "United States", "Mexico",
                      neutral=False, home="United States", **kw)
    )
