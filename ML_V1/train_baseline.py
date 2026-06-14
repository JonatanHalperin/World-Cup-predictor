"""Train the Elo-only Poisson baseline for the World Cup predictor.

Reads the CSVs from make_baseline_dataset.py and trains a small neural net
that predicts a goal rate (lambda) for one team from its Elo position. Both
teams' lambdas together define a full scoreline distribution, from which we
derive win/draw/loss probabilities and the most likely exact score.

Reports three benchmark metrics on the test set, each next to a "dumb floor"
(predict every team's historical average lambda) so you can see the model
clear the floor:
    1. Poisson NLL on goals          (the training objective)
    2. log-loss on derived W/D/L     (probabilistic outcome quality)
    3. exact-scoreline hit rate      (hardest target)

Run after: python elo.py && python make_baseline_dataset.py
Needs: torch, pandas, numpy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_DATA_DIR = Path(__file__).resolve().parent / "data"

VAL_START = "2015-01-01"    # must match make_baseline_dataset.py

FEATURES = [
    "team_elo_scaled", "opp_elo_scaled", "elo_diff_scaled", "is_home",
    "goals_for_avg_last_5_scaled", "goals_against_avg_last_5_scaled",
    "opp_goals_for_avg_last_5_scaled", "opp_goals_against_avg_last_5_scaled",
    "h2h_goal_diff_avg_60y_scaled",
    "is_competitive",
    "points_per_game_last_5_scaled", "opp_points_per_game_last_5_scaled",
]
TARGET = "team_goals"
GRID = 9            # goal values 0..8, matching the GOAL_CAP in the dataset
EPOCHS = 150
PATIENCE = 20       # stop if val NLL doesn't improve for this many epochs
BATCH = 512
N_ENSEMBLE = 5      # number of models trained with different seeds and averaged


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class PoissonNet(nn.Module):
    """n inputs -> two hidden layers with dropout -> softplus, output lambda > 0."""

    def __init__(self, n_in: int = len(FEATURES), hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_xy(path: str) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    df = pd.read_csv(path, parse_dates=["date"])
    x = torch.tensor(df[FEATURES].to_numpy(), dtype=torch.float32)
    y = torch.tensor(df[TARGET].to_numpy(), dtype=torch.float32)
    return x, y, df


def sample_weights(df: pd.DataFrame, cutoff_date: str, decay: float = 0.06) -> torch.Tensor:
    """Exponential recency decay × tournament importance.

    decay=0.08  → half-life ~8.7 years (recent data weighted ~3× more than data
    from 30 years ago).  World Cup rows get 2× extra weight; other competitive 1.5×.
    Weights are normalised to mean=1 so the effective learning rate is unchanged.
    """
    age_yr = (pd.Timestamp(cutoff_date) - df["date"]).dt.days.to_numpy() / 365.25
    time_w = np.exp(-decay * np.clip(age_yr, 0, None))
    tourn_w = np.where(df["is_world_cup"].to_numpy() > 0, 2.0,
                       np.where(df["is_competitive"].to_numpy() > 0, 1.5, 1.0))
    w = (time_w * tourn_w).astype(np.float32)
    w /= w.mean()
    return torch.tensor(w)


# --------------------------------------------------------------------------
# evaluation helpers
# --------------------------------------------------------------------------
def poisson_pmf_grid(lams: np.ndarray, k: int = GRID) -> np.ndarray:
    """P(goals = 0..k-1) for each lambda. Returns shape (len(lams), k)."""
    ks = np.arange(k)
    log_fact = np.array([np.sum(np.log(np.arange(1, i + 1))) for i in ks])
    # log pmf = -lam + ks*log(lam) - log(ks!), then exponentiate
    log_pmf = (
        -lams[:, None]
        + ks[None, :] * np.log(lams[:, None])
        - log_fact[None, :]
    )
    return np.exp(log_pmf)


def match_table(df: pd.DataFrame, lams: np.ndarray) -> pd.DataFrame:
    """Pair the two perspective-rows of each match back into one row.

    Uses an alphabetical canonical ordering of the two teams so each match
    collapses to a single row carrying both lambdas and both actual scores.
    """
    d = df.copy()
    d["lam"] = lams
    d["t_lo"] = d[["team", "opponent"]].min(axis=1)
    d["t_hi"] = d[["team", "opponent"]].max(axis=1)
    lo = d[d["team"] == d["t_lo"]]
    hi = d[d["team"] == d["t_hi"]]
    m = lo.merge(hi, on=["date", "t_lo", "t_hi"], suffixes=("_lo", "_hi"))
    return m


def dc_apply(joint: np.ndarray, lam_a: float, lam_b: float, rho: float) -> np.ndarray:
    """Apply Dixon-Coles correction to a single (GRID, GRID) joint matrix.

    Multiplies the four low-scoring cells by their tau factors, then renormalises
    so the result remains a valid probability distribution.
    """
    if rho == 0.0:
        return joint
    j = joint.copy()
    j[0, 0] *= max(1.0 - lam_a * lam_b * rho, 1e-12)
    j[0, 1] *= max(1.0 + lam_a * rho,          1e-12)
    j[1, 0] *= max(1.0 + lam_b * rho,          1e-12)
    j[1, 1] *= max(1.0 - rho,                  1e-12)
    total = j.sum()
    return j / total if total > 0 else j


def estimate_rho(m: pd.DataFrame) -> float:
    """Grid-search for the Dixon-Coles rho on paired match data.

    Maximises the sum of log-tau over all training matches (the Poisson terms
    don't depend on rho, so only the correction factors matter).  Uses only
    the training set to avoid leakage.
    """
    lam_lo = m["lam_lo"].to_numpy()
    lam_hi = m["lam_hi"].to_numpy()
    g_lo   = m["team_goals_lo"].to_numpy().astype(int)
    g_hi   = m["team_goals_hi"].to_numpy().astype(int)

    m00 = (g_lo == 0) & (g_hi == 0)
    m01 = (g_lo == 0) & (g_hi == 1)
    m10 = (g_lo == 1) & (g_hi == 0)
    m11 = (g_lo == 1) & (g_hi == 1)

    best_rho, best_ll = 0.0, -np.inf
    for rho in np.linspace(-0.5, 0.05, 500):
        tau = np.ones(len(m))
        tau[m00] = 1.0 - lam_lo[m00] * lam_hi[m00] * rho
        tau[m01] = 1.0 + lam_lo[m01] * rho
        tau[m10] = 1.0 + lam_hi[m10] * rho
        tau[m11] = 1.0 - rho
        if np.any(tau <= 0):
            continue
        ll = float(np.sum(np.log(tau)))
        if ll > best_ll:
            best_ll, best_rho = ll, rho
    return best_rho


def derived_metrics(m: pd.DataFrame, rho: float = 0.0) -> tuple[float, float]:
    """W/D/L log-loss and exact-score hit rate from paired lambdas.

    If rho != 0, applies the Dixon-Coles correction before computing metrics.
    """
    pmf_lo = poisson_pmf_grid(m["lam_lo"].to_numpy())   # (N, GRID)
    pmf_hi = poisson_pmf_grid(m["lam_hi"].to_numpy())
    joint = pmf_lo[:, :, None] * pmf_hi[:, None, :]      # (N, GRID, GRID)

    if rho != 0.0:
        lam_lo_arr = m["lam_lo"].to_numpy()
        lam_hi_arr = m["lam_hi"].to_numpy()
        joint = np.array([
            dc_apply(joint[k], lam_lo_arr[k], lam_hi_arr[k], rho)
            for k in range(len(m))
        ])

    iu = np.triu_indices(GRID, k=1)   # lo < hi  -> hi-team wins
    il = np.tril_indices(GRID, k=-1)  # lo > hi  -> lo-team wins
    p_lo_win = joint[:, il[0], il[1]].sum(axis=1)
    p_hi_win = joint[:, iu[0], iu[1]].sum(axis=1)
    p_draw = np.einsum("nii->n", joint)

    g_lo = m["team_goals_lo"].to_numpy()
    g_hi = m["team_goals_hi"].to_numpy()
    actual_lo_win = g_lo > g_hi
    actual_hi_win = g_lo < g_hi

    p_actual = np.where(actual_lo_win, p_lo_win,
                        np.where(actual_hi_win, p_hi_win, p_draw))
    log_loss = -np.mean(np.log(np.clip(p_actual, 1e-12, None)))

    # most likely exact score = argmax of the joint grid
    flat = joint.reshape(joint.shape[0], -1).argmax(axis=1)
    pred_lo, pred_hi = np.unravel_index(flat, (GRID, GRID))
    hit_rate = np.mean((pred_lo == g_lo) & (pred_hi == g_hi))
    return float(log_loss), float(hit_rate)


def report(name: str, nll: float, m: pd.DataFrame, rho: float = 0.0) -> None:
    log_loss, hit = derived_metrics(m, rho=rho)
    print(f"  {name:<22} NLL {nll:.4f}   W/D/L log-loss {log_loss:.4f}   "
          f"exact-score {hit*100:5.2f}%")


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------
def _train_one(
    seed: int,
    x_tr: torch.Tensor, y_tr: torch.Tensor, w_tr: torch.Tensor,
    x_va: torch.Tensor, y_va: torch.Tensor,
) -> tuple[dict, float]:
    """Train one PoissonNet. Returns (best_state_dict, best_val_nll)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    criterion_w   = nn.PoissonNLLLoss(log_input=False, full=True, reduction="none")
    criterion_nll = nn.PoissonNLLLoss(log_input=False, full=True)
    model = PoissonNet()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-2)
    g     = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_tr, y_tr, w_tr), batch_size=BATCH, shuffle=True, generator=g
    )

    best_val, best_state, waited = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb, wb in loader:
            opt.zero_grad()
            (criterion_w(model(xb), yb) * wb).mean().backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_nll = criterion_nll(model(x_va), y_va).item()
        if val_nll < best_val - 1e-5:
            best_val, best_state, waited = val_nll, model.state_dict(), 0
        else:
            waited += 1
            if waited >= PATIENCE:
                print(f"  seed {seed}: stop epoch {epoch:>3}  val {best_val:.4f}")
                break

    return best_state, best_val


def main() -> None:
    import json

    x_tr, y_tr, df_tr = load_xy(_DATA_DIR / "baseline_train.csv")
    x_va, y_va, df_va = load_xy(_DATA_DIR / "baseline_val.csv")
    x_te, y_te, df_te = load_xy(_DATA_DIR / "baseline_test.csv")

    w_tr = sample_weights(df_tr, VAL_START)
    criterion_nll = nn.PoissonNLLLoss(log_input=False, full=True)

    # Train ensemble of N_ENSEMBLE models, each with a different random seed
    print(f"Training ensemble of {N_ENSEMBLE} models...")
    states: list[dict] = []
    for i in range(N_ENSEMBLE):
        state, _ = _train_one(i, x_tr, y_tr, w_tr, x_va, y_va)
        states.append(state)
        torch.save(state, _DATA_DIR / f"baseline_model_{i}.pt")

    # seed-0 model saved as the default for callers that expect a single file
    torch.save(states[0], _DATA_DIR / "baseline_model.pt")
    with open(_DATA_DIR / "ensemble_info.json", "w") as f:
        json.dump({"n_models": N_ENSEMBLE}, f)

    # Collect per-model lambda predictions on train and test sets
    lams_tr_all: list[np.ndarray] = []
    lams_te_all: list[np.ndarray] = []
    for state in states:
        m = PoissonNet()
        m.load_state_dict(state)
        m.eval()
        with torch.no_grad():
            lams_tr_all.append(m(x_tr).numpy())
            lams_te_all.append(m(x_te).numpy())

    lams_te_ens = np.mean(lams_te_all, axis=0)  # averaged ensemble predictions
    lams_tr_ens = np.mean(lams_tr_all, axis=0)

    # Single-model (seed 0) reference
    m0 = PoissonNet()
    m0.load_state_dict(states[0])
    m0.eval()
    with torch.no_grad():
        nll_single = criterion_nll(m0(x_te), y_te).item()

    nll_ens = criterion_nll(torch.tensor(lams_te_ens), y_te).item()

    # Estimate Dixon-Coles rho from ensemble training predictions
    rho = estimate_rho(match_table(df_tr, lams_tr_ens))
    print(f"Dixon-Coles rho (estimated on train): {rho:.4f}")
    with open(_DATA_DIR / "dc_rho.json", "w") as f:
        json.dump({"rho": rho}, f)

    floor_lam = float(y_tr.mean())
    floor_nll = criterion_nll(torch.full_like(y_te, floor_lam), y_te).item()

    mt_te_s   = match_table(df_te, lams_te_all[0])
    mt_te_ens = match_table(df_te, lams_te_ens)
    mt_floor  = match_table(df_te, np.full(len(df_te), floor_lam))

    print("\nTest-set results (lower NLL & log-loss better, higher hit-rate better):")
    report("Single (no DC)",              nll_single, mt_te_s,   rho=0.0)
    report("Single + DC",                 nll_single, mt_te_s,   rho=rho)
    report(f"Ensemble x{N_ENSEMBLE} (no DC)", nll_ens,    mt_te_ens, rho=0.0)
    report(f"Ensemble x{N_ENSEMBLE} + DC",    nll_ens,    mt_te_ens, rho=rho)
    report("floor (mean goals)",          floor_nll,  mt_floor,  rho=0.0)


if __name__ == "__main__":
    main()