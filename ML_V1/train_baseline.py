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

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = ["team_elo_scaled", "opp_elo_scaled", "elo_diff_scaled", "is_home"]
TARGET = "team_goals"
GRID = 9            # goal values 0..8, matching the GOAL_CAP in the dataset
EPOCHS = 80
PATIENCE = 12       # stop if val NLL doesn't improve for this many epochs
BATCH = 512
SEED = 0


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class PoissonNet(nn.Module):
    """4 inputs -> one hidden layer -> softplus, so the output lambda > 0."""

    def __init__(self, n_in: int = len(FEATURES), hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
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


def derived_metrics(m: pd.DataFrame) -> tuple[float, float]:
    """W/D/L log-loss and exact-score hit rate from paired lambdas."""
    pmf_lo = poisson_pmf_grid(m["lam_lo"].to_numpy())   # (N, GRID)
    pmf_hi = poisson_pmf_grid(m["lam_hi"].to_numpy())
    joint = pmf_lo[:, :, None] * pmf_hi[:, None, :]      # (N, GRID, GRID)

    iu = np.triu_indices(GRID, k=1)   # lo < hi  -> hi-team wins
    il = np.tril_indices(GRID, k=-1)  # lo > hi  -> lo-team wins
    p_lo_win = joint[:, il[0], il[1]].sum(axis=1)
    p_hi_win = joint[:, iu[0], iu[1]].sum(axis=1)
    p_draw = np.einsum("nii->n", joint)

    g_lo = m["team_goals_lo"].to_numpy()
    g_hi = m["team_goals_hi"].to_numpy()
    actual_lo_win = g_lo > g_hi
    actual_hi_win = g_lo < g_hi
    actual_draw = g_lo == g_hi

    p_actual = np.where(actual_lo_win, p_lo_win,
                        np.where(actual_hi_win, p_hi_win, p_draw))
    log_loss = -np.mean(np.log(np.clip(p_actual, 1e-12, None)))

    # most likely exact score = argmax of the joint grid
    flat = joint.reshape(joint.shape[0], -1).argmax(axis=1)
    pred_lo, pred_hi = np.unravel_index(flat, (GRID, GRID))
    hit_rate = np.mean((pred_lo == g_lo) & (pred_hi == g_hi))
    return float(log_loss), float(hit_rate)


def report(name: str, nll: float, m: pd.DataFrame) -> None:
    log_loss, hit = derived_metrics(m)
    print(f"  {name:<18} NLL {nll:.4f}   W/D/L log-loss {log_loss:.4f}   "
          f"exact-score {hit*100:5.2f}%")


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x_tr, y_tr, df_tr = load_xy("baseline_train.csv")
    x_va, y_va, df_va = load_xy("baseline_val.csv")
    x_te, y_te, df_te = load_xy("baseline_test.csv")

    criterion = nn.PoissonNLLLoss(log_input=False, full=True)
    model = PoissonNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH, shuffle=True)

    best_val, best_state, waited = float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_nll = criterion(model(x_va), y_va).item()
        if val_nll < best_val - 1e-5:
            best_val, best_state, waited = val_nll, model.state_dict(), 0
        else:
            waited += 1
            if waited >= PATIENCE:
                print(f"early stop at epoch {epoch} (best val NLL {best_val:.4f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), "baseline_model.pt")  # reused by predict.py
    with torch.no_grad():
        test_nll = criterion(model(x_te), y_te).item()
        lams_te = model(x_te).numpy()

    # dumb floor: predict the training-set average goals for every team
    floor_lam = float(y_tr.mean())
    floor_nll = criterion(torch.full_like(y_te, floor_lam), y_te).item()
    floor_lams = np.full(len(df_te), floor_lam)

    print("\nTest-set results (lower NLL & log-loss better, higher hit-rate better):")
    report("Elo Poisson net", test_nll, match_table(df_te, lams_te))
    report("floor (mean goals)", floor_nll, match_table(df_te, floor_lams))


if __name__ == "__main__":
    main()