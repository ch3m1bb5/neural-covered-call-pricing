"""
vol_forecaster.py  ──  IMPROVEMENT #1: probabilistic volatility forecast
------------------------------------------------------------------------
Replaces the single fixed `IV` and the crude AR(1) "momentum" in the original
dashboard. An LSTM consumes the recent path of realized vol + returns and
forecasts the *average realized vol over the next h trading days* at three
quantiles (P10 / P50 / P90) via the pinball (quantile) loss.

The P50 becomes the sigma fed to Black-Scholes; P10..P90 becomes the premium /
Greeks uncertainty band. Quantiles are built cumulatively so they can never
cross. N-HiTS (neuralforecast) is a drop-in replacement for the LSTM core --
same inputs/outputs -- if you want the multi-rate hierarchical version.
"""
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cpu"
QUANTILES = (0.1, 0.5, 0.9)


class VolLSTM(nn.Module):
    def __init__(self, n_features=2, hidden=48, layers=2, n_q=3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, n_q)   # base + 2 positive increments

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]                    # last timestep
        raw = self.head(h)                   # (B, 3) in log-vol space
        base = raw[:, :1]
        inc = torch.nn.functional.softplus(raw[:, 1:])     # >= 0
        logv = torch.cat([base, base + inc[:, :1],
                          base + inc[:, :1] + inc[:, 1:]], dim=1)  # monotone
        return torch.exp(logv)               # vols, P10<=P50<=P90


def build_dataset(rv, log_ret, L=60, h=21):
    """
    Features per day: [log(rv), log_ret]. Target: mean RV over next h days.
    Returns X (N,L,2), y (N,1) and the feature-mean/std for inference scaling.
    """
    feats = np.stack([np.log(np.clip(rv, 1e-4, None)), log_ret], axis=1)
    mu, sd = feats.mean(0), feats.std(0) + 1e-8
    feats_n = (feats - mu) / sd
    X, y = [], []
    for t in range(L, len(rv) - h):
        X.append(feats_n[t - L:t])
        y.append(rv[t:t + h].mean())
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X, y, mu.astype(np.float32), sd.astype(np.float32)


def pinball_loss(pred, target, quantiles=QUANTILES):
    """pred (B, Q), target (B, 1)."""
    losses = []
    for i, q in enumerate(quantiles):
        e = target[:, 0] - pred[:, i]
        losses.append(torch.maximum(q * e, (q - 1) * e))
    return torch.stack(losses, 1).mean()


def train_vol_forecaster(rv, log_ret, L=60, h=21, epochs=400, lr=3e-3,
                         seed=0, verbose=True):
    torch.manual_seed(seed)
    X, y, mu, sd = build_dataset(rv, log_ret, L, h)
    n_val = max(16, int(0.15 * len(X)))
    Xtr, ytr = torch.tensor(X[:-n_val]), torch.tensor(y[:-n_val])
    Xva, yva = torch.tensor(X[-n_val:]), torch.tensor(y[-n_val:])

    model = VolLSTM().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = pinball_loss(model(Xtr), ytr)
        loss.backward(); opt.step(); sched.step()
        if ep % 50 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                vloss = pinball_loss(model(Xva), yva).item()
            if vloss < best:
                best, best_state = vloss, {k: v.clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  [vol] ep {ep:3d}  train {loss.item():.5f}  val {vloss:.5f}")
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"L": L, "h": h, "mu": mu.tolist(), "sd": sd.tolist()}


@torch.no_grad()
def forecast(model, rv_recent, ret_recent, meta):
    """
    rv_recent, ret_recent: arrays of at least meta['L'] most recent days.
    Returns {'p10','p50','p90'} annualized vols for the next h days.
    """
    L, mu, sd = meta["L"], np.asarray(meta["mu"]), np.asarray(meta["sd"])
    rv = np.asarray(rv_recent)[-L:]
    rt = np.asarray(ret_recent)[-L:]
    feats = np.stack([np.log(np.clip(rv, 1e-4, None)), rt], axis=1)
    feats_n = ((feats - mu) / sd).astype(np.float32)
    x = torch.tensor(feats_n).unsqueeze(0)
    q = model(x).cpu().numpy().reshape(-1)
    return {"p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2])}


def save(model, path):
    torch.save(model.state_dict(), path)


def load(path):
    m = VolLSTM()
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


if __name__ == "__main__":
    from data_synth import generate_price_series, realized_vol
    p, r, v = generate_price_series()
    rv = realized_vol(r)
    model, meta = train_vol_forecaster(rv, r, epochs=400)
    fc = forecast(model, rv, r, meta)
    print(f"forecast next {meta['h']}d RV  "
          f"P10={fc['p10']:.2%}  P50={fc['p50']:.2%}  P90={fc['p90']:.2%}  "
          f"(last RV={rv[-1]:.2%})")
