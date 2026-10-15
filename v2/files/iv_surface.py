"""
iv_surface.py  ──  IMPROVEMENT #2: deep implied-volatility surface
------------------------------------------------------------------
Replaces the single constant `IV` that the original dashboard fed into
Black-Scholes for *every* strike. Here a small MLP learns the whole surface

        (k = log-moneyness, T = years)  ->  w = total implied variance

with iv = sqrt(w / T). Two soft no-arbitrage penalties are added via autograd:

  * calendar  : w must be non-decreasing in T            -> penalize  relu(-dw/dT)
  * butterfly : convexity proxy in k (no wild wiggles)   -> penalize  relu(-d2w/dk2)

These are the same regularizers used in "deep SVI" calibration papers, in a
compact form. Output: a strike-specific IV so the 0.20-delta strike selection
and the Greeks in the dashboard are no longer biased by a flat sigma.
"""
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cpu"


class IVSurfaceNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, k, T):
        x = torch.cat([k, T], dim=1)
        # total variance is strictly positive
        return self.softplus(self.net(x)) + 1e-6   # w(k,T)

    # convenience: implied vol = sqrt(w / T)
    def iv(self, k, T):
        w = self.forward(k, T)
        return torch.sqrt(w / T.clamp_min(1e-6))


def _derivs(model, k, T):
    """First/second derivatives of w via autograd for the no-arb penalties."""
    k = k.clone().requires_grad_(True)
    T = T.clone().requires_grad_(True)
    w = model(k, T)
    dw_dk = torch.autograd.grad(w.sum(), k, create_graph=True)[0]
    d2w_dk2 = torch.autograd.grad(dw_dk.sum(), k, create_graph=True)[0]
    dw_dT = torch.autograd.grad(w.sum(), T, create_graph=True)[0]
    return d2w_dk2, dw_dT


def train_iv_surface(quotes, epochs=4000, lr=2e-3, lam_cal=5.0, lam_bf=5.0,
                     hidden=64, seed=0, verbose=True):
    """quotes: list of {k, T, iv}. Returns a trained IVSurfaceNet."""
    torch.manual_seed(seed)
    k = torch.tensor([[q["k"]] for q in quotes], dtype=torch.float32)
    T = torch.tensor([[q["T"]] for q in quotes], dtype=torch.float32)
    iv = torch.tensor([[q["iv"]] for q in quotes], dtype=torch.float32)
    w_target = iv ** 2 * T                       # total variance target

    model = IVSurfaceNet(hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    # collocation grid for the no-arb penalties (denser than the quotes)
    kk = torch.linspace(k.min().item(), k.max().item(), 40)
    TT = torch.linspace(T.min().item(), T.max().item(), 20)
    gk, gT = torch.meshgrid(kk, TT, indexing="ij")
    col_k = gk.reshape(-1, 1)
    col_T = gT.reshape(-1, 1)

    for ep in range(epochs):
        opt.zero_grad()
        iv_pred = model.iv(k, T)
        fit = ((iv_pred - iv) ** 2).mean()        # fit directly in IV space
        d2w_dk2, dw_dT = _derivs(model, col_k, col_T)
        pen_bf = torch.relu(-d2w_dk2).mean()      # butterfly (convexity) proxy
        pen_cal = torch.relu(-dw_dT).mean()       # calendar monotonicity
        loss = fit + lam_bf * pen_bf + lam_cal * pen_cal
        loss.backward()
        opt.step()
        sched.step()
        if verbose and (ep % 1000 == 0 or ep == epochs - 1):
            rmse_iv = torch.sqrt(((model.iv(k, T) - iv) ** 2).mean()).item()
            print(f"  [iv] ep {ep:4d}  loss {loss.item():.5f}  "
                  f"fit {fit.item():.5f}  ivRMSE {rmse_iv:.4f}  "
                  f"cal {pen_cal.item():.2e}  bf {pen_bf.item():.2e}")
    return model


@torch.no_grad()
def predict_iv(model, k, T):
    """k, T can be floats or arrays. Returns IV(s) as float/np.array."""
    k = np.atleast_1d(np.asarray(k, dtype=np.float32))
    T = np.atleast_1d(np.asarray(T, dtype=np.float32))
    n = max(k.size, T.size)
    k = np.broadcast_to(k, (n,)).copy()      # broadcast scalar <-> vector
    T = np.broadcast_to(T, (n,)).copy()
    kt = torch.tensor(k.reshape(-1, 1))
    Tt = torch.tensor(T.reshape(-1, 1))
    out = model.iv(kt, Tt).cpu().numpy().reshape(-1)
    return float(out[0]) if out.size == 1 else out


@torch.no_grad()
def iv_for_strike(model, K, spot, T):
    """IV for an absolute strike K given spot and maturity T (years)."""
    k = np.log(np.asarray(K, dtype=np.float64) / spot)
    return predict_iv(model, k, T)


def save(model, path):
    torch.save(model.state_dict(), path)


def load(path, hidden=64):
    m = IVSurfaceNet(hidden)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


# ───────────────────────── calibration to a real chain ──────────────────────
def _bs_call(S, K, T, r, sigma):
    from math import log, sqrt, erf, exp
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    ncdf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    return S * ncdf(d1) - K * exp(-r * T) * ncdf(d2)


def implied_vol(price, S, K, T, r=0.043, kind="call"):
    """Invert Black-Scholes for IV via bisection. Put uses call-put parity."""
    from math import exp
    if kind == "put":
        price = price + S - K * exp(-r * T)      # parity -> equivalent call
    lo, hi = 1e-3, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _bs_call(S, K, T, r, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def quotes_to_training(quotes, spot, r=0.043):
    """
    Normalize a posted chain into {k, T, iv} training rows. Each quote may give
    `iv` directly, or a `price`(+`kind`) to be inverted, with `K` and either
    `T` (years) or `T_days`.
    """
    rows = []
    for q in quotes:
        K = float(q["K"])
        T = float(q["T"]) if "T" in q else float(q["T_days"]) / 365.0
        if "iv" in q and q["iv"] is not None:
            iv = float(q["iv"])
        else:
            iv = implied_vol(float(q["price"]), spot, K, T, r,
                             q.get("kind", "call"))
        if iv <= 0 or not np.isfinite(iv):
            continue
        rows.append({"k": float(np.log(K / spot)), "T": T, "iv": iv})
    return rows


def calibrate_from_chain(quotes, spot, epochs=4000, **kw):
    """Fit a fresh IVSurfaceNet to a real posted chain. Returns (model, n_rows)."""
    rows = quotes_to_training(quotes, spot)
    if len(rows) < 6:
        raise ValueError("need at least ~6 valid quotes to fit a surface")
    model = train_iv_surface(rows, epochs=epochs, **kw)
    return model, len(rows)


if __name__ == "__main__":
    from data_synth import synthetic_option_chain, S0_ANCHOR
    ch = synthetic_option_chain()
    model = train_iv_surface(ch, epochs=2000)
    # check the 30-DTE skew is reproduced
    T = 30 / 365
    for K in [150, 167.76, 185, 200]:
        print(f"K={K:7.2f}  IV={iv_for_strike(model, K, S0_ANCHOR, T):.2%}")
