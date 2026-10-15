"""
scenario_gen.py  ──  IMPROVEMENT #3: generative scenario engine
---------------------------------------------------------------
Replaces the single deterministic "if called away at the 0.20-delta strike"
line in the original dashboard with a full Monte-Carlo *distribution* of the
covered-call P&L.

A VAE learns the SHAPE of H-day return windows from history (volatility
clustering, asymmetry, fat tails -- things a single GBM draw cannot reproduce).
At generation time the shapes are rescaled to a target daily vol that comes
straight from the vol forecaster (Improvement #1), so the three models compose:

        forecaster -> target vol -> VAE paths -> P&L distribution

Outputs: assignment probability, early-touch probability, P5/P50/P95 P&L and
CVaR(5%). TimeGAN is the heavier alternative if you want a discriminator-trained
generator; the VAE is chosen here for a self-contained, fast, stable demo.
"""
import numpy as np
import torch
import torch.nn as nn

DEVICE = "cpu"
TRADING_DAYS = 252


class PathVAE(nn.Module):
    def __init__(self, horizon=21, latent=8, hidden=64):
        super().__init__()
        self.H = horizon
        self.enc = nn.Sequential(
            nn.Linear(horizon, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, horizon),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.dec(z), mu, logvar


def build_windows(log_ret, H=21):
    """
    Per-window normalized VARIANCE profiles (mean ~1) of length H.
    Daily returns are near-unpredictable, but their *variance clustering* is
    structured and learnable -- so the VAE models the variance profile, and
    fresh Gaussian innovations are injected at generation time. This avoids the
    mean-collapse that a VAE trained on raw returns suffers from.
    """
    lr = np.asarray(log_ret)
    profs = []
    for t in range(0, len(lr) - H):
        w = lr[t:t + H]
        v = w ** 2
        v = v / (v.mean() + 1e-10)            # normalized variance profile
        profs.append(np.log(v + 1e-4))        # log space -> easier to model
    return np.asarray(profs, dtype=np.float32)


def train_scenario_vae(log_ret, H=21, latent=8, epochs=600, lr=2e-3,
                       beta=0.5, seed=0, verbose=True):
    torch.manual_seed(seed)
    X = torch.tensor(build_windows(log_ret, H))
    model = PathVAE(H, latent).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        recon, mu, logvar = model(X)
        rec = ((recon - X) ** 2).mean()
        kl = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        loss = rec + beta * kl
        loss.backward(); opt.step(); sched.step()
        if verbose and (ep % 150 == 0 or ep == epochs - 1):
            print(f"  [vae] ep {ep:3d}  loss {loss.item():.4f}  "
                  f"rec {rec.item():.4f}  kl {kl.item():.4f}")
    model.eval()
    return model, {"H": H, "latent": latent}


@torch.no_grad()
def generate_paths(model, n_paths, target_annual_vol, meta, drift_annual=0.0):
    """
    Sample a variance profile from the latent prior (clustering structure),
    then inject FRESH Gaussian innovations so paths are diverse and fat-tailed.
    Returns daily log-returns (n_paths, H) whose annualized vol ~ target.
    """
    H, latent = meta["H"], meta["latent"]
    z = torch.randn(n_paths, latent)
    log_vprof = model.dec(z).cpu().numpy()              # (n_paths, H) in log space
    vprof = np.exp(log_vprof)
    vprof = vprof / (vprof.mean(axis=1, keepdims=True) + 1e-10)   # mean ~1
    sigma_d = target_annual_vol / np.sqrt(TRADING_DAYS)
    eps = np.random.standard_normal((n_paths, H))      # fresh innovations
    local_var = (sigma_d ** 2) * vprof
    mu_d = drift_annual / TRADING_DAYS - 0.5 * local_var          # Ito (local)
    return mu_d + np.sqrt(local_var) * eps             # daily log-returns


def monte_carlo_covered_call(model, meta, *, spot, strike, premium,
                             target_annual_vol, cost_basis, shares=100,
                             n_paths=20000, drift_annual=0.0):
    """
    Full P&L distribution of a covered call held to expiry (H days).
    Assignment approximated at expiry (standard for OTM short calls); an
    early-touch probability is reported as an American-assignment proxy.
    """
    rets = generate_paths(model, n_paths, target_annual_vol, meta, drift_annual)
    price_paths = spot * np.exp(np.cumsum(rets, axis=1))       # (n_paths, H)
    S_T = price_paths[:, -1]
    touched = (price_paths.max(axis=1) >= strike)             # crossed strike
    assigned = S_T >= strike

    # P&L per position (shares + short call), in dollars
    capped_gain = (strike - cost_basis) * shares + premium * shares
    open_pnl = (S_T - cost_basis) * shares + premium * shares
    pnl = np.where(assigned, capped_gain, open_pnl)

    def pct(a, q): return float(np.percentile(a, q))
    cvar5 = float(pnl[pnl <= np.percentile(pnl, 5)].mean())
    # compact histogram for the dashboard
    counts, edges = np.histogram(pnl, bins=30)
    return {
        "n_paths": int(n_paths),
        "target_vol": float(target_annual_vol),
        "assignment_prob": float(assigned.mean()),
        "touch_prob": float(touched.mean()),
        "expected_pnl": float(pnl.mean()),
        "pnl_p5": pct(pnl, 5),
        "pnl_p50": pct(pnl, 50),
        "pnl_p95": pct(pnl, 95),
        "cvar_5pct": cvar5,
        "prob_profit": float((pnl > 0).mean()),
        "spot_p5": pct(S_T, 5),
        "spot_p50": pct(S_T, 50),
        "spot_p95": pct(S_T, 95),
        "hist_counts": counts.tolist(),
        "hist_edges": [float(e) for e in edges],
    }


def save(model, path):
    torch.save(model.state_dict(), path)


def load(path, H=21, latent=8):
    m = PathVAE(H, latent)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


if __name__ == "__main__":
    from data_synth import generate_price_series, S0_ANCHOR
    p, r, v = generate_price_series()
    model, meta = train_scenario_vae(r, epochs=600)
    res = monte_carlo_covered_call(
        model, meta, spot=S0_ANCHOR, strike=180.0, premium=2.10,
        target_annual_vol=0.26, cost_basis=197.0, shares=100, n_paths=20000)
    print(f"assignment={res['assignment_prob']:.1%}  touch={res['touch_prob']:.1%}  "
          f"E[P&L]=${res['expected_pnl']:.0f}  "
          f"P5=${res['pnl_p5']:.0f}  P95=${res['pnl_p95']:.0f}  "
          f"CVaR5%=${res['cvar_5pct']:.0f}  P(profit)={res['prob_profit']:.1%}")
