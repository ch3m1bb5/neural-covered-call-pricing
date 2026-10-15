"""
data_synth.py
-------------
Generates self-contained, market-realistic synthetic data so the three neural
models can train end-to-end without a live data feed (yfinance/CBOE etc.).

Two artifacts are produced:
  1) A daily UAL-style price path driven by a Heston-style stochastic-volatility
     process -> gives the vol forecaster a *time-varying* realized vol to learn.
  2) An option chain whose implied vols follow an SVI total-variance skew ->
     gives the IV-surface net a genuine smile/skew (not a flat sigma) to fit.

In production you swap these two functions for your real loaders (the rest of
the pipeline is untouched).
"""
import numpy as np

# Anchor to the dashboard's world
S0_ANCHOR = 167.76      # current UAL price used in the original dashboard
TRADING_DAYS = 252


# ───────────────────────── price series (Heston) ────────────────────────────
def generate_price_series(n_days=900, s0=120.0, seed=7, end_anchor=S0_ANCHOR):
    """
    Heston-style Euler discretization (full truncation on the variance).
    Returns (prices, daily_log_returns, inst_vol_annualized).
    The path is rescaled at the end so it terminates near `end_anchor`,
    matching the dashboard's current price.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS

    # Heston parameters (annualized)
    mu = 0.06          # drift
    kappa = 3.0        # mean reversion speed of variance
    theta = 0.09       # long-run variance  -> ~30% long-run vol
    xi = 0.45          # vol-of-vol
    rho = -0.65        # leverage effect (price down -> vol up)
    v0 = 0.08

    v = v0
    logp = np.log(s0)
    prices, rets, vols = [], [], []
    for _ in range(n_days):
        z1 = rng.standard_normal()
        z2 = rho * z1 + np.sqrt(1 - rho * rho) * rng.standard_normal()
        v_pos = max(v, 1e-8)
        # price log-return
        dlog = (mu - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z1
        logp += dlog
        # variance update (full truncation)
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * z2
        v = max(v, 1e-8)
        prices.append(np.exp(logp))
        rets.append(dlog)
        vols.append(np.sqrt(v) )  # annualized inst vol

    prices = np.asarray(prices)
    # rescale so the series ends near the anchor (keeps shape, fixes level)
    prices = prices * (end_anchor / prices[-1])
    return prices, np.asarray(rets), np.asarray(vols)


def realized_vol(log_returns, window=21):
    """Annualized rolling realized volatility from daily log returns."""
    lr = np.asarray(log_returns)
    out = np.full(lr.shape, np.nan)
    for i in range(window, len(lr) + 1):
        out[i - 1] = lr[i - window:i].std(ddof=1) * np.sqrt(TRADING_DAYS)
    # backfill the leading NaNs with first valid value
    first = out[~np.isnan(out)][0] if np.any(~np.isnan(out)) else 0.3
    out[np.isnan(out)] = first
    return out


def generate_exogenous(log_returns, inst_vol, seed=21):
    """
    Exogenous regressors for the N-HiTS forecaster:
      vix      : a VIX-style index ~ instantaneous vol + risk premium + noise
      log_vol  : log trading volume, mean-reverting with spikes on big moves
    Both are *historically observed* (hist_exog), correlated with vol the way
    real market features are.
    """
    rng = np.random.default_rng(seed)
    n = len(log_returns)
    # VIX proxy: inst vol plus a small premium and AR(1) noise
    prem = 0.03
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = 0.85 * noise[t - 1] + 0.01 * rng.standard_normal()
    vix = np.clip(inst_vol + prem + noise, 0.05, 2.0)
    # log volume: baseline + reaction to |return| + mean reversion
    base = 15.0
    lv = np.full(n, base)
    for t in range(1, n):
        shock = 6.0 * abs(log_returns[t])
        lv[t] = base + 0.6 * (lv[t - 1] - base) + shock + 0.05 * rng.standard_normal()
    return vix.astype(float), lv.astype(float)


# ───────────────────────── option chain (SVI skew) ──────────────────────────
def _svi_total_variance(k, a, b, rho, m, sigma):
    """Raw SVI: total implied variance w(k) = iv^2 * T."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def synthetic_option_chain(spot=S0_ANCHOR, seed=11,
                           expiries_days=(7, 14, 30, 45, 60, 90, 120),
                           n_strikes=21, strike_span=0.35, noise=0.004):
    """
    Build a chain with a real skew. For each expiry we evaluate an SVI slice
    (params drift mildly with maturity -> term structure) and convert total
    variance to implied vol. Returns a list of quotes:
        {K, T (years), iv, k (log-moneyness), spot}
    """
    rng = np.random.default_rng(seed)
    quotes = []
    for d in expiries_days:
        T = d / 365.0
        # SVI params with a gentle term-structure and a negative-rho equity skew
        a = 0.005 + 0.010 * T
        b = 0.040 + 0.020 * np.sqrt(T)
        rho = -0.62                      # downward (put) skew
        m = 0.01 + 0.0 * T
        sigma = 0.130 + 0.030 * T
        # liquid strike ladder widens with maturity (~sqrt(T)); keeps IVs sane
        span = strike_span * np.sqrt(T / (30.0 / 365.0))
        ks = np.linspace(-span, span, n_strikes)
        for k in ks:
            w = _svi_total_variance(k, a, b, rho, m, sigma)
            iv_true = np.sqrt(max(w, 1e-6) / T)
            iv = iv_true * (1.0 + noise * rng.standard_normal())  # quote noise
            K = float(spot * np.exp(k))
            quotes.append({
                "K": round(K, 2),
                "T": float(T),
                "iv": float(max(iv, 0.02)),
                "k": float(k),
                "spot": float(spot),
            })
    return quotes


if __name__ == "__main__":
    p, r, v = generate_price_series()
    rv = realized_vol(r)
    print(f"price path: n={len(p)}  last={p[-1]:.2f}  "
          f"RV range=[{rv.min():.2%},{rv.max():.2%}]")
    ch = synthetic_option_chain()
    ivs = [q["iv"] for q in ch]
    print(f"chain: n={len(ch)}  iv range=[{min(ivs):.2%},{max(ivs):.2%}]")
