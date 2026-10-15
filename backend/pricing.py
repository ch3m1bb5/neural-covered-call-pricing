"""
pricing.py — Modulo avancado de precificacao de opcoes
=======================================================
Melhorias implementadas:
  1. GARCH(1,1)            — forecast dinamico de volatilidade
  2. Binomial CRR          — pricing americano com exercicio antecipado + dividendos
  3. SABR (Hagan 2002)     — smile/skew de volatilidade por strike
  4. Term structure de IV  — IV varia com DTE + bump de earnings
  + Greeks de 2a ordem     — Vanna, Charm, Volga

Usado por app.py. Depende apenas de numpy e scipy.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


# ══════════════════════════════════════════════════════════════════════════════
# 1. GARCH(1,1) — VOLATILIDADE DINAMICA
# ══════════════════════════════════════════════════════════════════════════════

def fit_garch(returns, max_iter=200):
    """
    GARCH(1,1) via maxima verosimilhanca.
        sigma2_t = omega + alpha * r_{t-1}^2 + beta * sigma2_{t-1}
    Captura clustering de volatilidade (vol alta segue vol alta).
    Retorna parametros + forecast de vol anualizada.
    """
    r = np.asarray(returns, dtype=float)
    r = r - np.mean(r)
    if len(r) < 20:
        v = float(np.std(r) * np.sqrt(252))
        return {"omega": 0, "alpha": 0, "beta": 0, "persistence": 0,
                "next_vol_annual": v, "long_run_vol_annual": v, "converged": False}

    var_uncond = np.var(r)

    def neg_loglik(params):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e10
        sigma2 = np.empty(len(r))
        sigma2[0] = var_uncond
        for t in range(1, len(r)):
            sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
        sigma2 = np.maximum(sigma2, 1e-12)
        return 0.5 * np.sum(np.log(2*np.pi*sigma2) + r**2/sigma2)

    try:
        res = minimize(neg_loglik, [var_uncond*0.1, 0.1, 0.8],
                       bounds=[(1e-12, None), (0, 0.999), (0, 0.999)],
                       method="L-BFGS-B", options={"maxiter": max_iter})
        omega, alpha, beta = res.x
        converged = res.success
    except Exception:
        omega, alpha, beta, converged = var_uncond*0.1, 0.1, 0.8, False

    sigma2 = np.empty(len(r))
    sigma2[0] = var_uncond
    for t in range(1, len(r)):
        sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
    next_var = omega + alpha * r[-1]**2 + beta * sigma2[-1]

    persistence = alpha + beta
    long_run_var = omega / (1 - persistence) if persistence < 1 else var_uncond

    return {
        "omega": float(omega), "alpha": float(alpha), "beta": float(beta),
        "persistence": float(persistence),
        "next_vol_annual": float(np.sqrt(max(next_var, 1e-12) * 252)),
        "long_run_vol_annual": float(np.sqrt(max(long_run_var, 1e-12) * 252)),
        "converged": bool(converged),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. BINOMIAL CRR — PRICING AMERICANO + DIVIDENDOS
# ══════════════════════════════════════════════════════════════════════════════

def binomial_american_call(S, K, T, r, sigma, q=0.0, steps=160):
    """
    Cox-Ross-Rubinstein para call americana com dividend yield q.
    Modela exercicio antecipado (relevante perto de ex-dividend).
    """
    if T <= 0 or sigma <= 0:
        return float(max(S - K, 0))
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    p = np.clip(p, 0.0, 1.0)
    disc = np.exp(-r * dt)

    j = np.arange(steps + 1)
    ST = S * u**j * d**(steps - j)
    values = np.maximum(ST - K, 0.0)

    for i in range(steps - 1, -1, -1):
        ST = ST[:i+1] / u
        values = disc * (p * values[1:i+2] + (1 - p) * values[0:i+1])
        values = np.maximum(values, ST - K)  # exercicio antecipado americano
    return float(values[0])


def binomial_greeks(S, K, T, r, sigma, q=0.0, steps=160):
    """Greeks da call americana via diferencas finitas no binomial."""
    h  = max(S * 0.01, 0.01)
    p0 = binomial_american_call(S,     K, T, r, sigma, q, steps)
    pu = binomial_american_call(S + h, K, T, r, sigma, q, steps)
    pd = binomial_american_call(S - h, K, T, r, sigma, q, steps)
    delta = (pu - pd) / (2 * h)
    gamma = (pu - 2*p0 + pd) / (h**2)

    dt = 1.0 / 365
    pt = binomial_american_call(S, K, max(T - dt, 1e-6), r, sigma, q, steps)
    theta = pt - p0  # por dia

    dv = 0.01
    pv = binomial_american_call(S, K, T, r, sigma + dv, q, steps)
    vega = pv - p0   # por 1% de IV

    dr = 0.01
    pr = binomial_american_call(S, K, T, r + dr, sigma, q, steps)
    rho = pr - p0

    return {
        "engine": "Binomial-CRR (American)",
        "price": float(p0), "delta": float(delta), "gamma": float(gamma),
        "theta": float(theta), "vega": float(vega), "rho": float(rho),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. SABR — SMILE / SKEW DE VOLATILIDADE (Hagan 2002)
# ══════════════════════════════════════════════════════════════════════════════

def sabr_vol(F, K, T, alpha, beta, rho, nu):
    """Volatilidade implicita Hagan 2002 para um strike K."""
    if F <= 0 or K <= 0:
        return alpha
    if abs(F - K) < 1e-8:  # ATM
        FK = F**(1 - beta)
        term = 1 + ((((1-beta)**2/24)*(alpha**2/FK**2)) +
                    (rho*beta*nu*alpha/(4*FK)) +
                    ((2-3*rho**2)*nu**2/24)) * T
        return (alpha / FK) * term
    logFK = np.log(F / K)
    FK_beta = (F * K)**((1 - beta) / 2)
    z = (nu / alpha) * FK_beta * logFK
    sqrt_term = np.sqrt(1 - 2*rho*z + z**2)
    x_z = np.log((sqrt_term + z - rho) / (1 - rho))
    if abs(x_z) < 1e-10:
        x_z = z if abs(z) > 1e-10 else 1e-10
    denom = FK_beta * (1 + ((1-beta)**2/24)*logFK**2 + ((1-beta)**4/1920)*logFK**4)
    term = 1 + ((((1-beta)**2/24)*(alpha**2/FK_beta**2)) +
                (rho*beta*nu*alpha/(4*FK_beta)) +
                ((2-3*rho**2)*nu**2/24)) * T
    return (alpha / denom) * (z / x_z) * term


def calibrate_sabr(F, T, atm_vol, beta=0.5, rho=-0.3, nu=0.6):
    """
    Calibra alpha para que SABR ATM = atm_vol.
    rho < 0  => skew negativo (puts/ITM mais caras) — tipico de equities.
    beta=0.5 => regime entre lognormal (1) e normal (0).
    nu       => vol-of-vol (curvatura do smile).
    """
    alpha = atm_vol * F**(1 - beta)
    for _ in range(8):
        implied = sabr_vol(F, F, T, alpha, beta, rho, nu)
        if implied <= 0:
            break
        alpha *= atm_vol / implied
    return {"alpha": float(alpha), "beta": float(beta),
            "rho": float(rho), "nu": float(nu)}


# ══════════════════════════════════════════════════════════════════════════════
# 4. TERM STRUCTURE DE IV + EARNINGS
# ══════════════════════════════════════════════════════════════════════════════

def iv_term_structure(base_iv, dte, earnings_in_window=False):
    """
    Ajusta IV pela maturidade (DTE).
    - Curto prazo (<7d): premio de gamma.
    - Longo prazo (>60d): converge para vol estrutural (ligeiro desconto).
    - Earnings na janela: bump de +25%.
    """
    if   dte < 7:   tf = 1.15
    elif dte < 21:  tf = 1.05
    elif dte < 45:  tf = 1.00
    elif dte < 90:  tf = 0.97
    else:           tf = 0.95
    iv = base_iv * tf
    if earnings_in_window:
        iv *= 1.25
    return float(iv)


# ══════════════════════════════════════════════════════════════════════════════
# GREEKS DE 2A ORDEM (Black-Scholes com dividendos)
# ══════════════════════════════════════════════════════════════════════════════

def second_order_greeks(S, K, T, r, sigma, q=0.0):
    """Vanna, Charm e Volga — uteis para gestao activa (rolling/ajuste)."""
    if T <= 0 or sigma <= 0:
        return {"vanna": 0.0, "charm": 0.0, "volga": 0.0}
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    pdf = norm.pdf(d1)
    vanna = -np.exp(-q*T) * pdf * d2 / sigma
    charm = (q*np.exp(-q*T)*norm.cdf(d1)
             - np.exp(-q*T)*pdf*(2*(r-q)*T - d2*sigma*np.sqrt(T))
               / (2*T*sigma*np.sqrt(T)))
    volga = S*np.exp(-q*T)*pdf*np.sqrt(T)*d1*d2 / sigma
    return {"vanna": float(vanna), "charm": float(charm/365), "volga": float(volga/100)}


# ══════════════════════════════════════════════════════════════════════════════
# PRICER UNIFICADO
# ══════════════════════════════════════════════════════════════════════════════

def price_option(S, K, T, r, base_iv, q=0.0, dte=30,
                 earnings_in_window=False, sabr_params=None,
                 american=True, steps=160):
    """
    Pricing completo de uma call:
      1. IV do strike via SABR (se fornecido) — captura skew
      2. Ajuste de term structure pela maturidade
      3. Pricing americano (Binomial) ou europeu (BS)
      4. Greeks de 1a e 2a ordem
    """
    F = S * np.exp((r - q) * T)  # forward

    # 1. IV especifica do strike (smile)
    if sabr_params:
        strike_iv = sabr_vol(F, K, T, **sabr_params)
    else:
        strike_iv = base_iv

    # 2. Term structure
    strike_iv = iv_term_structure(strike_iv, dte, earnings_in_window)
    strike_iv = float(np.clip(strike_iv, 0.05, 3.0))

    # 3. Pricing
    if american:
        g = binomial_greeks(S, K, T, r, strike_iv, q, steps)
    else:
        g = _bs_greeks_q(S, K, T, r, strike_iv, q)

    # 4. Greeks 2a ordem
    g2 = second_order_greeks(S, K, T, r, strike_iv, q)
    g.update(g2)
    g["strike_iv"] = strike_iv
    g["dividend_yield"] = q
    return g


def _bs_greeks_q(S, K, T, r, sigma, q=0.0):
    """Black-Scholes europeu com dividend yield (fallback)."""
    if T <= 0 or sigma <= 0:
        return {"engine": "BS-European", "price": float(max(S-K, 0)),
                "delta": 1.0 if S > K else 0.0, "gamma": 0.0,
                "theta": 0.0, "vega": 0.0, "rho": 0.0}
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return {
        "engine": "BS-European",
        "price": float(max(price, 0)),
        "delta": float(np.exp(-q*T)*norm.cdf(d1)),
        "gamma": float(np.exp(-q*T)*norm.pdf(d1)/(S*sigma*np.sqrt(T))),
        "theta": float((-S*np.exp(-q*T)*norm.pdf(d1)*sigma/(2*np.sqrt(T))
                        - r*K*np.exp(-r*T)*norm.cdf(d2)
                        + q*S*np.exp(-q*T)*norm.cdf(d1))/365),
        "vega":  float(S*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)/100),
        "rho":   float(K*T*np.exp(-r*T)*norm.cdf(d2)/100),
    }


# ══════════════════════════════════════════════════════════════════════════════
# YFINANCE — OPTIONS CHAIN REAL (com timeout robusto)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_real_chain(ticker, target_dte=30, timeout=8):
    """
    Tenta obter a options chain real via yfinance, com timeout em thread
    separada (evita pendurar no Windows). Retorna None se falhar.
    """
    import threading
    result = {}

    def _worker():
        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)
            expiries = tk.options
            if not expiries:
                return
            from datetime import datetime
            today = datetime.today()
            best_exp, best_diff = None, 1e9
            for exp in expiries:
                d = abs((datetime.strptime(exp, "%Y-%m-%d") - today).days - target_dte)
                if d < best_diff:
                    best_diff, best_exp = d, exp
            chain = tk.option_chain(best_exp)
            calls = chain.calls
            spot = tk.history(period="1d")["Close"].iloc[-1]
            result["data"] = {
                "expiry": best_exp,
                "spot": float(spot),
                "calls": [
                    {"strike": float(row.strike),
                     "iv": float(row.impliedVolatility) if row.impliedVolatility else None,
                     "bid": float(row.bid), "ask": float(row.ask),
                     "last": float(row.lastPrice), "volume": int(row.volume or 0),
                     "oi": int(row.openInterest or 0)}
                    for row in calls.itertuples()
                ],
            }
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result.get("data")
