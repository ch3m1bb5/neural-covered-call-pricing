"""
N-HiTS Covered Call Engine — Flask Backend v4
==============================================
Novidades v4:
  - CORRIGIDO: bug critico na geracao de precos (Brownian Bridge propria)
    que estava a inflacionar toda a serie ~30x antes do ultimo ponto.
    Afectava RSI, Bollinger, S/R, N-HiTS e ARIMA em TODAS as versoes anteriores.
  - Varredura de vol expandida: [1h,4h,8h,12h,1d,3d,5d,10d,15d,20d,30d,45d]
    usando serie horaria sintetica separada (1800h ~ 75 dias)
  - Strike manual mantido (v3)

pip install flask flask-cors numpy pandas scipy requests
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import warnings
import requests

import pricing as P  # modulo avancado: GARCH, Binomial, SABR, term structure

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

YFINANCE_OK = False

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MSE
    NHITS_OK = True
except ImportError:
    NHITS_OK = False

try:
    import pmdarima as pm
    PMDARIMA_OK = True
except ImportError:
    PMDARIMA_OK = False

try:
    import QuantLib as ql
    QUANTLIB_OK = True
except ImportError:
    QUANTLIB_OK = False

CLAUDE_MODEL = "claude-opus-4-8"


# ══════════════════════════════════════════════════════════════════════════════
# GERACAO DE PRECOS — PONTE BROWNIANA (CORRIGIDO)
# ══════════════════════════════════════════════════════════════════════════════

def _brownian_bridge(start_price, end_price, n, vol, seed):
    """
    Gera caminho de precos que comeca EXACTAMENTE em start_price e termina
    EXACTAMENTE em end_price, com ruido Browniano realista no meio.

    Metodo: gera um random walk em log-espaco, subtrai a sua propria
    tendencia linear (start->end do random walk), depois soma a tendencia
    deterministica desejada entre log(start_price) e log(end_price).
    Isto evita o bug de versoes anteriores onde multiplicar por cost_basis
    E pelo factor de escala em simultaneo inflacionava toda a serie.
    """
    np.random.seed(seed)
    t = np.arange(n)
    raw_returns = np.random.normal(0, vol, n)
    W = np.cumsum(raw_returns)
    bridge_noise = W - W[0] - (t / (n - 1)) * (W[-1] - W[0])
    log_start, log_end = np.log(start_price), np.log(end_price)
    log_path = log_start + (log_end - log_start) * (t / (n - 1)) + bridge_noise
    prices = np.exp(log_path)
    prices[0] = start_price
    prices[-1] = end_price
    return prices


def fetch_prices(ticker, current_price, cost_basis):
    """Serie diaria (~126 dias uteis) para N-HiTS, ARIMA, RSI, Bollinger, S/R."""
    seed = abs(hash(ticker)) % (2 ** 31)
    dates = pd.bdate_range(end=datetime.today(), periods=126)
    n = len(dates)  # bdate_range pode devolver n-1 datas dependendo da hora actual
    drift_total = np.log(current_price / max(cost_basis, 0.01))
    implied_daily_vol = max(abs(drift_total) / np.sqrt(n) * 1.5, 0.015)
    prices = _brownian_bridge(cost_basis, current_price, n, implied_daily_vol, seed)
    return pd.DataFrame({"close": prices}, index=dates)


def fetch_hourly_prices(ticker, current_price, cost_basis, n_hours=1800):
    """
    Serie horaria sintetica (1800h ~ 75 dias, convencao 24h/dia de trading)
    dedicada a varredura de janelas sub-diarias [1h,4h,8h,12h].
    Calibrada ao mesmo nivel de vol implicita que a serie diaria.
    """
    seed = abs(hash(ticker + "_hourly")) % (2 ** 31)
    drift_total = np.log(current_price / max(cost_basis, 0.01))
    implied_daily_vol = max(abs(drift_total) / np.sqrt(126) * 1.5, 0.015)
    implied_hourly_vol = implied_daily_vol / np.sqrt(24)
    return _brownian_bridge(cost_basis, current_price, n_hours, implied_hourly_vol, seed)


# ══════════════════════════════════════════════════════════════════════════════
# VARREDURA DE VOLATILIDADE — 12 JANELAS (1h ATE 45d)
# ══════════════════════════════════════════════════════════════════════════════

def sweep_vol_window(hourly_prices: np.ndarray) -> dict:
    """
    Varredura: [1h,4h,8h,12h,1d,3d,5d,10d,15d,20d,30d,45d]
    Convencao: 24h sinteticas = 1 dia de trading => 252*24 = 6048 horas/ano.
    Janelas sub-diarias (1h-12h) capturam micro-regime / vol intraday.
    Janelas diarias (1d-45d) capturam vol estrutural.
    Escolhe a janela que minimiza o erro de previsao rolling 1-step.
    """
    HOURS_PER_YEAR = 252 * 24  # 6048

    window_defs = [
        ("1h", 1), ("4h", 4), ("8h", 8), ("12h", 12),
        ("1d", 24), ("3d", 72), ("5d", 120), ("10d", 240),
        ("15d", 360), ("20d", 480), ("30d", 720), ("45d", 1080),
    ]

    log_ret = np.log(hourly_prices[1:] / hourly_prices[:-1])
    results = {}

    # Pre-computa rolling std vectorizado (cumsum trick) para todas as janelas
    def rolling_std_mse(lr, w):
        """MSE entre vol prevista (rolling w) e |retorno| seguinte — vectorizado."""
        n = len(lr)
        if n < w + 2:
            return 999.0
        # Rolling variance via cumsum de x e x^2
        c1 = np.cumsum(np.insert(lr, 0, 0.0))
        c2 = np.cumsum(np.insert(lr**2, 0, 0.0))
        # janelas [i-w, i) para i de w ate n-1
        idx = np.arange(w, n)
        s1 = c1[idx] - c1[idx - w]
        s2 = c2[idx] - c2[idx - w]
        mean = s1 / w
        var = np.maximum(s2 / w - mean**2, 0.0)
        pred_vol = np.sqrt(var)
        actual = np.abs(lr[idx])  # retorno no instante i (mesma indexacao)
        # alinhar: prever lr[i] usando janela [i-w, i)
        m = min(len(pred_vol), len(actual))
        return float(np.mean((actual[:m] - pred_vol[:m]) ** 2)) if m else 999.0

    for label, w in window_defs:
        if len(log_ret) < w + 3:
            continue

        if w == 1:
            hv = float(np.abs(log_ret[-1]) * np.sqrt(HOURS_PER_YEAR))
            tail = log_ret[-31:]
            mse = float(np.mean((np.abs(tail[1:]) - np.abs(tail[:-1])) ** 2)) if len(tail) > 1 else 999.0
        else:
            hv = float(np.std(log_ret[-w:]) * np.sqrt(HOURS_PER_YEAR))
            mse = rolling_std_mse(log_ret, w)

        results[label] = {"hv": hv, "mse": mse, "hours": w}

    if not results:
        hv = float(np.std(log_ret) * np.sqrt(HOURS_PER_YEAR))
        return {"best_window": "1d", "best_window_hours": 24, "hv": hv,
                "iv": float(np.clip(hv * 1.15, 0.05, 2.0)), "by_window": {"1d": round(hv * 100, 2)}}

    best_label = min(results, key=lambda k: results[k]["mse"])
    best_hv    = results[best_label]["hv"]
    best_hours = results[best_label]["hours"]

    # Penalizacao leve para janelas muito curtas (estimativas mais ruidosas)
    window_penalty = 1.0 + (24 / (best_hours + 24)) * 0.15
    iv_premium = (1.10 + best_hv * 0.5) * window_penalty
    iv = float(np.clip(best_hv * iv_premium, 0.05, 2.0))

    return {
        "best_window": best_label,
        "best_window_hours": best_hours,
        "hv": best_hv,
        "iv": iv,
        "by_window": {label: round(v["hv"] * 100, 2) for label, v in results.items()},
    }


def sweep_ar_order(prices, max_p=6):
    best_p, best_aic = 1, np.inf
    n = len(prices)
    for p in range(1, max_p + 1):
        if n <= p + 5:
            break
        X = np.array([prices[i:n - p + i] for i in range(p)]).T
        y = prices[p:]
        try:
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            rss = float(np.sum((y - X @ coeffs) ** 2))
            aic = len(y) * np.log(max(rss / len(y), 1e-10)) + 2 * p
            if aic < best_aic:
                best_aic = aic
                best_p = p
        except Exception:
            continue
    return best_p


def sweep_fourier_harmonics(series, period=5):
    n = len(series)
    t = np.arange(n)
    best_h, best_rss = 1, np.inf
    for h_max in range(1, 4):
        Xf = [np.ones(n)]
        for h in range(1, h_max + 1):
            Xf += [np.sin(2*np.pi*h*t/period), np.cos(2*np.pi*h*t/period)]
        X = np.column_stack(Xf)
        try:
            c, *_ = np.linalg.lstsq(X, series, rcond=None)
            rss = float(np.sum((series - X @ c) ** 2))
            if rss < best_rss:
                best_rss = rss
                best_h = h_max
        except Exception:
            continue
    return best_h


# ══════════════════════════════════════════════════════════════════════════════
# N-HiTS
# ══════════════════════════════════════════════════════════════════════════════

def run_nhits(df, horizon=20, sweep=None, fast=True):
    if not NHITS_OK:
        return _nhits_fallback(df, horizon, sweep)

    input_size = min(48, len(df) - horizon - 1)
    nf_df = pd.DataFrame({"unique_id": "STOCK", "ds": df.index, "y": df["close"].values})
    # fast=True reduz drasticamente o custo de treino (200ms-2s em vez de 30-120s)
    max_steps = 60 if fast else 300
    mlp = [[128, 128]] * 3 if fast else [[256, 256]] * 3
    model = NHITS(
        h=horizon, input_size=input_size,
        stack_types=["identity", "identity", "identity"],
        n_stacks=3, n_blocks=[1, 1, 1],
        mlp_units=mlp,
        n_pool_kernel_size=[4, 2, 1], n_freq_downsample=[4, 2, 1],
        dropout_prob_theta=0.0, loss=MSE(),
        max_steps=max_steps, val_check_steps=max_steps,
        early_stop_patience_steps=-1,
        scaler_type="standard", random_seed=42, verbose_fit=False,
        enable_progress_bar=False,
    )
    nf = NeuralForecast(models=[model], freq="B")
    nf.fit(nf_df)
    fc = nf.predict()["NHITS"].values
    last = float(df["close"].iloc[-1])
    trend = _linear_trend(df["close"].values, horizon)
    seasonal = _fourier_seasonal(df["close"].values, horizon)
    return _package_nhits(fc, trend, seasonal, fc - trend - seasonal, last, horizon,
                          "N-HiTS (neuralforecast, fast)" if fast else "N-HiTS (neuralforecast)")


def _nhits_fallback(df, horizon, sweep=None):
    prices = df["close"].values.astype(float)
    n = len(prices)
    best_p = sweep.get("best_ar_p", 3) if sweep else 3
    best_h = sweep.get("best_harmonics", 2) if sweep else 2

    pool4 = _maxpool(prices, 4)
    trend_fc = _linear_interp(_ar_forecast(pool4, horizon // 4 + 2, p=min(best_p, max(1, len(pool4)//3))), horizon)[:horizon]
    res1 = prices - _linear_interp(_maxpool(prices, 4), n)[:n]

    pool2 = _maxpool(res1, 2)
    seas_fc = _linear_interp(_fourier_forecast(pool2, horizon // 2 + 2, period=5, n_harmonics=best_h), horizon)[:horizon]
    res2 = res1 - _linear_interp(_maxpool(res1, 2), n)[:n]

    short_fc = _ar_forecast(res2[-30:], horizon, p=min(best_p, 5))[:horizon]

    last = prices[-1]
    combined = trend_fc + seas_fc + short_fc
    combined = combined - combined[0] + last
    return _package_nhits(combined, trend_fc, seas_fc, short_fc, last, horizon,
                          f"N-HiTS numpy (AR p={best_p}, Fourier h={best_h})")


def _package_nhits(forecast, trend, seasonal, residual, last, horizon, model_name):
    h = len(forecast)
    idx_10 = min(9, h - 1)
    return {
        "model": model_name, "horizon": h,
        "forecast": list(map(float, forecast)),
        "trend":    list(map(float, trend[:h])),
        "seasonal": list(map(float, seasonal[:h])),
        "residual": list(map(float, residual[:h])),
        "last_price": float(last),
        "target_10d": float(forecast[idx_10]),
        "target_20d": float(forecast[-1]),
        "direction":  "UP" if forecast[-1] > last else "DOWN",
        "confidence": float(np.clip(1 - np.std(forecast) / (last + 1e-9) * 5, 0.30, 0.95)),
    }


def _maxpool(x, k):
    n = (len(x) // k) * k
    return x[:n].reshape(-1, k).max(axis=1)

def _linear_interp(x, target_len):
    return np.interp(np.linspace(0, len(x)-1, target_len), np.arange(len(x)), x)

def _ar_forecast(series, steps, p=3):
    s = series[-max(30, p + steps):]
    n = len(s)
    if n <= p:
        return np.full(steps, float(s[-1]))
    X = np.array([s[i:n-p+i] for i in range(p)]).T
    y = s[p:]
    try:
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return np.full(steps, float(s[-1]))
    buf = list(s[-p:])
    out = []
    for _ in range(steps):
        val = float(np.dot(coeffs, buf[-p:]))
        out.append(val)
        buf.append(val)
    return np.array(out)

def _fourier_forecast(series, steps, period=5, n_harmonics=2):
    n = len(series)
    t = np.arange(n)
    Xf = [np.ones(n)]
    for h in range(1, n_harmonics + 1):
        Xf += [np.sin(2*np.pi*h*t/period), np.cos(2*np.pi*h*t/period)]
    Xf = np.column_stack(Xf)
    try:
        c, *_ = np.linalg.lstsq(Xf, series, rcond=None)
    except Exception:
        return np.zeros(steps)
    tf = np.arange(n, n + steps)
    Xp = [np.ones(steps)]
    for h in range(1, n_harmonics + 1):
        Xp += [np.sin(2*np.pi*h*tf/period), np.cos(2*np.pi*h*tf/period)]
    return np.column_stack(Xp) @ c

def _linear_trend(prices, horizon):
    t = np.arange(len(prices))
    c = np.polyfit(t, prices, 1)
    return np.polyval(c, np.arange(len(prices), len(prices) + horizon))

def _fourier_seasonal(prices, horizon, period=5, n_harmonics=2):
    return _fourier_forecast(prices, horizon, period, n_harmonics)


# ══════════════════════════════════════════════════════════════════════════════
# ARIMA
# ══════════════════════════════════════════════════════════════════════════════

def run_arima(prices, horizon, best_p=3):
    if PMDARIMA_OK:
        try:
            model = pm.auto_arima(prices, seasonal=True, m=5, information_criterion="aic",
                                   stepwise=True, suppress_warnings=True, error_action="ignore",
                                   max_p=best_p, max_q=best_p)
            fc, ci = model.predict(n_periods=horizon, return_conf_int=True)
            return {"model": f"ARIMA{model.order}", "forecast": fc.tolist(),
                    "lower": ci[:,0].tolist(), "upper": ci[:,1].tolist(), "aic": float(model.aic())}
        except Exception:
            pass
    fc  = _ar_forecast(prices, horizon, p=best_p)
    std = np.std(np.diff(prices[-30:])) * np.sqrt(np.arange(1, horizon + 1))
    return {"model": f"AR({best_p}) fallback", "forecast": fc.tolist(),
            "lower": (fc - 1.96*std).tolist(), "upper": (fc + 1.96*std).tolist(), "aic": None}


# ══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES
# ══════════════════════════════════════════════════════════════════════════════

def black_scholes_greeks(S, K, T, r, sigma):
    if QUANTLIB_OK:
        try:
            today  = ql.Date.todaysDate()
            ql.Settings.instance().evaluationDate = today
            expiry = today + ql.Period(max(1, int(T*365)), ql.Days)
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(K))
            option = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry))
            spot   = ql.SimpleQuote(float(S))
            rTS = ql.YieldTermStructureHandle(ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(float(r))), ql.Actual365Fixed()))
            qTS = ql.YieldTermStructureHandle(ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(0.0)), ql.Actual365Fixed()))
            volTS = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(float(sigma))), ql.Actual365Fixed()))
            process = ql.BlackScholesMertonProcess(ql.QuoteHandle(spot), qTS, rTS, volTS)
            option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
            return {"engine": "QuantLib", "price": float(option.NPV()), "delta": float(option.delta()),
                    "gamma": float(option.gamma()), "theta": float(option.theta()/365),
                    "vega": float(option.vega()/100), "rho": float(option.rho()/100)}
        except Exception:
            pass
    if T <= 0:
        return {"engine": "scipy/BS", "price": float(max(S-K,0)),
                "delta": 1.0 if S>K else 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    try:
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return {
            "engine": "scipy/BS",
            "price": float(max(S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2), 0)),
            "delta": float(norm.cdf(d1)),
            "gamma": float(norm.pdf(d1) / (S*sigma*np.sqrt(T))),
            "theta": float((-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365),
            "vega":  float(S*norm.pdf(d1)*np.sqrt(T)/100),
            "rho":   float(K*T*np.exp(-r*T)*norm.cdf(d2)/100),
        }
    except Exception:
        return {"engine": "scipy/BS", "price": 0.0, "delta": 0.0,
                "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

def strike_ladder(S, T, r, sigma, strikes, q=0.0, dte=30,
                  sabr_params=None, american=True, earnings=False):
    """Ladder usando o pricer avancado (SABR + term structure + americano)."""
    out = []
    for K in strikes:
        g = P.price_option(S, K, T, r, sigma, q=q, dte=dte,
                           earnings_in_window=earnings, sabr_params=sabr_params,
                           american=american, steps=120)
        out.append({"strike": K, **g})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STRIKES ADAPTATIVOS
# ══════════════════════════════════════════════════════════════════════════════

def smart_strikes(current_price):
    S = current_price
    if S < 10:    step = 0.5
    elif S < 25:  step = 1.0
    elif S < 75:  step = 2.5
    elif S < 200: step = 5.0
    else:         step = 10.0
    base = round(S / step) * step
    return [round(base + i * step, 2) for i in range(-4, 16) if base + i * step > 0]


# ══════════════════════════════════════════════════════════════════════════════
# INDICADORES TECNICOS
# ══════════════════════════════════════════════════════════════════════════════

def rsi(prices, period=14):
    if len(prices) < period + 2:
        return 50.0
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    ag = np.mean(gains[:period])
    al = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        ag = (ag*(period-1) + gains[i]) / period
        al = (al*(period-1) + losses[i]) / period
    return float(100.0 if al == 0 else 100 - 100 / (1 + ag/al))

def bollinger(prices, window=20, k=2):
    if len(prices) < window:
        m = float(np.mean(prices)); s = float(np.std(prices))
        return {"upper": m+k*s, "mid": m, "lower": m-k*s, "bandwidth": 2*k*s/(m+1e-9)}
    s   = pd.Series(prices)
    mid = float(s.rolling(window).mean().iloc[-1])
    std = float(s.rolling(window).std().iloc[-1])
    return {"upper": mid+k*std, "mid": mid, "lower": mid-k*std, "bandwidth": 2*k*std/(mid+1e-9)}

def support_resistance(prices, n=5):
    highs, lows = [], []
    for i in range(2, len(prices)-2):
        if prices[i] > max(prices[i-2], prices[i-1], prices[i+1], prices[i+2]):
            highs.append(float(prices[i]))
        if prices[i] < min(prices[i-2], prices[i-1], prices[i+1], prices[i+2]):
            lows.append(float(prices[i]))
    return {
        "resistance": sorted(set(round(h,2) for h in highs), reverse=True)[:n],
        "support":    sorted(set(round(l,2) for l in lows))[:n],
    }


# ══════════════════════════════════════════════════════════════════════════════
# ANALISE DE STRIKE (AUTO OU MANUAL)
# ══════════════════════════════════════════════════════════════════════════════

def analyse_strike(S, K, T, r, iv, cost_basis, shares, nhits, daily_rsi, bb,
                   q=0.0, dte=30, sabr_params=None, american=True, earnings=False):
    greeks    = P.price_option(S, K, T, r, iv, q=q, dte=dte,
                               earnings_in_window=earnings, sabr_params=sabr_params,
                               american=american, steps=120)
    premium   = greeks["price"]
    ann_yield = (premium / S) * (365 / max(T*365, 1)) * 100
    total_pnl = (K - cost_basis + premium) * shares
    pnl_pct   = (K - cost_basis + premium) / cost_basis * 100
    breakeven = cost_basis - premium
    signals, score = [], 0

    def sig(label, stype, w):
        signals.append({"label": label, "type": stype, "weight": w})

    if   daily_rsi > 70: score += 3; sig(f"RSI {daily_rsi:.0f} — overbought forte", "bullish", 3)
    elif daily_rsi > 62: score += 2; sig(f"RSI {daily_rsi:.0f} — elevado", "bullish", 2)
    elif daily_rsi > 55: score += 1; sig(f"RSI {daily_rsi:.0f} — moderado", "neutral", 1)
    elif daily_rsi < 35: score -= 3; sig(f"RSI {daily_rsi:.0f} — oversold, aguardar bounce", "bearish", -3)
    elif daily_rsi < 45: score -= 1; sig(f"RSI {daily_rsi:.0f} — fraco", "bearish", -1)

    fc_chg = (nhits["target_10d"] - S) / S * 100
    if   fc_chg >  4: score -= 3; sig(f"N-HiTS: +{fc_chg:.1f}% — upside forte, nao vender", "bearish", -3)
    elif fc_chg >  2: score -= 2; sig(f"N-HiTS: +{fc_chg:.1f}% — upside moderado", "bearish", -2)
    elif fc_chg >  0.5: score -= 1; sig(f"N-HiTS: +{fc_chg:.1f}% — upside leve", "neutral", -1)
    elif fc_chg < -2: score += 3; sig(f"N-HiTS: {fc_chg:.1f}% — queda prevista", "bullish", 3)
    elif fc_chg < -0.5: score += 2; sig(f"N-HiTS: {fc_chg:.1f}% — tendencia negativa", "bullish", 2)
    else:             score += 1; sig(f"N-HiTS: {fc_chg:.1f}% — previsao lateral", "neutral", 1)

    gap = K - cost_basis
    if   gap >= cost_basis*0.05: score += 4; sig(f"Strike ${K:.2f} bem acima do custo (+${gap:.2f})", "bullish", 4)
    elif gap >= 0:               score += 2; sig(f"Strike ${K:.2f} acima do custo (+${gap:.2f})", "bullish", 2)
    elif gap >= -cost_basis*0.03:score -= 2; sig(f"Strike ${K:.2f} ligeiramente abaixo do custo", "bearish", -2)
    else:                        score -= 4; sig(f"Strike ${K:.2f} ABAIXO do custo (${gap:.2f})", "bearish", -4)

    if bb["upper"] > bb["lower"]:
        pct_b = (S - bb["lower"]) / (bb["upper"] - bb["lower"])
        if   pct_b > 0.85: score += 2; sig("Preco perto da banda superior Bollinger", "bullish", 2)
        elif pct_b > 0.65: score += 1; sig("Preco na metade superior Bollinger", "neutral", 1)
        elif pct_b < 0.15: score -= 2; sig("Preco perto da banda inferior", "bearish", -2)

    if total_pnl > 0: score += 2; sig(f"PnL se exercida: +${total_pnl:,.0f}", "bullish", 2)
    else:             score -= 3; sig(f"PnL se exercida: -${abs(total_pnl):,.0f} PREJUIZO", "bearish", -3)

    if   ann_yield > 20: score += 2; sig(f"Premio anualizado: {ann_yield:.1f}% excelente", "bullish", 2)
    elif ann_yield > 10: score += 1; sig(f"Premio anualizado: {ann_yield:.1f}% bom", "neutral", 1)
    elif ann_yield < 3:  score -= 1; sig(f"Premio anualizado: {ann_yield:.1f}% baixo", "bearish", -1)

    if   score >= 8: verdict, color, icon = "VENDER CALL", "#10b981", "OK"
    elif score >= 4: verdict, color, icon = "CONSIDERAR",  "#f59e0b", "?"
    elif score >= 0: verdict, color, icon = "AGUARDAR",    "#f59e0b", "..."
    else:            verdict, color, icon = "NAO VENDER",  "#ef4444", "X"

    rationale = {
        "VENDER CALL": "Setup excelente: RSI elevado, N-HiTS flat/negativo, strike acima do custo.",
        "CONSIDERAR":  "Condicoes favoraveis mas nao ideais. Verifique resistencias antes de agir.",
        "AGUARDAR":    "Sinais mistos. Aguarde RSI mais elevado ou strike acima do custo base.",
        "NAO VENDER":  "Vender agora seria prejudicial. N-HiTS preve upside ou strike abaixo do custo.",
    }[verdict]

    return {
        "strike": K, "greeks": greeks, "premium": premium,
        "ann_yield": ann_yield, "total_pnl": total_pnl,
        "pnl_pct": pnl_pct, "breakeven": breakeven,
        "score": score, "max_score": 16,
        "signals": signals, "verdict": verdict,
        "color": color, "icon": icon, "rationale": rationale,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ANALISE CLAUDE
# ══════════════════════════════════════════════════════════════════════════════

def claude_analysis(payload):
    api_key = payload.get("api_key", "")
    if not api_key or not api_key.startswith("sk-ant"):
        return "Configure a API key do Anthropic para activar a analise AI."
    d   = payload["data"]
    rec = d["auto_strike"]
    man = d.get("manual_strike")
    manual_section = ""
    if man:
        manual_section = f"""
STRIKE MANUAL: ${man['strike']:.2f}
  Delta: {man['greeks']['delta']:.3f} | Premio: ${man['premium']:.2f}/accao
  PnL se exercida: ${man['total_pnl']:+,.0f} ({man['pnl_pct']:.1f}%)
  Veredicto: {man['verdict']} (score {man['score']}/{man['max_score']})"""

    prompt = f"""Voce e um estrategista de opcoes profissional especializado em covered calls.

POSICAO:
- Ticker: {d['ticker']} | Preco actual: ${d['current_price']:.2f} | Custo base: ${d['cost_basis']:.2f}
- {d['shares']} accoes | PnL nao realizado: ${(d['current_price']-d['cost_basis'])*d['shares']:+,.0f}

STRIKE AUTO (delta alvo {d['target_delta']:.2f}): ${rec['strike']:.2f}
  Delta: {rec['greeks']['delta']:.3f} | Premio: ${rec['premium']:.2f}/accao
  PnL se exercida: ${rec['total_pnl']:+,.0f} ({rec['pnl_pct']:.1f}%)
  Yield anualizado: {rec['ann_yield']:.1f}% | Veredicto: {rec['verdict']} (score {rec['score']}/{rec['max_score']})
{manual_section}

N-HiTS FORECAST:
  10d: ${d['nhits']['target_10d']:.2f} ({(d['nhits']['target_10d']/d['current_price']-1)*100:+.1f}%)
  20d: ${d['nhits']['target_20d']:.2f} | Direccao: {d['nhits']['direction']} | Confianca: {d['nhits']['confidence']*100:.0f}%

VARREDURA VOL (12 janelas, 1h-45d): melhor={d['vol_sweep']['best_window']} | HV={d['vol_sweep']['hv']*100:.1f}% | IV={d['iv']*100:.1f}%
Por janela (HV%): {d['vol_sweep']['by_window']}
RSI: Diario {d['daily_rsi']:.1f} | Semanal {d['weekly_rsi']:.1f}

Responda EM PORTUGUES em 4 seccoes:
1. TIMING (vender agora ou aguardar? 1-2 frases directas)
2. STRIKE IDEAL (se ha strike manual, compare com o auto — qual e melhor e porque?)
3. RISCOS PRINCIPAIS (2-3 bullets)
4. CONCLUSAO (1 frase accionavel e decisiva)

Seja directo. Tome uma posicao clara."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": CLAUDE_MODEL, "max_tokens": 900,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()["content"][0]["text"]
        return f"Erro API Claude: {resp.status_code}"
    except Exception as e:
        return f"Erro de conexao: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "5.0",
                    "libs": {"yfinance": YFINANCE_OK, "neuralforecast_nhits": NHITS_OK,
                             "pmdarima": PMDARIMA_OK, "quantlib": QUANTLIB_OK,
                             "claude_model": CLAUDE_MODEL}})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body          = request.json or {}
    ticker        = str(body.get("ticker", "UAL")).upper()
    cost_basis    = float(body.get("cost_basis", 100))
    current_price = float(body.get("current_price", cost_basis))
    shares        = int(body.get("shares", 100))
    dte           = max(1, int(body.get("dte", 30)))
    risk_free     = float(body.get("risk_free", 0.053))
    target_delta  = float(body.get("target_delta", 0.20))
    horizon       = max(5, int(body.get("nhits_horizon", 20)))
    api_key       = str(body.get("api_key", ""))

    # NOVOS parametros v5
    div_yield     = float(body.get("dividend_yield", 0.0))       # ex: 0.015 = 1.5%
    earnings      = bool(body.get("earnings_in_window", False))  # earnings dentro do DTE?
    american      = bool(body.get("american", True))            # pricing americano (Binomial)
    use_sabr      = bool(body.get("use_sabr", True))            # smile de vol via SABR
    use_garch     = bool(body.get("use_garch", True))           # forecast de vol via GARCH

    try:
        manual_strike = float(body.get("manual_strike") or 0) or None
    except (TypeError, ValueError):
        manual_strike = None

    T = dte / 365

    # Series sinteticas (CORRIGIDAS via Brownian Bridge)
    df            = fetch_prices(ticker, current_price, cost_basis)
    prices        = df["close"].values
    hourly_prices = fetch_hourly_prices(ticker, current_price, cost_basis, n_hours=1800)

    # Varredura de parametros
    vol_sweep    = sweep_vol_window(hourly_prices)
    best_ar_p    = sweep_ar_order(prices)
    best_h_f     = sweep_fourier_harmonics(prices - np.mean(prices))
    iv, hv       = vol_sweep["iv"], vol_sweep["hv"]

    # GARCH(1,1): forecast dinamico de vol — substitui IV base se activado
    log_rets = np.log(prices[1:] / prices[:-1])
    garch = P.fit_garch(log_rets)
    if use_garch and garch["converged"]:
        # Combina IV do sweep com forecast GARCH (media ponderada)
        iv = float(np.clip(0.5 * iv + 0.5 * garch["next_vol_annual"], 0.05, 2.0))

    sweep_params = {"best_ar_p": best_ar_p, "best_harmonics": best_h_f,
                    "vol_window": vol_sweep["best_window"]}

    # SABR: calibra smile a partir da IV ATM
    forward = current_price * np.exp((risk_free - div_yield) * T)
    sabr_params = P.calibrate_sabr(forward, T, iv) if use_sabr else None

    all_strikes = smart_strikes(current_price)
    if manual_strike and manual_strike not in all_strikes:
        all_strikes = sorted(all_strikes + [round(manual_strike, 2)])

    ladder = strike_ladder(current_price, T, risk_free, iv, all_strikes,
                           q=div_yield, dte=dte, sabr_params=sabr_params,
                           american=american, earnings=earnings)
    nhits_result = run_nhits(df, horizon, sweep_params)
    arima_result = run_arima(prices, horizon, best_ar_p)
    daily_rsi    = rsi(prices, 14)
    weekly_rsi   = rsi(prices[::5], 14)
    bb           = bollinger(prices)
    sr           = support_resistance(prices)

    strike_kwargs = dict(q=div_yield, dte=dte, sabr_params=sabr_params,
                         american=american, earnings=earnings)

    rec_row       = min(ladder, key=lambda r: abs(r["delta"] - target_delta))
    auto_K        = rec_row["strike"]
    auto_analysis = analyse_strike(current_price, auto_K, T, risk_free, iv,
                                   cost_basis, shares, nhits_result, daily_rsi, bb,
                                   **strike_kwargs)

    manual_analysis = None
    if manual_strike and manual_strike > 0:
        manual_analysis = analyse_strike(current_price, manual_strike, T, risk_free, iv,
                                         cost_basis, shares, nhits_result, daily_rsi, bb,
                                         **strike_kwargs)

    result = {
        "ticker": ticker, "current_price": current_price, "cost_basis": cost_basis,
        "shares": shares, "hv30": hv, "iv": iv, "dte": dte,
        "risk_free": risk_free, "target_delta": target_delta,
        "dividend_yield": div_yield, "earnings_in_window": earnings,
        "american": american, "use_sabr": use_sabr, "use_garch": use_garch,
        "garch": garch,
        "sabr": sabr_params if sabr_params else {},
        "recommended_strike":   auto_K,
        "premium":              auto_analysis["premium"],
        "ann_yield":            auto_analysis["ann_yield"],
        "breakeven":            auto_analysis["breakeven"],
        "total_pnl_if_called":  auto_analysis["total_pnl"],
        "pnl_pct_if_called":    auto_analysis["pnl_pct"],
        "greeks":               auto_analysis["greeks"],
        "decision": {
            "verdict":   auto_analysis["verdict"],
            "color":     auto_analysis["color"],
            "icon":      auto_analysis["icon"],
            "score":     auto_analysis["score"],
            "max_score": auto_analysis["max_score"],
            "rationale": auto_analysis["rationale"],
            "signals":   auto_analysis["signals"],
        },
        "auto_strike":        auto_analysis,
        "manual_strike":      manual_analysis,
        "strike_ladder":      ladder,
        "nhits":              nhits_result,
        "arima":              arima_result,
        "daily_rsi":          daily_rsi,
        "weekly_rsi":         weekly_rsi,
        "bollinger":          bb,
        "support_resistance": sr,
        "vol_sweep":          vol_sweep,
        "sweep_params":       sweep_params,
        "price_history":      {"dates": [str(d.date()) for d in df.index],
                               "prices": prices.tolist()},
        "libs": {"nhits_model": nhits_result["model"],
                 "arima_model": arima_result["model"],
                 "greeks_engine": auto_analysis["greeks"]["engine"]},
        "ai_analysis": "",
        # Guarda contexto para o endpoint AI assincrono (chamado depois pelo frontend)
        "_ai_pending": bool(api_key and api_key.startswith("sk-ant")),
    }

    # NOTA: a analise Claude NAO e feita aqui (bloquearia o dashboard ate 30s).
    # O frontend chama /api/ai_analysis separadamente apos renderizar.

    return jsonify(result)


@app.route("/api/ai_analysis", methods=["POST"])
def ai_analysis_route():
    """Endpoint AI separado — chamado pelo frontend DEPOIS de renderizar o dashboard."""
    return jsonify({"analysis": claude_analysis(request.json or {})})


if __name__ == "__main__":
    print(f"N-HiTS:   {'PyTorch' if NHITS_OK else 'numpy fallback'}")
    print(f"QuantLib: {'OK' if QUANTLIB_OK else 'scipy fallback'}")
    print(f"pmdarima: {'OK' if PMDARIMA_OK else 'AR fallback'}")
    print(f"Claude:   {CLAUDE_MODEL}")
    print(f"Vol sweep: 12 janelas (1h-45d) | GARCH + SABR + Binomial Americano + Term Structure")
    app.run(debug=True, port=5050, use_reloader=False)
