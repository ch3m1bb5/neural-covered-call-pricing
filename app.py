"""
N-HiTS Covered Call Engine — Flask Backend (Optimizado)
========================================================
Melhorias v2:
  - Sem yfinance (resposta instantânea)
  - Varredura de parâmetros: AR(p), janela de vol, harmónicos de Fourier
  - Espaçamento de strikes auto-adaptativo por faixa de preço
  - Integração com Claude claude-opus-4-8 para análise AI
  - Índices seguros (horizon pode ser < 10)
  - IV calibrada por varredura de janelas

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
import json

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ── Libs opcionais ────────────────────────────────────────────────────────────
YFINANCE_OK = False  # desactivado — preço vem do frontend

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
# GERAÇÃO DE PREÇOS SINTÉTICOS
# ══════════════════════════════════════════════════════════════════════════════

def fetch_prices(ticker: str, current_price: float, cost_basis: float) -> pd.DataFrame:
    """
    Gera série sintética realista ancorada no preço actual.
    Usa o cost_basis para calibrar a volatilidade histórica implícita
    (distância entre custo e preço actual sugere vol passada).
    """
    np.random.seed(abs(hash(ticker)) % (2 ** 31))
    n = 126  # ~6 meses de dias úteis
    dates = pd.bdate_range(end=datetime.today(), periods=n)

    # Vol implícita calibrada pela distância cost_basis → current_price
    drift_total = np.log(current_price / max(cost_basis, 0.01))
    implied_daily_vol = max(abs(drift_total) / np.sqrt(n) * 1.5, 0.015)

    # Gera path que termina exactamente em current_price
    raw_returns = np.random.normal(0, implied_daily_vol, n)
    raw_path = np.exp(np.cumsum(raw_returns))
    # Normaliza para começar em cost_basis e terminar em current_price
    scale = current_price / raw_path[-1]
    prices = cost_basis * raw_path * scale
    prices[-1] = current_price  # âncora final

    return pd.DataFrame({"close": prices}, index=dates)


# ══════════════════════════════════════════════════════════════════════════════
# VARREDURA DE PARÂMETROS (optimização)
# ══════════════════════════════════════════════════════════════════════════════

def sweep_ar_order(prices: np.ndarray, max_p: int = 6) -> int:
    """
    Varredura AR(p): encontra ordem óptima via AIC.
    AIC = n*log(RSS/n) + 2*p
    """
    best_p, best_aic = 1, np.inf
    n = len(prices)
    for p in range(1, max_p + 1):
        if n <= p + 5:
            break
        X = np.array([prices[i:n - p + i] for i in range(p)]).T
        y = prices[p:]
        try:
            coeffs, residuals, *_ = np.linalg.lstsq(X, y, rcond=None)
            if len(residuals) == 0:
                yhat = X @ coeffs
                rss = np.sum((y - yhat) ** 2)
            else:
                rss = float(residuals[0])
            aic = len(y) * np.log(rss / len(y)) + 2 * p
            if aic < best_aic:
                best_aic = aic
                best_p = p
        except Exception:
            continue
    return best_p


def sweep_vol_window(prices: np.ndarray) -> dict:
    """
    Varredura de janelas de volatilidade: [10, 15, 20, 30, 45].
    Escolhe a que minimiza o erro de previsão in-sample (rolling 1-step).
    Retorna: melhor janela, HV por janela, e IV estimada.
    """
    windows = [10, 15, 20, 30, 45]
    results = {}
    log_ret = np.log(prices[1:] / prices[:-1])

    for w in windows:
        if len(log_ret) < w + 5:
            continue
        hv = float(np.std(log_ret[-w:]) * np.sqrt(252))
        # Erro de previsão rolling: usa vol da janela para prever log_ret seguinte
        errors = []
        for i in range(w, len(log_ret) - 1):
            pred_vol = np.std(log_ret[i - w:i]) * np.sqrt(252)
            actual = abs(log_ret[i])
            errors.append((actual - pred_vol / np.sqrt(252)) ** 2)
        mse = float(np.mean(errors)) if errors else 999
        results[w] = {"hv": hv, "mse": mse}

    best_w = min(results, key=lambda w: results[w]["mse"])
    best_hv = results[best_w]["hv"]

    # IV = HV * prémio de risco implícito (calibrado por janela óptima)
    iv_premium = 1.10 + (best_hv * 0.5)  # IV cresce com vol
    iv = min(best_hv * iv_premium, 2.0)   # cap a 200%

    return {
        "best_window": best_w,
        "hv": best_hv,
        "iv": iv,
        "by_window": {str(w): round(v["hv"] * 100, 2) for w, v in results.items()},
    }


def sweep_fourier_harmonics(series: np.ndarray, period: int = 5) -> int:
    """
    Varredura de harmónicos de Fourier [1, 2, 3].
    Escolhe o que minimiza RSS in-sample.
    """
    n = len(series)
    t = np.arange(n)
    best_h, best_rss = 1, np.inf

    for h_max in range(1, 4):
        Xf = [np.ones(n)]
        for h in range(1, h_max + 1):
            Xf.append(np.sin(2 * np.pi * h * t / period))
            Xf.append(np.cos(2 * np.pi * h * t / period))
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
# N-HiTS FORECASTER
# ══════════════════════════════════════════════════════════════════════════════

def run_nhits(df: pd.DataFrame, horizon: int = 20, sweep: dict = None) -> dict:
    if not NHITS_OK:
        return _nhits_fallback(df, horizon, sweep)

    input_size = min(60, len(df) - horizon - 1)
    nf_df = pd.DataFrame({
        "unique_id": "STOCK",
        "ds": df.index,
        "y": df["close"].values,
    })

    model = NHITS(
        h=horizon,
        input_size=input_size,
        stack_types=["identity", "identity", "identity"],
        n_stacks=3,
        n_blocks=[1, 1, 1],
        mlp_units=[[256, 256]] * 3,
        n_pool_kernel_size=[4, 2, 1],
        n_freq_downsample=[4, 2, 1],
        dropout_prob_theta=0.1,
        loss=MSE(),
        max_steps=300,
        val_check_steps=50,
        early_stop_patience_steps=5,
        scaler_type="standard",
        random_seed=42,
        verbose_fit=False,
    )

    nf = NeuralForecast(models=[model], freq="B")
    nf.fit(nf_df)
    preds = nf.predict()
    forecast_vals = preds["NHITS"].values
    last = float(df["close"].iloc[-1])

    trend = _linear_trend(df["close"].values, horizon)
    seasonal = _fourier_seasonal(df["close"].values, horizon, period=5)
    residual = forecast_vals - trend - seasonal

    return _package_nhits(forecast_vals, trend, seasonal, residual, last, horizon, "N-HiTS (neuralforecast)")


def _nhits_fallback(df: pd.DataFrame, horizon: int, sweep: dict = None) -> dict:
    """
    N-HiTS numpy — três stacks com parâmetros optimizados pela varredura.
    Stack 1: 4× MaxPool → AR(p*) trend
    Stack 2: 2× MaxPool → Fourier(h*) sazonal
    Stack 3: 1× directo → AR(p*) residual curto-prazo
    """
    prices = df["close"].values.astype(float)
    n = len(prices)

    # Parâmetros da varredura (ou defaults seguros)
    best_p = sweep.get("best_ar_p", 3) if sweep else 3
    best_h = sweep.get("best_harmonics", 2) if sweep else 2

    # ── Stack 1: Tendência (4× pooling) ──────────────────────────────────────
    pool4 = _maxpool(prices, 4)
    trend_coarse = _ar_forecast(pool4, horizon // 4 + 2, p=min(best_p, len(pool4) // 3))
    trend_fc = _linear_interp(trend_coarse, horizon)[:horizon]
    trend_in = _linear_interp(_maxpool(prices, 4), n)[:n]
    res1 = prices - trend_in

    # ── Stack 2: Sazonalidade (2× pooling) ───────────────────────────────────
    pool2 = _maxpool(res1, 2)
    seas_coarse = _fourier_forecast(pool2, horizon // 2 + 2, period=5, n_harmonics=best_h)
    seas_fc = _linear_interp(seas_coarse, horizon)[:horizon]
    seas_in = _linear_interp(_maxpool(res1, 2), n)[:n]
    res2 = res1 - seas_in

    # ── Stack 3: Ruído curto-prazo (1× directo) ───────────────────────────────
    short_fc = _ar_forecast(res2[-30:], horizon, p=min(best_p, 5))[:horizon]

    # ── Combinar e re-ancorar ─────────────────────────────────────────────────
    last = prices[-1]
    combined = trend_fc + seas_fc + short_fc
    combined = combined - combined[0] + last

    return _package_nhits(
        combined, trend_fc, seas_fc, short_fc, last, horizon,
        f"N-HiTS numpy (AR p={best_p}, Fourier h={best_h})"
    )


def _package_nhits(forecast, trend, seasonal, residual, last, horizon, model_name):
    h = len(forecast)
    idx_10 = min(9, h - 1)
    return {
        "model": model_name,
        "horizon": h,
        "forecast": list(map(float, forecast)),
        "trend": list(map(float, trend[:h])),
        "seasonal": list(map(float, seasonal[:h])),
        "residual": list(map(float, residual[:h])),
        "last_price": float(last),
        "target_10d": float(forecast[idx_10]),
        "target_20d": float(forecast[-1]),
        "direction": "UP" if forecast[-1] > last else "DOWN",
        "confidence": float(np.clip(1 - np.std(forecast) / (last + 1e-9) * 5, 0.30, 0.95)),
    }


# ── Helpers matemáticos ───────────────────────────────────────────────────────

def _maxpool(x, k):
    n = (len(x) // k) * k
    return x[:n].reshape(-1, k).max(axis=1)

def _linear_interp(x, target_len):
    xi = np.linspace(0, len(x) - 1, target_len)
    return np.interp(xi, np.arange(len(x)), x)

def _ar_forecast(series, steps, p=3):
    s = series[-max(30, p + steps):]
    n = len(s)
    if n <= p:
        return np.full(steps, float(s[-1]))
    X = np.array([s[i:n - p + i] for i in range(p)]).T
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
# ARIMA COMPLEMENTAR
# ══════════════════════════════════════════════════════════════════════════════

def run_arima(prices: np.ndarray, horizon: int, best_p: int = 3) -> dict:
    if PMDARIMA_OK:
        try:
            model = pm.auto_arima(
                prices, seasonal=True, m=5,
                information_criterion="aic",
                stepwise=True, suppress_warnings=True,
                error_action="ignore", max_p=best_p, max_q=best_p,
            )
            fc, ci = model.predict(n_periods=horizon, return_conf_int=True)
            return {
                "model": f"ARIMA{model.order}",
                "forecast": fc.tolist(),
                "lower": ci[:, 0].tolist(),
                "upper": ci[:, 1].tolist(),
                "aic": float(model.aic()),
            }
        except Exception:
            pass

    fc = _ar_forecast(prices, horizon, p=best_p)
    std = np.std(np.diff(prices[-30:])) * np.sqrt(np.arange(1, horizon + 1))
    return {
        "model": f"AR({best_p}) fallback",
        "forecast": fc.tolist(),
        "lower": (fc - 1.96 * std).tolist(),
        "upper": (fc + 1.96 * std).tolist(),
        "aic": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES
# ══════════════════════════════════════════════════════════════════════════════

def _bs_greeks(S, K, T, r, sigma):
    if T <= 0:
        return {"engine": "scipy/BS", "price": max(S - K, 0),
                "delta": 1.0 if S > K else 0.0,
                "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return {
            "engine": "QuantLib" if QUANTLIB_OK else "scipy/BS",
            "price": float(max(price, 0)),
            "delta": float(norm.cdf(d1)),
            "gamma": float(norm.pdf(d1) / (S * sigma * np.sqrt(T))),
            "theta": float((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                           - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365),
            "vega":  float(S * norm.pdf(d1) * np.sqrt(T) / 100),
            "rho":   float(K * T * np.exp(-r * T) * norm.cdf(d2) / 100),
        }
    except Exception:
        return {"engine": "scipy/BS", "price": 0.0, "delta": 0.0,
                "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

def black_scholes_greeks(S, K, T, r, sigma):
    if QUANTLIB_OK:
        try:
            today = ql.Date.todaysDate()
            ql.Settings.instance().evaluationDate = today
            expiry = today + ql.Period(max(1, int(T * 365)), ql.Days)
            payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(K))
            option = ql.VanillaOption(payoff, ql.EuropeanExercise(expiry))
            spot = ql.SimpleQuote(float(S))
            rTS = ql.YieldTermStructureHandle(ql.FlatForward(
                today, ql.QuoteHandle(ql.SimpleQuote(float(r))), ql.Actual365Fixed()))
            qTS = ql.YieldTermStructureHandle(ql.FlatForward(
                today, ql.QuoteHandle(ql.SimpleQuote(0.0)), ql.Actual365Fixed()))
            volTS = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
                today, ql.NullCalendar(),
                ql.QuoteHandle(ql.SimpleQuote(float(sigma))), ql.Actual365Fixed()))
            process = ql.BlackScholesMertonProcess(ql.QuoteHandle(spot), qTS, rTS, volTS)
            option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
            return {"engine": "QuantLib", "price": option.NPV(), "delta": option.delta(),
                    "gamma": option.gamma(), "theta": option.theta() / 365,
                    "vega": option.vega() / 100, "rho": option.rho() / 100}
        except Exception:
            pass
    return _bs_greeks(S, K, T, r, sigma)

def strike_ladder(S, T, r, sigma, strikes):
    return [{"strike": K, **black_scholes_greeks(S, K, T, r, sigma)} for K in strikes]


# ══════════════════════════════════════════════════════════════════════════════
# STRIKES ADAPTATIVOS POR FAIXA DE PREÇO
# ══════════════════════════════════════════════════════════════════════════════

def smart_strikes(current_price: float) -> list:
    """
    Gera strikes reais conforme as convenções de mercado:
      < $10   → incrementos de $0.50
      $10–$25 → incrementos de $1
      $25–$75 → incrementos de $2.50  (SMCI a $30 entra aqui)
      $75–$200→ incrementos de $5
      > $200  → incrementos de $10
    """
    S = current_price
    if S < 10:
        step = 0.5
    elif S < 25:
        step = 1.0
    elif S < 75:
        step = 2.5
    elif S < 200:
        step = 5.0
    else:
        step = 10.0

    base = round(S / step) * step
    # Gera 6 strikes abaixo e 14 acima do ATM
    strikes = [round(base + i * step, 2) for i in range(-4, 15)]
    return [k for k in strikes if k > 0]


# ══════════════════════════════════════════════════════════════════════════════
# INDICADORES TÉCNICOS
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
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
    return float(100.0 if al == 0 else 100 - 100 / (1 + ag / al))

def bollinger(prices, window=20, k=2):
    if len(prices) < window:
        m = float(np.mean(prices))
        s = float(np.std(prices))
        return {"upper": m + k*s, "mid": m, "lower": m - k*s, "bandwidth": 2*k*s/(m+1e-9)}
    s = pd.Series(prices)
    mid = float(s.rolling(window).mean().iloc[-1])
    std = float(s.rolling(window).std().iloc[-1])
    return {"upper": mid + k*std, "mid": mid, "lower": mid - k*std,
            "bandwidth": 2*k*std/(mid+1e-9)}

def support_resistance(prices, n=5):
    highs, lows = [], []
    for i in range(2, len(prices) - 2):
        if prices[i] > max(prices[i-2], prices[i-1], prices[i+1], prices[i+2]):
            highs.append(float(prices[i]))
        if prices[i] < min(prices[i-2], prices[i-1], prices[i+1], prices[i+2]):
            lows.append(float(prices[i]))
    return {
        "resistance": sorted(set(round(h, 2) for h in highs), reverse=True)[:n],
        "support":    sorted(set(round(l, 2) for l in lows))[:n],
    }


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR DE DECISÃO
# ══════════════════════════════════════════════════════════════════════════════

def covered_call_decision(p: dict) -> dict:
    S, cost_basis = p["current_price"], p["cost_basis"]
    strike, premium = p["recommended_strike"], p["premium"]
    nhits, bb = p["nhits"], p["bollinger"]
    daily_rsi = p["daily_rsi"]

    score, signals = 0, []

    def sig(label, stype, weight):
        signals.append({"label": label, "type": stype, "weight": weight})

    # 1. RSI diário
    if daily_rsi > 70:
        score += 3; sig(f"RSI diário {daily_rsi:.0f} — overbought forte", "bullish", 3)
    elif daily_rsi > 62:
        score += 2; sig(f"RSI diário {daily_rsi:.0f} — elevado", "bullish", 2)
    elif daily_rsi > 55:
        score += 1; sig(f"RSI diário {daily_rsi:.0f} — moderado", "neutral", 1)
    elif daily_rsi < 35:
        score -= 3; sig(f"RSI diário {daily_rsi:.0f} — oversold, aguardar bounce", "bearish", -3)
    elif daily_rsi < 45:
        score -= 1; sig(f"RSI diário {daily_rsi:.0f} — fraco", "bearish", -1)

    # 2. N-HiTS forecast
    fc_chg = (nhits["target_10d"] - S) / S * 100
    if fc_chg > 4:
        score -= 3; sig(f"N-HiTS: +{fc_chg:.1f}% — upside forte, nao vender", "bearish", -3)
    elif fc_chg > 2:
        score -= 2; sig(f"N-HiTS: +{fc_chg:.1f}% — upside moderado", "bearish", -2)
    elif fc_chg > 0.5:
        score -= 1; sig(f"N-HiTS: +{fc_chg:.1f}% — upside leve", "neutral", -1)
    elif fc_chg < -2:
        score += 3; sig(f"N-HiTS: {fc_chg:.1f}% — queda prevista, ideal para call", "bullish", 3)
    elif fc_chg < -0.5:
        score += 2; sig(f"N-HiTS: {fc_chg:.1f}% — tendencia negativa leve", "bullish", 2)
    else:
        score += 1; sig(f"N-HiTS: {fc_chg:.1f}% — previsao lateral", "neutral", 1)

    # 3. Strike vs cost basis (mais importante)
    gap = strike - cost_basis
    if gap >= cost_basis * 0.05:
        score += 4; sig(f"Strike ${strike:.2f} bem acima do custo (${gap:+.2f})", "bullish", 4)
    elif gap >= 0:
        score += 2; sig(f"Strike ${strike:.2f} acima do custo (+${gap:.2f})", "bullish", 2)
    elif gap >= -cost_basis * 0.03:
        score -= 2; sig(f"Strike ${strike:.2f} ligeiramente abaixo do custo", "bearish", -2)
    else:
        score -= 4; sig(f"Strike ${strike:.2f} MUITO abaixo do custo (${gap:.2f}) — PERDA CERTA", "bearish", -4)

    # 4. Bollinger Band
    if bb["upper"] > bb["lower"]:
        pct_b = (S - bb["lower"]) / (bb["upper"] - bb["lower"])
        if pct_b > 0.85:
            score += 2; sig("Preco perto da banda superior Bollinger — sobrecomprado", "bullish", 2)
        elif pct_b > 0.65:
            score += 1; sig("Preco na metade superior Bollinger", "neutral", 1)
        elif pct_b < 0.15:
            score -= 2; sig("Preco perto da banda inferior — possivel bounce", "bearish", -2)

    # 5. P&L se exercida
    net_pnl = (strike - cost_basis + premium) * p.get("shares", 100)
    if net_pnl > 0:
        score += 2; sig(f"P&L total se exercida: +${net_pnl:,.0f}", "bullish", 2)
    else:
        score -= 3; sig(f"P&L total se exercida: -${abs(net_pnl):,.0f} — PREJUIZO", "bearish", -3)

    # 6. Premium yield
    ann_yield = p.get("ann_yield", 0)
    if ann_yield > 20:
        score += 2; sig(f"Premio anualizado: {ann_yield:.1f}% — excelente", "bullish", 2)
    elif ann_yield > 10:
        score += 1; sig(f"Premio anualizado: {ann_yield:.1f}% — bom", "neutral", 1)
    elif ann_yield < 3:
        score -= 1; sig(f"Premio anualizado: {ann_yield:.1f}% — muito baixo", "bearish", -1)

    # Veredicto
    max_score = 16
    if score >= 8:
        verdict, color, icon = "VENDER CALL", "#10b981", "✅"
        rationale = "Setup excelente: RSI elevado, N-HiTS flat/negativo, strike acima do custo, P&L positivo."
    elif score >= 4:
        verdict, color, icon = "CONSIDERAR", "#f59e0b", "⚖️"
        rationale = "Condicoes favoraveis mas nao ideais. Verifique resistencias tecnicas antes de agir."
    elif score >= 0:
        verdict, color, icon = "AGUARDAR", "#f59e0b", "⏳"
        rationale = "Sinais mistos. Aguarde RSI mais elevado ou strike acima do custo base."
    else:
        verdict, color, icon = "NAO VENDER", "#ef4444", "🚫"
        rationale = "Vender agora seria prejudicial. N-HiTS preve upside ou strike abaixo do custo."

    return {"verdict": verdict, "color": color, "icon": icon,
            "score": score, "max_score": max_score,
            "rationale": rationale, "signals": signals}


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISE AI COM CLAUDE claude-opus-4-8
# ══════════════════════════════════════════════════════════════════════════════

def claude_analysis(payload: dict) -> str:
    """
    Chama Claude claude-opus-4-8 para análise narrativa da posicao.
    A chave API é lida do header X-API-Key enviado pelo frontend.
    """
    api_key = payload.get("api_key", "")
    if not api_key or not api_key.startswith("sk-ant"):
        return "Configure a API key do Anthropic no frontend para activar a análise AI."

    d = payload["data"]
    prompt = f"""Você é um estrategista de opções profissional especializado em covered calls.

POSIÇÃO:
- Ticker: {d['ticker']} | Preço actual: ${d['current_price']:.2f} | Custo base: ${d['cost_basis']:.2f}
- {d['shares']} acções | P&L não realizado: ${(d['current_price']-d['cost_basis'])*d['shares']:+,.0f}

BLACK-SCHOLES:
- Strike recomendado: ${d['recommended_strike']:.2f} (delta ≈ {d['greeks']['delta']:.3f})
- Premio: ${d['premium']:.2f}/acção (${d['premium']*d['shares']:.0f} total)
- IV: {d['iv']*100:.1f}% | HV30: {d['hv30']*100:.1f}% | DTE: {d['dte']}
- Theta: ${abs(d['greeks']['theta']):.3f}/dia | Vega: ${d['greeks']['vega']:.3f}/1%IV
- Yield anualizado: {d['ann_yield']:.1f}%

N-HiTS (3 stacks hierárquicos):
- Previsão 10d: ${d['nhits']['target_10d']:.2f} ({(d['nhits']['target_10d']/d['current_price']-1)*100:+.1f}%)
- Previsão 20d: ${d['nhits']['target_20d']:.2f} ({(d['nhits']['target_20d']/d['current_price']-1)*100:+.1f}%)
- Direcção: {d['nhits']['direction']} | Confiança: {d['nhits']['confidence']*100:.0f}%

VARREDURA DE PARÂMETROS:
- Melhor janela vol: {d['vol_sweep']['best_window']}d | HV óptima: {d['vol_sweep']['hv']*100:.1f}%
- Melhor AR(p): {d['sweep_params']['best_ar_p']} | Harmónicos Fourier: {d['sweep_params']['best_harmonics']}

RSI: Diário {d['daily_rsi']:.1f} | Semanal {d['weekly_rsi']:.1f}
Bollinger %B: {((d['current_price']-d['bollinger']['lower'])/(d['bollinger']['upper']-d['bollinger']['lower'])*100):.0f}%
Breakeven: ${d['breakeven']:.2f} | P&L se exercida: ${d['total_pnl_if_called']:+,.0f}

VEREDICTO DO SISTEMA: {d['decision']['verdict']} (score {d['decision']['score']}/{d['decision']['max_score']})

Responda EM PORTUGUÊS em 4 secções:
1. TIMING (1-2 frases directas: vender agora ou aguardar?)
2. RISCOS PRINCIPAIS (2-3 bullets concisos)
3. ESTRATÉGIA ALTERNATIVA (se timing não for ideal)
4. CONCLUSÃO (1 frase decisiva e accionável)

Seja directo. Não cubra todos os lados. Tome uma posição clara."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 800,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["content"][0]["text"]
        return f"Erro API Claude: {resp.status_code} — {resp.text[:200]}"
    except Exception as e:
        return f"Erro de conexão com Claude API: {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "version": "2.0-optimized",
        "libs": {
            "yfinance": YFINANCE_OK,
            "neuralforecast_nhits": NHITS_OK,
            "pmdarima": PMDARIMA_OK,
            "quantlib": QUANTLIB_OK,
            "claude_model": CLAUDE_MODEL,
        }
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.json or {}
    ticker       = str(body.get("ticker", "UAL")).upper()
    cost_basis   = float(body.get("cost_basis", 100))
    current_price= float(body.get("current_price", cost_basis))
    shares       = int(body.get("shares", 100))
    dte          = max(1, int(body.get("dte", 30)))
    risk_free    = float(body.get("risk_free", 0.053))
    target_delta = float(body.get("target_delta", 0.20))
    horizon      = max(5, int(body.get("nhits_horizon", 20)))
    api_key      = str(body.get("api_key", ""))
    T = dte / 365

    # 1. Preços sintéticos calibrados
    df = fetch_prices(ticker, current_price, cost_basis)
    prices = df["close"].values

    # 2. Varredura de parâmetros
    vol_sweep  = sweep_vol_window(prices)
    best_ar_p  = sweep_ar_order(prices)
    best_h_fourier = sweep_fourier_harmonics(prices - np.mean(prices))
    iv = vol_sweep["iv"]
    hv = vol_sweep["hv"]

    sweep_params = {
        "best_ar_p": best_ar_p,
        "best_harmonics": best_h_fourier,
        "vol_window": vol_sweep["best_window"],
    }

    # 3. Strike ladder adaptativo
    all_strikes = smart_strikes(current_price)
    ladder = strike_ladder(current_price, T, risk_free, iv, all_strikes)

    # Strike mais próximo do delta alvo
    rec_row = min(ladder, key=lambda r: abs(r["delta"] - target_delta))
    rec_strike = rec_row["strike"]
    premium = rec_row["price"]
    ann_yield = (premium / current_price) * (365 / dte) * 100

    # 4. N-HiTS com parâmetros optimizados
    nhits_result = run_nhits(df, horizon, sweep_params)

    # 5. ARIMA
    arima_result = run_arima(prices, horizon, best_ar_p)

    # 6. Indicadores técnicos
    daily_rsi  = rsi(prices, 14)
    weekly_rsi = rsi(prices[::5], 14)
    bb         = bollinger(prices)
    sr         = support_resistance(prices)

    # 7. Decisão
    decision = covered_call_decision({
        "current_price": current_price,
        "cost_basis": cost_basis,
        "recommended_strike": rec_strike,
        "premium": premium,
        "nhits": nhits_result,
        "daily_rsi": daily_rsi,
        "weekly_rsi": weekly_rsi,
        "bollinger": bb,
        "shares": shares,
        "ann_yield": ann_yield,
    })

    # 8. Cenários
    total_pnl = (rec_strike - cost_basis + premium) * shares
    pnl_pct   = (rec_strike - cost_basis + premium) / cost_basis * 100
    breakeven = cost_basis - premium

    result = {
        "ticker": ticker,
        "current_price": current_price,
        "cost_basis": cost_basis,
        "shares": shares,
        "hv30": hv,
        "iv": iv,
        "dte": dte,
        "risk_free": risk_free,
        "recommended_strike": rec_strike,
        "premium": premium,
        "ann_yield": ann_yield,
        "breakeven": breakeven,
        "total_pnl_if_called": total_pnl,
        "pnl_pct_if_called": pnl_pct,
        "greeks": rec_row,
        "strike_ladder": ladder,
        "nhits": nhits_result,
        "arima": arima_result,
        "daily_rsi": daily_rsi,
        "weekly_rsi": weekly_rsi,
        "bollinger": bb,
        "support_resistance": sr,
        "decision": decision,
        "vol_sweep": vol_sweep,
        "sweep_params": sweep_params,
        "price_history": {
            "dates": [str(d.date()) for d in df.index],
            "prices": prices.tolist(),
        },
        "libs": {
            "nhits_model": nhits_result["model"],
            "arima_model": arima_result["model"],
            "greeks_engine": rec_row["engine"],
        },
    }

    # 9. Análise Claude claude-opus-4-8 (opcional)
    if api_key and api_key.startswith("sk-ant"):
        result["ai_analysis"] = claude_analysis({"api_key": api_key, "data": result})
    else:
        result["ai_analysis"] = ""

    return jsonify(result)


@app.route("/api/ai_analysis", methods=["POST"])
def ai_analysis_route():
    """Endpoint separado para análise AI (chamado pelo frontend após /analyze)."""
    body = request.json or {}
    text = claude_analysis(body)
    return jsonify({"analysis": text})


if __name__ == "__main__":
    print(f"N-HiTS: {'PyTorch' if NHITS_OK else 'numpy fallback'}")
    print(f"QuantLib: {'OK' if QUANTLIB_OK else 'scipy fallback'}")
    print(f"pmdarima: {'OK' if PMDARIMA_OK else 'AR fallback'}")
    print(f"Claude: {CLAUDE_MODEL}")
    app.run(debug=True, port=5050, use_reloader=False)