"""
app.py  ──  Flask API that serves the three neural upgrades to the React
dashboard. Endpoints:

  GET  /api/health
  GET  /api/state            -> current spot + recent RV (for the UI)
  POST /api/iv-surface       -> IV(k,T) grid + IV for a list of strikes
  POST /api/forecast-vol     -> P10/P50/P90 RV forecast for next h days
  POST /api/simulate         -> Monte-Carlo covered-call P&L distribution

Run:
  python train_all.py        # once, to produce artifacts/
  python app.py              # serves on http://localhost:5001
"""
import json
import os
import shutil
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

import iv_surface as IV
import vol_forecaster as VF
import scenario_gen as SG

# Artifacts live in the SAME folder as this script (flat layout).
ART = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app)


def _ensure_nhits_dir(art):
    """
    neuralforecast.load() needs a DIRECTORY. In a flat layout the three N-HiTS
    files sit loose next to the scripts; assemble them into an 'nhits' subfolder
    once so the pre-trained forecaster loads without retraining.
    """
    d = os.path.join(art, "nhits")
    loose = ["NHITS_0.ckpt", "alias_to_model.pkl", "configuration.pkl"]
    if not os.path.isdir(d) and all(os.path.exists(os.path.join(art, f)) for f in loose):
        os.makedirs(d, exist_ok=True)
        for f in loose:
            shutil.copy(os.path.join(art, f), os.path.join(d, f))
    return d if os.path.isdir(d) else None

# ── load artifacts once at startup ───────────────────────────────────────────
with open(os.path.join(ART, "meta.json")) as f:
    META = json.load(f)
_series = np.load(os.path.join(ART, "series.npz"))
PRICES, RETS, RV = _series["prices"], _series["rets"], _series["rv"]
VIX = _series["vix"] if "vix" in _series else None
LOG_VOL = _series["log_vol"] if "log_vol" in _series else None
SPOT = float(META["spot"])

IV_MODEL = IV.load(os.path.join(ART, "iv_surface.pt"))     # mutable (calibration)
VF_MODEL = VF.load(os.path.join(ART, "vol_forecaster.pt"))
VF_META = META["vol_forecaster"]
SG_MODEL = SG.load(os.path.join(ART, "scenario_vae.pt"),
                   H=META["scenario_vae"]["H"], latent=META["scenario_vae"]["latent"])
SG_META = META["scenario_vae"]

# N-HiTS is primary if it was trained; LSTM is the fallback
NHITS_NF, NHITS_DF, NH = None, None, None
_nhits_dir = _ensure_nhits_dir(ART)
if META.get("forecaster") == "nhits" and _nhits_dir:
    try:
        import vol_forecaster_nhits as NH
        NHITS_NF = NH.load(_nhits_dir)
        NHITS_DF = NH.build_frame(RV, RETS, VIX, LOG_VOL)
        print("forecaster: N-HiTS (neuralforecast, +exog)")
    except Exception as e:
        print(f"forecaster: LSTM fallback (N-HiTS load failed: {e})")
else:
    print("forecaster: LSTM")


def forecast_vol():
    """Unified forecaster: N-HiTS if available, else the LSTM."""
    if NHITS_NF is not None:
        fc = NH.forecast_nhits(NHITS_NF, NHITS_DF)
        fc["model"] = "nhits"
        return fc
    fc = VF.forecast(VF_MODEL, RV, RETS, VF_META)
    fc.update({"last_rv": float(RV[-1]), "horizon_days": VF_META["h"], "model": "lstm"})
    return fc


@app.get("/api/health")
def health():
    return jsonify(status="ok", spot=SPOT,
                   forecaster="nhits" if NHITS_NF is not None else "lstm",
                   models=["iv_surface", "vol_forecaster", "scenario_vae"])


@app.get("/api/state")
def state():
    return jsonify(spot=SPOT,
                   last_rv=float(RV[-1]),
                   rv_recent=[float(x) for x in RV[-60:]],
                   price_recent=[float(x) for x in PRICES[-60:]])


@app.post("/api/iv-surface")
def iv_surface():
    """body: {spot?, strikes:[...], T_days:30}"""
    b = request.get_json(force=True)
    spot = float(b.get("spot", SPOT))
    T = float(b.get("T_days", 30)) / 365.0
    strikes = b.get("strikes")
    if not strikes:
        strikes = [round(spot * m, 2) for m in np.linspace(0.85, 1.25, 17)]
    ivs = IV.iv_for_strike(IV_MODEL, np.array(strikes, dtype=float), spot, T)
    ivs = np.atleast_1d(ivs)
    surface = [{"K": float(k), "iv": float(v)} for k, v in zip(strikes, ivs)]
    ks = np.linspace(-0.25, 0.25, 41)
    smile = [{"k": float(k), "iv": float(IV.predict_iv(IV_MODEL, k, T))} for k in ks]
    return jsonify(T_days=round(T * 365), spot=spot, strikes=surface, smile=smile)


@app.post("/api/forecast-vol")
def forecast_vol_route():
    fc = forecast_vol()
    return jsonify(fc)


@app.post("/api/calibrate-surface")
def calibrate_surface():
    """
    Refit the IV surface to YOUR real chain and hot-swap the live model.
    body: {spot, quotes:[{K, T_days|T, iv} | {K, T_days|T, price, kind}], epochs?}
    Returns a fit report (rows used, IV RMSE, sample skew).
    """
    global IV_MODEL
    b = request.get_json(force=True)
    spot = float(b.get("spot", SPOT))
    quotes = b["quotes"]
    epochs = int(b.get("epochs", 4000))
    try:
        new_model, n_rows = IV.calibrate_from_chain(quotes, spot, epochs=epochs,
                                                    verbose=False)
    except Exception as e:
        return jsonify(error=str(e)), 400

    # fit quality on the provided quotes
    rows = IV.quotes_to_training(quotes, spot)
    err = [IV.predict_iv(new_model, r["k"], r["T"]) - r["iv"] for r in rows]
    rmse = float(np.sqrt(np.mean(np.square(err))))

    IV_MODEL = new_model                       # hot-swap
    IV.save(new_model, os.path.join(ART, "iv_surface.pt"))   # persist
    T0 = rows[0]["T"]
    sample = [{"K": round(spot * np.exp(k), 2),
               "iv": round(float(IV.predict_iv(new_model, k, T0)), 4)}
              for k in (-0.1, 0.0, 0.1)]
    return jsonify(calibrated=True, rows_used=n_rows, iv_rmse=round(rmse, 4),
                   sample_30d=sample)


@app.post("/api/simulate")
def simulate():
    """
    body: {spot?, strike, premium, cost_basis, shares?, target_vol?,
           n_paths?, drift_annual?}
    If target_vol is omitted, the vol forecaster's P50 is used -> the three
    models compose automatically.
    """
    b = request.get_json(force=True)
    spot = float(b.get("spot", SPOT))
    strike = float(b["strike"])
    premium = float(b["premium"])
    cost_basis = float(b.get("cost_basis", 197.0))
    shares = int(b.get("shares", 100))
    n_paths = int(b.get("n_paths", 20000))
    drift = float(b.get("drift_annual", 0.0))
    target_vol = b.get("target_vol")
    if target_vol is None:
        target_vol = forecast_vol()["p50"]      # unified (N-HiTS or LSTM)
    target_vol = float(target_vol)

    res = SG.monte_carlo_covered_call(
        SG_MODEL, SG_META, spot=spot, strike=strike, premium=premium,
        target_annual_vol=target_vol, cost_basis=cost_basis, shares=shares,
        n_paths=n_paths, drift_annual=drift)
    res["used_target_vol"] = target_vol
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
