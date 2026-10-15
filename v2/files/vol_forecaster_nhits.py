"""
vol_forecaster_nhits.py  ──  IMPROVEMENT #1, the N-HiTS version
---------------------------------------------------------------
Drop-in replacement for the LSTM forecaster, using Nixtla's neuralforecast
NHITS with multi-rate pooling + hierarchical interpolation and *exogenous*
regressors (returns, a VIX-style index, log volume). Output is the same
contract as the LSTM module: P10 / P50 / P90 of the average realized vol over
the next h trading days, so app.py can use either interchangeably.

MQLoss(quantiles=[0.1,0.5,0.9]) -> columns NHITS-lo-80.0 / NHITS-median /
NHITS-hi-80.0, which map to P10 / P50 / P90.
"""
import os
import warnings
import logging
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
for noisy in ("pytorch_lightning", "lightning.pytorch", "lightning_fabric"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss

EXOG = ["log_ret", "vix", "log_vol"]
START = "2022-01-03"


def build_frame(rv, rets, vix, log_vol):
    n = len(rv)
    return pd.DataFrame({
        "unique_id": "UAL_RV",
        "ds": pd.date_range(START, periods=n, freq="B"),
        "y": np.asarray(rv, dtype=float),
        "log_ret": np.asarray(rets, dtype=float),
        "vix": np.asarray(vix, dtype=float),
        "log_vol": np.asarray(log_vol, dtype=float),
    })


def train_nhits(df, h=21, input_size=126, max_steps=600, seed=0, verbose=False):
    model = NHITS(
        h=h, input_size=input_size,
        n_pool_kernel_size=[8, 4, 1],     # macro / intermediate / high-freq stacks
        n_freq_downsample=[24, 12, 1],    # hierarchical output resolution
        hist_exog_list=EXOG,              # observed exogenous regressors
        loss=MQLoss(quantiles=[0.1, 0.5, 0.9]),
        scaler_type="robust",
        max_steps=max_steps,
        val_check_steps=max(50, max_steps // 6),
        random_seed=seed,
        enable_progress_bar=verbose, logger=False,
    )
    nf = NeuralForecast(models=[model], freq="B")
    val = max(h * 2, int(0.12 * len(df)))
    nf.fit(df, val_size=val)
    return nf


def forecast_nhits(nf, df):
    """
    Forecast from the end of `df`. Returns averaged quantiles over the horizon
    (comparable to 'average RV over the next h days') + the last observed RV.
    """
    fc = nf.predict(df=df)
    p10 = float(fc["NHITS-lo-80.0"].mean())
    p50 = float(fc["NHITS-median"].mean())
    p90 = float(fc["NHITS-hi-80.0"].mean())
    # guard monotonicity (interpolation can rarely cross at the tails)
    p10, p50, p90 = sorted([p10, p50, p90])
    h = len(fc)
    return {"p10": p10, "p50": p50, "p90": p90,
            "last_rv": float(df["y"].iloc[-1]), "horizon_days": h,
            "path_median": [float(x) for x in fc["NHITS-median"].values]}


def save(nf, path):
    nf.save(path, overwrite=True, save_dataset=False)


def load(path):
    return NeuralForecast.load(path)


if __name__ == "__main__":
    from data_synth import (generate_price_series, realized_vol,
                            generate_exogenous)
    p, r, v = generate_price_series(n_days=900, seed=7)
    rv = realized_vol(r); vix, lv = generate_exogenous(r, v)
    df = build_frame(rv, r, vix, lv)
    nf = train_nhits(df, max_steps=400)
    fc = forecast_nhits(nf, df)
    print(f"N-HiTS forecast next {fc['horizon_days']}d RV  "
          f"P10={fc['p10']:.2%}  P50={fc['p50']:.2%}  P90={fc['p90']:.2%}  "
          f"(last RV={fc['last_rv']:.2%})")
