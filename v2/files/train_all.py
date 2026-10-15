"""
train_all.py  ──  trains the three models once and saves everything to
artifacts/. Run this before starting the API:   python train_all.py
On CPU it finishes in well under a minute with the default sizes.
"""
import json
import os
import numpy as np

import data_synth as D
import iv_surface as IV
import vol_forecaster as VF
import scenario_gen as SG

# Artifacts are written into the SAME folder as this script (flat layout).
ART = os.path.dirname(os.path.abspath(__file__))
os.makedirs(ART, exist_ok=True)


def main():
    print("== generating synthetic market data ==")
    prices, rets, inst_vol = D.generate_price_series(n_days=900, seed=7)
    rv = D.realized_vol(rets, window=21)
    vix, log_vol = D.generate_exogenous(rets, inst_vol)
    chain = D.synthetic_option_chain(spot=D.S0_ANCHOR, seed=11)
    np.savez(os.path.join(ART, "series.npz"),
             prices=prices, rets=rets, rv=rv, vix=vix, log_vol=log_vol)

    print("== [1/4] IV surface net ==")
    iv_model = IV.train_iv_surface(chain, epochs=4000, verbose=True)
    IV.save(iv_model, os.path.join(ART, "iv_surface.pt"))

    print("== [2/4] vol forecaster — N-HiTS (neuralforecast, +exog) ==")
    forecaster = "lstm"
    try:
        import vol_forecaster_nhits as NH
        df = NH.build_frame(rv, rets, vix, log_vol)
        nf = NH.train_nhits(df, h=21, input_size=126, max_steps=300, verbose=False)
        NH.save(nf, os.path.join(ART, "nhits"))
        fc = NH.forecast_nhits(nf, df)
        forecaster = "nhits"
        print(f"   N-HiTS ready: P10/P50/P90 = "
              f"{fc['p10']:.2%}/{fc['p50']:.2%}/{fc['p90']:.2%}")
    except Exception as e:
        print(f"   N-HiTS unavailable ({e}); will rely on the LSTM.")

    print("== [3/4] vol forecaster — LSTM (fallback) ==")
    vf_model, vf_meta = VF.train_vol_forecaster(rv, rets, L=60, h=21,
                                                epochs=400, verbose=False)
    VF.save(vf_model, os.path.join(ART, "vol_forecaster.pt"))

    print("== [4/4] scenario VAE ==")
    sg_model, sg_meta = SG.train_scenario_vae(rets, H=21, latent=8,
                                              epochs=600, verbose=False)
    SG.save(sg_model, os.path.join(ART, "scenario_vae.pt"))

    meta = {
        "spot": float(prices[-1]),
        "forecaster": forecaster,            # "nhits" if available else "lstm"
        "vol_forecaster": vf_meta,
        "scenario_vae": sg_meta,
        "chain_expiries_days": sorted({round(q["T"] * 365) for q in chain}),
    }
    with open(os.path.join(ART, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved artifacts to", ART)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
