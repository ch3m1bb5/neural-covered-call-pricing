# UAL Covered Call — Neural Pricing v2

Upgrade of the *Covered Call Timing Analysis for UAL* dashboard. The original
priced every strike with a single constant IV and used an AR(1) "momentum"
proxy. This version adds three trained neural models behind a Flask API; the
React dashboard consumes them.

```
                 ┌──────────────────────────────────────────────┐
   option chain ─┤ ① IV surface net   (k,T) → IV   [deep SVI]    ├─ strike-specific IV
                 │                                               │
   price history ┤ ② vol forecaster   LSTM → P10/P50/P90 RV      ├─ target vol ─┐
                 │                                               │              │
                 │ ③ scenario VAE     z → variance profile ──────┼──────────────┘
                 │                    + fresh innovations → MC    │
                 └──────────────────────────────────────────────┘ → P&L distribution
```

The three compose: the forecaster's P50 becomes the target vol the scenario
generator scales its paths to, so `/api/simulate` needs only a strike + premium.

## Why each model (mapped to the original's weaknesses)

| # | Model | Replaces | Robustness gained |
|---|-------|----------|-------------------|
| 1 | `vol_forecaster_nhits.py` — **N-HiTS** (neuralforecast) + exog; `vol_forecaster.py` LSTM fallback | fixed `IV` + AR(1) | forward vol with P10/P50/P90, multi-rate + exogenous (returns, VIX, volume) |
| 2 | `iv_surface.py` — MLP w(k,T) + no-arb penalties + **live calibration** | one sigma for all strikes | correct skew → unbiased strike & Greeks; refit to your own chain |
| 3 | `scenario_gen.py` — VAE + Monte Carlo | single "if called away" line | full P&L distribution, assignment %, CVaR |

The forecaster is **N-HiTS** when `neuralforecast` is installed (multi-rate
pooling `n_pool_kernel_size=[8,4,1]`, hierarchical interpolation
`n_freq_downsample=[24,12,1]`, `hist_exog_list=[log_ret, vix, log_vol]`,
`MQLoss` quantiles). If neuralforecast is missing, `train_all.py` falls back to
the self-contained LSTM and the API serves that instead — same response shape.

No-arbitrage in #2 is enforced softly via autograd: calendar monotonicity
(`relu(-∂w/∂T)`) and a butterfly/convexity proxy (`relu(-∂²w/∂k²)`).
#3 avoids VAE mean-collapse by learning the **variance-clustering profile** and
injecting fresh Gaussian innovations at generation (fat tails, vol clustering).

## Setup

```bash
cd backend
pip install -r requirements.txt
python train_all.py      # ~1 min on CPU; writes artifacts/
python app.py            # serves http://localhost:5001
```

Then run the dashboard in your Vite project (drop in `covered_call_dashboard_v2.jsx`):

```bash
npm run dev
```

`API_BASE` at the top of the JSX points to `http://localhost:5001`.

## Endpoints

| Method | Path | Returns |
|--------|------|---------|
| GET  | `/api/health` | status + loaded models |
| GET  | `/api/state` | spot, recent RV & prices |
| POST | `/api/iv-surface` | `{strikes:[{K,iv}], smile:[{k,iv}]}` |
| POST | `/api/forecast-vol` | `{model, p10,p50,p90,last_rv,horizon_days}` |
| POST | `/api/calibrate-surface` | refit IV net to your chain + hot-swap; `{rows_used, iv_rmse, sample_30d}` |
| POST | `/api/simulate` | P&L distribution + assignment/touch/CVaR + histogram |

`/api/calibrate-surface` body: `{spot, quotes:[{K, T_days|T, iv}]}` — or pass
`{K, T_days, price, kind:"call"|"put"}` and the IV is inverted from the price.
The dashboard's "Calibrate surface to your chain" card posts here and refreshes
the smile in place.

`/api/simulate` body: `{strike, premium, cost_basis, shares?, target_vol?, n_paths?}`.
Omit `target_vol` to auto-use the forecaster's P50.

## Going to real data / production

* Replace `data_synth.py` with your live loaders (price series + option chain).
  Everything downstream is untouched.
* Swap the LSTM core in `vol_forecaster.py` for **N-HiTS** (`neuralforecast`) —
  already wired in `vol_forecaster_nhits.py` and used by default when installed.
* Swap the VAE for **TimeGAN** in `scenario_gen.py` if you want a
  discriminator-trained generator (heavier; VAE chosen for a stable demo).
* The Greeks remain client-side (fast UI), but now consume the strike-specific
  IV from the surface instead of a flat sigma.

## Notes / honest caveats

* The synthetic Heston series and SVI chain make the repo self-contained and
  trainable end-to-end; numbers are illustrative, not a UAL forecast.
* American early assignment is approximated (expiry assignment + an early-touch
  probability as a proxy). Fine for OTM short calls.
* The dashboard is decision *support*, not advice. Validate any forecaster
  walk-forward against a HAR-RV / EGARCH baseline before trusting it.
