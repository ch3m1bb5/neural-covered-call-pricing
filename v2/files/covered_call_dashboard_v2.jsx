import { useState, useEffect, useCallback } from "react";

// ════════════════════════════════════════════════════════════════════════════
// Covered Call Dashboard v2 — neural pricing upgrades
//   #1 vol forecaster (quantile)   -> /api/forecast-vol
//   #2 IV surface (deep SVI)       -> /api/iv-surface
//   #3 scenario generator (VAE MC) -> /api/simulate
// Backend: the Flask app in ./backend (see README). Start it, then run this.
// ════════════════════════════════════════════════════════════════════════════
const API_BASE = "http://localhost:5001";

// ── position constants (from the original UAL setup) ────────────────────────
const COST_BASIS = 197;
const SHARES = 100;
const DTE = 30;
const RISK_FREE = 0.043;
const RESISTANCE_LEVELS = [
  { price: 172, label: "R1 – Jan swing high", strength: "Strong" },
  { price: 175, label: "R2 – Dec consolidation", strength: "Moderate" },
  { price: 182, label: "R3 – prior breakdown", strength: "Moderate" },
];

// ── Black-Scholes (QuantLib-style) — Greeks computed client-side ────────────
function erf(x) {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741,
        a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + p * x);
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-x * x);
  return sign * y;
}
const normCDF = (x) => 0.5 * (1 + erf(x / Math.sqrt(2)));
const normPDF = (x) => Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);

function blackScholes(S, K, T, r, sigma) {
  if (T <= 0) return { call: Math.max(S - K, 0), delta: S > K ? 1 : 0, gamma: 0, theta: 0, vega: 0 };
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  const call = S * normCDF(d1) - K * Math.exp(-r * T) * normCDF(d2);
  const delta = normCDF(d1);
  const gamma = normPDF(d1) / (S * sigma * Math.sqrt(T));
  const theta = (-S * normPDF(d1) * sigma / (2 * Math.sqrt(T)) - r * K * Math.exp(-r * T) * normCDF(d2)) / 365;
  const vega = S * normPDF(d1) * Math.sqrt(T) / 100;
  return { call, delta, gamma, theta, vega };
}

function calcRSI(prices, period = 14) {
  if (prices.length < period + 1) return null;
  let gains = 0, losses = 0;
  for (let i = 1; i <= period; i++) {
    const d = prices[i] - prices[i - 1];
    if (d > 0) gains += d; else losses -= d;
  }
  let avgGain = gains / period, avgLoss = losses / period;
  for (let i = period + 1; i < prices.length; i++) {
    const d = prices[i] - prices[i - 1];
    avgGain = (avgGain * (period - 1) + (d > 0 ? d : 0)) / period;
    avgLoss = (avgLoss * (period - 1) + (d < 0 ? -d : 0)) / period;
  }
  if (avgLoss === 0) return 100;
  return 100 - 100 / (1 + avgGain / avgLoss);
}

const fmt$ = (x) => (x < 0 ? "-$" : "$") + Math.abs(x).toLocaleString(undefined, { maximumFractionDigits: 0 });
const pct = (x) => (x * 100).toFixed(1) + "%";

async function api(path, body) {
  const res = await fetch(API_BASE + path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

// ── tiny inline SVG line/area chart for the smile and forecast ──────────────
function Smile({ smile, atmStrikeIV }) {
  if (!smile?.length) return null;
  const W = 320, H = 150, pad = 28;
  const ks = smile.map((p) => p.k), ivs = smile.map((p) => p.iv);
  const kMin = Math.min(...ks), kMax = Math.max(...ks);
  const ivMin = Math.min(...ivs) * 0.97, ivMax = Math.max(...ivs) * 1.03;
  const X = (k) => pad + ((k - kMin) / (kMax - kMin)) * (W - 2 * pad);
  const Y = (v) => H - pad - ((v - ivMin) / (ivMax - ivMin)) * (H - 2 * pad);
  const path = smile.map((p, i) => `${i ? "L" : "M"}${X(p.k).toFixed(1)},${Y(p.iv).toFixed(1)}`).join(" ");
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`}>
      <line x1={X(0)} y1={pad} x2={X(0)} y2={H - pad} stroke="#334155" strokeDasharray="3 3" />
      <text x={X(0)} y={H - 8} fill="#64748b" fontSize="9" textAnchor="middle">ATM</text>
      <path d={path} fill="none" stroke="#38bdf8" strokeWidth="2" />
      {atmStrikeIV != null && (
        <text x={W - pad} y={pad + 4} fill="#94a3b8" fontSize="9" textAnchor="end">
          skew: puts richer →
        </text>
      )}
      <text x={pad} y={pad} fill="#64748b" fontSize="9">{pct(ivMax)}</text>
      <text x={pad} y={H - pad + 8} fill="#64748b" fontSize="9">{pct(ivMin)}</text>
    </svg>
  );
}

function ForecastBand({ fc }) {
  if (!fc) return null;
  const W = 320, H = 120, pad = 30;
  const lo = Math.min(fc.p10, fc.last_rv) * 0.9, hi = Math.max(fc.p90, fc.last_rv) * 1.1;
  const X = (v) => pad + ((v - lo) / (hi - lo)) * (W - 2 * pad);
  const y = 56;
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`}>
      <line x1={X(fc.p10)} y1={y} x2={X(fc.p90)} y2={y} stroke="#475569" strokeWidth="8" strokeLinecap="round" />
      <line x1={X(fc.p10)} y1={y} x2={X(fc.p50)} y2={y} stroke="#22d3ee" strokeWidth="8" strokeLinecap="round" />
      <circle cx={X(fc.p50)} cy={y} r="6" fill="#0ea5e9" stroke="#e0f2fe" strokeWidth="1.5" />
      <line x1={X(fc.last_rv)} y1={y - 18} x2={X(fc.last_rv)} y2={y + 18} stroke="#f59e0b" strokeWidth="2" strokeDasharray="3 2" />
      <text x={X(fc.last_rv)} y={y - 22} fill="#f59e0b" fontSize="9" textAnchor="middle">last RV {pct(fc.last_rv)}</text>
      <text x={X(fc.p10)} y={y + 26} fill="#64748b" fontSize="9" textAnchor="middle">P10 {pct(fc.p10)}</text>
      <text x={X(fc.p50)} y={y + 26} fill="#22d3ee" fontSize="10" textAnchor="middle">P50 {pct(fc.p50)}</text>
      <text x={X(fc.p90)} y={y + 26} fill="#64748b" fontSize="9" textAnchor="middle">P90 {pct(fc.p90)}</text>
    </svg>
  );
}

function PnLHist({ sim }) {
  if (!sim?.hist_counts) return null;
  const W = 340, H = 150, pad = 26;
  const counts = sim.hist_counts, edges = sim.hist_edges;
  const maxC = Math.max(...counts);
  const x0 = edges[0], x1 = edges[edges.length - 1];
  const X = (v) => pad + ((v - x0) / (x1 - x0)) * (W - 2 * pad);
  const bw = (W - 2 * pad) / counts.length;
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`}>
      {counts.map((c, i) => {
        const h = (c / maxC) * (H - 2 * pad);
        const mid = (edges[i] + edges[i + 1]) / 2;
        return <rect key={i} x={pad + i * bw} y={H - pad - h} width={bw - 1} height={h}
          fill={mid >= 0 ? "#34d399" : "#f87171"} opacity="0.85" />;
      })}
      <line x1={X(0)} y1={pad - 6} x2={X(0)} y2={H - pad} stroke="#94a3b8" strokeDasharray="3 3" />
      <text x={X(0)} y={pad - 8} fill="#94a3b8" fontSize="9" textAnchor="middle">break-even</text>
      <line x1={X(sim.pnl_p5)} y1={H - pad} x2={X(sim.pnl_p5)} y2={pad} stroke="#ef4444" strokeWidth="1" opacity="0.6" />
      <text x={X(sim.pnl_p5)} y={H - 8} fill="#ef4444" fontSize="8" textAnchor="middle">P5</text>
      <text x={X(sim.pnl_p95)} y={H - 8} fill="#10b981" fontSize="8" textAnchor="middle">P95</text>
    </svg>
  );
}

export default function Dashboard() {
  const [state, setState] = useState(null);
  const [strike, setStrike] = useState(180);
  const [surface, setSurface] = useState(null);
  const [fc, setFc] = useState(null);
  const [sim, setSim] = useState(null);
  const [loading, setLoading] = useState(true);
  const [simLoading, setSimLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [ai, setAi] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [calibText, setCalibText] = useState(
    '[{"K":150,"T_days":30,"iv":0.52},{"K":167.76,"T_days":30,"iv":0.42},{"K":185,"T_days":30,"iv":0.37}]'
  );
  const [calibMsg, setCalibMsg] = useState("");
  const [calibLoading, setCalibLoading] = useState(false);

  const spot = state?.spot ?? 167.76;

  const calibrate = useCallback(async () => {
    setCalibLoading(true); setCalibMsg("");
    try {
      const quotes = JSON.parse(calibText);
      const r = await api("/api/calibrate-surface", { spot, quotes, epochs: 3000 });
      if (r.error) { setCalibMsg("Error: " + r.error); }
      else {
        setCalibMsg(`Calibrated to ${r.rows_used} quotes · IV RMSE ${pct(r.iv_rmse)}`);
        const surf = await api("/api/iv-surface", {
          spot, T_days: DTE,
          strikes: Array.from({ length: 17 }, (_, i) => +(spot * (0.85 + i * 0.025)).toFixed(2)),
        });
        setSurface(surf);   // refresh the smile with the calibrated surface
      }
    } catch (e) { setCalibMsg("Invalid JSON or request failed: " + e.message); }
    finally { setCalibLoading(false); }
  }, [calibText, spot]);

  const T = DTE / 365;

  // strike-specific IV from the surface (the #2 upgrade); fallback 0.40
  const strikeIV = surface?.strikes?.find((s) => Math.abs(s.K - strike) < 0.5)?.iv
    ?? (surface?.strikes?.length
      ? surface.strikes.reduce((a, b) => Math.abs(b.K - strike) < Math.abs(a.K - strike) ? b : a).iv
      : 0.40);

  const bs = blackScholes(spot, strike, T, RISK_FREE, strikeIV);
  const premium = bs.call;
  const annualizedYield = (premium / spot) * (365 / DTE) * 100;
  const gainLossPerShare = strike - COST_BASIS;
  const totalPnL = gainLossPerShare * SHARES + premium * SHARES;
  const totalPnLPct = (totalPnL / (COST_BASIS * SHARES)) * 100;
  const breakeven = COST_BASIS - premium;

  const dailyRSI = state ? calcRSI(state.price_recent) : null;

  // initial load: state + forecast + surface
  useEffect(() => {
    (async () => {
      try {
        const [st, f] = await Promise.all([api("/api/state"), api("/api/forecast-vol", {})]);
        setState(st); setFc(f);
        const surf = await api("/api/iv-surface", {
          spot: st.spot, T_days: DTE,
          strikes: Array.from({ length: 17 }, (_, i) => +(st.spot * (0.85 + i * 0.025)).toFixed(2)),
        });
        setSurface(surf);
      } catch (e) { setErr(e.message); }
      finally { setLoading(false); }
    })();
  }, []);

  const runSim = useCallback(async () => {
    setSimLoading(true);
    try {
      const r = await api("/api/simulate", {
        spot, strike, premium: +premium.toFixed(2),
        cost_basis: COST_BASIS, shares: SHARES, n_paths: 20000,
        // target_vol omitted -> backend uses the forecaster P50 automatically
      });
      setSim(r);
    } catch (e) { setErr(e.message); }
    finally { setSimLoading(false); }
  }, [spot, strike, premium]);

  // recommendation logic — now gated on cost basis, RSI, AND assignment risk
  let rec = "NEUTRAL — mixed conditions.", recColor = "#8b5cf6";
  if (strike < COST_BASIS) {
    rec = "WAIT — strike is below your cost basis. Selling here locks in a capped loss.";
    recColor = "#ef4444";
  } else if (dailyRSI != null && dailyRSI > 65) {
    rec = "FAVORABLE — daily RSI elevated and strike above cost basis.";
    recColor = "#10b981";
  } else if (sim && sim.prob_profit < 0.25) {
    rec = "CAUTION — modeled probability of profit is low for this setup.";
    recColor = "#f59e0b";
  }

  const fetchAI = useCallback(async () => {
    setAiLoading(true); setAi("");
    const prompt = `You are a professional options strategist. Analyze this covered call on UAL.

POSITION: ${SHARES} shares @ cost ${COST_BASIS}, spot ${spot.toFixed(2)}, DTE ${DTE}.
NEURAL IV SURFACE (#2): strike ${strike} IV = ${(strikeIV * 100).toFixed(1)}% (strike-specific, not flat).
BLACK-SCHOLES: premium ${premium.toFixed(2)}/sh, delta ${bs.delta.toFixed(3)}, theta ${bs.theta.toFixed(3)}, vega ${bs.vega.toFixed(3)}, ann.yield ${annualizedYield.toFixed(1)}%.
VOL FORECAST (#1): next ${fc?.horizon_days}d RV P10/P50/P90 = ${fc ? `${pct(fc.p10)}/${pct(fc.p50)}/${pct(fc.p90)}` : "n/a"} vs last RV ${fc ? pct(fc.last_rv) : "n/a"}.
MONTE-CARLO (#3): assignment ${sim ? pct(sim.assignment_prob) : "run sim"}, touch ${sim ? pct(sim.touch_prob) : "-"}, E[P&L] ${sim ? fmt$(sim.expected_pnl) : "-"}, P5/P95 ${sim ? `${fmt$(sim.pnl_p5)}/${fmt$(sim.pnl_p95)}` : "-"}, CVaR5% ${sim ? fmt$(sim.cvar_5pct) : "-"}, P(profit) ${sim ? pct(sim.prob_profit) : "-"}.
RSI daily: ${dailyRSI?.toFixed(1)}.

Give: 1) TIMING VERDICT (1-2 sentences), 2) KEY RISKS (2-3 bullets), 3) ALTERNATIVE if poor timing, 4) BOTTOM LINE (one bold sentence). Be direct.`;
    try {
      const res = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          // each user sets their OWN key in .env (VITE_ANTHROPIC_API_KEY)
          "x-api-key": import.meta.env.VITE_ANTHROPIC_API_KEY,
          "anthropic-version": "2023-06-01",
          "anthropic-dangerous-direct-browser-access": "true",
        },
        body: JSON.stringify({ model: "claude-sonnet-4-6", max_tokens: 1000, messages: [{ role: "user", content: prompt }] }),
      });
      const data = await res.json();
      setAi(data.content.filter((c) => c.type === "text").map((c) => c.text).join("\n"));
    } catch (e) { setAi("AI call failed locally — set your Anthropic key (x-api-key header) or proxy. " + e.message); }
    finally { setAiLoading(false); }
  }, [spot, strike, strikeIV, premium, bs, fc, sim, dailyRSI, annualizedYield]);

  if (loading) return <div style={S.wrap}><div style={S.muted}>Connecting to neural backend at {API_BASE} …</div></div>;
  if (err) return <div style={S.wrap}><div style={{ ...S.card, borderColor: "#ef4444" }}>
    <b style={{ color: "#ef4444" }}>Backend not reachable.</b>
    <div style={S.muted}>Start it: <code>cd backend && python train_all.py && python app.py</code></div>
    <div style={S.muted}>{err}</div></div></div>;

  return (
    <div style={S.wrap}>
      <style>{`*{box-sizing:border-box} button:hover{filter:brightness(1.12)}`}</style>
      <div style={S.header}>
        <div>
          <div style={S.title}>UAL Covered Call · <span style={{ color: "#38bdf8" }}>Neural Pricing v2</span></div>
          <div style={S.muted}>spot ${spot.toFixed(2)} · cost ${COST_BASIS} · {DTE} DTE · models: IV-surface · vol-forecast · scenario-MC</div>
        </div>
      </div>

      <div style={{ ...S.banner, borderColor: recColor }}>
        <span style={{ color: recColor, fontWeight: 700 }}>{rec}</span>
      </div>

      <div style={S.grid}>
        {/* Position + Greeks (now fed strike-specific IV) */}
        <div style={S.card}>
          <div style={S.cardTitle}>Black-Scholes Greeks · ${strike} strike</div>
          <div style={S.row}>
            <label style={S.muted}>Strike</label>
            <input type="range" min={Math.round(spot)} max={Math.round(spot * 1.25)} value={strike}
              onChange={(e) => setStrike(+e.target.value)} style={{ flex: 1 }} />
            <b>${strike}</b>
          </div>
          <div style={S.metricRow}><span style={S.muted}>IV (surface, strike-specific)</span><b style={{ color: "#38bdf8" }}>{pct(strikeIV)}</b></div>
          <div style={S.metricRow}><span style={S.muted}>Premium</span><b>${premium.toFixed(2)}/sh · ${(premium * 100).toFixed(0)}</b></div>
          <div style={S.metricRow}><span style={S.muted}>Δ Delta</span><b>{bs.delta.toFixed(3)}</b></div>
          <div style={S.metricRow}><span style={S.muted}>Γ / Θ / V</span><b>{bs.gamma.toFixed(4)} / {bs.theta.toFixed(3)} / {bs.vega.toFixed(3)}</b></div>
          <div style={S.metricRow}><span style={S.muted}>Ann. yield</span><b>{annualizedYield.toFixed(1)}%</b></div>
          <div style={S.metricRow}><span style={S.muted}>If called P&L</span><b style={{ color: totalPnL >= 0 ? "#10b981" : "#f87171" }}>{fmt$(totalPnL)} ({totalPnLPct.toFixed(1)}%)</b></div>
        </div>

        {/* #2 IV surface skew */}
        <div style={S.card}>
          <div style={S.cardTitle}>① IV surface skew · {DTE} DTE <span style={S.tag}>net #2</span></div>
          <Smile smile={surface?.smile} atmStrikeIV={strikeIV} />
          <div style={S.muted}>Each strike priced with its own IV — the put wing is richer, so the 0.20Δ strike and Greeks are no longer biased by a flat sigma.</div>
        </div>

        {/* #1 vol forecast */}
        <div style={S.card}>
          <div style={S.cardTitle}>② Volatility forecast · next {fc?.horizon_days}d <span style={S.tag}>net #1</span></div>
          <ForecastBand fc={fc} />
          <div style={S.muted}>{fc?.model === "nhits" ? "N-HiTS (neuralforecast, +exog)" : "LSTM"} quantile forecast. P50 = {fc ? pct(fc.p50) : "—"} feeds the Monte-Carlo sizing automatically.</div>
        </div>

        {/* IV-surface calibration to a real chain */}
        <div style={S.card}>
          <div style={S.cardTitle}>Calibrate surface to your chain <span style={S.tag}>refit #2</span></div>
          <textarea value={calibText} onChange={(e) => setCalibText(e.target.value)}
            rows={4} style={S.textarea}
            placeholder='[{"K":150,"T_days":30,"iv":0.52}, ...]  (or use "price"+"kind")' />
          <button style={{ ...S.btn, marginTop: 8 }} onClick={calibrate} disabled={calibLoading}>
            {calibLoading ? "Fitting…" : "Calibrate to my chain"}
          </button>
          {calibMsg && <div style={{ ...S.muted, color: calibMsg.startsWith("Calibrated") ? "#10b981" : "#f87171" }}>{calibMsg}</div>}
          <div style={S.muted}>Posts quotes, refits the IV net with no-arb penalties, hot-swaps the live surface above.</div>
        </div>

        {/* #3 scenario MC */}
        <div style={S.card}>
          <div style={S.cardTitle}>③ Monte-Carlo P&L · 20k paths <span style={S.tag}>net #3</span></div>
          {!sim && <button style={S.btn} onClick={runSim} disabled={simLoading}>{simLoading ? "Simulating…" : "Run simulation"}</button>}
          {sim && <>
            <PnLHist sim={sim} />
            <div style={S.metricRow}><span style={S.muted}>Assignment / touch</span><b>{pct(sim.assignment_prob)} / {pct(sim.touch_prob)}</b></div>
            <div style={S.metricRow}><span style={S.muted}>E[P&L] · P5 · P95</span><b>{fmt$(sim.expected_pnl)} · {fmt$(sim.pnl_p5)} · {fmt$(sim.pnl_p95)}</b></div>
            <div style={S.metricRow}><span style={S.muted}>CVaR 5% · P(profit)</span><b style={{ color: "#f87171" }}>{fmt$(sim.cvar_5pct)}</b> · <b>{pct(sim.prob_profit)}</b></div>
            <button style={{ ...S.btn, marginTop: 8 }} onClick={runSim} disabled={simLoading}>{simLoading ? "…" : "↻ Re-run"}</button>
          </>}
        </div>

        {/* RSI (kept) */}
        <div style={S.card}>
          <div style={S.cardTitle}>RSI · daily(14)</div>
          <div style={{ fontSize: 34, fontWeight: 700, color: dailyRSI > 70 ? "#ef4444" : dailyRSI > 60 ? "#f59e0b" : dailyRSI < 40 ? "#38bdf8" : "#94a3b8" }}>
            {dailyRSI?.toFixed(1)}
          </div>
          <div style={S.muted}>{dailyRSI > 70 ? "Overbought — favorable for call selling" : dailyRSI > 60 ? "Elevated" : dailyRSI < 40 ? "Oversold — may bounce" : "Neutral"}</div>
          <div style={{ marginTop: 10 }}>
            {RESISTANCE_LEVELS.map((r) => (
              <div key={r.price} style={S.metricRow}><span style={S.muted}>{r.label}</span><b>${r.price}</b></div>
            ))}
          </div>
        </div>

        {/* AI strategist (kept, enriched) */}
        <div style={{ ...S.card, gridColumn: "1 / -1" }}>
          <div style={S.cardTitle}>AI Strategist <span style={S.tag}>claude-sonnet-4-6</span></div>
          <button style={S.btn} onClick={fetchAI} disabled={aiLoading}>{aiLoading ? "Analyzing…" : sim ? "Get verdict" : "Run simulation first for full context"}</button>
          {ai && <pre style={S.pre}>{ai}</pre>}
        </div>
      </div>
      <div style={S.muted}>Synthetic data demo. Swap data_synth.py for your live chain + price feed; the models and API stay identical.</div>
    </div>
  );
}

const S = {
  wrap: { background: "#0b1220", minHeight: "100vh", color: "#e2e8f0", fontFamily: "Inter, system-ui, sans-serif", padding: 20 },
  header: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 },
  title: { fontSize: 22, fontWeight: 700 },
  muted: { color: "#64748b", fontSize: 12, marginTop: 4 },
  banner: { border: "1px solid", borderRadius: 10, padding: "10px 14px", marginBottom: 16, background: "#0f172a" },
  grid: { display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))", gap: 14 },
  card: { background: "#0f172a", border: "1px solid #1e293b", borderRadius: 12, padding: 16 },
  cardTitle: { fontSize: 13, fontWeight: 600, marginBottom: 10, color: "#cbd5e1", display: "flex", justifyContent: "space-between", alignItems: "center" },
  tag: { fontSize: 10, color: "#38bdf8", background: "#0c2535", borderRadius: 6, padding: "2px 6px", fontWeight: 600 },
  row: { display: "flex", alignItems: "center", gap: 10, marginBottom: 12 },
  metricRow: { display: "flex", justifyContent: "space-between", padding: "4px 0", fontSize: 13, borderBottom: "1px solid #16202e" },
  btn: { background: "#0ea5e9", color: "#04141f", border: "none", borderRadius: 8, padding: "8px 14px", fontWeight: 600, cursor: "pointer", fontSize: 13 },
  pre: { whiteSpace: "pre-wrap", fontSize: 13, lineHeight: 1.55, marginTop: 12, color: "#cbd5e1", fontFamily: "inherit" },
  textarea: { width: "100%", background: "#0b1424", color: "#cbd5e1", border: "1px solid #1e293b", borderRadius: 8, padding: 8, fontSize: 11, fontFamily: "ui-monospace, monospace", resize: "vertical" },
};
