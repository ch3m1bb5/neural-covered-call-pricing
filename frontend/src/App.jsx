import { useState, useEffect, useCallback } from "react";
import ForecastChart from "./components/ForecastChart";
import StrikeLadder from "./components/StrikeLadder";
import SignalPanel from "./components/SignalPanel";
import GreeksPanel from "./components/GreeksPanel";
import TechPanel from "./components/TechPanel";
import DecisionBanner from "./components/DecisionBanner";
import DecompositionChart from "./components/DecompositionChart";

const API = "http://localhost:5050";

const DEFAULT_PARAMS = {
  ticker: "SMCI",
  current_price: 30.85,
  cost_basis: 31.07,
  shares: 1000,
  dte: 30,
  risk_free: 0.053,
  target_delta: 0.20,
  nhits_horizon: 20,
  manual_strike: "",
  dividend_yield: 0.0,
  earnings_in_window: false,
  american: true,
  use_sabr: true,
  use_garch: true,
  api_key: "",
};

function useAnalysis(params) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [libs, setLibs] = useState(null);
  const [aiText, setAiText] = useState("");
  const [aiLoading, setAiLoading] = useState(false);

  const run = useCallback(async () => {
    setLoading(true); setError(null); setAiText("");
    try {
      const [health, analysis] = await Promise.all([
        fetch(`${API}/api/health`).then(r => r.json()),
        fetch(`${API}/api/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        }).then(r => r.json()),
      ]);
      setLibs(health.libs);
      setData(analysis);
      setLoading(false);  // dashboard aparece JA, sem esperar pela AI

      // Analise Claude carrega DEPOIS, em separado (nao bloqueia o dashboard)
      if (analysis._ai_pending) {
        setAiLoading(true);
        try {
          const aiResp = await fetch(`${API}/api/ai_analysis`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              api_key: params.api_key,
              data: { ...analysis, auto_strike: analysis.auto_strike, manual_strike: analysis.manual_strike },
            }),
          }).then(r => r.json());
          setAiText(aiResp.analysis || "");
        } catch (e) {
          setAiText("Erro ao obter analise AI.");
        }
        setAiLoading(false);
      }
    } catch (e) {
      setError("Nao foi possivel ligar ao backend Flask. Confirma que esta a correr em :5050");
      setLoading(false);
    }
  }, [JSON.stringify(params)]);

  useEffect(() => { run(); }, [run]);
  return { data, loading, error, libs, aiText, aiLoading, refresh: run };
}

export default function App() {
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [draft, setDraft] = useState(DEFAULT_PARAMS);
  const { data, loading, error, libs, aiText, aiLoading, refresh } = useAnalysis(params);

  const apply = () => setParams({ ...draft });

  const updateDraft = (key, value, isNumber) => {
    setDraft(d => ({ ...d, [key]: isNumber ? (value === "" ? "" : parseFloat(value)) : value }));
  };

  return (
    <div style={styles.root}>
      <style>{css}</style>

      {/* Header */}
      <header style={styles.header}>
        <div>
          <div style={styles.ticker}>{params.ticker}</div>
          <div style={styles.headerSub}>N-HiTS Covered Call Engine</div>
        </div>
        {data && (
          <div style={styles.headerPrice}>
            <span style={styles.priceVal}>${data.current_price.toFixed(2)}</span>
            <span style={{ ...styles.priceDiff, color: data.current_price >= data.cost_basis ? "#10b981" : "#ef4444" }}>
              {data.current_price >= data.cost_basis ? "+" : "−"}${Math.abs(data.cost_basis - data.current_price).toFixed(2)} vs cost
            </span>
          </div>
        )}
        <button className="btn-refresh" onClick={refresh} disabled={loading}>
          {loading ? "Calculando..." : "Re-run"}
        </button>
      </header>

      {/* Lib status bar */}
      {libs && (
        <div style={styles.libBar}>
          {[["N-HiTS", libs.neuralforecast_nhits], ["pmdarima", libs.pmdarima],
            ["QuantLib", libs.quantlib], ["yfinance", libs.yfinance]].map(([name, ok]) => (
            <span key={name} className={ok ? "lib-ok" : "lib-warn"}>{ok ? "●" : "○"} {name}</span>
          ))}
          {data && (
            <span style={{ color: "#4a5568", fontSize: "10px", marginLeft: "auto" }}>
              {data.libs.nhits_model} · {data.libs.arima_model} · Greeks: {data.libs.greeks_engine}
            </span>
          )}
        </div>
      )}

      {/* Param bar */}
      <div style={styles.paramBar}>
        <label style={styles.paramLabel}>
          <span style={styles.paramName}>Ticker</span>
          <input type="text" value={draft.ticker}
            onChange={e => updateDraft("ticker", e.target.value, false)}
            className="param-input" />
        </label>

        <label style={styles.paramLabel}>
          <span style={styles.paramName}>Current Price $</span>
          <input type="number" step="0.01" value={draft.current_price}
            onChange={e => updateDraft("current_price", e.target.value, true)}
            className="param-input" />
        </label>

        <label style={styles.paramLabel}>
          <span style={styles.paramName}>Cost Basis $</span>
          <input type="number" step="0.01" value={draft.cost_basis}
            onChange={e => updateDraft("cost_basis", e.target.value, true)}
            className="param-input" />
        </label>

        <label style={styles.paramLabel}>
          <span style={styles.paramName}>Shares</span>
          <input type="number" step="1" value={draft.shares}
            onChange={e => updateDraft("shares", e.target.value, true)}
            className="param-input" />
        </label>

        <label style={styles.paramLabel}>
          <span style={styles.paramName}>DTE</span>
          <input type="number" step="1" value={draft.dte}
            onChange={e => updateDraft("dte", e.target.value, true)}
            className="param-input" />
        </label>

        <label style={styles.paramLabel}>
          <span style={styles.paramName}>Risk-Free %</span>
          <input type="number" step="0.001" value={draft.risk_free}
            onChange={e => updateDraft("risk_free", e.target.value, true)}
            className="param-input" />
        </label>

        <label style={styles.paramLabel}>
          <span style={styles.paramName}>Target Δ (auto)</span>
          <input type="number" step="0.01" value={draft.target_delta}
            onChange={e => updateDraft("target_delta", e.target.value, true)}
            className="param-input" />
        </label>

        <label style={{ ...styles.paramLabel, position: "relative" }}>
          <span style={{ ...styles.paramName, color: "#818cf8" }}>★ Manual Strike $</span>
          <input type="number" step="0.5" placeholder="ex: 32.50" value={draft.manual_strike}
            onChange={e => updateDraft("manual_strike", e.target.value, true)}
            className="param-input param-input-highlight" />
        </label>

        <label style={styles.paramLabel}>
          <span style={styles.paramName}>Horizon (dias)</span>
          <input type="number" step="1" value={draft.nhits_horizon}
            onChange={e => updateDraft("nhits_horizon", e.target.value, true)}
            className="param-input" />
        </label>

        <button className="btn-apply" onClick={apply}>Apply</button>
      </div>

      {/* Advanced pricing controls */}
      <div style={styles.advBar}>
        <label style={styles.paramLabel}>
          <span style={styles.paramName}>Dividend Yield %</span>
          <input type="number" step="0.001" value={draft.dividend_yield}
            onChange={e => updateDraft("dividend_yield", e.target.value, true)}
            className="param-input" />
        </label>

        <label style={styles.toggleLabel}>
          <input type="checkbox" checked={draft.earnings_in_window}
            onChange={e => updateDraft("earnings_in_window", e.target.checked, false)} />
          <span>Earnings na janela (+25% IV)</span>
        </label>

        <label style={styles.toggleLabel}>
          <input type="checkbox" checked={draft.american}
            onChange={e => updateDraft("american", e.target.checked, false)} />
          <span>Pricing Americano (Binomial)</span>
        </label>

        <label style={styles.toggleLabel}>
          <input type="checkbox" checked={draft.use_sabr}
            onChange={e => updateDraft("use_sabr", e.target.checked, false)} />
          <span>SABR smile/skew</span>
        </label>

        <label style={styles.toggleLabel}>
          <input type="checkbox" checked={draft.use_garch}
            onChange={e => updateDraft("use_garch", e.target.checked, false)} />
          <span>GARCH(1,1) vol</span>
        </label>
      </div>

      {/* API key bar (optional, for AI analysis) */}
      <div style={styles.apiBar}>
        <span style={styles.paramName}>Claude API Key (opcional, para analise AI)</span>
        <input
          type="password"
          placeholder="sk-ant-..."
          value={draft.api_key}
          onChange={e => updateDraft("api_key", e.target.value, false)}
          className="param-input"
          style={{ width: "320px" }}
        />
      </div>

      {/* Error */}
      {error && (
        <div style={styles.errorBox}>
          ⚠ {error}
          <div style={{ fontSize: "11px", marginTop: "6px", color: "#94a3b8" }}>
            Terminal: <code>cd backend &amp;&amp; python app.py</code>
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div style={styles.loadingBox}>
          <div className="pulse-dot" />
          <span>A treinar stacks N-HiTS e a calcular varredura de parametros...</span>
        </div>
      )}

      {/* Main */}
      {data && !loading && (
        <div style={styles.main}>

          <DecisionBanner decision={data.decision} />

          {/* Manual Strike Result — destaque se inserido */}
          {data.manual_strike && (
            <div style={{
              background: `${data.manual_strike.color}12`,
              border: `2px solid ${data.manual_strike.color}50`,
              borderRadius: "12px", padding: "18px 24px",
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "10px" }}>
                <div>
                  <div style={{ fontSize: "10px", letterSpacing: "0.15em", textTransform: "uppercase", color: "#818cf8", marginBottom: "4px" }}>
                    ★ Strike Manual: ${data.manual_strike.strike.toFixed(2)}
                  </div>
                  <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: "20px", fontWeight: "700", color: data.manual_strike.color }}>
                    {data.manual_strike.verdict}
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: "9px", color: "#4a5568", textTransform: "uppercase" }}>Score</div>
                  <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: "18px", fontWeight: "700", color: data.manual_strike.color }}>
                    {data.manual_strike.score > 0 ? "+" : ""}{data.manual_strike.score}/{data.manual_strike.max_score}
                  </div>
                </div>
              </div>
              <p style={{ margin: "0 0 12px", fontSize: "13px", color: "#94a3b8" }}>{data.manual_strike.rationale}</p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "10px" }}>
                <MiniStat label="Delta" val={data.manual_strike.greeks.delta.toFixed(3)} />
                <MiniStat label="Premio/sh" val={`$${data.manual_strike.premium.toFixed(2)}`} />
                <MiniStat label="Total Premio" val={`$${(data.manual_strike.premium * data.shares).toFixed(0)}`} />
                <MiniStat label="Ann. Yield" val={`${data.manual_strike.ann_yield.toFixed(1)}%`} />
                <MiniStat label="P&L se exercida" val={`${data.manual_strike.total_pnl >= 0 ? "+" : ""}$${data.manual_strike.total_pnl.toFixed(0)}`}
                  color={data.manual_strike.total_pnl >= 0 ? "#10b981" : "#ef4444"} />
                <MiniStat label="P&L %" val={`${data.manual_strike.pnl_pct.toFixed(1)}%`}
                  color={data.manual_strike.pnl_pct >= 0 ? "#10b981" : "#ef4444"} />
                <MiniStat label="Breakeven" val={`$${data.manual_strike.breakeven.toFixed(2)}`} />
                <MiniStat label="vs Strike Auto" val={`${data.manual_strike.strike >= data.recommended_strike ? "+" : ""}$${(data.manual_strike.strike - data.recommended_strike).toFixed(2)}`} />
              </div>
            </div>
          )}

          {/* Row 1: Forecast + Decomposition */}
          <div style={styles.row2}>
            <div style={styles.card}>
              <div style={styles.cardTitle}>N-HiTS Price Forecast · {data.nhits.horizon}d horizon</div>
              <ForecastChart
                history={data.price_history}
                nhits={data.nhits}
                arima={data.arima}
                costBasis={data.cost_basis}
                strike={data.manual_strike ? data.manual_strike.strike : data.recommended_strike}
              />
              <div style={styles.fcMeta}>
                <Stat label="10d Target" val={`$${data.nhits.target_10d.toFixed(2)}`}
                  color={data.nhits.target_10d > data.current_price ? "#10b981" : "#ef4444"} />
                <Stat label="20d Target" val={`$${data.nhits.target_20d.toFixed(2)}`}
                  color={data.nhits.target_20d > data.current_price ? "#10b981" : "#ef4444"} />
                <Stat label="Direction" val={data.nhits.direction}
                  color={data.nhits.direction === "UP" ? "#10b981" : "#ef4444"} />
                <Stat label="Confidence" val={`${(data.nhits.confidence * 100).toFixed(0)}%`} color="#6366f1" />
              </div>
            </div>

            <div style={styles.card}>
              <div style={styles.cardTitle}>Varredura de Volatilidade (12 janelas: 1h–45d)</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "7px", marginBottom: "12px" }}>
                {Object.entries(data.vol_sweep.by_window).map(([label, hv]) => {
                  const isBest = label === data.vol_sweep.best_window;
                  return (
                    <div key={label} style={{
                      background: isBest ? "#6366f120" : "#090913",
                      border: isBest ? "1px solid #6366f160" : "1px solid #1a1a28",
                      borderRadius: "8px", padding: "9px", textAlign: "center"
                    }}>
                      <div style={{ fontSize: "9px", color: "#4a5568" }}>{label}</div>
                      <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: "13px", fontWeight: "700", color: isBest ? "#818cf8" : "#94a3b8" }}>
                        {hv}%
                      </div>
                    </div>
                  );
                })}
              </div>
              <div style={{ fontSize: "11px", color: "#4a5568", lineHeight: "1.6" }}>
                Melhor janela: <b style={{ color: "#818cf8" }}>{data.vol_sweep.best_window}</b> ({data.vol_sweep.best_window_hours}h, menor erro de previsao rolling)<br />
                HV optima: <b>{(data.vol_sweep.hv * 100).toFixed(1)}%</b> · IV calibrada: <b>{(data.iv * 100).toFixed(1)}%</b><br />
                AR optimo: p={data.sweep_params.best_ar_p} · Fourier: h={data.sweep_params.best_harmonics}
              </div>
            </div>
          </div>

          {/* Advanced pricing models panel */}
          <div style={styles.card}>
            <div style={styles.cardTitle}>Modelos Avancados de Precificacao</div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px" }}>
              {/* Engine */}
              <div style={styles.modelBox}>
                <div style={styles.modelLabel}>Engine de Pricing</div>
                <div style={styles.modelVal}>{data.greeks.engine}</div>
                <div style={styles.modelSub}>{data.american ? "Exercicio antecipado ON" : "Europeu"}</div>
              </div>
              {/* GARCH */}
              <div style={styles.modelBox}>
                <div style={styles.modelLabel}>GARCH(1,1)</div>
                <div style={styles.modelVal}>
                  {data.use_garch && data.garch.converged ? `${(data.garch.next_vol_annual * 100).toFixed(1)}%` : "OFF"}
                </div>
                <div style={styles.modelSub}>
                  {data.garch.converged ? `persist ${data.garch.persistence.toFixed(2)} · LR ${(data.garch.long_run_vol_annual * 100).toFixed(0)}%` : "nao convergiu"}
                </div>
              </div>
              {/* SABR */}
              <div style={styles.modelBox}>
                <div style={styles.modelLabel}>SABR Smile</div>
                <div style={styles.modelVal}>
                  {data.use_sabr && data.sabr.alpha ? `α=${data.sabr.alpha.toFixed(2)}` : "OFF"}
                </div>
                <div style={styles.modelSub}>
                  {data.sabr.rho !== undefined ? `ρ=${data.sabr.rho} (skew) · ν=${data.sabr.nu}` : "sem skew"}
                </div>
              </div>
              {/* IV do strike */}
              <div style={styles.modelBox}>
                <div style={styles.modelLabel}>IV @ Strike Auto</div>
                <div style={styles.modelVal}>{(data.greeks.strike_iv * 100).toFixed(1)}%</div>
                <div style={styles.modelSub}>
                  base {(data.iv * 100).toFixed(1)}% {data.earnings_in_window ? "· earnings+" : ""}
                </div>
              </div>
            </div>
            {/* Greeks de 2a ordem */}
            <div style={{ marginTop: "14px", paddingTop: "14px", borderTop: "1px solid #1a1a28" }}>
              <div style={{ fontSize: "9px", color: "#4a5568", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "10px" }}>
                Greeks de 2a Ordem (gestao activa)
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px" }}>
                <div style={styles.modelBox}>
                  <div style={styles.modelLabel}>Vanna (∂Δ/∂σ)</div>
                  <div style={styles.modelVal}>{data.greeks.vanna?.toFixed(4)}</div>
                  <div style={styles.modelSub}>delta vs IV</div>
                </div>
                <div style={styles.modelBox}>
                  <div style={styles.modelLabel}>Charm (∂Δ/∂t)</div>
                  <div style={styles.modelVal}>{data.greeks.charm?.toFixed(5)}</div>
                  <div style={styles.modelSub}>delta/dia decay</div>
                </div>
                <div style={styles.modelBox}>
                  <div style={styles.modelLabel}>Volga (∂Vega/∂σ)</div>
                  <div style={styles.modelVal}>{data.greeks.volga?.toFixed(4)}</div>
                  <div style={styles.modelSub}>convexidade vega</div>
                </div>
              </div>
            </div>
          </div>

          {/* Row 2: Greeks + Tech */}
          <div style={styles.row2}>
            <GreeksPanel
              greeks={data.greeks}
              strike={data.recommended_strike}
              premium={data.premium}
              annYield={data.ann_yield}
              iv={data.iv}
              hv={data.hv30}
              currentPrice={data.current_price}
              costBasis={data.cost_basis}
              dte={data.dte}
              breakeven={data.breakeven}
              totalPnl={data.total_pnl_if_called}
              pnlPct={data.pnl_pct_if_called}
              shares={data.shares}
            />
            <TechPanel
              dailyRsi={data.daily_rsi}
              weeklyRsi={data.weekly_rsi}
              bollinger={data.bollinger}
              sr={data.support_resistance}
              currentPrice={data.current_price}
              strike={data.recommended_strike}
            />
          </div>

          {/* AI Analysis (carrega assincronamente) */}
          {(aiLoading || aiText) && (
            <div style={styles.card}>
              <div style={styles.cardTitle}>Analise Claude ({libs?.claude_model || "AI"})</div>
              {aiLoading ? (
                <div style={{ display: "flex", alignItems: "center", gap: "10px", color: "#4a5568", fontSize: "12px" }}>
                  <div className="pulse-dot" />
                  <span>A consultar o estrategista AI... (pode demorar ate 30s)</span>
                </div>
              ) : (
                <div style={{ fontSize: "13px", color: "#94a3b8", lineHeight: "1.7", whiteSpace: "pre-wrap" }}>
                  {aiText}
                </div>
              )}
            </div>
          )}

          {/* Row 3: Signals + Strike ladder */}
          <div style={styles.row2}>
            <SignalPanel signals={data.decision.signals} score={data.decision.score} />
            <div style={styles.card}>
              <div style={styles.cardTitle}>Strike Ladder · {data.dte} DTE · IV {(data.iv * 100).toFixed(1)}%</div>
              <StrikeLadder
                ladder={data.strike_ladder}
                recommended={data.manual_strike ? data.manual_strike.strike : data.recommended_strike}
                costBasis={data.cost_basis}
                currentPrice={data.current_price}
              />
            </div>
          </div>

        </div>
      )}
    </div>
  );
}

function Stat({ label, val, color }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ fontSize: "9px", letterSpacing: "0.12em", textTransform: "uppercase", color: "#4a5568" }}>{label}</div>
      <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: "16px", fontWeight: "700", color }}>{val}</div>
    </div>
  );
}

function MiniStat({ label, val, color = "#e2e8f0" }) {
  return (
    <div style={{ background: "#090913", borderRadius: "8px", padding: "10px", textAlign: "center" }}>
      <div style={{ fontSize: "9px", color: "#4a5568", textTransform: "uppercase", letterSpacing: "0.08em" }}>{label}</div>
      <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: "15px", fontWeight: "700", color }}>{val}</div>
    </div>
  );
}

const styles = {
  root: { background: "#080810", minHeight: "100vh", color: "#e2e8f0", fontFamily: "'DM Mono', monospace", padding: "0 0 40px" },
  header: { display: "flex", alignItems: "center", gap: "20px", padding: "20px 28px", borderBottom: "1px solid #12121e", background: "#0a0a14" },
  ticker: { fontFamily: "'Space Grotesk', sans-serif", fontSize: "28px", fontWeight: "700", letterSpacing: "-0.02em" },
  headerSub: { fontSize: "10px", color: "#4a5568", letterSpacing: "0.15em", textTransform: "uppercase", marginTop: "2px" },
  headerPrice: { marginLeft: "auto", textAlign: "right" },
  priceVal: { fontFamily: "'Space Grotesk', sans-serif", fontSize: "32px", fontWeight: "700" },
  priceDiff: { display: "block", fontSize: "12px", marginTop: "2px" },
  libBar: { display: "flex", gap: "16px", alignItems: "center", padding: "8px 28px", background: "#09090f", borderBottom: "1px solid #12121e", flexWrap: "wrap" },
  paramBar: { display: "flex", gap: "12px", alignItems: "flex-end", padding: "14px 28px", background: "#0a0a14", borderBottom: "1px solid #12121e", flexWrap: "wrap" },
  apiBar: { display: "flex", gap: "12px", alignItems: "center", padding: "10px 28px", background: "#09090f", borderBottom: "1px solid #12121e" },
  advBar: { display: "flex", gap: "20px", alignItems: "center", padding: "12px 28px", background: "#0a0a14", borderBottom: "1px solid #12121e", flexWrap: "wrap" },
  toggleLabel: { display: "flex", alignItems: "center", gap: "7px", fontSize: "11px", color: "#94a3b8", cursor: "pointer", userSelect: "none" },
  modelBox: { background: "#090913", border: "1px solid #1a1a28", borderRadius: "8px", padding: "12px", textAlign: "center" },
  modelLabel: { fontSize: "9px", color: "#4a5568", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "4px" },
  modelVal: { fontFamily: "'Space Grotesk', sans-serif", fontSize: "18px", fontWeight: "700", color: "#818cf8" },
  modelSub: { fontSize: "9px", color: "#4a5568", marginTop: "3px" },
  paramLabel: { display: "flex", flexDirection: "column", gap: "4px" },
  paramName: { fontSize: "9px", color: "#4a5568", letterSpacing: "0.1em", textTransform: "uppercase" },
  errorBox: { margin: "20px 28px", background: "#1a0a0a", border: "1px solid #ef444440", borderRadius: "10px", padding: "16px 20px", color: "#fca5a5" },
  loadingBox: { display: "flex", alignItems: "center", gap: "12px", padding: "40px 28px", color: "#4a5568" },
  main: { padding: "20px 28px", display: "flex", flexDirection: "column", gap: "16px" },
  row2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" },
  card: { background: "#0f0f1a", border: "1px solid #1a1a28", borderRadius: "12px", padding: "20px" },
  cardTitle: { fontSize: "10px", letterSpacing: "0.15em", textTransform: "uppercase", color: "#4a5568", marginBottom: "14px", borderBottom: "1px solid #1a1a28", paddingBottom: "8px" },
  fcMeta: { display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "8px", marginTop: "12px" },
};

const css = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Space+Grotesk:wght@500;600;700&display=swap');
  * { box-sizing: border-box; }
  .btn-refresh { background: #1a1a2e; border: 1px solid #2d2d45; color: #94a3b8; border-radius: 7px; padding: 8px 18px; cursor: pointer; font-family: 'DM Mono', monospace; font-size: 11px; letter-spacing: 0.08em; }
  .btn-refresh:hover { border-color: #6366f1; color: #e2e8f0; }
  .btn-refresh:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-apply { background: #6366f120; border: 1px solid #6366f140; color: #818cf8; border-radius: 7px; padding: 6px 16px; cursor: pointer; font-family: 'DM Mono', monospace; font-size: 11px; align-self: flex-end; }
  .btn-apply:hover { background: #6366f130; }
  .param-input { background: #0d0d18; border: 1px solid #1e1e30; border-radius: 6px; color: #e2e8f0; font-family: 'DM Mono', monospace; font-size: 12px; padding: 5px 10px; width: 90px; }
  .param-input:focus { outline: none; border-color: #6366f1; }
  .param-input-highlight { border-color: #6366f160; background: #6366f108; }
  .param-input-highlight:focus { border-color: #818cf8; box-shadow: 0 0 0 2px #6366f130; }
  .lib-ok { font-size: 10px; color: #10b981; letter-spacing: 0.08em; }
  .lib-warn { font-size: 10px; color: #f59e0b; letter-spacing: 0.08em; }
  .pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: #6366f1; animation: pulse 1.2s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.3;transform:scale(0.8)} }
  @media (max-width: 900px) { .row2 { grid-template-columns: 1fr !important; } }
`;
