import os

base = r"C:\Users\Rodrigo\Documents\Quantitative finance\frontend\src\components"
os.makedirs(base, exist_ok=True)

# ── ForecastChart.jsx ─────────────────────────────────────────────────────
with open(os.path.join(base, "ForecastChart.jsx"), "w") as f:
    f.write('''import { useMemo } from "react";

export default function ForecastChart({ history, nhits, arima, costBasis, strike }) {
  const W = 560, H = 180, PAD = { t: 10, r: 10, b: 30, l: 48 };
  const IW = W - PAD.l - PAD.r, IH = H - PAD.t - PAD.b;

  const allVals = useMemo(() => [
    ...history.prices, ...nhits.forecast, ...arima.forecast, costBasis, strike,
  ], [history, nhits, arima, costBasis, strike]);

  const minV = Math.min(...allVals) * 0.98;
  const maxV = Math.max(...allVals) * 1.02;
  const totalPoints = history.prices.length + nhits.horizon;
  const xScale = i => (i / (totalPoints - 1)) * IW + PAD.l;
  const yScale = v => IH - ((v - minV) / (maxV - minV)) * IH + PAD.t;

  const histPath = history.prices.map((p, i) => `${i === 0 ? "M" : "L"}${xScale(i)},${yScale(p)}`).join(" ");
  const hLen = history.prices.length;
  const nhitsPath = nhits.forecast.map((p, i) => `${i === 0 ? "M" : "L"}${xScale(hLen - 1 + i)},${yScale(p)}`).join(" ");
  const arimaPath = arima.forecast.map((p, i) => `${i === 0 ? "M" : "L"}${xScale(hLen - 1 + i)},${yScale(p)}`).join(" ");

  const bandPath = [
    ...arima.upper.map((v, i) => `${i === 0 ? "M" : "L"}${xScale(hLen - 1 + i)},${yScale(v)}`),
    ...[...arima.lower].reverse().map((v, i) => `L${xScale(hLen - 1 + (arima.lower.length - 1 - i))},${yScale(v)}`),
    "Z"
  ].join(" ");

  const gridY = [minV, (minV + maxV) / 2, maxV].map(v => ({ v, y: yScale(v) }));
  const costY = yScale(costBasis);
  const strikeY = yScale(strike);
  const dividerX = xScale(hLen - 1);

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: "visible" }}>
      <defs>
        <linearGradient id="histGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#6366f1" stopOpacity="0.15" />
          <stop offset="100%" stopColor="#6366f1" stopOpacity="0" />
        </linearGradient>
        <clipPath id="forecastClip">
          <rect x={dividerX} y={PAD.t} width={IW} height={IH} />
        </clipPath>
      </defs>
      {gridY.map(({ v, y }) => (
        <g key={v}>
          <line x1={PAD.l} y1={y} x2={W - PAD.r} y2={y} stroke="#1a1a28" strokeWidth="1" />
          <text x={PAD.l - 4} y={y + 4} fontSize="8" fill="#4a5568" textAnchor="end">${v.toFixed(0)}</text>
        </g>
      ))}
      <path d={`${histPath} V${yScale(minV)} H${PAD.l} Z`} fill="url(#histGrad)" />
      <path d={bandPath} fill="#f59e0b" fillOpacity="0.08" clipPath="url(#forecastClip)" />
      <path d={histPath} fill="none" stroke="#6366f1" strokeWidth="1.8" />
      <line x1={dividerX} y1={PAD.t} x2={dividerX} y2={H - PAD.b} stroke="#2d2d45" strokeWidth="1" strokeDasharray="4,3" />
      <text x={dividerX + 3} y={PAD.t + 10} fontSize="8" fill="#4a5568">Forecast</text>
      <path d={arimaPath} fill="none" stroke="#f59e0b" strokeWidth="1.2" strokeDasharray="5,3" opacity="0.7" clipPath="url(#forecastClip)" />
      <path d={nhitsPath} fill="none" stroke="#10b981" strokeWidth="2" clipPath="url(#forecastClip)" />
      <line x1={PAD.l} y1={costY} x2={W - PAD.r} y2={costY} stroke="#ef4444" strokeWidth="1" strokeDasharray="4,3" opacity="0.7" />
      <text x={W - PAD.r} y={costY - 3} fontSize="8" fill="#ef4444" textAnchor="end">Cost ${costBasis}</text>
      <line x1={PAD.l} y1={strikeY} x2={W - PAD.r} y2={strikeY} stroke="#10b981" strokeWidth="1" strokeDasharray="4,3" opacity="0.6" />
      <text x={W - PAD.r} y={strikeY - 3} fontSize="8" fill="#10b981" textAnchor="end">Strike ${strike}</text>
      <g transform={`translate(${PAD.l}, ${H - 12})`}>
        <line x1="0" y1="0" x2="16" y2="0" stroke="#6366f1" strokeWidth="2" />
        <text x="20" y="4" fontSize="8" fill="#4a5568">Price</text>
        <line x1="55" y1="0" x2="71" y2="0" stroke="#10b981" strokeWidth="2" />
        <text x="75" y="4" fontSize="8" fill="#4a5568">N-HiTS</text>
        <line x1="120" y1="0" x2="136" y2="0" stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="4,2" />
        <text x="140" y="4" fontSize="8" fill="#4a5568">ARIMA</text>
      </g>
    </svg>
  );
}
''')
print("✓ ForecastChart.jsx")

# ── DecompositionChart.jsx ────────────────────────────────────────────────
with open(os.path.join(base, "DecompositionChart.jsx"), "w") as f:
    f.write('''export default function DecompositionChart({ nhits }) {
  const W = 480, H = 160, PAD = { t: 8, r: 8, b: 24, l: 44 };
  const IW = W - PAD.l - PAD.r, IH = H - PAD.t - PAD.b;
  const n = nhits.horizon;

  const series = [
    { key: "trend",    label: "Trend",    color: "#6366f1", data: nhits.trend },
    { key: "seasonal", label: "Seasonal", color: "#f59e0b", data: nhits.seasonal },
    { key: "residual", label: "Residual", color: "#10b981", data: nhits.residual },
  ];

  const allVals = series.flatMap(s => s.data);
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const range = maxV - minV || 1;
  const xScale = i => (i / (n - 1)) * IW + PAD.l;
  const yScale = v => IH - ((v - minV) / range) * IH + PAD.t;
  const zero = yScale(0);

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ overflow: "visible" }}>
      {zero > PAD.t && zero < H - PAD.b && (
        <line x1={PAD.l} y1={zero} x2={W - PAD.r} y2={zero} stroke="#2d2d45" strokeWidth="1" />
      )}
      {[minV, (minV + maxV) / 2, maxV].map(v => (
        <g key={v}>
          <line x1={PAD.l} y1={yScale(v)} x2={W - PAD.r} y2={yScale(v)} stroke="#141420" strokeWidth="1" />
          <text x={PAD.l - 4} y={yScale(v) + 3} fontSize="7" fill="#4a5568" textAnchor="end">
            {v >= 0 ? "+" : ""}{v.toFixed(1)}
          </text>
        </g>
      ))}
      {series.map(({ key, color, data }) => {
        const path = data.map((v, i) => `${i === 0 ? "M" : "L"}${xScale(i)},${yScale(v)}`).join(" ");
        const fill = path + ` L${xScale(n - 1)},${zero} L${xScale(0)},${zero} Z`;
        return (
          <g key={key}>
            <path d={fill} fill={color} fillOpacity="0.06" />
            <path d={path} fill="none" stroke={color} strokeWidth="1.5" />
          </g>
        );
      })}
      <g transform={`translate(${PAD.l}, ${H - 10})`}>
        {series.map(({ label, color }, i) => (
          <g key={label} transform={`translate(${i * 100}, 0)`}>
            <line x1="0" y1="0" x2="14" y2="0" stroke={color} strokeWidth="2" />
            <text x="18" y="4" fontSize="8" fill="#4a5568">{label}</text>
          </g>
        ))}
      </g>
    </svg>
  );
}
''')
print("✓ DecompositionChart.jsx")

# ── DecisionBanner.jsx ────────────────────────────────────────────────────
with open(os.path.join(base, "DecisionBanner.jsx"), "w") as f:
    f.write('''function ScoreBar({ score, max, color }) {
  const pct = ((score + max) / (max * 2)) * 100;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <div style={{ width: "100px", height: "5px", background: "#1e1e2e", borderRadius: "3px", overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: "3px" }} />
      </div>
      <span style={{ fontSize: "10px", color: "#4a5568" }}>{score > 0 ? "+" : ""}{score}/{max}</span>
    </div>
  );
}

export default function DecisionBanner({ decision }) {
  const c = decision.color;
  return (
    <div style={{
      background: `${c}12`, border: `1px solid ${c}35`,
      borderRadius: "12px", padding: "18px 24px",
      display: "flex", alignItems: "flex-start", gap: "16px",
    }}>
      <span style={{ fontSize: "28px", lineHeight: 1 }}>{decision.icon}</span>
      <div style={{ flex: 1 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px", marginBottom: "6px" }}>
          <span style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "22px", fontWeight: "700", color: c }}>
            {decision.verdict}
          </span>
          <ScoreBar score={decision.score} max={decision.max_score} color={c} />
        </div>
        <p style={{ margin: 0, fontSize: "13px", color: "#94a3b8", lineHeight: 1.55 }}>{decision.rationale}</p>
      </div>
    </div>
  );
}
''')
print("✓ DecisionBanner.jsx")

# ── GreeksPanel.jsx ───────────────────────────────────────────────────────
with open(os.path.join(base, "GreeksPanel.jsx"), "w") as f:
    f.write('''export default function GreeksPanel({ greeks, strike, premium, annYield, iv, hv, costBasis, dte, breakeven, totalPnl, pnlPct, shares }) {
  return (
    <div style={{ background: "#0f0f1a", border: "1px solid #1a1a28", borderRadius: "12px", padding: "20px" }}>
      <div style={{ fontSize: "10px", letterSpacing: "0.15em", textTransform: "uppercase", color: "#4a5568", marginBottom: "14px", borderBottom: "1px solid #1a1a28", paddingBottom: "8px" }}>
        Black-Scholes Greeks · ${strike} Strike · {dte} DTE
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginBottom: "14px" }}>
        {[
          { name: "Delta", val: greeks.delta?.toFixed(3), color: "#6366f1" },
          { name: "Gamma", val: greeks.gamma?.toFixed(5), color: "#8b5cf6" },
          { name: "Theta/day", val: `$${Math.abs(greeks.theta || 0).toFixed(3)}`, color: "#10b981" },
          { name: "Vega /1%IV", val: `$${(greeks.vega || 0).toFixed(3)}`, color: "#f59e0b" },
        ].map(g => (
          <div key={g.name} style={{ background: "#090913", border: "1px solid #1a1a28", borderRadius: "8px", padding: "12px", textAlign: "center" }}>
            <div style={{ fontSize: "10px", color: g.color, fontWeight: "600", marginBottom: "3px" }}>{g.name}</div>
            <div style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "18px", fontWeight: "700" }}>{g.val}</div>
          </div>
        ))}
      </div>
      <div style={{ background: "#090913", borderRadius: "8px", padding: "14px", marginBottom: "10px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
          <div>
            <div style={{ fontSize: "9px", color: "#4a5568", textTransform: "uppercase", letterSpacing: "0.1em" }}>Premium · {shares} shares</div>
            <div style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "26px", fontWeight: "700", color: "#10b981" }}>
              ${(premium * shares).toFixed(0)}<span style={{ fontSize: "13px", color: "#4a5568" }}> (${premium.toFixed(2)}/sh)</span>
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div style={{ fontSize: "9px", color: "#4a5568", textTransform: "uppercase", letterSpacing: "0.1em" }}>Ann. Yield</div>
            <div style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "20px", fontWeight: "700", color: "#10b981" }}>{annYield.toFixed(1)}%</div>
          </div>
        </div>
        <div style={{ marginTop: "8px", paddingTop: "8px", borderTop: "1px solid #1a1a28", display: "flex", justifyContent: "space-between" }}>
          <span style={{ fontSize: "11px", color: "#4a5568" }}>IV {(iv * 100).toFixed(1)}% / HV {(hv * 100).toFixed(1)}%</span>
          <span style={{ fontSize: "11px", color: "#4a5568" }}>Breakeven: ${breakeven.toFixed(2)}</span>
        </div>
      </div>
      <div style={{ background: "#090913", borderRadius: "8px", padding: "14px", borderLeft: `3px solid ${totalPnl >= 0 ? "#10b981" : "#ef4444"}` }}>
        <div style={{ fontSize: "9px", color: "#4a5568", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "4px" }}>If Called Away at ${strike}</div>
        <div style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "20px", fontWeight: "700", color: totalPnl >= 0 ? "#10b981" : "#ef4444" }}>
          {totalPnl >= 0 ? "+" : ""}${totalPnl.toFixed(0)} ({pnlPct.toFixed(1)}%)
        </div>
        <div style={{ fontSize: "11px", color: "#4a5568", marginTop: "3px" }}>
          Cap. {strike - costBasis >= 0 ? "gain" : "loss"} ${((strike - costBasis) * shares).toFixed(0)} + premium ${(premium * shares).toFixed(0)}
        </div>
      </div>
    </div>
  );
}
''')
print("✓ GreeksPanel.jsx")

# ── TechPanel.jsx ─────────────────────────────────────────────────────────
with open(os.path.join(base, "TechPanel.jsx"), "w") as f:
    f.write('''export default function TechPanel({ dailyRsi, weeklyRsi, bollinger, sr, currentPrice, strike }) {
  const rsiColor = r => r > 70 ? "#ef4444" : r > 60 ? "#f59e0b" : r < 40 ? "#06b6d4" : "#10b981";
  const rsiLabel = r => r > 70 ? "Overbought" : r > 60 ? "Elevated" : r < 40 ? "Oversold" : "Neutral";
  const pctB = bollinger ? (currentPrice - bollinger.lower) / (bollinger.upper - bollinger.lower) : 0.5;

  return (
    <div style={{ background: "#0f0f1a", border: "1px solid #1a1a28", borderRadius: "12px", padding: "20px" }}>
      <div style={{ fontSize: "10px", letterSpacing: "0.15em", textTransform: "uppercase", color: "#4a5568", marginBottom: "14px", borderBottom: "1px solid #1a1a28", paddingBottom: "8px" }}>
        Technical Indicators
      </div>
      {[{ label: "Daily RSI (14)", val: dailyRsi }, { label: "Weekly RSI (14)", val: weeklyRsi }].map(r => (
        <div key={r.label} style={{ marginBottom: "14px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "5px" }}>
            <span style={{ fontSize: "11px", color: "#94a3b8" }}>{r.label}</span>
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <span style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "17px", fontWeight: "700", color: rsiColor(r.val) }}>{r.val?.toFixed(1)}</span>
              <span style={{ fontSize: "9px", padding: "2px 8px", borderRadius: "99px", background: `${rsiColor(r.val)}20`, color: rsiColor(r.val), border: `1px solid ${rsiColor(r.val)}40` }}>{rsiLabel(r.val)}</span>
            </div>
          </div>
          <div style={{ height: "5px", background: "#1e1e2e", borderRadius: "3px", overflow: "hidden" }}>
            <div style={{ width: `${r.val}%`, height: "100%", background: rsiColor(r.val), borderRadius: "3px" }} />
          </div>
        </div>
      ))}
      {bollinger && (
        <div style={{ background: "#090913", borderRadius: "8px", padding: "12px", marginBottom: "12px" }}>
          <div style={{ fontSize: "9px", color: "#4a5568", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "8px" }}>Bollinger Bands (20,2)</div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "#4a5568", marginBottom: "6px" }}>
            <span>Low: ${bollinger.lower.toFixed(2)}</span>
            <span>Mid: ${bollinger.mid.toFixed(2)}</span>
            <span>Up: ${bollinger.upper.toFixed(2)}</span>
          </div>
          <div style={{ height: "6px", background: "#1e1e2e", borderRadius: "3px", position: "relative" }}>
            <div style={{ position: "absolute", left: `${Math.max(0, Math.min(100, pctB * 100))}%`, top: -2, width: "10px", height: "10px", background: "#6366f1", borderRadius: "50%", transform: "translateX(-50%)" }} />
          </div>
          <div style={{ fontSize: "9px", color: "#4a5568", marginTop: "4px" }}>%B: {(pctB * 100).toFixed(0)}% · BW: {(bollinger.bandwidth * 100).toFixed(1)}%</div>
        </div>
      )}
      {sr && (
        <div>
          <div style={{ fontSize: "9px", color: "#4a5568", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "6px" }}>Key Levels</div>
          <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
            {sr.resistance.slice(0, 4).map(r => (
              <span key={r} style={{ fontSize: "10px", padding: "3px 8px", borderRadius: "5px", background: Math.abs(r - strike) < 5 ? "#10b98120" : "#1a1a28", color: Math.abs(r - strike) < 5 ? "#10b981" : "#94a3b8", border: `1px solid ${Math.abs(r - strike) < 5 ? "#10b98140" : "#2d2d45"}` }}>
                R ${r}
              </span>
            ))}
            {sr.support.slice(0, 3).map(s => (
              <span key={s} style={{ fontSize: "10px", padding: "3px 8px", borderRadius: "5px", background: "#1a1a28", color: "#4a5568", border: "1px solid #2d2d45" }}>S ${s}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
''')
print("✓ TechPanel.jsx")

# ── SignalPanel.jsx ───────────────────────────────────────────────────────
with open(os.path.join(base, "SignalPanel.jsx"), "w") as f:
    f.write('''export default function SignalPanel({ signals, score }) {
  const typeColor = { bullish: "#10b981", bearish: "#ef4444", neutral: "#f59e0b" };
  const typeLabel = { bullish: "FOR", bearish: "AGAINST", neutral: "NEUTRAL" };
  return (
    <div style={{ background: "#0f0f1a", border: "1px solid #1a1a28", borderRadius: "12px", padding: "20px" }}>
      <div style={{ fontSize: "10px", letterSpacing: "0.15em", textTransform: "uppercase", color: "#4a5568", marginBottom: "14px", borderBottom: "1px solid #1a1a28", paddingBottom: "8px" }}>
        Signal Breakdown · Net Score {score > 0 ? "+" : ""}{score}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        {signals.map((s, i) => (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: "10px", background: "#090913", borderRadius: "8px", padding: "10px 12px" }}>
            <div style={{ width: "6px", height: "6px", borderRadius: "50%", background: typeColor[s.type], flexShrink: 0 }} />
            <span style={{ flex: 1, fontSize: "12px", color: "#94a3b8" }}>{s.label}</span>
            <span style={{ fontSize: "9px", padding: "2px 7px", borderRadius: "99px", background: `${typeColor[s.type]}15`, color: typeColor[s.type], border: `1px solid ${typeColor[s.type]}30`, flexShrink: 0 }}>
              {typeLabel[s.type]} {s.weight > 0 ? "+" : ""}{s.weight}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
''')
print("✓ SignalPanel.jsx")

# ── StrikeLadder.jsx ──────────────────────────────────────────────────────
with open(os.path.join(base, "StrikeLadder.jsx"), "w") as f:
    f.write('''export default function StrikeLadder({ ladder, recommended, costBasis, currentPrice }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "11px" }}>
        <thead>
          <tr style={{ color: "#4a5568", borderBottom: "1px solid #1e1e2e" }}>
            {["Strike", "Delta", "Premium", "Yield/mo", "Ann.Yield", "vs Cost"].map(h => (
              <th key={h} style={{ padding: "7px 10px", textAlign: "right", fontWeight: "400", fontSize: "9px", letterSpacing: "0.1em", textTransform: "uppercase" }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ladder.filter(r => r.strike >= currentPrice * 0.96 && r.strike <= currentPrice * 1.28).map(r => {
            const isRec = r.strike === recommended;
            const yield1m = (r.price / currentPrice) * 100;
            const annYld = yield1m * 12;
            return (
              <tr key={r.strike} style={{ borderBottom: "1px solid #0d0d18", background: isRec ? "#6366f110" : "transparent", color: isRec ? "#e2e8f0" : "#94a3b8" }}>
                <td style={{ padding: "7px 10px", textAlign: "right", fontFamily: "Space Grotesk, sans-serif", fontWeight: isRec ? "600" : "400", color: isRec ? "#6366f1" : "inherit" }}>
                  ${r.strike}{isRec ? " ★" : ""}
                </td>
                <td style={{ padding: "7px 10px", textAlign: "right" }}>{r.delta?.toFixed(3)}</td>
                <td style={{ padding: "7px 10px", textAlign: "right" }}>${r.price?.toFixed(2)}</td>
                <td style={{ padding: "7px 10px", textAlign: "right" }}>{yield1m.toFixed(2)}%</td>
                <td style={{ padding: "7px 10px", textAlign: "right", color: annYld > 15 ? "#10b981" : annYld > 8 ? "#f59e0b" : "inherit" }}>{annYld.toFixed(1)}%</td>
                <td style={{ padding: "7px 10px", textAlign: "right", color: r.strike >= costBasis ? "#10b981" : "#ef4444" }}>
                  {r.strike >= costBasis ? `+$${(r.strike - costBasis).toFixed(0)}` : `-$${(costBasis - r.strike).toFixed(0)}`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
''')
print("✓ StrikeLadder.jsx")

print("\n✅ Todos os componentes criados em:")
print(f"   {base}")
print("\nAgora no terminal:")
print("   cd frontend")
print("   npm run dev")
