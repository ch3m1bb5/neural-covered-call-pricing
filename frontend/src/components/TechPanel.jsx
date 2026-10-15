export default function TechPanel({ dailyRsi, weeklyRsi, bollinger, sr, currentPrice, strike }) {
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
