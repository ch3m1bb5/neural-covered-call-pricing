export default function GreeksPanel({ greeks, strike, premium, annYield, iv, hv, costBasis, dte, breakeven, totalPnl, pnlPct, shares }) {
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
