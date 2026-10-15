export default function SignalPanel({ signals, score }) {
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
