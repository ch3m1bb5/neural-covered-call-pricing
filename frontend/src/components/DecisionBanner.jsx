function ScoreBar({ score, max, color }) {
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
