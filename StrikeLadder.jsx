export default function StrikeLadder({ ladder, recommended, costBasis, currentPrice }) {
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
                  ${r.strike}{isRec ? " [REC]" : ""}
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