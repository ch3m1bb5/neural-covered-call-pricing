export default function DecompositionChart({ nhits }) {
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
