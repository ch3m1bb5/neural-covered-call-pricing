import { useMemo } from "react";

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
