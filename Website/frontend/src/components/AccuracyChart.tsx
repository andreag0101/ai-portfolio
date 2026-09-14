import { useMemo, useState } from "react";

interface AccuracyChartProps {
  k: number[];
  pca: number[];
  lda: number[];
}

const WIDTH = 560;
const HEIGHT = 220;
const PAD = { top: 16, right: 34, bottom: 24, left: 34 };

export default function AccuracyChart({ k, pca, lda }: AccuracyChartProps) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const plotW = WIDTH - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const xMin = k[0];
  const xMax = k[k.length - 1];

  const xScale = (v: number) => PAD.left + ((v - xMin) / (xMax - xMin)) * plotW;
  const yScale = (v: number) => PAD.top + (1 - v / 100) * plotH;

  const pcaPath = useMemo(() => k.map((kv, i) => `${xScale(kv)},${yScale(pca[i])}`).join(" L "), [k, pca]);
  const ldaPath = useMemo(() => k.map((kv, i) => `${xScale(kv)},${yScale(lda[i])}`).join(" L "), [k, lda]);

  function handleMove(e: React.MouseEvent<SVGRectElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * WIDTH;
    const frac = (px - PAD.left) / plotW;
    const idx = Math.round(frac * (k.length - 1));
    setHoverIdx(Math.max(0, Math.min(k.length - 1, idx)));
  }

  return (
    <div className="viz-root">
      <style>{`
        .viz-root {
          color-scheme: light;
          --surface-1: transparent;
          --text-secondary: #52514e;
          --muted: #898781;
          --grid: #e1e0d9;
          --series-pca: #2a78d6;
          --series-lda: #eb6834;
        }
        @media (prefers-color-scheme: dark) {
          :root:not([data-theme="light"]) .viz-root {
            color-scheme: dark;
            --text-secondary: #c3c2b7;
            --muted: #898781;
            --grid: #2c2c2a;
            --series-pca: #3987e5;
            --series-lda: #d95926;
          }
        }
        :root[data-theme="dark"] .viz-root {
          color-scheme: dark;
          --text-secondary: #c3c2b7;
          --muted: #898781;
          --grid: #2c2c2a;
          --series-pca: #3987e5;
          --series-lda: #d95926;
        }
      `}</style>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full select-none">
        {[0, 25, 50, 75, 100].map((tick) => (
          <g key={tick}>
            <line x1={PAD.left} x2={WIDTH - PAD.right} y1={yScale(tick)} y2={yScale(tick)} stroke="var(--grid)" strokeWidth={1} />
            <text x={PAD.left - 8} y={yScale(tick)} textAnchor="end" dominantBaseline="middle" fontSize={10} fill="var(--muted)">
              {tick}
            </text>
          </g>
        ))}
        {[1, 10, 20, 29].map((kv) => (
          <text key={kv} x={xScale(kv)} y={HEIGHT - 6} textAnchor="middle" fontSize={10} fill="var(--muted)">
            {kv}
          </text>
        ))}
        <text x={WIDTH / 2} y={HEIGHT - 6} textAnchor="middle" fontSize={10} fill="var(--muted)" opacity={0} />

        <path d={`M ${pcaPath}`} fill="none" stroke="var(--series-pca)" strokeWidth={2} strokeLinecap="round" />
        <path d={`M ${ldaPath}`} fill="none" stroke="var(--series-lda)" strokeWidth={2} strokeLinecap="round" />

        {(() => {
          const pcaY = yScale(pca[pca.length - 1]);
          const ldaY = yScale(lda[lda.length - 1]);
          const collide = Math.abs(pcaY - ldaY) < 12;
          const pcaLabelY = collide ? pcaY - 7 : pcaY;
          const ldaLabelY = collide ? ldaY + 7 : ldaY;
          return (
            <>
              <text x={xScale(k[k.length - 1]) + 4} y={pcaLabelY} fontSize={11} fill="var(--series-pca)" dominantBaseline="middle">
                PCA
              </text>
              <text x={xScale(k[k.length - 1]) + 4} y={ldaLabelY} fontSize={11} fill="var(--series-lda)" dominantBaseline="middle">
                LDA
              </text>
            </>
          );
        })()}

        {hoverIdx !== null && (
          <>
            <line x1={xScale(k[hoverIdx])} x2={xScale(k[hoverIdx])} y1={PAD.top} y2={HEIGHT - PAD.bottom} stroke="var(--muted)" strokeWidth={1} strokeDasharray="3 3" />
            <circle cx={xScale(k[hoverIdx])} cy={yScale(pca[hoverIdx])} r={3.5} fill="var(--series-pca)" />
            <circle cx={xScale(k[hoverIdx])} cy={yScale(lda[hoverIdx])} r={3.5} fill="var(--series-lda)" />
          </>
        )}

        <rect
          x={PAD.left}
          y={PAD.top}
          width={plotW}
          height={plotH}
          fill="transparent"
          onMouseMove={handleMove}
          onMouseLeave={() => setHoverIdx(null)}
        />
      </svg>
      <div className="flex items-center justify-between text-xs" style={{ color: "var(--text-secondary)" }}>
        <span>K (subspace dimensions)</span>
        {hoverIdx !== null && (
          <span>
            K={k[hoverIdx]}: PCA {pca[hoverIdx].toFixed(1)}%, LDA {lda[hoverIdx].toFixed(1)}%
          </span>
        )}
      </div>
    </div>
  );
}
