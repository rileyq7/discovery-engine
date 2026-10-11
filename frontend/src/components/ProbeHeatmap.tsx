import { useMemo, useState } from "react";
import { probeCounts, type ProbeRow } from "../lib/api";
import { fmtNum } from "../lib/utils";
import { Tabs } from "./ui";

type Mode = "delta" | "r2";

// Diverging ramp: red for below-control, neutral at 0, blue -> gold as encoding strengthens.
const POS = ["#262b38", "#2f4a8a", "#3f6fd8", "#7c9cff", "#c5b8ff", "#f5d76e"];
const NEG = ["#262b38", "#6b2a33", "#c0392b"];

function lerp(a: string, b: string, t: number) {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * t)).join(",")})`;
}
function ramp(stops: string[], t: number) {
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  return lerp(stops[i], stops[i + 1], x - i);
}
export function heatColor(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "transparent";
  return v >= 0 ? ramp(POS, v) : ramp(NEG, Math.min(1, -v / 0.5));
}

export function ProbeHeatmap({ rows, target, margin = 0.1, height = 260, compact = false }: { rows: ProbeRow[]; target: string; margin?: number; height?: number; compact?: boolean }) {
  const [mode, setMode] = useState<Mode>("delta");
  const [hover, setHover] = useState<{ step: number; layer: number; r2: number; control: number | null } | null>(null);

  const { steps, layers, cell, control } = useMemo(() => {
    const rs = rows.filter((r) => r.target === target);
    const steps = [...new Set(rs.map((r) => r.checkpoint_step))].sort((a, b) => a - b);
    const layers = [...new Set(rs.map((r) => r.layer))].sort((a, b) => b - a); // top = deepest
    const cell = new Map(rs.map((r) => [`${r.checkpoint_step}:${r.layer}`, r]));
    const control = new Map(rs.map((r) => [r.layer, r.control_r2]));
    return { steps, layers, cell, control };
  }, [rows, target]);

  if (!steps.length) return <div className="text-sm text-muted">No probe results for {target} yet.</div>;

  const W = 760, left = 46, right = 64, top = 8, bottom = 30;
  const H = height;
  const cw = (W - left - right) / steps.length;
  const ch = (H - top - bottom) / layers.length;
  const tickEvery = Math.max(1, Math.ceil(steps.length / 10));
  const value = (r: ProbeRow) => (mode === "delta" ? r.r2 - (r.control_r2 ?? 0) : r.r2);

  return (
    <div>
      {!compact && (
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <Tabs value={mode} onChange={setMode} options={[{ value: "delta", label: "R² − control" }, { value: "r2", label: "raw R²" }]} />
          <div className="text-xs text-muted tabular-nums h-4">
            {hover
              ? `step ${hover.step} · layer ${hover.layer} · R² ${fmtNum(hover.r2)} · control ${fmtNum(hover.control)} · Δ ${fmtNum(hover.r2 - (hover.control ?? 0))}`
              : `outlined cells beat the best random-model control by ≥ max(${margin}, control spread)`}
          </div>
        </div>
      )}
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`Probe R² heatmap for ${target}`} onMouseLeave={() => setHover(null)}>
        {layers.map((l, yi) => (
          <text key={l} x={left - 8} y={top + yi * ch + ch / 2 + 4} textAnchor="end" fontSize="11" fill="var(--muted)">
            {l === 0 ? "emb" : `L${l}`}
          </text>
        ))}
        {steps.map((s, xi) =>
          layers.map((l, yi) => {
            const r = cell.get(`${s}:${l}`);
            if (!r) return null;
            const v = value(r);
            const ok = probeCounts(r, margin);
            return (
              <rect
                key={`${s}:${l}`}
                x={left + xi * cw + 0.5}
                y={top + yi * ch + 0.5}
                width={Math.max(cw - 1, 0.5)}
                height={ch - 1}
                fill={heatColor(v)}
                stroke={ok ? "var(--fg)" : "none"}
                strokeOpacity={0.55}
                strokeWidth={ok ? 1 : 0}
                rx={1.5}
                onMouseEnter={() => setHover({ step: s, layer: l, r2: r.r2, control: r.control_r2 })}
              />
            );
          }),
        )}
        {/* control overlay: the random-model probe, one column per layer */}
        <text x={W - right + 30} y={H - bottom + 18} textAnchor="middle" fontSize="10" fill="var(--muted)">random</text>
        {layers.map((l, yi) => {
          const c = control.get(l);
          return (
            <g key={`c${l}`}>
              <rect x={W - right + 12} y={top + yi * ch + 0.5} width={36} height={ch - 1} rx={1.5}
                fill={c === null || c === undefined ? "transparent" : heatColor(mode === "delta" ? 0 : c)} stroke="var(--border)" />
              <text x={W - right + 30} y={top + yi * ch + ch / 2 + 4} textAnchor="middle" fontSize="10" fill="var(--fg)">
                {c === null || c === undefined ? "–" : c.toFixed(2)}
              </text>
            </g>
          );
        })}
        {steps.map((s, xi) =>
          xi % tickEvery === 0 || xi === steps.length - 1 ? (
            <text key={s} x={left + xi * cw + cw / 2} y={H - bottom + 18} textAnchor="middle" fontSize="10" fill="var(--muted)">
              {s >= 1000 ? `${(s / 1000).toFixed(s % 1000 ? 1 : 0)}k` : s}
            </text>
          ) : null,
        )}
      </svg>
      {!compact && (
        <div className="mt-1 flex items-center gap-2 text-[11px] text-muted">
          <span>{mode === "delta" ? "−0.5" : ""}</span>
          <div className="h-2 w-40 rounded" style={{ background: `linear-gradient(90deg, ${mode === "delta" ? NEG.slice().reverse().join(",") + "," : ""}${POS.join(",")})` }} />
          <span>1.0</span>
          <span className="ml-3">x: training step · y: layer</span>
        </div>
      )}
    </div>
  );
}
