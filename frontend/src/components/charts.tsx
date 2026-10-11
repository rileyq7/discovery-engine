import { useMemo } from "react";
import {
  CartesianGrid, ComposedChart, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { probeCounts, type Metric, type ProbeRow } from "../lib/api";
import { fmtNum } from "../lib/utils";

const C = {
  train: "#7c9cff", val: "#f5d76e", base: "#8b929c",
  force: "#51cf66", accel_step: "#20c997", distance: "#74c0fc", velocity: "#ff922b", energy: "#da77f2",
};
const axis = { stroke: "var(--muted)", fontSize: 11 };
const tooltipStyle = { background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12, color: "var(--fg)" };

function bySteps(metrics: Metric[], names: string[]) {
  const m = new Map<number, Record<string, number>>();
  for (const r of metrics) {
    if (!names.includes(r.name) || !(r.value > 0 || r.name === "val_mse_ratio")) continue;
    const row = m.get(r.step) ?? { step: r.step };
    row[r.name] = r.value;
    m.set(r.step, row);
  }
  return [...m.values()].sort((a, b) => a.step - b.step);
}

export function LossChart({ metrics, height = 220, mini = false }: { metrics: Metric[]; height?: number; mini?: boolean }) {
  const data = useMemo(() => bySteps(metrics, ["train_loss", "val_mse", "baseline_cv_mse"]), [metrics]);
  if (!data.length) return <div className="flex items-center justify-center text-xs text-muted" style={{ height }}>waiting for metrics…</div>;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: mini ? 6 : 16, bottom: 0, left: mini ? -18 : 0 }}>
        {!mini && <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />}
        <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} {...axis} tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)} />
        <YAxis yAxisId="loss" scale="log" domain={["auto", "auto"]} {...axis} tickFormatter={(v) => fmtNum(v, 2)} width={mini ? 52 : 64} hide={mini} />
        <YAxis yAxisId="mse" orientation="right" scale="log" domain={["auto", "auto"]} {...axis} tickFormatter={(v) => fmtNum(v, 1)} width={64} hide={mini} />
        <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmtNum(Number(v), 4)} labelFormatter={(l) => `step ${l}`} />
        {!mini && <Legend wrapperStyle={{ fontSize: 12 }} />}
        <Line yAxisId="loss" dataKey="train_loss" name="train loss (norm.)" stroke={C.train} dot={false} strokeWidth={1.5} connectNulls isAnimationActive={false} />
        <Line yAxisId="mse" dataKey="val_mse" name="val MSE" stroke={C.val} dot={!mini} strokeWidth={2} connectNulls isAnimationActive={false} />
        <Line yAxisId="mse" dataKey="baseline_cv_mse" name="const-velocity MSE" stroke={C.base} strokeDasharray="5 4" dot={false} connectNulls isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Loss and probe R² on one shared step axis: does understanding come before or after accuracy? */
export function FormationChart({ metrics, probes, margin = 0.1, height = 300 }: { metrics: Metric[]; probes: ProbeRow[]; margin?: number; height?: number }) {
  const { data, targets, formed, accurate } = useMemo(() => {
    const steps = [...new Set(probes.map((p) => p.checkpoint_step))].sort((a, b) => a - b);
    const targets = [...new Set(probes.map((p) => p.target))];
    const final = steps[steps.length - 1];
    const bestLayer: Record<string, number> = {};
    for (const t of targets) {
      const fin = probes.filter((p) => p.target === t && p.checkpoint_step === final);
      fin.sort((a, b) => b.r2 - (b.control_r2 ?? 0) - (a.r2 - (a.control_r2 ?? 0)));
      if (fin[0]) bestLayer[t] = fin[0].layer;
    }
    const ratio = new Map(metrics.filter((m) => m.name === "val_mse_ratio").map((m) => [m.step, m.value]));
    const rows = steps.map((s) => {
      const row: Record<string, number> = { step: s };
      const r = ratio.get(s);
      if (r !== undefined) row.ratio = r;
      for (const t of targets) {
        const p = probes.find((q) => q.target === t && q.checkpoint_step === s && q.layer === bestLayer[t]);
        if (p) row[t] = p.r2 - (p.control_r2 ?? 0);
      }
      return row;
    });
    // formation: first step after which the best layer keeps beating its controls (max(margin, control spread))
    const formed: Record<string, number | null> = {};
    for (const t of targets) {
      formed[t] = null;
      const series = steps.map((s) => probes.find((q) => q.target === t && q.checkpoint_step === s && q.layer === bestLayer[t]));
      for (let i = 0; i < series.length; i++) {
        if (series.slice(i).every((p) => p && probeCounts(p, margin))) { formed[t] = steps[i]; break; }
      }
    }
    const acc = rows.find((r) => r.ratio !== undefined && r.ratio < 0.1);
    return { data: rows, targets, formed, accurate: acc ? acc.step : null, bestLayer };
  }, [metrics, probes, margin]);

  if (!data.length) return <div className="text-sm text-muted">Formation chart needs probe results on several checkpoints.</div>;
  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 10, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} {...axis} tickFormatter={(v) => (v >= 1000 ? `${v / 1000}k` : v)} />
          <YAxis yAxisId="ratio" scale="log" domain={["auto", "auto"]} {...axis} width={56} tickFormatter={(v) => fmtNum(v, 2)}
            label={{ value: "MSE / const-velocity", angle: -90, position: "insideLeft", fill: "var(--muted)", fontSize: 11, dy: 60 }} />
          <YAxis yAxisId="r2" orientation="right" domain={[-0.2, 1]} {...axis} width={44}
            label={{ value: "probe R² − control", angle: 90, position: "insideRight", fill: "var(--muted)", fontSize: 11, dy: 60 }} />
          <Tooltip contentStyle={tooltipStyle} formatter={(v) => fmtNum(Number(v), 3)} labelFormatter={(l) => `step ${l}`} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ReferenceLine yAxisId="ratio" y={1} stroke="var(--muted)" strokeDasharray="2 4" />
          <ReferenceLine yAxisId="r2" y={margin} stroke="var(--muted)" strokeDasharray="2 4" />
          {accurate !== null && <ReferenceLine yAxisId="ratio" x={accurate} stroke={C.val} strokeDasharray="4 3" label={{ value: "accurate", fill: C.val, fontSize: 10, position: "top" }} />}
          {formed.force != null && <ReferenceLine yAxisId="r2" x={formed.force} stroke={C.force} strokeDasharray="4 3" label={{ value: "force formed", fill: C.force, fontSize: 10, position: "insideTopRight" }} />}
          {formed.accel_step != null && <ReferenceLine yAxisId="r2" x={formed.accel_step} stroke={C.accel_step} strokeDasharray="2 3" label={{ value: "accel formed", fill: C.accel_step, fontSize: 10, position: "insideBottomRight" }} />}
          <Line yAxisId="ratio" dataKey="ratio" name="val MSE ratio" stroke={C.val} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
          {targets.map((t) => (
            <Line key={t} yAxisId="r2" dataKey={t} name={`${t} R² − ctrl`} stroke={C[t as keyof typeof C] ?? "#aaa"} strokeWidth={1.75} dot={false} isAnimationActive={false} connectNulls />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
      <div className="mt-1 text-xs text-muted">
        Accuracy (ratio &lt; 0.1) at step {accurate ?? "–"} ·{" "}
        {targets.map((t) => `${t} formed at ${formed[t] ?? "never"}`).join(" · ")}
      </div>
    </div>
  );
}
