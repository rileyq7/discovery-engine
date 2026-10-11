import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, type DataConfig, type DatasetStats, type Hist, type OrbitSample } from "../lib/api";
import { fmtNum, fmtTime } from "../lib/utils";
import { PageHeader } from "../App";
import { Button, Card, CardContent, CardHeader, CardTitle, Dialog, Empty, ErrorText, Input, Label, Select } from "../components/ui";

const A = "#7c9cff", B = "#f5d76e";
const BODY = ["#f5b942", "#7c9cff", "#51cf66"];

function OrbitPlot({ s }: { s: OrbitSample }) {
  const all = s.bodies.flat();
  const ext = Math.max(...all.map(([x, y]) => Math.max(Math.abs(x), Math.abs(y)))) * 1.1 || 1;
  const path = (pts: [number, number][]) => pts.map(([x, y], i) => `${i ? "L" : "M"}${(x / ext) * 48 + 50},${50 - (y / ext) * 48}`).join("");
  return (
    <div className="rounded-md border border-border p-2">
      <svg viewBox="0 0 100 100" className="aspect-square w-full">
        <line x1="50" y1="2" x2="50" y2="98" stroke="var(--border)" strokeWidth="0.3" />
        <line x1="2" y1="50" x2="98" y2="50" stroke="var(--border)" strokeWidth="0.3" />
        {s.bodies.map((b, i) => (
          <g key={i}>
            <path d={path(b)} fill="none" stroke={BODY[i]} strokeWidth="0.8" strokeOpacity="0.85" />
            <circle cx={(b[0][0] / ext) * 48 + 50} cy={50 - (b[0][1] / ext) * 48} r={1.6} fill={BODY[i]} />
          </g>
        ))}
      </svg>
      <div className="text-[10px] text-muted tabular-nums">m={s.masses.map((m) => m.toFixed(2)).join("/")} · e={s.e.toFixed(2)} · a={s.a.toFixed(2)}</div>
    </div>
  );
}

function histRows(a: Hist, b?: Hist) {
  if (!b) return a.counts.map((c, i) => ({ x: (a.edges[i] + a.edges[i + 1]) / 2, A: c / Math.max(1, a.counts.reduce((s, v) => s + v, 0)) }));
  // Re-bin both onto a shared grid so they can be compared.
  const lo = Math.min(a.edges[0], b.edges[0]), hi = Math.max(a.edges[a.edges.length - 1], b.edges[b.edges.length - 1]);
  const n = 24, w = (hi - lo) / n || 1;
  const rebin = (h: Hist) => {
    const out = new Array(n).fill(0);
    const tot = h.counts.reduce((s, v) => s + v, 0) || 1;
    h.counts.forEach((c, i) => { const mid = (h.edges[i] + h.edges[i + 1]) / 2; out[Math.min(n - 1, Math.max(0, Math.floor((mid - lo) / w)))] += c / tot; });
    return out;
  };
  const ra = rebin(a), rb = rebin(b);
  return ra.map((v, i) => ({ x: lo + (i + 0.5) * w, A: v, B: rb[i] }));
}

function HistChart({ title, a, b }: { title: string; a: Hist; b?: Hist }) {
  const data = histRows(a, b);
  return (
    <Card>
      <CardHeader><CardTitle>{title}</CardTitle><span className="text-[11px] text-muted tabular-nums">μ={fmtNum(a.mean)}{b ? ` vs ${fmtNum(b.mean)}` : ""}</span></CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={150}>
          <BarChart data={data} barGap={0} barCategoryGap={1} margin={{ left: -24, right: 4 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="x" tickFormatter={(v) => fmtNum(v, 2)} stroke="var(--muted)" fontSize={10} />
            <YAxis stroke="var(--muted)" fontSize={10} tickFormatter={(v) => `${Math.round(v * 100)}%`} />
            <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)", fontSize: 12 }} formatter={(v) => `${(Number(v) * 100).toFixed(1)}%`} labelFormatter={(l) => fmtNum(Number(l), 3)} />
            {b && <Legend wrapperStyle={{ fontSize: 11 }} />}
            <Bar dataKey="A" name="dataset A" fill={A} isAnimationActive={false} />
            {b && <Bar dataKey="B" name="dataset B" fill={B} isAnimationActive={false} />}
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

function GenerateForm({ onDone }: { onDone: () => void }) {
  const [c, setC] = useState<DataConfig>({ source: "rebound", n_bodies: 2, n_systems: 2000, steps_per_orbit: 64, mass_ratio_range: [0.1, 1.0], eccentricity_range: [0, 0.5], data_seed: 0 });
  const [nSteps, setNSteps] = useState(129);
  const gen = useMutation({ mutationFn: () => api.generateDataset(c, nSteps), onSuccess: onDone });
  const num = (label: string, v: number, set: (n: number) => void, step = "1") => (
    <div><Label>{label}</Label><Input type="number" step={step} value={v} onChange={(e) => set(Number(e.target.value))} /></div>
  );
  return (
    <form className="grid grid-cols-2 gap-3" onSubmit={(e) => { e.preventDefault(); gen.mutate(); }}>
      <div><Label>Bodies</Label><Select className="w-full" value={c.n_bodies} onChange={(e) => setC({ ...c, n_bodies: Number(e.target.value) as 2 | 3 })}><option value={2}>2</option><option value={3}>3 (hierarchical)</option></Select></div>
      {num("Systems", c.n_systems, (v) => setC({ ...c, n_systems: v }))}
      {num("Snapshots per orbit", c.steps_per_orbit, (v) => setC({ ...c, steps_per_orbit: v }))}
      {num("Steps per system", nSteps, setNSteps)}
      {num("Mass ratio min", c.mass_ratio_range[0], (v) => setC({ ...c, mass_ratio_range: [v, c.mass_ratio_range[1]] }), "0.05")}
      {num("Mass ratio max", c.mass_ratio_range[1], (v) => setC({ ...c, mass_ratio_range: [c.mass_ratio_range[0], v] }), "0.05")}
      {num("Eccentricity min", c.eccentricity_range![0], (v) => setC({ ...c, eccentricity_range: [v, c.eccentricity_range![1]] }), "0.05")}
      {num("Eccentricity max", c.eccentricity_range![1], (v) => setC({ ...c, eccentricity_range: [c.eccentricity_range![0], v] }), "0.05")}
      {num("Data seed", c.data_seed ?? 0, (v) => setC({ ...c, data_seed: v }))}
      <div className="col-span-2"><ErrorText error={gen.error} /><Button type="submit" disabled={gen.isPending}>Generate</Button></div>
    </form>
  );
}

export default function Data() {
  const qc = useQueryClient();
  const ds = useQuery({ queryKey: ["datasets"], queryFn: api.datasets, refetchInterval: (q) => (q.state.data?.some((d) => d.status === "generating") ? 3000 : false) });
  const ready = (ds.data ?? []).filter((d) => d.status !== "generating");
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [seed, setSeed] = useState(0);
  const [genOpen, setGenOpen] = useState(false);
  useEffect(() => { if (!a && ready[0]) setA(ready[0].id); }, [ready.length]);
  const sa = useQuery({ queryKey: ["dsstats", a], queryFn: () => api.datasetStats(a), enabled: !!a });
  const sb = useQuery({ queryKey: ["dsstats", b], queryFn: () => api.datasetStats(b), enabled: !!b });
  const sample = useQuery({ queryKey: ["dssample", a, seed], queryFn: () => api.datasetSample(a, 8, seed), enabled: !!a });
  const meta = useMemo(() => new Map((ds.data ?? []).map((d) => [d.id, d])), [ds.data]);
  const label = (id: string) => {
    const d = meta.get(id);
    return d?.config ? `${id} · ${d.config.n_bodies}-body · ${d.config.n_systems} sys · q∈[${d.config.mass_ratio_range.join(", ")}]` : id;
  };
  const stats: [string, keyof DatasetStats][] = [["Mass ratio m₂/m₁", "mass_ratio"], ["Period", "period"], ["Eccentricity", "eccentricity"], ["Semi-major axis", "semi_major_axis"]];

  return (
    <div>
      <PageHeader title="Data" sub="REBOUND datasets: regenerated from config + seed, cached locally (gitignored).">
        <Button onClick={() => setGenOpen(true)}>Generate dataset</Button>
      </PageHeader>
      {!ready.length ? <Empty>{ds.isLoading ? "Loading…" : "No datasets on this machine yet. Running an experiment generates one, or generate one here."}</Empty> : (
        <>
          <div className="mb-4 flex flex-wrap items-end gap-3">
            <div><Label>Dataset A</Label><Select className="block min-w-80" value={a} onChange={(e) => setA(e.target.value)}>{ready.map((d) => <option key={d.id} value={d.id}>{label(d.id)}</option>)}</Select></div>
            <div><Label>Compare with (B)</Label><Select className="block min-w-80" value={b} onChange={(e) => setB(e.target.value)}><option value="">— none —</option>{ready.filter((d) => d.id !== a).map((d) => <option key={d.id} value={d.id}>{label(d.id)}</option>)}</Select></div>
            {(ds.data ?? []).filter((d) => d.status === "generating").map((d) => <span key={d.id} className="text-xs text-warn">generating {d.id}…</span>)}
          </div>
          {meta.get(a) && (
            <p className="mb-3 text-xs text-muted">
              {sa.data && `${sa.data.n_systems} systems × ${sa.data.n_steps} steps · `}generated {fmtTime(meta.get(a)!.created_at)} in {meta.get(a)!.generated_seconds}s · used by {meta.get(a)!.used_by_runs?.length ?? 0} run(s)
            </p>
          )}
          <Card className="mb-4">
            <CardHeader><CardTitle>Sample orbits (centre-of-mass frame)</CardTitle><Button size="sm" variant="outline" onClick={() => setSeed((s) => s + 1)}>Shuffle</Button></CardHeader>
            <CardContent className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-8">
              {(sample.data ?? []).map((s) => <OrbitPlot key={s.index} s={s} />)}
            </CardContent>
          </Card>
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {sa.data && stats.map(([t, k]) => <HistChart key={k} title={t} a={sa.data![k] as Hist} b={b && sb.data ? (sb.data[k] as Hist) : undefined} />)}
          </div>
          <ErrorText error={sa.error || sample.error} />
        </>
      )}
      <Dialog open={genOpen} onClose={() => setGenOpen(false)} title="Generate a REBOUND dataset">
        <GenerateForm onDone={() => { setGenOpen(false); qc.invalidateQueries({ queryKey: ["datasets"] }); }} />
      </Dialog>
    </div>
  );
}
