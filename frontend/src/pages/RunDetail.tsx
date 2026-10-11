import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, type RunDetail } from "../lib/api";
import { fmtHours, fmtNum, fmtTime, shortId } from "../lib/utils";
import { PageHeader } from "../App";
import { FormationChart, LossChart } from "../components/charts";
import { ProbeHeatmap } from "../components/ProbeHeatmap";
import { StateBadge, VerdictBadge } from "../components/badges";
import { Badge, Button, Card, CardContent, CardHeader, CardTitle, Empty, ErrorText, Input, Stat, Table, Tabs, Td, Th } from "../components/ui";

const TARGET_LABEL: Record<string, string> = {
  force: "force (physical)", accel_step: "accel / snapshot", distance: "distance r", velocity: "rel. velocity", energy: "energy",
};

function RunList() {
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.runs });
  return (
    <div>
      <PageHeader title="Runs" sub="Pick a run to see its curves, probes, equations and logs." />
      <Card>
        {runs.data?.length ? (
          <Table>
            <thead><tr><Th>Run</Th><Th>Experiment</Th><Th>Seed</Th><Th>Status</Th><Th>MSE / const-vel</Th><Th>Best force R² (ctrl)</Th><Th>Best accel R² (ctrl)</Th><Th>Compute</Th><Th>Started</Th></tr></thead>
            <tbody>
              {runs.data.map((r) => {
                const f = r.summary?.probes?.targets?.force, a = r.summary?.probes?.targets?.accel_step;
                return (
                  <tr key={r.id} className="cursor-pointer hover:bg-panel-2/60" onClick={() => (window.location.hash = `#/runs/${r.id}`)}>
                    <Td className="font-mono text-xs">{shortId(r.id)} {r.bad && <Badge tone="bad">bad</Badge>}</Td>
                    <Td className="font-mono text-xs text-muted">{shortId(r.experiment_id)}</Td>
                    <Td>{r.seed}</Td>
                    <Td><StateBadge state={r.status} /></Td>
                    <Td className="tabular-nums">{fmtNum(r.summary?.val_mse_ratio)}</Td>
                    <Td className="tabular-nums">{f ? `${fmtNum(f.r2, 2)} (${fmtNum(f.control_r2, 2)})` : "–"}</Td>
                    <Td className="tabular-nums">{a ? `${fmtNum(a.r2, 2)} (${fmtNum(a.control_r2, 2)})` : "–"}</Td>
                    <Td className="tabular-nums text-xs">{fmtHours(r.compute_hours)}</Td>
                    <Td className="text-xs text-muted">{fmtTime(r.started_at)}</Td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        ) : <div className="p-4"><Empty>{runs.isLoading ? "Loading…" : "No runs yet."}</Empty></div>}
      </Card>
    </div>
  );
}

function Equations({ run }: { run: RunDetail }) {
  const eqs = run.equations;
  if (!eqs) return <p className="text-sm text-muted">Symbolic regression was not enabled for this run (set probes.symbolic).</p>;
  return (
    <div className="grid gap-3 md:grid-cols-2">
      {Object.entries(eqs).map(([t, e]) => (
        <div key={t} className="rounded-md border border-border p-3">
          <div className="mb-1 flex items-center gap-2 text-xs">
            <span className="font-semibold">|{TARGET_LABEL[t] ?? t}|</span>
            <span className="text-muted">probe at layer {e.layer} {e.probe_r2 !== undefined && `(R² ${fmtNum(e.probe_r2, 2)})`}</span>
            {e.law_recovered !== undefined && <Badge tone={e.law_recovered ? "good" : "warn"}>{e.law_recovered ? "law recovered" : "law not recovered"}</Badge>}
          </div>
          {e.error ? <ErrorText error={e.error} /> : (
            <>
              <div className="rounded bg-panel-2 px-2 py-1.5 font-mono text-sm">{e.equation}</div>
              <div className="mt-1.5 text-xs text-muted">true law: <span className="font-mono">{e.true_law}</span></div>
              <div className="mt-1 text-xs text-muted">
                fit error vs probe {fmtNum(e.fit_error_vs_probe)} · vs true law {fmtNum(e.fit_error_vs_true_law)} · true law vs probe {fmtNum(e.true_law_error_vs_probe)}
              </div>
              {e.pareto && (
                <details className="mt-2 text-xs">
                  <summary className="cursor-pointer text-muted">Pareto front ({e.pareto.length})</summary>
                  <ul className="mt-1 space-y-0.5 font-mono text-[11px]">
                    {e.pareto.map((p, i) => <li key={i}><span className="text-muted">c={p.complexity} loss={fmtNum(p.loss)}</span> {p.equation}</li>)}
                  </ul>
                </details>
              )}
            </>
          )}
        </div>
      ))}
    </div>
  );
}

function Detail({ runId }: { runId: string }) {
  const qc = useQueryClient();
  const run = useQuery({ queryKey: ["run", runId], queryFn: () => api.run(runId), refetchInterval: (q) => (q.state.data?.status === "running" ? 10_000 : false) });
  const metrics = useQuery({ queryKey: ["metrics", runId], queryFn: () => api.metrics(runId) });
  const probes = useQuery({ queryKey: ["probes", runId], queryFn: () => api.probes(runId), enabled: run.data?.status === "completed" });
  const targets = useMemo(() => [...new Set((probes.data ?? []).map((p) => p.target))], [probes.data]);
  const [target, setTarget] = useState("force");
  const [note, setNote] = useState("");
  const [tab, setTab] = useState<"log" | "config" | "artifacts">("log");
  const inv = () => qc.invalidateQueries({ queryKey: ["run", runId] });
  const annotate = useMutation({ mutationFn: (b: { note?: string; bad?: boolean }) => api.annotateRun(runId, b), onSuccess: () => { setNote(""); inv(); } });
  const rerun = useMutation({ mutationFn: () => api.rerun(runId), onSuccess: () => qc.invalidateQueries({ queryKey: ["experiments"] }) });

  const r = run.data;
  if (!r) return <div className="text-sm text-muted">{run.error ? <ErrorText error={run.error} /> : "Loading…"}</div>;
  const s = r.summary ?? {};
  const margin = r.experiment.config.probes.control_margin ?? 0.1;
  const activeTarget = targets.includes(target) ? target : targets[0] ?? "force";

  return (
    <div>
      <PageHeader
        title={`Run ${shortId(r.id)}`}
        sub={<span>{r.experiment.family} · seed {r.seed} · <a className="text-accent hover:underline" href="#/experiments">{shortId(r.experiment_id)}</a> · {r.experiment.hypothesis}</span>}
      >
        <StateBadge state={r.status} />
        {r.bad && <Badge tone="bad">marked bad</Badge>}
        <Button variant="secondary" size="sm" onClick={() => rerun.mutate()} disabled={rerun.isPending}>{rerun.isSuccess ? "Re-run queued" : "Re-run with new seed"}</Button>
        <Button variant="outline" size="sm" onClick={() => annotate.mutate({ bad: !r.bad })}>{r.bad ? "Unmark bad" : "Mark as bad"}</Button>
      </PageHeader>
      <ErrorText error={rerun.error || annotate.error} />
      {r.error && <ErrorText error={r.error} />}

      <div className="mb-4 grid grid-cols-2 gap-4 rounded-lg border border-border bg-panel p-4 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Val MSE" value={fmtNum(s.final_val_mse)} sub={`const-velocity ${fmtNum(s.baseline_cv_mse)}`} />
        <Stat label="MSE ratio" value={<span className={s.beats_baseline ? "text-good" : ""}>{fmtNum(s.val_mse_ratio)}</span>} sub={s.beats_baseline ? "beats baseline" : s.val_mse_ratio !== undefined ? "does not beat baseline" : ""} />
        {(["force", "accel_step"] as const).map((t) => {
          const ts = s.probes?.targets?.[t];
          return <Stat key={t} label={`${TARGET_LABEL[t]} R²`} value={ts ? fmtNum(ts.r2, 3) : "–"}
            sub={ts ? `L${ts.best_layer} · ctrl ${fmtNum(ts.control_r2, 2)} · ${ts.beats_control ? "counts" : "below margin"}` : ""} />;
        })}
        <Stat label="Compute" value={fmtHours(r.compute_hours)} sub={`${r.device ?? s.device ?? ""} · ${s.n_params?.toLocaleString() ?? "?"} params`} />
        <Stat label="Commit" value={<span className="font-mono text-base">{r.git_commit ?? "–"}</span>} sub={fmtTime(r.started_at)} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Loss curves</CardTitle></CardHeader>
          <CardContent><LossChart metrics={metrics.data ?? []} height={260} /></CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Formation timing</CardTitle><span className="text-xs text-muted">does understanding come before or after accuracy?</span></CardHeader>
          <CardContent>
            {probes.data?.length ? <FormationChart metrics={metrics.data ?? []} probes={probes.data} margin={margin} height={260} />
              : <Empty>{r.status === "completed" ? "No probe results." : "Probes run after training finishes."}</Empty>}
          </CardContent>
        </Card>
      </div>

      <Card className="mt-4">
        <CardHeader className="flex-wrap">
          <CardTitle>Probe R² heatmap (layer × checkpoint)</CardTitle>
          {targets.length > 0 && <Tabs value={activeTarget} onChange={setTarget} options={targets.map((t) => ({ value: t, label: TARGET_LABEL[t] ?? t }))} />}
        </CardHeader>
        <CardContent>
          {probes.data?.length ? <ProbeHeatmap rows={probes.data} target={activeTarget} margin={margin} />
            : <Empty>{r.status === "completed" ? "No probe results." : "Waiting for the probe suite."}</Empty>}
          {activeTarget === "accel_step" && (
            <p className="mt-2 text-xs text-muted">
              accel / snapshot is body 2's acceleration in the model's own time units, (F/m₂)·dt². Orbits are sampled at a fixed
              number of snapshots per lap, so dt grows with orbit size: compare this with the physical force to tell
              "learned gravity" from "learned a per-step version of gravity".
            </p>
          )}
        </CardContent>
      </Card>

      <Card className="mt-4">
        <CardHeader><CardTitle>Extracted equations</CardTitle></CardHeader>
        <CardContent><Equations run={r} /></CardContent>
      </Card>

      {(r.patching || r.eval) && (
        <div className="mt-4 grid gap-4 xl:grid-cols-2">
          {r.patching && (
            <Card>
              <CardHeader><CardTitle>Activation patching</CardTitle><span className="text-xs text-muted">clean MSE {fmtNum(r.patching.clean_mse)}</span></CardHeader>
              <CardContent>
                <Table>
                  <thead><tr><Th>Head</Th><Th>Force R²</Th><Th>Patched MSE</Th><Th>Effect</Th></tr></thead>
                  <tbody>{r.patching.patched.map((p) => (
                    <tr key={`${p.layer}-${p.head}`}><Td>L{p.layer}H{p.head} {p.control && <Badge>control</Badge>}</Td><Td>{fmtNum(p.force_r2)}</Td><Td>{fmtNum(p.patched_mse)}</Td><Td className={p.effect > 0.5 ? "text-bad" : ""}>{p.effect >= 0 ? "+" : ""}{(p.effect * 100).toFixed(0)}%</Td></tr>
                  ))}</tbody>
                </Table>
              </CardContent>
            </Card>
          )}
          {r.eval && (
            <Card>
              <CardHeader><CardTitle>Out-of-distribution evaluation</CardTitle></CardHeader>
              <CardContent>
                <Table>
                  <thead><tr><Th>Set</Th><Th>Bodies</Th><Th>MSE ratio</Th><Th>Best force R²</Th><Th>Best accel R²</Th></tr></thead>
                  <tbody>{r.eval.map((e) => {
                    const best = (t: string) => { const v = Object.values(e.probe_r2[t] ?? {}); return v.length ? Math.max(...v) : null; };
                    return <tr key={e.name}><Td>{e.name}</Td><Td>{e.n_bodies}</Td><Td>{fmtNum(e.val_mse_ratio)}</Td><Td>{fmtNum(best("force"))}</Td><Td>{fmtNum(best("accel_step"))}</Td></tr>;
                  })}</tbody>
                </Table>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      <div className="mt-4 grid gap-4 xl:grid-cols-[2fr_1fr]">
        <Card>
          <CardHeader><Tabs value={tab} onChange={setTab} options={[{ value: "log", label: "Log" }, { value: "config", label: "Config" }, { value: "artifacts", label: "Artifacts" }]} /></CardHeader>
          <CardContent>
            {tab === "log" && <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded-md bg-panel-2 p-3 font-mono text-[11px] leading-relaxed">{r.log || "No log."}</pre>}
            {tab === "config" && <pre className="max-h-96 overflow-auto rounded-md bg-panel-2 p-3 font-mono text-[11px]">{JSON.stringify(r.experiment.config, null, 2)}</pre>}
            {tab === "artifacts" && (r.artifacts_present ? (
              <ul className="max-h-96 overflow-auto font-mono text-[11px]">
                {r.artifacts.map((a) => <li key={a.path} className="flex justify-between gap-4"><span title={a.path}>{a.name}</span><span className="text-muted">{(a.bytes / 1024).toFixed(0)} KB</span></li>)}
              </ul>
            ) : <p className="text-sm text-muted">Artifacts are not on this machine ({r.artifact_dir}). They are gitignored and can be regenerated from the config and seed.</p>)}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Notes</CardTitle></CardHeader>
          <CardContent>
            <ul className="mb-3 space-y-1.5 text-sm">
              {r.notes.map((n, i) => <li key={i}><span className="text-xs text-muted">{fmtTime(n.at)} {n.actor}:</span> {n.text}</li>)}
              {!r.notes.length && <li className="text-muted">No notes.</li>}
            </ul>
            <form className="flex gap-1.5" onSubmit={(e) => { e.preventDefault(); if (note.trim()) annotate.mutate({ note }); }}>
              <Input className="h-8" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Add a note…" />
              <Button size="sm" type="submit">Add</Button>
            </form>
            {r.experiment.verdict && <div className="mt-4 text-xs text-muted">Experiment verdict: <VerdictBadge verdict={r.experiment.verdict} /></div>}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default function RunDetailPage({ runId }: { runId?: string }) {
  return runId ? <Detail runId={runId} /> : <RunList />;
}
