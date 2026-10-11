import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pause, Play, Power } from "lucide-react";
import { api, type Run } from "../lib/api";
import { fmtHours, fmtNum, fmtTime, shortId } from "../lib/utils";
import { PageHeader } from "../App";
import { LossChart } from "../components/charts";
import { FindingCard } from "../components/FindingCard";
import { StateBadge } from "../components/badges";
import { Button, Card, CardContent, CardHeader, CardTitle, ConfirmButton, Empty, ErrorText, Input, Label, Stat } from "../components/ui";

function LiveRun({ run }: { run: Run & { experiment: { family: string; hypothesis: string } | null } }) {
  const m = useQuery({ queryKey: ["metrics", run.id], queryFn: () => api.metrics(run.id) });
  const last = [...(m.data ?? [])].reverse().find((x) => x.name === "val_mse_ratio");
  return (
    <a href={`#/runs/${run.id}`} className="block rounded-md border border-border p-3 hover:bg-panel-2">
      <div className="flex items-center gap-2 text-xs">
        <span className="font-semibold">{run.experiment?.family}</span>
        <span className="font-mono text-muted">{shortId(run.id)} · seed {run.seed}</span>
        <span className="ml-auto"><StateBadge state={run.progress === "probing" ? "analyzing" : "running"} /></span>
      </div>
      <p className="mt-1 line-clamp-1 text-xs text-muted">{run.experiment?.hypothesis}</p>
      <LossChart metrics={m.data ?? []} height={120} mini />
      <div className="text-[11px] text-muted">
        {run.progress === "probing" ? "running probe suite…" : `MSE / const-velocity: ${fmtNum(last?.value)}`}
      </div>
    </a>
  );
}

function BudgetForm({ s }: { s: { daily_budget_hours: number; max_concurrent_runs: number; per_experiment_cap_hours: number; max_cycles_per_day: number } }) {
  const qc = useQueryClient();
  const [f, setF] = useState(s);
  useEffect(() => setF(s), [s.daily_budget_hours, s.max_concurrent_runs, s.per_experiment_cap_hours, s.max_cycles_per_day]);
  const save = useMutation({ mutationFn: () => api.setBudget(f), onSuccess: () => qc.invalidateQueries({ queryKey: ["dashboard"] }) });
  const field = (k: keyof typeof f, label: string, step: string) => (
    <div>
      <Label>{label}</Label>
      <Input type="number" step={step} min={0} value={f[k]} onChange={(e) => setF({ ...f, [k]: Number(e.target.value) })} />
    </div>
  );
  return (
    <form className="grid grid-cols-2 gap-2" onSubmit={(e) => { e.preventDefault(); save.mutate(); }}>
      {field("daily_budget_hours", "Daily compute (h)", "0.1")}
      {field("max_concurrent_runs", "Max concurrent runs", "1")}
      {field("per_experiment_cap_hours", "Auto-approve cap (h)", "0.05")}
      {field("max_cycles_per_day", "Max cycles / day", "1")}
      <div className="col-span-2 flex items-center gap-2">
        <Button size="sm" type="submit" disabled={save.isPending}>Save budget</Button>
        {save.isSuccess && <span className="text-xs text-good">saved</span>}
      </div>
      <div className="col-span-2"><ErrorText error={save.error} /></div>
    </form>
  );
}

export default function Dashboard() {
  const qc = useQueryClient();
  const d = useQuery({ queryKey: ["dashboard"], queryFn: api.dashboard, refetchInterval: 10_000 });
  const qs = useQuery({ queryKey: ["questions"], queryFn: api.questions });
  const refresh = () => { qc.invalidateQueries({ queryKey: ["dashboard"] }); qc.invalidateQueries({ queryKey: ["loop"] }); };
  const start = useMutation({ mutationFn: api.startLoop, onSuccess: refresh });
  const pause = useMutation({ mutationFn: api.pauseLoop, onSuccess: refresh });
  const kill = useMutation({ mutationFn: api.kill, onSuccess: refresh });
  const [q, setQ] = useState("");
  const addQ = useMutation({ mutationFn: () => api.addQuestion(q), onSuccess: () => { setQ(""); qc.invalidateQueries({ queryKey: ["questions"] }); } });

  if (!d.data) return <div className="text-sm text-muted">{d.error ? <ErrorText error={d.error} /> : "Loading…"}</div>;
  const { loop, running_runs, latest_findings, open_questions, counts } = d.data;
  const used = loop.compute_used_today, budget = loop.settings.daily_budget_hours;
  const pct = Math.min(100, (used / Math.max(budget, 1e-9)) * 100);

  return (
    <div>
      <PageHeader title="Dashboard" sub="A self-running lab: train small transformers on orbits, probe them for gravity.">
        {loop.paused
          ? <Button onClick={() => start.mutate()} disabled={start.isPending}><Play size={14} /> Start loop</Button>
          : <Button variant="secondary" onClick={() => pause.mutate()} disabled={pause.isPending}><Pause size={14} /> Pause loop</Button>}
        <ConfirmButton variant="destructive" onConfirm={() => kill.mutate()} confirmLabel="Confirm: pause + cancel queue">
          <Power size={14} /> Kill switch
        </ConfirmButton>
      </PageHeader>
      {kill.data && <p className="mb-3 text-xs text-muted">Kill switch: loop paused, cancelled {kill.data.cancelled.length} queued experiment(s). Running jobs continue.</p>}

      <div className="grid gap-4 lg:grid-cols-3">
        <Card>
          <CardHeader><CardTitle>Loop</CardTitle></CardHeader>
          <CardContent className="grid grid-cols-2 gap-4">
            <Stat label="Status" value={<span className={loop.paused ? "text-warn" : "text-good"}>{loop.paused ? "Paused" : "Running"}</span>}
              sub={`worker ${loop.worker.online ? "online" : "offline"}`} />
            <Stat label="Cycles today" value={`${loop.cycles_today} / ${loop.settings.max_cycles_per_day}`}
              sub={loop.last_cycle ? `${loop.last_cycle.type.replace("cycle.", "")} ${fmtTime(loop.last_cycle.timestamp)}` : "no cycles yet"} />
            <Stat label="Awaiting approval" value={<a href="#/experiments" className="hover:underline">{counts.proposed ?? 0}</a>} sub={`${counts.approved + counts.queued} approved/queued`} />
            <Stat label="Open questions" value={open_questions} sub={`${d.data.n_findings} findings total`} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Compute today</CardTitle><span className="text-xs text-muted tabular-nums">{fmtHours(used)} / {fmtHours(budget)}</span></CardHeader>
          <CardContent>
            <div className="mb-4 h-2.5 w-full overflow-hidden rounded-full bg-panel-2">
              <div className="h-full rounded-full" style={{ width: `${pct}%`, background: pct > 90 ? "var(--bad)" : pct > 70 ? "var(--warn)" : "var(--accent)" }} />
            </div>
            <BudgetForm s={loop.settings} />
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Open questions</CardTitle></CardHeader>
          <CardContent>
            <ul className="max-h-48 space-y-1.5 overflow-y-auto text-sm">
              {(qs.data ?? []).filter((x) => x.status === "open").map((x) => (
                <li key={x.id} className="flex items-start gap-2">
                  <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-info" />
                  <span className="flex-1">{x.text}</span>
                  <button className="text-[11px] text-muted hover:text-fg cursor-pointer" onClick={() => api.updateQuestion(x.id, "dropped").then(() => qc.invalidateQueries({ queryKey: ["questions"] }))}>drop</button>
                </li>
              ))}
              {!qs.data?.some((x) => x.status === "open") && <li className="text-muted">No open questions.</li>}
            </ul>
            <form className="mt-3 flex gap-1.5" onSubmit={(e) => { e.preventDefault(); if (q.trim()) addQ.mutate(); }}>
              <Input className="h-8" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Add a question for the agent…" />
              <Button size="sm" type="submit">Add</Button>
            </form>
          </CardContent>
        </Card>
      </div>

      <h2 className="mb-2 mt-6 text-sm font-semibold">Runs in progress</h2>
      {running_runs.length ? (
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">{running_runs.map((r) => <LiveRun key={r.id} run={r} />)}</div>
      ) : <Empty>No runs in progress. {loop.paused ? "The loop is paused." : "Waiting for approved experiments."}</Empty>}

      <div className="mb-2 mt-6 flex items-center justify-between">
        <h2 className="text-sm font-semibold">Latest findings</h2>
        <a href="#/findings" className="text-xs text-accent hover:underline">All findings →</a>
      </div>
      {latest_findings.length ? (
        <div className="grid gap-3 lg:grid-cols-2">{latest_findings.map((f) => <FindingCard key={f.id} f={f} actions={false} />)}</div>
      ) : <Empty>No findings yet. Findings appear once the agent analyzes completed experiments.</Empty>}
    </div>
  );
}
