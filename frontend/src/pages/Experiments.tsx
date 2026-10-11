import { Fragment, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, ChevronRight, Plus } from "lucide-react";
import { api, type Experiment, type Validation } from "../lib/api";
import { fmtHours, fmtTime, shortId } from "../lib/utils";
import { PageHeader } from "../App";
import { StateBadge, VerdictBadge } from "../components/badges";
import { RunLink } from "../components/FindingCard";
import { Badge, Button, Card, Dialog, Empty, ErrorText, Label, Select, Table, Td, Textarea, Th } from "../components/ui";

function ConfigEditor({ initial, onSubmit, submitLabel, showApprove }: {
  initial: string; onSubmit: (cfg: unknown, approve: boolean) => Promise<unknown>; submitLabel: string; showApprove?: boolean;
}) {
  const [text, setText] = useState(initial);
  const [v, setV] = useState<Validation | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { setText(initial); setV(null); setErr(null); }, [initial]);

  const parse = () => { try { return JSON.parse(text); } catch (e) { setErr(new Error(`Invalid JSON: ${(e as Error).message}`)); return null; } };
  const validate = async () => {
    setErr(null);
    const cfg = parse();
    if (cfg) try { setV(await api.validate(cfg)); } catch (e) { setErr(e); }
  };
  const submit = async (approve: boolean) => {
    setErr(null);
    const cfg = parse();
    if (!cfg) return;
    setBusy(true);
    try { await onSubmit(cfg, approve); } catch (e) { setErr(e); } finally { setBusy(false); }
  };
  return (
    <div className="space-y-2">
      <Textarea className="h-[420px] font-mono text-xs leading-relaxed" spellCheck={false} value={text} onChange={(e) => { setText(e.target.value); setV(null); }} />
      {v && (
        <div className={`rounded-md border p-2 text-xs ${v.valid ? "border-good/40 bg-good/10" : "border-bad/40 bg-bad/10"}`}>
          {v.valid ? (
            <>Valid · estimated {fmtHours(v.estimate_hours)} of compute · {v.auto_approvable ? "would be auto-approved" : `needs human approval (${v.auto_reason})`}</>
          ) : <ul className="list-disc pl-4 text-bad">{v.errors.map((e, i) => <li key={i}>{e}</li>)}</ul>}
        </div>
      )}
      <ErrorText error={err} />
      <div className="flex flex-wrap gap-2">
        <Button variant="secondary" onClick={validate}>Validate</Button>
        <Button variant={showApprove ? "secondary" : "default"} onClick={() => submit(false)} disabled={busy}>{submitLabel}</Button>
        {showApprove && <Button onClick={() => submit(true)} disabled={busy}>Propose and approve</Button>}
      </div>
    </div>
  );
}

function ExperimentDetail({ e }: { e: Experiment }) {
  const qc = useQueryClient();
  const [editing, setEditing] = useState(false);
  const inv = () => { qc.invalidateQueries({ queryKey: ["experiments"] }); qc.invalidateQueries({ queryKey: ["dashboard"] }); };
  const approve = useMutation({ mutationFn: () => api.approve(e.id), onSuccess: inv });
  const reject = useMutation({ mutationFn: () => api.reject(e.id), onSuccess: inv });
  const cancel = useMutation({ mutationFn: () => api.cancel(e.id), onSuccess: inv });
  return (
    <div className="grid gap-4 bg-panel-2/40 p-4 lg:grid-cols-[1fr_1fr]">
      <div className="space-y-3 text-sm">
        <div><Label>Hypothesis</Label><p>{e.hypothesis}</p></div>
        <div className="grid gap-3 sm:grid-cols-2">
          <div><Label>Expected if true</Label><p className="text-xs">{e.expected_if_true}</p></div>
          <div><Label>Expected if false</Label><p className="text-xs">{e.expected_if_false}</p></div>
        </div>
        <div>
          <Label>Runs</Label>
          {e.runs?.length ? (
            <ul className="mt-1 space-y-1 text-xs">
              {e.runs.map((r) => (
                <li key={r.id} className="flex items-center gap-2">
                  <RunLink id={r.id} /> <span className="text-muted">seed {r.seed}</span> <StateBadge state={r.status} />
                  {r.bad && <Badge tone="bad">marked bad</Badge>}
                  <span className="text-muted">{fmtHours(r.compute_hours)}</span>
                </li>
              ))}
            </ul>
          ) : <p className="text-xs text-muted">No runs yet.</p>}
        </div>
        <div>
          <Label>History</Label>
          <ol className="mt-1 space-y-0.5 text-xs text-muted">
            {e.history.map((h, i) => <li key={i}>{fmtTime(h.at)} · <span className="text-fg">{h.state}</span> by {h.actor}{h.note ? ` — ${h.note}` : ""}</li>)}
          </ol>
        </div>
        {e.notes?.length > 0 && (
          <div><Label>Notes</Label><ul className="text-xs">{e.notes.map((n, i) => <li key={i}>{n.actor}: {n.text}</li>)}</ul></div>
        )}
        {e.state === "proposed" && (
          <div className="flex flex-wrap gap-2">
            <Button variant="good" onClick={() => approve.mutate()} disabled={approve.isPending}>Approve</Button>
            <Button variant="outline" onClick={() => setEditing((x) => !x)}>{editing ? "Close editor" : "Edit config"}</Button>
            <Button variant="destructive" onClick={() => reject.mutate()} disabled={reject.isPending}>Reject</Button>
          </div>
        )}
        {(e.state === "approved" || e.state === "queued") && <Button variant="outline" size="sm" onClick={() => cancel.mutate()}>Cancel</Button>}
        <ErrorText error={approve.error || reject.error || cancel.error} />
      </div>
      <div>
        {editing ? (
          <ConfigEditor initial={JSON.stringify(e.config, null, 2)} submitLabel="Save config"
            onSubmit={async (cfg) => { await api.editConfig(e.id, cfg); setEditing(false); inv(); }} />
        ) : (
          <pre className="max-h-[440px] overflow-auto rounded-md border border-border bg-panel p-3 text-[11px] leading-relaxed">{JSON.stringify(e.config, null, 2)}</pre>
        )}
      </div>
    </div>
  );
}

export default function Experiments() {
  const qc = useQueryClient();
  const exps = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  const templates = useQuery({ queryKey: ["templates"], queryFn: api.templates });
  const [family, setFamily] = useState("");
  const [state, setState] = useState("");
  const [verdict, setVerdict] = useState("");
  const [open, setOpen] = useState<string | null>(null);
  const [proposing, setProposing] = useState(false);
  const [tpl, setTpl] = useState(0);

  const rows = useMemo(() => (exps.data ?? []).filter((e) =>
    (!family || e.family === family) && (!state || e.state === state) && (!verdict || (e.verdict ?? "none") === verdict)), [exps.data, family, state, verdict]);

  return (
    <div>
      <PageHeader title="Experiments" sub="Every proposal, its state, verdict and compute. Approve or edit anything waiting at the gate.">
        <Button onClick={() => setProposing(true)}><Plus size={14} /> Propose experiment</Button>
      </PageHeader>
      <div className="mb-3 flex flex-wrap gap-2">
        <Select value={family} onChange={(e) => setFamily(e.target.value)}>
          <option value="">All families</option>
          {["E1", "E2", "E3", "E4", "E5", "E6", "custom"].map((f) => <option key={f}>{f}</option>)}
        </Select>
        <Select value={state} onChange={(e) => setState(e.target.value)}>
          <option value="">All states</option>
          {["proposed", "approved", "queued", "running", "analyzing", "complete", "failed", "cancelled", "rejected"].map((f) => <option key={f}>{f}</option>)}
        </Select>
        <Select value={verdict} onChange={(e) => setVerdict(e.target.value)}>
          <option value="">All verdicts</option>
          {["supported", "refuted", "inconclusive", "none"].map((f) => <option key={f}>{f}</option>)}
        </Select>
      </div>
      <Card>
        {rows.length ? (
          <Table>
            <thead><tr><Th className="w-6" /><Th>Family</Th><Th>Hypothesis</Th><Th>State</Th><Th>Verdict</Th><Th>Compute</Th><Th>By</Th><Th>Date</Th></tr></thead>
            <tbody>
              {rows.map((e) => (
                <Fragment key={e.id}>
                  <tr className="cursor-pointer hover:bg-panel-2/60" onClick={() => setOpen(open === e.id ? null : e.id)}>
                    <Td>{open === e.id ? <ChevronDown size={14} /> : <ChevronRight size={14} />}</Td>
                    <Td className="font-semibold">{e.family}</Td>
                    <Td className="max-w-[520px]"><div className="line-clamp-2">{e.hypothesis}</div><div className="font-mono text-[10px] text-muted">{shortId(e.id)}{e.replicates_finding_id ? ` · replicates ${shortId(e.replicates_finding_id)}` : ""}</div></Td>
                    <Td><StateBadge state={e.state} /></Td>
                    <Td><VerdictBadge verdict={e.verdict} /></Td>
                    <Td className="tabular-nums text-xs">{e.compute_hours ? fmtHours(e.compute_hours) : <span className="text-muted">est. {fmtHours(e.estimate_hours)}</span>}</Td>
                    <Td className="text-xs text-muted">{e.proposed_by}</Td>
                    <Td className="whitespace-nowrap text-xs text-muted">{fmtTime(e.created_at)}</Td>
                  </tr>
                  {open === e.id && <tr><td colSpan={8} className="border-b border-border p-0"><ExperimentDetail e={e} /></td></tr>}
                </Fragment>
              ))}
            </tbody>
          </Table>
        ) : <div className="p-4"><Empty>{exps.isLoading ? "Loading…" : "No experiments match. Propose one to get started."}</Empty></div>}
      </Card>

      <Dialog open={proposing} onClose={() => setProposing(false)} title="Propose an experiment" wide>
        <div className="mb-3 flex items-center gap-2">
          <Label>Start from</Label>
          <Select value={tpl} onChange={(e) => setTpl(Number(e.target.value))}>
            {(templates.data ?? []).map((t, i) => <option key={t.name} value={i}>{t.name} — {t.config.hypothesis.slice(0, 70)}</option>)}
          </Select>
        </div>
        {templates.data?.length ? (
          <ConfigEditor initial={JSON.stringify(templates.data[tpl].config, null, 2)} submitLabel="Propose" showApprove
            onSubmit={async (cfg, approve) => { await api.propose(cfg, approve); setProposing(false); qc.invalidateQueries({ queryKey: ["experiments"] }); }} />
        ) : <ErrorText error={templates.error} />}
      </Dialog>
    </div>
  );
}
