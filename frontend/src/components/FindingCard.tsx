import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api, type Finding } from "../lib/api";
import { fmtTime, shortId } from "../lib/utils";
import { VERDICT_COLOR, VerdictBadge } from "./badges";
import { Badge, Button, ErrorText, Input, Select } from "./ui";

export function RunLink({ id }: { id: string }) {
  return (
    <a href={`#/runs/${id}`} className="font-mono text-[11px] text-accent hover:underline">
      {shortId(id)}
    </a>
  );
}

export function FindingCard({ f, actions = true, draggable = false }: { f: Finding; actions?: boolean; draggable?: boolean }) {
  const qc = useQueryClient();
  const [note, setNote] = useState("");
  const [showNote, setShowNote] = useState(false);
  const done = () => { qc.invalidateQueries({ queryKey: ["findings"] }); setNote(""); setShowNote(false); };
  const promote = useMutation({ mutationFn: (s: string) => api.promote(f.id, s), onSuccess: done });
  const annotate = useMutation({ mutationFn: (b: { note?: string; disputed?: boolean }) => api.annotateFinding(f.id, b), onSuccess: done });
  const replicate = useMutation({ mutationFn: () => api.replicate(f.id), onSuccess: () => qc.invalidateQueries({ queryKey: ["experiments"] }) });
  const color = VERDICT_COLOR[f.status] ?? "var(--muted)";

  return (
    <div
      className="rounded-lg border border-border bg-panel p-3.5"
      style={{ borderLeft: `3px solid ${color}` }}
      draggable={draggable}
      onDragStart={(e) => {
        e.dataTransfer.setData("text/plain", `${f.claim} [${f.id}]`);
        e.dataTransfer.setData("application/x-finding", f.id);
      }}
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <VerdictBadge verdict={f.verdict} />
        {f.status !== f.verdict && <VerdictBadge verdict={f.status} />}
        {f.disputed && <Badge tone="bad">disputed</Badge>}
        {(f.replication_count ?? 0) > 0 && <Badge tone="accent">{f.replication_count} replication{f.replication_count === 1 ? "" : "s"}</Badge>}
        <span className="ml-auto text-[11px] text-muted">{fmtTime(f.created_at)} · {f.created_by}</span>
      </div>
      <p className="mt-2 text-sm leading-snug">{f.claim}</p>
      <p className="mt-1.5 text-xs text-muted">{f.confidence_note}</p>
      <div className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted">
        <span>evidence:</span>
        {f.evidence_run_ids.map((r) => <RunLink key={r} id={r} />)}
        {f.replicates && <span>· replicates <span className="font-mono">{shortId(f.replicates)}</span></span>}
        <span className="ml-auto font-mono text-[10px]">{f.id}</span>
      </div>
      {f.notes.length > 0 && (
        <ul className="mt-2 space-y-1 border-t border-border pt-2 text-xs">
          {f.notes.map((n, i) => <li key={i}><span className="text-muted">{n.actor}:</span> {n.text}</li>)}
        </ul>
      )}
      {actions && (
        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <Select className="h-7 text-xs" value="" onChange={(e) => e.target.value && promote.mutate(e.target.value)}>
            <option value="">Promote / set status…</option>
            {["replicated", "supported", "refuted", "inconclusive", "hypothesis"].filter((s) => s !== f.status).map((s) => <option key={s} value={s}>{s}</option>)}
          </Select>
          <Button size="sm" variant="outline" onClick={() => annotate.mutate({ disputed: !f.disputed })}>{f.disputed ? "Clear dispute" : "Dispute"}</Button>
          <Button size="sm" variant="outline" onClick={() => setShowNote((v) => !v)}>Add note</Button>
          <Button size="sm" variant="secondary" onClick={() => replicate.mutate()} disabled={replicate.isPending}>
            {replicate.isSuccess ? "Replication queued" : "Start replication"}
          </Button>
        </div>
      )}
      {showNote && (
        <form className="mt-2 flex gap-1.5" onSubmit={(e) => { e.preventDefault(); if (note.trim()) annotate.mutate({ note }); }}>
          <Input className="h-8" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Note…" autoFocus />
          <Button size="sm" type="submit">Save</Button>
        </form>
      )}
      <ErrorText error={promote.error || annotate.error || replicate.error} />
    </div>
  );
}
