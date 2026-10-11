import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import ReactMarkdown, { defaultUrlTransform } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { Download, FileText, Plus, Sparkles } from "lucide-react";
import { api, type Finding } from "../lib/api";
import { fmtTime, shortId } from "../lib/utils";
import { PageHeader } from "../App";
import { MarkdownEditor } from "../components/MarkdownEditor";
import { ProbeHeatmap } from "../components/ProbeHeatmap";
import { FormationChart } from "../components/charts";
import { VerdictBadge } from "../components/badges";
import { Badge, Button, Card, Dialog, Empty, ErrorText, Input, Label, Textarea } from "../components/ui";

const FID = /\[(f-\d{8}-\d{6}-[0-9a-f]{4})\]/g;

/** Figures are embedded as ![caption](figure:heatmap/<run_id>/<target>) or figure:formation/<run_id>. */
function Figure({ src, alt }: { src: string; alt: string }) {
  const [, kind, runId, target] = src.match(/^figure:(\w+)\/([\w-]+)(?:\/(\w+))?/) ?? [];
  const probes = useQuery({ queryKey: ["probes", runId], queryFn: () => api.probes(runId), enabled: !!runId });
  const metrics = useQuery({ queryKey: ["metrics", runId], queryFn: () => api.metrics(runId), enabled: !!runId && kind === "formation" });
  if (!runId) return <span className="text-bad">[bad figure reference {src}]</span>;
  return (
    <figure className="my-4 rounded-md border border-border p-3 not-prose">
      {kind === "heatmap" && probes.data && <ProbeHeatmap rows={probes.data} target={target ?? "force"} compact height={200} />}
      {kind === "formation" && probes.data && <FormationChart probes={probes.data} metrics={metrics.data ?? []} height={220} />}
      <figcaption className="mt-1 text-xs text-muted">{alt} · run <a className="text-accent" href={`#/runs/${runId}`}>{shortId(runId)}</a></figcaption>
    </figure>
  );
}

function Preview({ md, findings }: { md: string; findings: Finding[] }) {
  const { text, refs } = useMemo(() => {
    const order: string[] = [];
    const text = md.replace(FID, (_m, id: string) => {
      if (!order.includes(id)) order.push(id);
      return `[[${order.indexOf(id) + 1}]](#/findings)`;
    });
    return { text, refs: order };
  }, [md]);
  const byId = new Map(findings.map((f) => [f.id, f]));
  return (
    <div className="prose-de">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        urlTransform={(url) => (url.startsWith("figure:") ? url : defaultUrlTransform(url))}
        components={{
          img: ({ src, alt }) => (typeof src === "string" && src.startsWith("figure:") ? <Figure src={src} alt={alt ?? ""} /> : <img src={src as string} alt={alt} />),
          p: ({ children, node }) => {
            const only = node?.children?.length === 1 && node.children[0].type === "element" && (node.children[0] as { tagName?: string }).tagName === "img";
            return only ? <div>{children}</div> : <p>{children}</p>;
          },
        }}
      >
        {text}
      </ReactMarkdown>
      {refs.length > 0 && (
        <div className="mt-8 border-t border-border pt-3">
          <h2>References</h2>
          <ol className="text-sm">
            {refs.map((id) => {
              const f = byId.get(id);
              return (
                <li key={id} className="mb-1">
                  {f ? <>{f.claim} <span className="text-muted">({f.status}; </span>runs {f.evidence_run_ids.map((r, i) => <span key={r}>{i > 0 && ", "}<a href={`#/runs/${r}`} className="font-mono text-xs">{r}</a></span>)}<span className="text-muted">)</span> <span className="font-mono text-[10px] text-muted">{id}</span></>
                    : <span className="text-bad">{id} (unknown finding)</span>}
                </li>
              );
            })}
          </ol>
        </div>
      )}
    </div>
  );
}

function FindingsSidebar({ findings, overrides, onToggleOverride }: { findings: Finding[]; overrides: string[]; onToggleOverride: (id: string) => void }) {
  return (
    <div className="space-y-2">
      <p className="text-[11px] text-muted">Drag a claim or figure into the editor. Only <b>replicated</b> findings may be cited unless you allow one.</p>
      {findings.map((f) => {
        const allowed = f.status === "replicated" || overrides.includes(f.id);
        return (
          <div key={f.id} className="rounded-md border border-border bg-panel p-2 text-xs">
            <div draggable onDragStart={(e) => e.dataTransfer.setData("text/plain", `${f.claim} [${f.id}]`)} className="cursor-grab">
              <div className="mb-1 flex items-center gap-1"><VerdictBadge verdict={f.status} />{!allowed && <Badge tone="warn">not citable</Badge>}</div>
              <p className="leading-snug">{f.claim}</p>
            </div>
            <div className="mt-1.5 flex flex-wrap gap-1">
              {f.evidence_run_ids.slice(0, 2).map((r) => (
                <span key={r} className="flex gap-1">
                  <span draggable onDragStart={(e) => e.dataTransfer.setData("text/plain", `\n![Probe R² heatmap (force) for ${shortId(r)}](figure:heatmap/${r}/force)\n`)}
                    className="cursor-grab rounded border border-border px-1 py-0.5 text-[10px] text-muted hover:text-fg">▦ heatmap</span>
                  <span draggable onDragStart={(e) => e.dataTransfer.setData("text/plain", `\n![Formation timing for ${shortId(r)}](figure:formation/${r})\n`)}
                    className="cursor-grab rounded border border-border px-1 py-0.5 text-[10px] text-muted hover:text-fg">⟋ formation</span>
                </span>
              ))}
            </div>
            {f.status !== "replicated" && (
              <label className="mt-1.5 flex items-center gap-1 text-[10px] text-muted">
                <input type="checkbox" checked={overrides.includes(f.id)} onChange={() => onToggleOverride(f.id)} /> allow citing (override)
              </label>
            )}
          </div>
        );
      })}
      {!findings.length && <Empty>No findings yet.</Empty>}
    </div>
  );
}

function Editor({ id }: { id: string }) {
  const qc = useQueryClient();
  const paper = useQuery({ queryKey: ["paper", id], queryFn: () => api.paper(id), refetchInterval: (q) => (q.state.data?.drafting?.status === "running" ? 3000 : false) });
  const findings = useQuery({ queryKey: ["findings"], queryFn: api.findings });
  const [md, setMd] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [dirty, setDirty] = useState(false);
  const [draftOpen, setDraftOpen] = useState(false);
  const [section, setSection] = useState("Results");
  const [instructions, setInstructions] = useState("");
  const saveTimer = useRef<ReturnType<typeof setTimeout>>(undefined);
  const drafting = paper.data?.drafting?.status === "running";

  useEffect(() => {
    if (paper.data && (!dirty || md === null)) { setMd(paper.data.markdown); setTitle(paper.data.title); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paper.data?.markdown, paper.data?.title]);

  const save = useMutation({
    mutationFn: (p: { md: string; title: string; overrides?: string[] }) => api.savePaper(id, p.title, p.md, p.overrides),
    onSuccess: (p) => { setDirty(false); qc.setQueryData(["paper", id], p); },
  });
  const scheduleSave = (nextMd: string, nextTitle: string) => {
    setDirty(true);
    clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => save.mutate({ md: nextMd, title: nextTitle }), 1200);
  };
  const overrides = paper.data?.cite_overrides ?? [];
  const toggleOverride = (fid: string) =>
    save.mutate({ md: md ?? "", title, overrides: overrides.includes(fid) ? overrides.filter((x) => x !== fid) : [...overrides, fid] });
  const draft = useMutation({
    mutationFn: () => api.draftSection(id, section, instructions, overrides),
    onSuccess: () => { setDraftOpen(false); qc.invalidateQueries({ queryKey: ["paper", id] }); },
  });

  if (!paper.data || md === null) return <div className="text-sm text-muted">{paper.error ? <ErrorText error={paper.error} /> : "Loading…"}</div>;
  return (
    <div className="flex h-[calc(100vh-40px)] flex-col">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <a href="#/papers" className="text-xs text-muted hover:text-fg">← Papers</a>
        <Input className="h-8 max-w-md font-semibold" value={title} onChange={(e) => { setTitle(e.target.value); scheduleSave(md, e.target.value); }} />
        <span className="text-xs text-muted">{save.isPending ? "saving…" : dirty ? "unsaved" : `saved ${fmtTime(paper.data.updated_at)}`}</span>
        <div className="ml-auto flex flex-wrap gap-2">
          <Button size="sm" onClick={() => setDraftOpen(true)} disabled={drafting}><Sparkles size={13} /> {drafting ? `Drafting ${paper.data.drafting?.section}…` : "Ask agent to draft a section"}</Button>
          <a href={`/api/papers/${id}/export?format=markdown`}><Button size="sm" variant="outline"><Download size={13} /> Markdown</Button></a>
          <a href={`/api/papers/${id}/export?format=latex`}><Button size="sm" variant="outline"><Download size={13} /> LaTeX</Button></a>
        </div>
      </div>
      {paper.data.drafting?.status === "error" && <ErrorText error={`Drafting failed: ${paper.data.drafting.error}`} />}
      {paper.data.citation_warnings.length > 0 && (
        <div className="mb-2 rounded-md border border-warn/40 bg-warn/10 p-2 text-xs text-warn">
          {paper.data.citation_warnings.map((w) => <div key={w}>⚠ {w}</div>)}
        </div>
      )}
      <ErrorText error={save.error} />
      <div className="grid min-h-0 flex-1 gap-3 lg:grid-cols-[1fr_1fr_240px]">
        <Card className="min-h-[300px] overflow-hidden">
          <MarkdownEditor value={md} readOnly={drafting} onChange={(v) => { setMd(v); scheduleSave(v, title); }} />
        </Card>
        <Card className="min-h-[300px] overflow-y-auto p-5"><Preview md={md} findings={findings.data ?? []} /></Card>
        <div className="min-h-0 overflow-y-auto"><Label>Findings</Label><div className="mt-1"><FindingsSidebar findings={findings.data ?? []} overrides={overrides} onToggleOverride={toggleOverride} /></div></div>
      </div>
      <Dialog open={draftOpen} onClose={() => setDraftOpen(false)} title="Ask the agent to draft a section">
        <div className="space-y-3">
          <div><Label>Section</Label><Input value={section} onChange={(e) => setSection(e.target.value)} placeholder="e.g. Results, Methods, Discussion" /></div>
          <div><Label>Instructions (optional)</Label><Textarea rows={4} value={instructions} onChange={(e) => setInstructions(e.target.value)} placeholder="Focus on the force vs per-snapshot acceleration comparison…" /></div>
          <p className="text-xs text-muted">The agent runs through <code>claude -p</code> on your subscription. It cites only replicated findings{overrides.length ? ` plus ${overrides.length} you allowed` : ""}, and the section is appended to the paper.</p>
          <ErrorText error={draft.error} />
          <Button onClick={() => draft.mutate()} disabled={draft.isPending || !section.trim()}>Draft section</Button>
        </div>
      </Dialog>
    </div>
  );
}

export default function Papers({ paperId }: { paperId?: string }) {
  const qc = useQueryClient();
  const papers = useQuery({ queryKey: ["papers"], queryFn: api.papers, enabled: !paperId });
  const [title, setTitle] = useState("");
  const create = useMutation({ mutationFn: () => api.createPaper(title), onSuccess: (p) => { qc.invalidateQueries({ queryKey: ["papers"] }); window.location.hash = `#/papers/${p.id}`; } });
  if (paperId) return <Editor id={paperId} />;
  return (
    <div>
      <PageHeader title="Papers" sub="Drafts built from findings. Default rule: only replicated findings can be cited.">
        <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (title.trim()) create.mutate(); }}>
          <Input className="w-64" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="New paper title" />
          <Button type="submit"><Plus size={14} /> New paper</Button>
        </form>
      </PageHeader>
      <ErrorText error={create.error} />
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {(papers.data ?? []).map((p) => (
          <a key={p.id} href={`#/papers/${p.id}`}>
            <Card className="p-4 hover:bg-panel-2">
              <div className="flex items-center gap-2"><FileText size={16} className="text-muted" /><span className="font-medium">{p.title}</span></div>
              <div className="mt-2 text-xs text-muted">updated {fmtTime(p.updated_at)} · cites {p.cited_finding_ids.length} finding(s)</div>
            </Card>
          </a>
        ))}
      </div>
      {!papers.data?.length && <Empty>{papers.isLoading ? "Loading…" : "No papers yet. Create one, then drag findings in or ask the agent to draft a section."}</Empty>}
    </div>
  );
}
