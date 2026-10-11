import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, type Finding } from "../lib/api";
import { PageHeader } from "../App";
import { FindingCard } from "../components/FindingCard";
import { VERDICT_COLOR } from "../components/badges";
import { Empty, Select, Tabs } from "../components/ui";

function Timeline({ findings }: { findings: Finding[] }) {
  // Oldest first, so the research story reads top to bottom.
  const sorted = [...findings].sort((a, b) => a.created_at.localeCompare(b.created_at));
  let lastDay = "";
  return (
    <div className="relative pl-8">
      <div className="absolute bottom-2 left-[11px] top-2 w-px bg-border" />
      {sorted.map((f) => {
        const day = f.created_at.slice(0, 10);
        const header = day !== lastDay;
        lastDay = day;
        return (
          <div key={f.id}>
            {header && (
              <div className="relative -ml-8 mb-2 mt-4 flex items-center gap-2 first:mt-0">
                <span className="z-10 rounded-full border border-border bg-panel px-2 py-0.5 text-[11px] text-muted">
                  {new Date(day + "T00:00").toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" })}
                </span>
              </div>
            )}
            <div className="relative mb-3">
              <span className="absolute -left-[26px] top-4 h-3 w-3 rounded-full border-2 border-panel" style={{ background: VERDICT_COLOR[f.status] }} />
              <FindingCard f={f} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default function Findings() {
  const fs = useQuery({ queryKey: ["findings"], queryFn: api.findings });
  const [view, setView] = useState<"timeline" | "list">("timeline");
  const [status, setStatus] = useState("");
  const rows = useMemo(() => (fs.data ?? []).filter((f) => !status || f.status === status || (status === "disputed" && f.disputed)), [fs.data, status]);
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const f of fs.data ?? []) c[f.status] = (c[f.status] ?? 0) + 1;
    return c;
  }, [fs.data]);

  return (
    <div>
      <PageHeader title="Findings" sub="Every claim with its verdict and the runs behind it. Null and negative results stay here too.">
        <Select value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="">All statuses</option>
          {["replicated", "supported", "refuted", "inconclusive", "hypothesis", "disputed"].map((s) => <option key={s} value={s}>{s}{counts[s] ? ` (${counts[s]})` : ""}</option>)}
        </Select>
        <Tabs value={view} onChange={setView} options={[{ value: "timeline", label: "Timeline" }, { value: "list", label: "List" }]} />
      </PageHeader>
      <div className="mb-4 flex flex-wrap gap-3 text-xs text-muted">
        {["supported", "refuted", "inconclusive", "replicated"].map((s) => (
          <span key={s} className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-full" style={{ background: VERDICT_COLOR[s] }} />{s} {counts[s] ?? 0}</span>
        ))}
      </div>
      {!rows.length ? <Empty>{fs.isLoading ? "Loading…" : "No findings yet."}</Empty>
        : view === "timeline" ? <div className="max-w-3xl"><Timeline findings={rows} /></div>
        : <div className="grid gap-3 lg:grid-cols-2">{rows.map((f) => <FindingCard key={f.id} f={f} />)}</div>}
    </div>
  );
}
