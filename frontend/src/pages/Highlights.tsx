import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { cn, fmtTime } from "../lib/utils";
import { PageHeader } from "../App";
import { Markdown } from "../components/Markdown";
import { Card, CardContent, Empty } from "../components/ui";

export default function Highlights() {
  const q = useQuery({ queryKey: ["digests"], queryFn: api.digests });
  const [date, setDate] = useState<string | null>(null);
  const list = q.data ?? [];
  const d = list.find((x) => x.date === date) ?? list[0];
  return (
    <div>
      <PageHeader title="Highlights" sub="A daily highlight reel of what the lab did, why, and what it means, written each morning." />
      {!list.length ? (
        <Empty>{q.isLoading ? "Loading…" : "No highlight reels yet. The research loop writes one each morning at 07:00 (settings.digest_hour)."}</Empty>
      ) : (
        <div className="flex flex-col gap-4 lg:flex-row">
          <nav className="flex shrink-0 gap-1 overflow-x-auto lg:w-44 lg:flex-col">
            {list.map((x) => (
              <button key={x.date} onClick={() => setDate(x.date)}
                className={cn("cursor-pointer rounded-md px-3 py-2 text-left text-sm", x.date === d?.date ? "bg-panel-2 font-medium" : "text-muted hover:bg-panel-2")}>
                {new Date(`${x.date}T12:00:00`).toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" })}
                <div className="text-[11px] text-muted">{x.counts.findings} findings · {x.counts.runs} runs</div>
              </button>
            ))}
          </nav>
          {d && (
            <Card className="min-w-0 flex-1">
              <CardContent className="py-5">
                <div className="mb-3 text-xs text-muted">Covers {fmtTime(d.since)} → {fmtTime(d.until)}</div>
                <Markdown>{d.markdown}</Markdown>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
