import { Badge } from "./ui";
import type { ExpState, FindingStatus, Verdict } from "../lib/api";

const STATE_TONE: Record<string, "neutral" | "good" | "bad" | "warn" | "info" | "accent"> = {
  proposed: "warn", approved: "accent", queued: "accent", running: "info", analyzing: "info",
  complete: "good", failed: "bad", cancelled: "neutral", rejected: "neutral",
  pending: "neutral", completed: "good",
};

export function StateBadge({ state }: { state: ExpState | string }) {
  return (
    <Badge tone={STATE_TONE[state] ?? "neutral"}>
      {(state === "running" || state === "analyzing") && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />}
      {state}
    </Badge>
  );
}

export const VERDICT_COLOR: Record<string, string> = {
  supported: "var(--good)", refuted: "var(--bad)", inconclusive: "var(--warn)", replicated: "var(--accent)", hypothesis: "var(--muted)",
};

export function VerdictBadge({ verdict }: { verdict: Verdict | FindingStatus | null | undefined }) {
  if (!verdict) return <span className="text-xs text-muted">–</span>;
  const tone = verdict === "supported" ? "good" : verdict === "refuted" ? "bad" : verdict === "inconclusive" ? "warn" : verdict === "replicated" ? "accent" : "neutral";
  return <Badge tone={tone}>{verdict}</Badge>;
}
