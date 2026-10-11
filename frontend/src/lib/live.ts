import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { STATIC, type Metric } from "./api";

/** Subscribe to the backend's Server-Sent Events stream and keep TanStack Query caches fresh. */
export function useLiveUpdates(): boolean {
  const qc = useQueryClient();
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (STATIC) return; // the hosted snapshot has no backend to stream from
    let es: EventSource | null = null;
    let retry: ReturnType<typeof setTimeout> | undefined;
    let pending = new Set<string>();
    let flush: ReturnType<typeof setTimeout> | undefined;

    const invalidate = (keys: string[]) => {
      keys.forEach((k) => pending.add(k));
      if (flush) return;
      flush = setTimeout(() => {
        const batch = pending;
        pending = new Set();
        flush = undefined;
        batch.forEach((k) => qc.invalidateQueries({ queryKey: [k] }));
      }, 250);
    };

    const connect = () => {
      es = new EventSource("/api/stream");
      es.addEventListener("hello", () => setConnected(true));
      es.addEventListener("change", (ev) => {
        const e = JSON.parse((ev as MessageEvent).data) as { type: string; payload: Record<string, string> };
        const t = e.type.split(".")[0];
        const keys = ["dashboard", "loop"];
        if (t === "experiment") keys.push("experiments", "experiment");
        if (t === "run") keys.push("runs", "run", "experiments", "experiment");
        if (t === "finding") keys.push("findings", "experiments", "paper");
        if (t === "question") keys.push("questions");
        if (t === "review" || t === "loop") keys.push("reviews", "loop");
        if (t === "paper") keys.push("paper", "papers");
        if (t === "dataset") keys.push("datasets");
        if (t === "cycle" || t === "settings") keys.push("loop");
        invalidate(keys);
      });
      es.addEventListener("metrics", (ev) => {
        const { run_id, rows } = JSON.parse((ev as MessageEvent).data) as { run_id: string; rows: Metric[] };
        qc.setQueryData<Metric[]>(["metrics", run_id], (old) => (old ? [...old, ...rows] : old));
      });
      es.onerror = () => {
        setConnected(false);
        es?.close();
        retry = setTimeout(connect, 3000);
      };
    };
    connect();
    return () => {
      es?.close();
      if (retry) clearTimeout(retry);
      if (flush) clearTimeout(flush);
    };
  }, [qc]);

  return connected;
}

export function useHashRoute(): string[] {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const on = () => setHash(window.location.hash);
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const path = hash.replace(/^#\/?/, "").split("?")[0];
  return path ? path.split("/") : [];
}
