import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Activity, BookOpen, Database, FlaskConical, Lightbulb, LayoutDashboard, Menu, MessageCircle, Moon, Sparkles, Sun } from "lucide-react";
import { api, snapshotMeta, STATIC } from "./lib/api";
import { useHashRoute, useLiveUpdates } from "./lib/live";
import { cn } from "./lib/utils";
import Dashboard from "./pages/Dashboard";
import Experiments from "./pages/Experiments";
import RunDetailPage from "./pages/RunDetail";
import Findings from "./pages/Findings";
import Papers from "./pages/Papers";
import Data from "./pages/Data";
import Chat from "./pages/Chat";
import Highlights from "./pages/Highlights";

const NAV = [
  { path: "", label: "Dashboard", icon: LayoutDashboard },
  { path: "highlights", label: "Highlights", icon: Sparkles },
  { path: "chat", label: "Ask the lab", icon: MessageCircle },
  { path: "experiments", label: "Experiments", icon: FlaskConical },
  { path: "runs", label: "Run detail", icon: Activity },
  { path: "findings", label: "Findings", icon: Lightbulb },
  { path: "papers", label: "Papers", icon: BookOpen },
  { path: "data", label: "Data", icon: Database },
];

export default function App() {
  const route = useHashRoute();
  const connected = useLiveUpdates();
  const [dark, setDark] = useState(() => {
    try { return localStorage.getItem("de-theme") !== "light"; } catch { return true; }
  });
  const [navOpen, setNavOpen] = useState(false);
  const loop = useQuery({ queryKey: ["loop"], queryFn: api.loop, refetchInterval: 15_000 });
  const pending = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  const meta = useQuery({ queryKey: ["snapshot-meta"], queryFn: snapshotMeta, enabled: STATIC, refetchInterval: 60_000 });
  const nProposed = pending.data?.filter((e) => e.state === "proposed").length ?? 0;

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    try { localStorage.setItem("de-theme", dark ? "dark" : "light"); } catch { /* storage unavailable */ }
  }, [dark]);
  useEffect(() => setNavOpen(false), [route.join("/")]);

  const section = route[0] ?? "";
  let page;
  switch (section) {
    case "experiments": page = <Experiments />; break;
    case "runs": page = <RunDetailPage runId={route[1]} />; break;
    case "findings": page = <Findings />; break;
    case "papers": page = <Papers paperId={route[1]} />; break;
    case "data": page = <Data />; break;
    case "highlights": page = <Highlights />; break;
    case "chat": page = <Chat threadId={route[1]} />; break;
    default: page = <Dashboard />;
  }

  return (
    <div className="flex h-full">
      <aside className={cn(
        "fixed inset-y-0 left-0 z-40 flex w-56 flex-col border-r border-border bg-panel transition-transform md:static md:translate-x-0",
        navOpen ? "translate-x-0" : "-translate-x-full",
      )}>
        <div className="flex items-center gap-2 px-4 py-4">
          <svg viewBox="0 0 32 32" className="h-6 w-6"><circle cx="16" cy="16" r="4" fill="#f5b942" /><ellipse cx="16" cy="16" rx="13" ry="6" fill="none" stroke="#7c9cff" strokeWidth="2" transform="rotate(-20 16 16)" /></svg>
          <div>
            <div className="text-sm font-semibold leading-tight">Discovery Engine</div>
            <div className="text-[11px] text-muted">orbits → laws</div>
          </div>
        </div>
        <nav className="flex-1 space-y-0.5 px-2">
          {NAV.filter((n) => !(STATIC && n.path === "chat")).map(({ path, label, icon: Icon }) => (
            <a key={path} href={`#/${path}`}
              className={cn("flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm",
                section === path ? "bg-panel-2 font-medium text-fg" : "text-muted hover:bg-panel-2 hover:text-fg")}>
              <Icon size={16} />
              <span className="flex-1">{label}</span>
              {path === "experiments" && nProposed > 0 && (
                <span className="rounded-full bg-warn/20 px-1.5 text-[10px] font-semibold text-warn">{nProposed}</span>
              )}
            </a>
          ))}
        </nav>
        <div className="space-y-2 border-t border-border p-3 text-xs">
          <div className="flex items-center gap-2">
            <span className={cn("h-2 w-2 rounded-full", loop.data?.paused ? "bg-warn" : "bg-good animate-pulse")} />
            <span>Loop {loop.data ? (loop.data.paused ? "paused" : "running") : "…"}</span>
          </div>
          <div className="flex items-center gap-2 text-muted">
            <span className={cn("h-2 w-2 rounded-full", loop.data?.worker.online ? "bg-good" : "bg-muted")} />
            Worker {loop.data?.worker.online ? "online" : "offline"}
          </div>
          {STATIC ? (
            <div className="text-muted" title="Rebuilt from the repo after every agent cycle">
              Snapshot {meta.data ? new Date(meta.data.generated_at).toLocaleString([], { dateStyle: "short", timeStyle: "short" }) : "…"}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-muted">
              <span className={cn("h-2 w-2 rounded-full", connected ? "bg-good" : "bg-bad")} />
              {connected ? "Live" : "Reconnecting…"}
            </div>
          )}
          <button className="flex items-center gap-1.5 text-muted hover:text-fg cursor-pointer" onClick={() => setDark((d) => !d)}>
            {dark ? <Sun size={13} /> : <Moon size={13} />} {dark ? "Light mode" : "Dark mode"}
          </button>
        </div>
      </aside>
      {navOpen && <div className="fixed inset-0 z-30 bg-black/50 md:hidden" onClick={() => setNavOpen(false)} />}
      <main className="min-w-0 flex-1 overflow-y-auto">
        <div className="sticky top-0 z-20 flex items-center gap-2 border-b border-border bg-panel px-4 py-2 md:hidden">
          <button onClick={() => setNavOpen(true)} aria-label="Open navigation" className="cursor-pointer"><Menu size={18} /></button>
          <span className="text-sm font-semibold">Discovery Engine</span>
        </div>
        {STATIC && (
          <div className="border-b border-border bg-panel-2 px-4 py-1.5 text-center text-xs text-muted md:px-6">
            Read-only snapshot of the lab, updated after each hourly agent cycle. Approving, pausing and editing
            happen in the local app.
          </div>
        )}
        <div className="mx-auto max-w-[1400px] px-4 py-5 md:px-6">{page}</div>
      </main>
    </div>
  );
}

export function PageHeader({ title, sub, children }: { title: string; sub?: React.ReactNode; children?: React.ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {sub && <p className="mt-0.5 text-sm text-muted">{sub}</p>}
      </div>
      <div className="flex flex-wrap items-center gap-2">{children}</div>
    </div>
  );
}
