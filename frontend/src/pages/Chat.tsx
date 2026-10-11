import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Brain, ChevronRight, Globe, MessageSquarePlus, Search, Send, Square, Terminal, Trash2, FileText } from "lucide-react";
import { api, sendChat, type ChatMessage, type ChatTool } from "../lib/api";
import { cn, fmtTime } from "../lib/utils";
import { Markdown } from "../components/Markdown";
import { Button, Textarea } from "../components/ui";

const STARTERS = [
  "Explain the two-body problem to me like I know ML but not physics.",
  "What is accel_step, and why isn't it the same thing as force?",
  "Walk me through what the lab is running right now and why.",
  "Why would high eccentricity make orbits harder to learn and to probe?",
  "What did the reviewer flag, and which findings should I actually trust?",
  "The symbolic fits keep giving a²/r instead of a³/r². What could that mean physically?",
];

const TOOL_ICON: Record<string, typeof FileText> = { Read: FileText, Grep: Search, Glob: Search, Bash: Terminal, WebSearch: Globe, WebFetch: Globe };

function ToolChips({ tools }: { tools: ChatTool[] }) {
  if (!tools.length) return null;
  return (
    <div className="mb-2 flex flex-wrap gap-1">
      {tools.map((t) => {
        const Icon = TOOL_ICON[t.name] ?? Terminal;
        return (
          <span key={t.id} title={`${t.name}: ${t.summary}`}
            className={cn("inline-flex max-w-full items-center gap-1 rounded border border-border bg-panel-2 px-1.5 py-0.5 font-mono text-[11px] text-muted",
              t.error && "border-bad/40 text-bad")}>
            <Icon size={11} className="shrink-0" /><span className="truncate">{t.summary || t.name}</span>
          </span>
        );
      })}
    </div>
  );
}

function Thinking({ text, live }: { text: string; live?: boolean }) {
  const [open, setOpen] = useState(false);
  if (!text) return null;
  return (
    <div className="mb-2">
      <button onClick={() => setOpen((o) => !o)} className="flex cursor-pointer items-center gap-1 text-xs text-muted hover:text-fg">
        <ChevronRight size={12} className={cn("transition-transform", open && "rotate-90")} />
        <Brain size={12} /> {live ? "Thinking…" : "Thinking"}
      </button>
      {open && <div className="mt-1 whitespace-pre-wrap border-l-2 border-border pl-3 text-xs text-muted">{text}</div>}
    </div>
  );
}

function Bubble({ m, live }: { m: ChatMessage; live?: boolean }) {
  if (m.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-lg bg-accent/15 px-3 py-2 text-sm">
          {m.content}
          {m.web && <div className="mt-1 flex items-center gap-1 text-[11px] text-muted"><Globe size={11} /> web search allowed</div>}
        </div>
      </div>
    );
  }
  return (
    <div className="max-w-full">
      <Thinking text={m.thinking ?? ""} live={live && !m.content} />
      <ToolChips tools={m.tools ?? []} />
      {m.content ? <Markdown>{m.content}</Markdown> : live ? <div className="text-sm text-muted">Looking into it…</div> : null}
      {m.error && <div className="mt-1 text-xs text-bad">{m.error}</div>}
    </div>
  );
}

function readAsk(): string | null {
  const q = window.location.hash.split("?")[1];
  return q ? new URLSearchParams(q).get("ask") : null;
}

export default function Chat({ threadId }: { threadId?: string }) {
  const qc = useQueryClient();
  const threads = useQuery({ queryKey: ["chat-threads"], queryFn: api.chatThreads });
  const thread = useQuery({ queryKey: ["chat-thread", threadId], queryFn: () => api.chatThread(threadId!), enabled: !!threadId });
  const [text, setText] = useState("");
  const [web, setWeb] = useState(false);
  const [draft, setDraft] = useState<ChatMessage | null>(null);
  const [pending, setPending] = useState<ChatMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const busy = draft !== null;

  // "Ask about this" links arrive as #/chat?ask=...: prefill the composer of a fresh chat
  useEffect(() => {
    const ask = readAsk();
    if (ask) {
      setText(ask);
      window.history.replaceState(null, "", "#/chat");
    }
  }, []);
  useEffect(() => { bottom.current?.scrollIntoView({ block: "end" }); }, [thread.data?.messages.length, draft?.content, pending]);

  const send = async (msg: string) => {
    const body = msg.trim();
    if (!body || busy) return;
    setError(null);
    let tid = threadId;
    if (!tid) {
      tid = (await api.newChat()).id;
      window.location.hash = `#/chat/${tid}`;
      qc.invalidateQueries({ queryKey: ["chat-threads"] });
    }
    setText("");
    setPending({ role: "user", content: body, at: new Date().toISOString(), web });
    const live: ChatMessage = { role: "assistant", content: "", thinking: "", tools: [], at: new Date().toISOString() };
    setDraft({ ...live });
    abort.current = new AbortController();
    try {
      await sendChat(tid, { text: body, web }, (e) => {
        if (e.type === "text") live.content += e.delta;
        else if (e.type === "thinking") live.thinking = (live.thinking ?? "") + e.delta;
        else if (e.type === "tool") live.tools = [...(live.tools ?? []), { id: e.id, name: e.name, summary: e.summary }];
        else if (e.type === "tool_result") live.tools = (live.tools ?? []).map((t) => (t.id === e.id ? { ...t, error: e.error } : t));
        else if (e.type === "error") setError(e.message);
        setDraft({ ...live });
      }, abort.current.signal);
    } catch (err) {
      if ((err as Error).name !== "AbortError") setError(String((err as Error).message ?? err));
    } finally {
      await qc.invalidateQueries({ queryKey: ["chat-thread", tid] });
      await qc.invalidateQueries({ queryKey: ["chat-threads"] });
      setDraft(null);
      setPending(null);
      abort.current = null;
    }
  };

  const messages = thread.data?.messages ?? [];
  const shown = pending && !messages.some((m) => m.role === "user" && m.content === pending.content && m.at >= pending.at.slice(0, 19))
    ? [...messages, pending] : messages;

  return (
    <div className="-my-5 flex h-[100dvh] gap-4">
      <aside className="hidden w-60 shrink-0 flex-col border-r border-border py-5 pr-3 md:flex">
        <Button size="sm" variant="outline" onClick={() => { window.location.hash = "#/chat"; setText(""); }}>
          <MessageSquarePlus size={14} /> New chat
        </Button>
        <div className="mt-3 flex-1 space-y-0.5 overflow-y-auto">
          {(threads.data ?? []).map((t) => (
            <div key={t.id} className={cn("group flex items-center rounded-md", t.id === threadId ? "bg-panel-2" : "hover:bg-panel-2")}>
              <a href={`#/chat/${t.id}`} className="min-w-0 flex-1 px-2 py-1.5">
                <div className="truncate text-sm">{t.title}</div>
                <div className="text-[11px] text-muted">{fmtTime(t.updated_at)}</div>
              </a>
              <button title="Delete chat" className="mr-1 hidden cursor-pointer p-1 text-muted hover:text-bad group-hover:block"
                onClick={async () => {
                  await api.deleteChat(t.id);
                  await qc.invalidateQueries({ queryKey: ["chat-threads"] });
                  if (t.id === threadId) window.location.hash = "#/chat";
                }}><Trash2 size={13} /></button>
            </div>
          ))}
          {!threads.data?.length && <p className="px-2 text-xs text-muted">No chats yet.</p>}
        </div>
      </aside>

      <section className="flex min-w-0 flex-1 flex-col py-5">
        <div className="mb-3">
          <h1 className="text-lg font-semibold">{thread.data?.title ?? "Ask the lab"}</h1>
          <p className="text-xs text-muted">
            A tutor that reads the whole project, the live lab state and the agents' reasoning, and explains the physics from an ML starting point. Read-only.
          </p>
        </div>

        <div className="min-h-0 flex-1 space-y-5 overflow-y-auto pr-1">
          {!shown.length && !draft && (
            <div className="grid gap-2 sm:grid-cols-2">
              {STARTERS.map((s) => (
                <button key={s} onClick={() => send(s)}
                  className="cursor-pointer rounded-lg border border-border bg-panel px-3 py-2.5 text-left text-sm hover:border-accent/60">{s}</button>
              ))}
            </div>
          )}
          {shown.map((m, i) => <Bubble key={i} m={m} />)}
          {draft && <Bubble m={draft} live />}
          {error && <div className="text-sm text-bad">{error}</div>}
          <div ref={bottom} />
        </div>

        <form className="mt-3 rounded-lg border border-border bg-panel p-2" onSubmit={(e) => { e.preventDefault(); send(text); }}>
          <Textarea value={text} onChange={(e) => setText(e.target.value)} rows={2} placeholder="Ask about the experiments, the physics, a run or a finding…"
            className="resize-none border-0 bg-transparent focus-visible:ring-0"
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(text); } }} />
          <div className="flex items-center justify-between gap-2 px-1">
            <label className="flex cursor-pointer items-center gap-1.5 text-xs text-muted" title="Lets the tutor search the web for this message; queries leave your machine">
              <input type="checkbox" checked={web} onChange={(e) => setWeb(e.target.checked)} /> <Globe size={12} /> allow web search
            </label>
            {busy
              ? <Button type="button" size="sm" variant="outline" onClick={() => abort.current?.abort()}><Square size={12} /> Stop</Button>
              : <Button type="submit" size="sm" disabled={!text.trim()}><Send size={13} /> Send</Button>}
          </div>
        </form>
      </section>
    </div>
  );
}
