"""Lab tutor chat: a read-only `claude -p` session per thread, streamed to the UI.

Each thread keeps its Claude Code session id, so follow-ups resume the same conversation. Every message
carries a live <lab_state> snapshot. The tutor's tools are Read/Grep/Glob inside the repo plus the
read-only `scripts/lab` command. It can't edit anything, and only project settings are loaded, so user-level
permission rules don't leak in. Web search is off unless the user enables it for a message. Threads are
stored locally in runs/chats/ (gitignored): they are personal, not lab records.
"""
from __future__ import annotations

import json
import secrets
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

from agent.claude_cli import check_billing, claude_bin, clean_env
from app import store
from engine.paths import REPO_ROOT, runs_dir

TUTOR_PROMPT = Path(__file__).resolve().parents[1] / "agent" / "prompts" / "tutor.md"
TOOLS = ["Read", "Grep", "Glob", "Bash"]
ALLOWED = ["Read", "Grep", "Glob", "Bash(scripts/lab:*)"]
WEB = ["WebSearch", "WebFetch"]
TIMEOUT_S = 900
_locks: dict[str, threading.Lock] = {}


# ------------------------------------------------------------------ threads

def _dir() -> Path:
    d = runs_dir() / "chats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(tid: str) -> Path:
    if not tid.startswith("c-") or "/" in tid:
        raise KeyError(tid)
    return _dir() / f"{tid}.json"


def list_threads() -> list[dict]:
    out = []
    for p in _dir().glob("c-*.json"):
        t = json.loads(p.read_text())
        out.append({k: t[k] for k in ("id", "title", "created_at", "updated_at")} | {"n_messages": len(t["messages"])})
    return sorted(out, key=lambda t: t["updated_at"], reverse=True)


def get_thread(tid: str) -> dict:
    p = _path(tid)
    if not p.exists():
        raise KeyError(tid)
    return json.loads(p.read_text())


def create_thread(title: str = "") -> dict:
    t = {"id": f"c-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}", "title": title.strip()[:120] or "New chat",
         "created_at": store.now(), "updated_at": store.now(), "session_id": None, "messages": []}
    _save(t)
    return t


def delete_thread(tid: str) -> None:
    _path(tid).unlink(missing_ok=True)


def _save(t: dict) -> None:
    t["updated_at"] = store.now()
    store._write_json(_path(t["id"]), t)


# ------------------------------------------------------------------ live context

def lab_state(page: Optional[str] = None) -> dict:
    """What is happening in the lab right now, compactly."""
    from agent import review
    from agent.tools import status
    from app.api.loop import worker_status

    st = status()
    pf = store.lab_dir() / "PAUSED"
    running = []
    for r in store.list_runs("running"):
        e = store.get_experiment(r["experiment_id"])
        ms = store.get_metrics(r["id"], ["val_mse_ratio"])
        tl = store.get_metrics(r["id"], ["train_loss"])
        running.append({"run_id": r["id"], "experiment_id": e["id"], "family": e["family"], "seed": r["seed"],
                        "step": tl[-1]["step"] if tl else 0, "of_steps": e["config"]["training"]["steps"],
                        "latest_val_mse_ratio": round(ms[-1]["value"], 4) if ms else None,
                        "progress": r.get("progress")})
    exps = store.list_experiments()
    waiting = [{"id": e["id"], "family": e["family"], "state": e["state"], "hypothesis": e["hypothesis"][:400]}
               for e in exps if e["state"] in ("proposed", "approved", "queued", "running", "analyzing")]
    events = store.list_events(after_id=0, limit=100000, types=["cycle.finished", "cycle.failed"])
    last_cycle = events[-1] if events else None
    reviews = review.list_reviews()
    return {
        "now": store.now(),
        "loop": {"paused": st["paused"], "paused_note": pf.read_text().strip() if pf.exists() else None,
                 "cycles_today": st["cycles_today"], "compute_used_today_h": st["compute_used_today"],
                 "daily_budget_h": st["daily_budget_hours"], "worker_online": worker_status()["online"],
                 "last_cycle": {"at": last_cycle["timestamp"], "summary": str(last_cycle["payload"].get("summary"))[:800]}
                 if last_cycle else None},
        "running_runs": running,
        "in_flight_experiments": waiting,
        "recent_findings": [{"id": f["id"], "status": f["status"], "verdict": f["verdict"], "claim": f["claim"][:300]}
                            for f in store.list_findings()[-6:]],
        "reviewer": {"latest": {"id": reviews[-1]["id"], "verdict": reviews[-1]["verdict"],
                                "summary": reviews[-1]["summary"][:600]} if reviews else None,
                     "open_issues": len(review.open_issues())},
        "counts": {"experiments": len(exps), "findings": len(store.list_findings())},
        "user_is_viewing": page or None,
    }


def _prompt(t: dict, text: str, page: Optional[str], recap: bool) -> str:
    parts = ["<lab_state>", json.dumps(lab_state(page), indent=1, default=str), "</lab_state>"]
    if not t["messages"] or recap:
        others = [x for x in list_threads() if x["id"] != t["id"]][:12]
        if others:
            parts += ["<earlier_chats>", "Topics from the user's earlier chats (they may build on them): "
                      + "; ".join(x["title"] for x in others), "</earlier_chats>"]
    if recap and t["messages"]:  # the Claude session was lost: replay the conversation so far
        hist = "\n\n".join(f"{m['role'].upper()}: {m['content'][:3000]}" for m in t["messages"][-12:])
        parts += ["<conversation_so_far>", hist, "</conversation_so_far>"]
    parts.append(text)
    return "\n".join(parts)


# ------------------------------------------------------------------ one turn, streamed

def _cmd(resume: Optional[str], web: bool) -> list[str]:
    cmd = [claude_bin(), "-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
           "--append-system-prompt", TUTOR_PROMPT.read_text(), "--setting-sources", "project",
           "--strict-mcp-config", "--tools", ",".join(TOOLS + (WEB if web else [])),
           "--allowedTools", " ".join(ALLOWED + (WEB if web else []))]
    if resume:
        cmd += ["--resume", resume]
    return cmd


def _tool_summary(name: str, inp: dict) -> str:
    if name == "Bash":
        return inp.get("command", "")[:200]
    if name in ("Read",):
        return str(inp.get("file_path", "")).replace(str(REPO_ROOT) + "/", "")
    if name in ("Grep", "Glob"):
        return f"{inp.get('pattern', '')} {inp.get('path', '') or ''}".strip()[:200]
    if name in ("WebSearch",):
        return inp.get("query", "")[:200]
    if name == "WebFetch":
        return inp.get("url", "")[:200]
    return json.dumps(inp)[:200]


def _run(t: dict, prompt: str, resume: Optional[str], web: bool, out: dict) -> Iterator[dict]:
    """Run one claude -p turn and yield UI events. Fills `out` with text, thinking, tools, session_id, error."""
    proc = subprocess.Popen(_cmd(resume, web), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, cwd=REPO_ROOT, env=clean_env(), bufsize=1)
    killer = threading.Timer(TIMEOUT_S, proc.kill)
    killer.start()
    try:
        proc.stdin.write(prompt)
        proc.stdin.close()
        open_blocks: dict[int, str] = {}
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            typ = ev.get("type")
            if typ == "system" and ev.get("subtype") == "init":
                out["session_id"] = ev.get("session_id")
            elif typ == "stream_event":
                e = ev["event"]
                if e["type"] == "content_block_start":
                    kind = e["content_block"]["type"]
                    open_blocks[e["index"]] = kind
                    if kind == "text" and out["text"] and not out["text"].endswith("\n\n"):
                        out["text"] += "\n\n"
                        yield {"type": "text", "delta": "\n\n"}
                elif e["type"] == "content_block_delta":
                    d = e["delta"]
                    if d.get("type") == "text_delta":
                        out["text"] += d["text"]
                        yield {"type": "text", "delta": d["text"]}
                    elif d.get("type") == "thinking_delta":
                        out["thinking"] += d["thinking"]
                        yield {"type": "thinking", "delta": d["thinking"]}
            elif typ == "assistant":
                for b in ev["message"].get("content", []):
                    if b.get("type") == "tool_use":
                        tool = {"id": b["id"], "name": b["name"], "summary": _tool_summary(b["name"], b.get("input") or {})}
                        out["tools"].append(tool)
                        yield {"type": "tool", **tool}
            elif typ == "user":
                for b in ev["message"].get("content", []):
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        for tool in out["tools"]:
                            if tool["id"] == b.get("tool_use_id"):
                                tool["error"] = bool(b.get("is_error"))
                        yield {"type": "tool_result", "id": b.get("tool_use_id"), "error": bool(b.get("is_error"))}
            elif typ == "result":
                out["session_id"] = ev.get("session_id") or out.get("session_id")
                if ev.get("is_error"):
                    out["error"] = str(ev.get("result") or ev.get("subtype"))[:1000]
        proc.wait(timeout=30)
        if proc.returncode and not out.get("error"):
            out["error"] = (proc.stderr.read() or f"claude exited {proc.returncode}")[-1000:]
    finally:
        killer.cancel()
        if proc.poll() is None:
            proc.kill()


def stream_reply(tid: str, text: str, web: bool = False, page: Optional[str] = None) -> Iterator[str]:
    """Server-sent events for one user message: text/thinking deltas, tool calls, then done or error."""
    lock = _locks.setdefault(tid, threading.Lock())
    if not lock.acquire(blocking=False):
        yield _sse({"type": "error", "message": "this chat is already answering; wait for it to finish"})
        return
    try:
        check_billing()
        t = get_thread(tid)
        if not t["messages"] and t["title"] == "New chat":
            t["title"] = text.strip().splitlines()[0][:80]
        t["messages"].append({"role": "user", "content": text, "at": store.now(), "web": web, "page": page})
        _save(t)
        out = {"text": "", "thinking": "", "tools": [], "session_id": None, "error": None}
        resume = t.get("session_id")
        for ev in _run(t, _prompt(t, text, page, recap=False), resume, web, out):
            yield _sse(ev)
        if out["error"] and resume and not out["text"]:  # stale session: start fresh with a recap
            out = {"text": "", "thinking": "", "tools": [], "session_id": None, "error": None}
            t["messages"].pop()
            for ev in _run(t, _prompt(t, text, page, recap=True), None, web, out):
                yield _sse(ev)
            t["messages"].append({"role": "user", "content": text, "at": store.now(), "web": web, "page": page})
        t = {**get_thread(tid), "messages": t["messages"], "title": t["title"]}
        t["session_id"] = out["session_id"] or t.get("session_id")
        t["messages"].append({"role": "assistant", "content": out["text"], "thinking": out["thinking"],
                              "tools": out["tools"], "error": out["error"], "at": store.now()})
        _save(t)
        yield _sse({"type": "error", "message": out["error"]} if out["error"] and not out["text"]
                   else {"type": "done", "title": t["title"]})
    except Exception as e:  # noqa: BLE001
        yield _sse({"type": "error", "message": str(e)[:500]})
    finally:
        lock.release()


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"
