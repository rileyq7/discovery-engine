"""The only way the system talks to a model: headless Claude Code (`claude -p`) on the Max subscription.

There is no API key anywhere. If ANTHROPIC_API_KEY (or CLAUDE_CODE_OAUTH_TOKEN) is set, Claude Code
would bill differently, so every call refuses to run.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time

FORBIDDEN_ENV = ("ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN")


class BillingError(RuntimeError):
    pass


class ClaudeError(RuntimeError):
    pass


def check_billing() -> None:
    bad = [k for k in FORBIDDEN_ENV if os.environ.get(k)]
    if bad:
        raise BillingError(f"refusing to run: {', '.join(bad)} is set. Discovery Engine runs on the Max "
                           f"subscription only; unset it (and remove it from .env / CI / the scheduled task).")


def claude_bin() -> str:
    b = os.environ.get("DE_CLAUDE_BIN") or shutil.which("claude")
    if not b:
        raise ClaudeError("`claude` CLI not found on PATH; install Claude Code and log in with the Max account")
    return b


def clean_env() -> dict:
    return {k: v for k, v in os.environ.items() if k not in FORBIDDEN_ENV}


def run_claude(prompt: str, timeout: int = 900, system: str | None = None, no_tools: bool = False,
               role: str = "agent") -> str:
    """Run `claude -p` with the prompt on stdin and return the text result. `no_tools` gives a fresh,
    tool-less session (no files, shell or MCP servers) that can only reason over the prompt.

    Output is read as stream-json so any thinking blocks are kept. Every call is saved to
    lab/transcripts/<date>.jsonl (reply, thinking, timing), the record of what the agents were thinking."""
    check_billing()
    cmd = [claude_bin(), "-p", "--output-format", "stream-json", "--verbose"]
    if no_tools:
        cmd += ["--tools", "", "--strict-mcp-config", "--no-session-persistence"]
    if system:
        cmd += ["--append-system-prompt", system]
    t0 = time.time()
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, env=clean_env())
    if proc.returncode != 0:
        raise ClaudeError(f"claude -p exited {proc.returncode}: {(proc.stderr or proc.stdout)[-2000:]}")
    result, thinking, final = "", [], {}
    for line in proc.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "assistant":
            thinking += [b.get("thinking", "") for b in ev["message"].get("content", []) if b.get("type") == "thinking"]
        elif ev.get("type") == "result":
            final = ev
            result = ev.get("result", "") or ""
    if not final:  # not stream-json (e.g. an older CLI): treat stdout as the reply
        result = proc.stdout
    if final.get("is_error"):
        raise ClaudeError(f"claude -p error: {str(final.get('result'))[:2000]}")
    _transcript(role, prompt, result, [t for t in thinking if t], final, time.time() - t0)
    return result


def _transcript(role: str, prompt: str, reply: str, thinking: list[str], final: dict, seconds: float) -> None:
    try:
        from engine.paths import lab_dir

        d = lab_dir() / "transcripts"
        d.mkdir(parents=True, exist_ok=True)
        task = next((l.strip("# ").strip() for l in prompt.splitlines() if l.strip()), "")[:200]
        rec = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "role": role, "task": task, "prompt_chars": len(prompt),
               "reply": reply, "thinking": thinking, "seconds": round(seconds, 1),
               "model": next(iter(final.get("modelUsage") or {}), None)}
        with open(d / f"{time.strftime('%Y-%m-%d')}.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001 - a transcript must never break a cycle
        pass


def extract_json(text: str):
    """Pull the first JSON object or array out of a model response (fenced or bare)."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidates = [m.group(1)] if m else []
    candidates.append(text)
    for c in candidates:
        c = c.strip()
        for opener, closer in (("{", "}"), ("[", "]")):
            i, j = c.find(opener), c.rfind(closer)
            if i != -1 and j > i:
                try:
                    return json.loads(c[i : j + 1])
                except json.JSONDecodeError:
                    continue
    raise ValueError("no JSON found in model output")
