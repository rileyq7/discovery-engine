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


def run_claude(prompt: str, timeout: int = 900, system: str | None = None) -> str:
    """Run `claude -p` with the prompt on stdin and return the text result."""
    check_billing()
    cmd = [claude_bin(), "-p", "--output-format", "json"]
    if system:
        cmd += ["--append-system-prompt", system]
    env = {k: v for k, v in os.environ.items() if k not in FORBIDDEN_ENV}
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode != 0:
        raise ClaudeError(f"claude -p exited {proc.returncode}: {(proc.stderr or proc.stdout)[-2000:]}")
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return proc.stdout
    if out.get("is_error"):
        raise ClaudeError(f"claude -p error: {str(out.get('result'))[:2000]}")
    return out.get("result", "")


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
