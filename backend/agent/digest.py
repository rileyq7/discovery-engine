"""Daily highlight reel: what the lab did since the last digest, written for the owner over breakfast.

Generated once a day (at `digest_hour`, default 07:00 local) by the research loop, even while the lab is
paused. Facts are gathered here from lab/ files; a tool-less `claude -p` session writes them up in the
tutor's voice (ML-fluent reader, physics explained through the night's results) and must cite IDs.
Saved as lab/digests/<date>.md (+ .json metadata) and shown on the Highlights page.

    python -m agent.digest            # write today's digest now (if missing)
    python -m agent.digest --preview  # print a digest without saving it
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Optional

from app import store
from engine.paths import lab_dir

DEFAULT_HOUR = 7
FIRST_WINDOW_H = 12  # the first digest covers the last 12 hours

PROMPT = """Write the lab owner's morning highlight reel for Discovery Engine, covering {since} to {until}.

The reader is fluent in ML (training, loss, MSE, probes) but shaky on physics, and wants to learn it as the
experiments happen. Write in Markdown, about 600-900 words, with these sections:

# Highlights: {date}
**TL;DR**: 3 bullets. The most important things, with numbers.
## What the lab did and why
Experiments run or proposed, and the reasoning behind them (use the agent's reasoning excerpts). Group them
by question rather than listing every ID.
## Findings
New or changed findings: status (replicated / supported / refuted / inconclusive), what was measured, and
what it means in plain words. Say clearly which results are solid and which are shaky (reviewer flags,
single seeds, early stops).
## The reviewer
Verdicts, the issues that matter, and how the research agent answered them.
## Needs you
Approvals waiting, pauses, failed runs, or anything odd. Write "Nothing" if there is nothing. Problems
already fixed by a pipeline change (see `pipeline_changes_since`) are not open; mention them only briefly,
as fixed, if they shaped the night's results.
## Physics concept of the day
One concept that last night's results hinge on. Build the intuition from ML first, then the physics,
then tie it back to a specific number from the night. Use $...$ math sparingly.
## What's next
1-3 bullets on what the lab plans or should try.

Rules: use only the facts below; never invent numbers or IDs; take every count from `counts_use_these_exactly` rather than counting yourself; cite IDs inline (`f-...`, `exp-...`, `run-...`)
so they can be clicked. If little happened, keep it short and say so. Reply with only the Markdown.

## Facts (JSON)
```json
{facts}
```
"""


def digests_dir() -> Path:
    return lab_dir() / "digests"


def list_digests() -> list[dict]:
    d = digests_dir()
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json"), reverse=True):
        meta = json.loads(p.read_text())
        md = p.with_suffix(".md")
        out.append({**meta, "markdown": md.read_text() if md.exists() else ""})
    return out


def _since() -> str:
    prev = list_digests()
    if prev:
        return prev[0]["until"]
    return (dt.datetime.now() - dt.timedelta(hours=FIRST_WINDOW_H)).isoformat(timespec="seconds")


def _events(since: str) -> list[dict]:
    out = []
    for f in sorted((lab_dir() / "events").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                if e["timestamp"] > since:
                    out.append(e)
    return out


def _transcripts(since: str, roles: tuple[str, ...]) -> list[dict]:
    out = []
    for f in sorted((lab_dir() / "transcripts").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["at"] > since and r["role"].startswith(roles):
                    out.append(r)
    return out


def facts(since: str, until: str) -> dict:
    from agent import review

    store.rebuild_index()
    ev = _events(since)
    changed = {e["payload"].get("id") for e in ev if e["type"].startswith("finding.")}
    findings = [f for f in store.list_findings() if f["created_at"] > since or f["id"] in changed]
    exps = [e for e in store.list_experiments() if e["created_at"] > since or e["updated_at"] > since]
    runs = [r for r in store.list_runs() if (r.get("ended_at") or "") > since]
    reviews = [r for r in review.list_reviews() if r["created_at"] > since or
               any(x["at"] > since for x in r.get("responses", []))]
    reasoning = []
    for r in _transcripts(since, ("research.propose",)):
        try:
            from agent.claude_cli import extract_json

            reasoning.append({"at": r["at"], "reasoning": (extract_json(r["reply"]) or {}).get("reasoning", "")[:1200],
                              "thinking_excerpt": (r.get("thinking") or [""])[0][:1500]})
        except Exception:  # noqa: BLE001
            continue
    st = store.get_settings()
    cyc = [e for e in ev if e["type"] == "cycle.finished"]
    return {
        "window": {"since": since, "until": until},
        "counts_use_these_exactly": {
            "cycles_finished": len(cyc), "runs_finished": len(runs),
            "runs_by_status": {s: sum(r["status"] == s for r in runs) for s in sorted({r["status"] for r in runs})},
            "new_findings": sum(f["created_at"] > since for f in findings),
            "reviews": len(reviews),
            "reviews_by_verdict": {v: sum(r["verdict"] == v for r in reviews) for v in sorted({r["verdict"] for r in reviews})}},
        "pipeline_changes_since (code commits: bugs fixed, features added)": _code_commits(since),
        "cycles": [{"at": e["timestamp"], "type": e["type"], "summary": str(e["payload"].get("summary"))[:600]}
                   for e in ev if e["type"] in ("cycle.finished", "cycle.failed", "cycle.skipped")],
        "pauses": [{"at": e["timestamp"], "type": e["type"], "by": e["actor"], "reason": e["payload"].get("reason")}
                   for e in ev if e["type"] in ("loop.paused", "loop.started", "loop.killed")],
        "agent_reasoning_when_proposing": reasoning,
        "experiments": [{"id": e["id"], "family": e["family"], "state": e["state"], "hypothesis": e["hypothesis"][:500],
                         "expected_if_true": e["expected_if_true"][:300]} for e in exps],
        "runs_finished": [{"id": r["id"], "experiment_id": r["experiment_id"], "seed": r["seed"], "status": r["status"],
                           "val_mse_ratio": (r.get("summary") or {}).get("val_mse_ratio"),
                           "stopped_early": (r.get("summary") or {}).get("stopped"),
                           "error": (r.get("error") or "")[-200:] or None} for r in runs],
        "findings": [{"id": f["id"], "status": f["status"], "verdict": f["verdict"], "claim": f["claim"][:900],
                      "confidence_note": f["confidence_note"][:400],
                      "notes": [n["text"][:300] for n in f.get("notes", [])][-3:]} for f in findings],
        "reviews": [{"id": r["id"], "verdict": r["verdict"], "summary": r["summary"][:800],
                     "issues": [{"severity": i["severity"], "category": i["category"], "finding_id": i.get("finding_id"),
                                 "message": i["message"][:400]} for i in r["issues"] if i["severity"] != "info"],
                     "responses": [x["response"][:400] for x in r.get("responses", [])]} for r in reviews],
        "waiting_for_human": [{"id": e["id"], "family": e["family"], "hypothesis": e["hypothesis"][:300]}
                              for e in store.list_experiments("proposed")],
        "open_review_issues": len(review.open_issues()),
        "lab_paused_now": store.is_paused(),
        "compute": {"used_today_h": round(store.compute_used_today(), 3), "daily_budget_h": st["daily_budget_hours"]},
        "totals": {"findings": len(store.list_findings()),
                   "replicated": sum(f["status"] == "replicated" for f in store.list_findings())},
    }


def _code_commits(since: str) -> list[str]:
    import subprocess

    from engine.paths import REPO_ROOT

    p = subprocess.run(["git", "log", f"--since={since}", "--pretty=%ad %s%n%b", "--date=format:%H:%M"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    out, cur = [], None
    for line in p.stdout.splitlines():
        if line[:5].replace(":", "").isdigit() and len(line) > 6 and line[2] == ":":
            subj = line[6:]
            cur = None if subj.startswith(("cycle ", "review ", "highlights ", "pause")) else subj
            if cur:
                out.append(cur)
        elif cur and line.strip() and not line.startswith("Co-Authored-By"):
            out[-1] += " | " + line.strip()
    return [c[:500] for c in out]


def generate(save: bool = True, now: Optional[dt.datetime] = None) -> dict:
    from agent.claude_cli import run_claude

    now = now or dt.datetime.now()
    until = now.isoformat(timespec="seconds")
    since = _since()
    f = facts(since, until)
    system = (Path(__file__).parent / "prompts" / "tutor.md").read_text()
    md = run_claude(PROMPT.format(since=since, until=until, date=now.strftime("%A %-d %B %Y"),
                                  facts=json.dumps(f, indent=1, default=str)),
                    system=system, no_tools=True, role="digest").strip()
    md = md.removeprefix("```markdown").removeprefix("```").removesuffix("```").strip()
    meta = {"date": now.strftime("%Y-%m-%d"), "since": since, "until": until, "generated_at": until,
            "counts": {"findings": len(f["findings"]), "experiments": len(f["experiments"]),
                       "runs": len(f["runs_finished"]), "reviews": len(f["reviews"])}}
    if save:
        digests_dir().mkdir(parents=True, exist_ok=True)
        (digests_dir() / f"{meta['date']}.md").write_text(md + "\n")
        store._write_json(digests_dir() / f"{meta['date']}.json", meta)
        store.log_event("agent", "digest.created", {"date": meta["date"], **meta["counts"]})
    return {**meta, "markdown": md}


def due(now: Optional[dt.datetime] = None) -> bool:
    now = now or dt.datetime.now()
    hour = store.get_settings().get("digest_hour", DEFAULT_HOUR)
    if hour is None or now.hour < int(hour):
        return False
    return not (digests_dir() / f"{now.strftime('%Y-%m-%d')}.md").exists()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agent.digest")
    ap.add_argument("--preview", action="store_true", help="print without saving")
    a = ap.parse_args(argv)
    d = generate(save=not a.preview)
    print(d["markdown"])


if __name__ == "__main__":
    main()
