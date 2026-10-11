"""Draft a paper section with `claude -p`. Cites only replicated findings unless overridden per claim."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from agent.claude_cli import run_claude
from app import store
from engine.paths import REPO_ROOT

PROMPTS = Path(__file__).parent / "prompts"
FID = re.compile(r"\bf-\d{8}-\d{6}-[0-9a-f]{4}\b")

PROMPT = """# Task: draft the "{section}" section of a research paper

Paper title: {title}

Write the body of the section in Markdown (start with the heading `## {section}`). Rules:
- Every empirical claim must cite a finding inline as [f-...] using ONLY these IDs: {allowed_ids}.
  If that list is empty, say plainly that no results have been replicated yet and describe what is pending.
- Report numbers exactly as given; do not invent results, run IDs or citations.
- Include null and negative results; they matter.
- You may embed figures for evidence runs as ![caption](figure:heatmap/<run_id>/force),
  ![caption](figure:heatmap/<run_id>/accel_step) or ![caption](figure:formation/<run_id>).
- Use LaTeX math with $...$ where helpful (e.g. $F = GMm/r^2$).
- Counts (how many findings, how many supported/refuted, how many runs) must come from "Facts" below; do not
  recount them yourself.
- If an evidence run has `stopped: true`, it was cut short by the lab's budget hard stop at `completion` of its
  planned steps. Say so where its numbers are used.
- Describe methods only as documented in "Lab protocol" below and in the findings' evidence. Do not invent
  details.
- Extra instructions from the author: {instructions}

Reply with ONLY the Markdown for this section.

## Facts (computed from the lab records)
```json
{facts}
```

## Lab protocol (documented methods)
{protocol}

## Citable findings (with their evidence)
```json
{findings}
```

## The paper so far (for context and consistency; do not repeat it)
```markdown
{paper}
```
"""


def _evidence(f: dict) -> dict:
    runs = []
    for rid in f["evidence_run_ids"]:
        try:
            r = store.get_run(rid)
        except KeyError:
            continue
        s = r.get("summary") or {}
        run = {"run_id": rid, "seed": r["seed"], "val_mse_ratio": s.get("val_mse_ratio"),
               "probes_final": (s.get("probes") or {}).get("targets"), "equations": s.get("equations")}
        if s.get("stopped"):
            run.update(stopped=True, completion=round(store.run_completion(r), 3))
        runs.append(run)
    return {"id": f["id"], "claim": f["claim"], "verdict": f["verdict"], "status": f["status"],
            "confidence_note": f["confidence_note"], "evidence": runs}


SECTION_ORDER = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Limitations", "Conclusion"]


def facts(allowed: list[dict]) -> dict:
    by_verdict: dict[str, int] = {}
    runs = set()
    for f in allowed:
        by_verdict[f["verdict"]] = by_verdict.get(f["verdict"], 0) + 1
        runs.update(f["evidence_run_ids"])
    truncated = [r for r in runs if (store.get_run(r).get("summary") or {}).get("stopped")]
    return {"citable_findings": len(allowed), "by_verdict": by_verdict,
            "by_status": {st: sum(f["status"] == st for f in allowed) for st in {f["status"] for f in allowed}},
            "evidence_runs": len(runs), "evidence_runs_cut_short_by_budget_stop": len(truncated)}


def protocol() -> str:
    """The documented science protocol (the 'Science and engine' part of DECISIONS.md)."""
    try:
        text = (REPO_ROOT / "DECISIONS.md").read_text()
        return text.split("## Science and engine", 1)[1].split("\n## ", 1)[0].strip()
    except (OSError, IndexError):
        return "(not available)"


def replace_or_append_section(md: str, section: str, body: str) -> str:
    """Replace a `## section`, or insert it in conventional paper order (Abstract first, ...)."""
    body = body.strip()
    if not body.startswith("## "):
        body = f"## {section}\n\n{body}"
    pat = re.compile(rf"^## {re.escape(section)}\s*$.*?(?=^## |\Z)", re.M | re.S)
    if pat.search(md):
        return pat.sub(lambda _m: body + "\n\n", md, count=1).rstrip() + "\n"
    if section in SECTION_ORDER:
        for later in SECTION_ORDER[SECTION_ORDER.index(section) + 1:]:
            m = re.search(rf"^## {re.escape(later)}\s*$", md, re.M)
            if m:
                return md[:m.start()] + body + "\n\n" + md[m.start():]
    return md.rstrip() + "\n\n" + body + "\n"


def draft_section(pid: str, section: str, instructions: str = "", allow_unreplicated: Optional[list[str]] = None,
                  actor: str = "agent", extra_meta: Optional[dict] = None) -> dict:
    paper = store.get_paper(pid)
    overrides = set(allow_unreplicated or []) | set(paper.get("cite_overrides", []))
    allowed = [f for f in store.list_findings() if f["status"] == "replicated" or f["id"] in overrides]
    allowed_ids = [f["id"] for f in allowed]
    prompt = PROMPT.format(section=section, title=paper["title"], allowed_ids=json.dumps(allowed_ids),
                           instructions=instructions or "none", findings=json.dumps([_evidence(f) for f in allowed],
                                                                                    indent=2, default=str),
                           paper=paper["markdown"][-12000:], facts=json.dumps(facts(allowed), indent=2),
                           protocol=protocol())
    system = (PROMPTS / "system.md").read_text()
    body = run_claude(prompt, system=system)
    bad = sorted(set(FID.findall(body)) - set(allowed_ids))
    if bad:
        body = run_claude(prompt + f"\n\nYour previous draft cited findings that are not allowed: {bad}. "
                                   "Rewrite it citing only the allowed IDs.", system=system)
        bad = sorted(set(FID.findall(body)) - set(allowed_ids))
        for b in bad:  # last resort: mark rather than silently cite
            body = body.replace(b, f"{b} (NOT CITABLE)")
    body = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", body.strip())
    md = replace_or_append_section(paper["markdown"], section, body)
    meta = {k: v for k, v in paper.items() if k != "markdown"}
    meta["cite_overrides"] = sorted(overrides)
    if extra_meta:
        meta.update(extra_meta)
    saved = store.save_paper(pid, paper["title"], md, actor, meta=meta)
    store.log_event(actor, "paper.section_drafted", {"id": pid, "section": section, "cited": allowed_ids})
    return saved
