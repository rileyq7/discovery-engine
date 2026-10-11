"""Draft a paper section with `claude -p`. Cites only replicated findings unless overridden per claim."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from agent.claude_cli import run_claude
from app import store

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
- Extra instructions from the author: {instructions}

Reply with ONLY the Markdown for this section.

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
        runs.append({"run_id": rid, "seed": r["seed"], "val_mse_ratio": s.get("val_mse_ratio"),
                     "probes_final": (s.get("probes") or {}).get("targets"), "equations": s.get("equations")})
    return {"id": f["id"], "claim": f["claim"], "verdict": f["verdict"], "status": f["status"],
            "confidence_note": f["confidence_note"], "evidence": runs}


def replace_or_append_section(md: str, section: str, body: str) -> str:
    body = body.strip()
    if not body.startswith("## "):
        body = f"## {section}\n\n{body}"
    pat = re.compile(rf"^## {re.escape(section)}\s*$.*?(?=^## |\Z)", re.M | re.S)
    if pat.search(md):
        return pat.sub(lambda _m: body + "\n\n", md, count=1).rstrip() + "\n"
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
                           paper=paper["markdown"][-12000:])
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
