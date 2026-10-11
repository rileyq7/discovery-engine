from __future__ import annotations

import threading
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from app import store
from app.latex import markdown_to_latex

router = APIRouter(prefix="/papers", tags=["papers"])
_drafting: dict[str, dict] = {}


class PaperBody(BaseModel):
    title: str
    markdown: Optional[str] = None
    cite_overrides: Optional[list[str]] = None
    actor: str = "human"


class DraftBody(BaseModel):
    section: str
    instructions: str = ""
    allow_unreplicated: list[str] = []
    actor: str = "human"


def citation_warnings(paper: dict) -> list[str]:
    overrides = set(paper.get("cite_overrides", []))
    warnings = []
    for fid in paper.get("cited_finding_ids", []):
        try:
            f = store.get_finding(fid)
        except KeyError:
            warnings.append(f"{fid} does not exist")
            continue
        if f["status"] != "replicated" and fid not in overrides:
            warnings.append(f"{fid} is {f['status']}, not replicated (override per claim to allow)")
    return warnings


def _get(pid: str) -> dict:
    try:
        p = store.get_paper(pid)
    except KeyError:
        raise HTTPException(404, "paper not found")
    return {**p, "citation_warnings": citation_warnings(p), "drafting": _drafting.get(pid)}


@router.get("")
def list_papers():
    return store.list_papers()


@router.post("")
def create_paper(body: PaperBody):
    return store.create_paper(body.title, body.actor, body.markdown or "")


@router.get("/{pid}")
def get_paper(pid: str):
    return _get(pid)


@router.put("/{pid}")
def save_paper(pid: str, body: PaperBody):
    p = _get(pid)
    meta = {k: v for k, v in p.items() if k not in ("markdown", "citation_warnings", "drafting")}
    if body.cite_overrides is not None:
        meta["cite_overrides"] = body.cite_overrides
    store.save_paper(pid, body.title, body.markdown if body.markdown is not None else p["markdown"], body.actor,
                     meta=meta)
    return _get(pid)


@router.post("/{pid}/draft-section", status_code=202)
def draft_section(pid: str, body: DraftBody):
    _get(pid)
    if _drafting.get(pid, {}).get("status") == "running":
        raise HTTPException(409, "a section is already being drafted for this paper")
    _drafting[pid] = {"status": "running", "section": body.section, "started_at": store.now()}
    store.log_event(body.actor, "paper.draft_started", {"id": pid, "section": body.section})

    def work():
        from agent.papers import draft_section as do_draft

        try:
            do_draft(pid, body.section, body.instructions, body.allow_unreplicated, actor="agent")
            _drafting[pid] = {"status": "done", "section": body.section, "finished_at": store.now()}
        except Exception as e:
            _drafting[pid] = {"status": "error", "section": body.section, "error": str(e)[:1000]}
            store.log_event("agent", "paper.draft_failed", {"id": pid, "error": str(e)[:500]})

    threading.Thread(target=work, daemon=True).start()
    return {"status": "running", "section": body.section}


@router.get("/{pid}/export")
def export(pid: str, format: str = "markdown"):
    p = _get(pid)
    if format == "latex":
        return PlainTextResponse(markdown_to_latex(p["markdown"], p["title"]), media_type="application/x-tex",
                                 headers={"Content-Disposition": f'attachment; filename="{pid}.tex"'})
    if format == "markdown":
        return PlainTextResponse(p["markdown"], media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="{pid}.md"'})
    raise HTTPException(400, "format must be markdown or latex")
