from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import lab, store

router = APIRouter(prefix="/experiments", tags=["experiments"])


class ProposeBody(BaseModel):
    config: dict[str, Any]
    approve: bool = False
    actor: str = "human"


class NoteBody(BaseModel):
    note: str = ""
    actor: str = "human"


def _with_runs(e: dict) -> dict:
    runs = store.runs_for(e["id"])
    return {**e, "runs": [{k: r.get(k) for k in ("id", "seed", "status", "compute_hours", "started_at", "ended_at",
                                                 "bad", "summary", "progress")} for r in runs]}


@router.get("")
def list_experiments(state: Optional[str] = None, family: Optional[str] = None, verdict: Optional[str] = None):
    exps = store.list_experiments(state, family)
    if verdict:
        exps = [e for e in exps if (e.get("verdict") or "none") == verdict]
    return [_with_runs(e) for e in reversed(exps)]


@router.get("/templates")
def templates():
    """Seeded experiment configs from configs/ (E1-E6) to start a proposal from."""
    import json as _json

    from engine.paths import REPO_ROOT

    out = []
    root = REPO_ROOT / "configs"
    for p in sorted(root.glob("*.json")) + sorted((root / "e6").glob("*.json")):
        try:
            out.append({"name": str(p.relative_to(root)), "config": _json.loads(p.read_text())})
        except ValueError:
            continue
    return out


@router.post("/validate")
def validate(body: ProposeBody):
    cfg, errors, est = lab.check(body.config)
    auto = lab.auto_approvable(cfg, est) if cfg and not errors else (False, "invalid")
    return {"valid": not errors, "errors": errors, "estimate_hours": est, "auto_approvable": auto[0],
            "auto_reason": auto[1]}


@router.post("")
def propose(body: ProposeBody):
    try:
        rec = lab.propose(body.config, body.actor, auto_approve=False)
        if body.approve:
            rec = lab.approve(rec["id"], body.actor, "approved at proposal")
        return rec
    except lab.ProposalError as e:
        raise HTTPException(422, {"errors": e.errors})


@router.get("/{exp_id}")
def get_experiment(exp_id: str):
    try:
        return _with_runs(store.get_experiment(exp_id))
    except KeyError:
        raise HTTPException(404, "experiment not found")


@router.put("/{exp_id}/config")
def edit_config(exp_id: str, body: ProposeBody):
    try:
        return lab.edit_config(exp_id, body.config, body.actor)
    except lab.ProposalError as e:
        raise HTTPException(422, {"errors": e.errors})
    except store.StoreError as e:
        raise HTTPException(409, str(e))


@router.post("/{exp_id}/approve")
def approve(exp_id: str, body: NoteBody = NoteBody()):
    try:
        return lab.approve(exp_id, body.actor, body.note)
    except lab.ProposalError as e:
        raise HTTPException(422, {"errors": e.errors})
    except store.StoreError as e:
        raise HTTPException(409, str(e))


@router.post("/{exp_id}/reject")
def reject(exp_id: str, body: NoteBody = NoteBody()):
    try:
        return lab.reject(exp_id, body.actor, body.note)
    except store.StoreError as e:
        raise HTTPException(409, str(e))


@router.post("/{exp_id}/cancel")
def cancel(exp_id: str, body: NoteBody = NoteBody()):
    try:
        return store.transition(exp_id, "cancelled", body.actor, body.note)
    except store.StoreError as e:
        raise HTTPException(409, str(e))


@router.post("/{exp_id}/note")
def note(exp_id: str, body: NoteBody):
    return store.add_experiment_note(exp_id, body.note, body.actor)
