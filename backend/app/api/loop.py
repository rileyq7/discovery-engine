from __future__ import annotations

import json
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import store
from engine.paths import runs_dir

router = APIRouter(prefix="/loop", tags=["loop"])


class BudgetBody(BaseModel):
    daily_budget_hours: Optional[float] = None
    max_concurrent_runs: Optional[int] = None
    per_experiment_cap_hours: Optional[float] = None
    auto_approve_families: Optional[list[str]] = None
    max_cycles_per_day: Optional[int] = None
    actor: str = "human"


class ActorBody(BaseModel):
    actor: str = "human"
    reason: str = ""


def worker_status() -> dict:
    hb = runs_dir() / ".worker_heartbeat"
    try:
        t = float(hb.read_text().strip())
        return {"online": time.time() - t < 15, "last_seen_s": round(time.time() - t, 1)}
    except (FileNotFoundError, ValueError):
        return {"online": False, "last_seen_s": None}


def cycles_today() -> list[dict]:
    d = store.today()
    evs = [e for e in store.list_events(0, 100000) if e["type"] in ("cycle.started", "cycle.finished", "cycle.failed",
                                                                    "cycle.skipped")]
    return [e for e in evs if e["timestamp"].startswith(d)]


def status() -> dict:
    s = store.get_settings()
    cyc = cycles_today()
    finished = [e for e in cyc if e["type"] == "cycle.finished"]
    last = cyc[-1] if cyc else None
    pf = store.lab_dir() / "PAUSED"
    note = pf.read_text().strip() if pf.exists() else None
    return {"paused": store.is_paused(), "paused_note": note, "settings": s, "compute_used_today": round(store.compute_used_today(), 4),
            "cycles_today": len(finished), "last_cycle": last, "worker": worker_status()}


@router.get("")
def get_loop():
    return status()


@router.post("/start")
def start(body: ActorBody = ActorBody()):
    store.set_paused(False, body.actor, body.reason)
    return status()


@router.post("/pause")
def pause(body: ActorBody = ActorBody()):
    store.set_paused(True, body.actor, body.reason)
    return status()


@router.post("/kill")
def kill(body: ActorBody = ActorBody()):
    cancelled = store.kill_switch(body.actor)
    return {**status(), "cancelled": cancelled}


@router.put("/budget")
def budget(body: BudgetBody):
    patch = body.model_dump(exclude_none=True, exclude={"actor"})
    if not patch:
        raise HTTPException(422, "nothing to update")
    store.update_settings(patch, body.actor)
    return status()
