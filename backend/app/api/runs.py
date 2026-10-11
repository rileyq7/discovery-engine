from __future__ import annotations

import copy
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import lab, store

router = APIRouter(prefix="/runs", tags=["runs"])


class AnnotateBody(BaseModel):
    note: Optional[str] = None
    bad: Optional[bool] = None
    actor: str = "human"


def _get(run_id: str) -> dict:
    try:
        return store.get_run(run_id)
    except KeyError:
        raise HTTPException(404, "run not found")


@router.get("")
def list_runs(status: Optional[str] = None, limit: int = 200):
    runs = store.list_runs(status)
    return list(reversed(runs))[:limit]


@router.get("/{run_id}")
def get_run(run_id: str):
    run = _get(run_id)
    exp = store.get_experiment(run["experiment_id"])
    log = store.run_log_path(run_id)
    from pathlib import Path

    art = Path(run["artifact_dir"])
    artifacts = []
    if art.exists():
        for p in sorted(art.rglob("*")):
            if p.is_file():
                artifacts.append({"path": str(p), "name": str(p.relative_to(art)), "bytes": p.stat().st_size})
    return {**run, "experiment": exp,
            "equations": store.read_run_result(run_id, "equations"),
            "patching": store.read_run_result(run_id, "patching"),
            "eval": store.read_run_result(run_id, "eval"),
            "log": log.read_text()[-20000:] if log.exists() else "",
            "artifacts": artifacts[:200],
            "artifacts_present": art.exists()}


@router.get("/{run_id}/metrics")
def metrics(run_id: str, names: Optional[str] = None):
    _get(run_id)
    return store.get_metrics(run_id, names.split(",") if names else None)


@router.get("/{run_id}/probes")
def probes(run_id: str):
    _get(run_id)
    return store.get_probe_results(run_id)


@router.post("/{run_id}/annotate")
def annotate(run_id: str, body: AnnotateBody):
    _get(run_id)
    return store.annotate_run(run_id, body.actor, body.note, body.bad)


@router.post("/{run_id}/rerun")
def rerun(run_id: str, body: AnnotateBody = AnnotateBody()):
    """Re-run with a new seed: a new experiment with the same config and one fresh seed."""
    run = _get(run_id)
    exp = store.get_experiment(run["experiment_id"])
    cfg = copy.deepcopy(exp["config"])
    seeds_used = {r["seed"] for r in store.runs_for(exp["id"])} | set(cfg["seeds"])
    cfg["seeds"] = [max(seeds_used) + 1]
    cfg["budget"]["max_compute_hours"] = round(max(0.01, cfg["budget"]["max_compute_hours"] /
                                                   max(1, len(exp["config"]["seeds"]))) * 1.5, 3)
    try:
        rec = lab.propose(cfg, body.actor, auto_approve=False, note=f"re-run of {run_id} with new seed")
        return lab.approve(rec["id"], body.actor, "re-run")
    except lab.ProposalError as e:
        raise HTTPException(422, {"errors": e.errors})
