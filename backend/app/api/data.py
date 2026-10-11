from __future__ import annotations

import threading

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import store
from app.schemas.config import DataConfig, dataset_key
from engine.sim.rebound_gen import dataset_path, list_datasets, load_or_generate

router = APIRouter(prefix="/data", tags=["data"])
_generating: dict[str, str] = {}


class GenerateBody(BaseModel):
    data: DataConfig
    n_steps: int = 129
    actor: str = "human"


def _load(key: str) -> dict:
    p = dataset_path(key)
    if not p.exists():
        raise HTTPException(404, "dataset not found (it may need regenerating from its config)")
    z = np.load(p)
    return {k: z[k] for k in z.files}


def _usage() -> dict[str, list[str]]:
    use: dict[str, list[str]] = {}
    for r in store.list_runs():
        did = (r.get("summary") or {}).get("dataset_id") or r.get("dataset_id")
        if did:
            use.setdefault(did, []).append(r["id"])
    return use


@router.get("/datasets")
def datasets():
    use = _usage()
    out = [{**d, "used_by_runs": use.get(d["id"], [])} for d in list_datasets()]
    out += [{"id": k, "status": "generating"} for k in _generating]
    return out


@router.post("/datasets", status_code=202)
def generate(body: GenerateBody):
    key = dataset_key(body.data, body.n_steps)
    if dataset_path(key).exists():
        return {"id": key, "status": "exists"}
    _generating[key] = "running"

    def work():
        try:
            load_or_generate(body.data, body.n_steps, log=lambda m: None)
            store.log_event(body.actor, "dataset.generated", {"id": key})
        finally:
            _generating.pop(key, None)

    threading.Thread(target=work, daemon=True).start()
    store.log_event(body.actor, "dataset.requested", {"id": key, "config": body.data.model_dump(mode="json")})
    return {"id": key, "status": "generating"}


@router.get("/datasets/{key}/sample")
def sample(key: str, n: int = 6, seed: int = 0):
    a = _load(key)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(a["pos"]), size=min(n, len(a["pos"])), replace=False)
    nb = int(a["body_mask"].sum())
    return [{"index": int(i), "masses": a["masses"][i][:nb].round(4).tolist(), "e": round(float(a["e"][i]), 4),
             "a": round(float(a["a"][i]), 4), "period": round(float(a["period"][i]), 4),
             "bodies": [a["pos"][i][:, b].round(5).tolist() for b in range(nb)]} for i in idx]


def _hist(x: np.ndarray, bins: int = 24) -> dict:
    counts, edges = np.histogram(x, bins=bins)
    return {"counts": counts.tolist(), "edges": edges.round(5).tolist(), "mean": float(x.mean()),
            "std": float(x.std())}


@router.get("/datasets/{key}/stats")
def stats(key: str):
    a = _load(key)
    return {"id": key, "n_systems": int(len(a["pos"])), "n_steps": int(a["pos"].shape[1]),
            "n_bodies": int(a["body_mask"].sum()),
            "mass_ratio": _hist(a["masses"][:, 1]), "period": _hist(a["period"]),
            "eccentricity": _hist(a["e"]), "semi_major_axis": _hist(a["a"]),
            "energy": _hist(a["energy"][:, 0])}
