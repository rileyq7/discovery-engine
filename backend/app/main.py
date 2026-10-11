"""FastAPI app: REST endpoints plus a Server-Sent Events stream of state changes and run progress."""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import store
from app.api import data, experiments, findings, loop, papers, runs


@asynccontextmanager
async def lifespan(_app: FastAPI):
    counts = store.rebuild_index()
    print(f"[api] index rebuilt from lab/: {counts}", flush=True)
    yield


app = FastAPI(title="Discovery Engine", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
for r in (experiments.router, runs.router, findings.router, papers.router, loop.router, data.router):
    app.include_router(r)
    app.include_router(r, prefix="/api")


@app.get("/api/dashboard")
def dashboard():
    running = store.list_runs("running")
    fs = list(reversed(store.list_findings()))
    exps = store.list_experiments()
    return {"loop": loop.status(),
            "running_runs": [{**r, "experiment": next((e for e in exps if e["id"] == r["experiment_id"]), None)}
                             for r in running],
            "latest_findings": fs[:5],
            "open_questions": len(store.list_questions("open")),
            "counts": {s: sum(1 for e in exps if e["state"] == s) for s in store.EXPERIMENT_STATES},
            "n_findings": len(fs)}


@app.get("/api/events")
def events(after: int = 0, limit: int = 200):
    return store.list_events(after, limit)


async def _stream(request: Request):
    last_event = store.last_event_id()
    last_metric = store.max_metric_rowid()
    yield f"event: hello\ndata: {json.dumps({'event_id': last_event})}\n\n"
    ticks = 0
    while True:
        if await request.is_disconnected():
            return
        for e in store.list_events(last_event, 500):
            last_event = e["id"]
            yield f"event: change\ndata: {json.dumps(e)}\n\n"
        ms = store.metrics_since(last_metric)
        if ms:
            last_metric = ms[-1]["rowid"]
            by_run: dict[str, list] = {}
            for m in ms:
                by_run.setdefault(m["run_id"], []).append({k: m[k] for k in ("step", "name", "value")})
            for rid, rows in by_run.items():
                yield f"event: metrics\ndata: {json.dumps({'run_id': rid, 'rows': rows})}\n\n"
        ticks += 1
        if ticks % 15 == 0:
            yield ": keepalive\n\n"
        await asyncio.sleep(1.0)


@app.get("/stream")
@app.get("/api/stream")
def stream(request: Request):
    return StreamingResponse(_stream(request), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# Serve the built frontend when present (make dev uses the Vite dev server instead).
_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.exists() and not os.environ.get("DE_NO_STATIC"):
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = _dist / path
        return FileResponse(f if path and f.is_file() else _dist / "index.html")
