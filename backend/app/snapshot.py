"""Export a read-only snapshot of the lab for the hosted dashboard (Vercel).

Every GET endpoint the UI uses is called through the real API and written as JSON under snapshot/api/, so the
static build (VITE_STATIC=1) shows exactly what the local UI shows. The agent cycle runs this before committing.

    python -m app.snapshot [--out ../snapshot]
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from engine.paths import REPO_ROOT

SAMPLE_N = 8  # matches the Data page's sample size


def _write(root: Path, path: str, body) -> None:
    f = root / "api" / f"{path.strip('/')}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    # sorted, compact and stable, so unchanged data produces no git diff
    f.write_text(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str))


def export(out: Path) -> dict:
    from app.main import app

    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    counts = {}
    with TestClient(app) as c:
        def get(path: str):
            r = c.get(f"/api{path}")
            r.raise_for_status()
            return r

        for path in ("/dashboard", "/loop", "/experiments", "/experiments/templates", "/runs", "/findings",
                     "/questions", "/papers", "/data/datasets", "/reviews"):
            _write(tmp, path, get(path).json())
        exps = get("/experiments").json()
        for e in exps:
            _write(tmp, f"/experiments/{e['id']}", get(f"/experiments/{e['id']}").json())
        runs = get("/runs").json()
        for r in runs:
            for sub in ("", "/metrics", "/probes"):
                _write(tmp, f"/runs/{r['id']}{sub}", get(f"/runs/{r['id']}{sub}").json())
        papers = get("/papers").json()
        for p in papers:
            _write(tmp, f"/papers/{p['id']}", get(f"/papers/{p['id']}").json())
            d = tmp / "api" / "papers" / p["id"]
            d.mkdir(parents=True, exist_ok=True)
            (d / "export.md").write_text(get(f"/papers/{p['id']}/export?format=markdown").text)
            (d / "export.tex").write_text(get(f"/papers/{p['id']}/export?format=latex").text)
        datasets = get("/data/datasets").json()
        for ds in datasets:
            key = ds.get("id") or ds.get("key")
            try:
                _write(tmp, f"/data/datasets/{key}/stats", get(f"/data/datasets/{key}/stats").json())
                _write(tmp, f"/data/datasets/{key}/sample", get(f"/data/datasets/{key}/sample?n={SAMPLE_N}&seed=0").json())
            except Exception as e:  # dataset still generating or missing locally
                print(f"[snapshot] skipped dataset {key}: {e}")
        counts = {"experiments": len(exps), "runs": len(runs), "papers": len(papers), "datasets": len(datasets)}
    (tmp / "meta.json").write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **counts}, sort_keys=True))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(prog="app.snapshot")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "snapshot")
    a = ap.parse_args(argv)
    print(f"[snapshot] {export(a.out)} -> {a.out}")


if __name__ == "__main__":
    main()
