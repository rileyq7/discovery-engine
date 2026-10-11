"""Command line entry points for running experiments without the server.

    python -m engine.cli run ../configs/e1.json          # propose, approve and run every seed inline
    python -m engine.cli status <experiment_id>
    python -m engine.cli datasets
"""
from __future__ import annotations

import argparse
import json
import sys

from app import lab, store
from engine.run_job import execute_run
from engine.sim.rebound_gen import list_datasets


def cmd_run(a):
    store.rebuild_index()
    raw = json.load(open(a.config))
    if a.seeds:
        raw["seeds"] = [int(s) for s in a.seeds.split(",")]
    rec = lab.propose(raw, "cli", auto_approve=False, note=f"from {a.config}")
    exp_id = rec["id"]
    lab.approve(exp_id, "cli", "run from CLI")
    store.transition(exp_id, "queued", "cli")
    runs = [store.create_run(exp_id, s, "cli") for s in rec["config"]["seeds"]]
    store.transition(exp_id, "running", "cli")
    results = [execute_run(r["id"]) for r in runs]
    hours = round(sum(r.get("compute_hours", 0) for r in results), 4)
    store.set_experiment_fields(exp_id, compute_hours=hours)
    ok = any(r["status"] == "completed" for r in results)
    store.transition(exp_id, "analyzing" if ok else "failed", "cli")
    print(json.dumps({"experiment_id": exp_id, "runs": [{"id": r["id"], "status": r["status"], "seed": r["seed"],
                                                          "summary": r.get("summary")} for r in results]}, indent=2))
    return 0 if ok else 1


def cmd_status(a):
    exp = store.get_experiment(a.experiment_id)
    print(json.dumps({"experiment": {k: exp[k] for k in ("id", "family", "state", "estimate_hours", "compute_hours")},
                      "runs": store.runs_for(a.experiment_id)}, indent=2))


def cmd_datasets(_a):
    print(json.dumps(list_datasets(), indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="engine.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("config")
    p.add_argument("--seeds", default=None)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("status")
    p.add_argument("experiment_id")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("datasets")
    p.set_defaults(fn=cmd_datasets)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
