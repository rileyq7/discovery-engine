"""Execute one run: train, probe, record. Used by the worker (as a subprocess) and the CLI.

    python -m engine.run_job <run_id>
"""
from __future__ import annotations

import os
import sys
import time
import traceback

from app import store
from app.schemas.config import validate_config
from engine.paths import device_name, run_artifact_dir
from engine.probes.suite import run_suite
from engine.train import prepare, train

HARD_STOP_FACTOR = 2.0


def execute_run(run_id: str, echo: bool = True) -> dict:
    run = store.get_run(run_id)
    exp = store.get_experiment(run["experiment_id"])
    cfg = validate_config(exp["config"])
    art = run_artifact_dir(run_id)
    art.mkdir(parents=True, exist_ok=True)
    log_path = store.run_log_path(run_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with open(log_path, "a") as f:
            f.write(line + "\n")
        if echo:
            print(f"{run_id}: {line}", flush=True)

    t0 = time.time()
    # Runaway guard: stop training at HARD_STOP_FACTOR x this seed's share of the experiment budget. The clock
    # starts at the first training step, so dataset generation and loading don't count against it.
    seed_budget_s = cfg.budget.max_compute_hours / len(cfg.seeds) * 3600 * HARD_STOP_FACTOR
    train_t0: list[float] = []

    def should_stop() -> bool:
        if not train_t0:
            train_t0.append(time.time())
        return time.time() - train_t0[0] > seed_budget_s
    store.update_run(run_id, {"status": "running", "started_at": store.now(), "device": device_name(),
                              "pid": os.getpid()})
    try:
        log(f"experiment {exp['id']} ({cfg.family}) seed {run['seed']}: {cfg.hypothesis}")
        summary = train(cfg, run["seed"], art, log=log, on_metrics=lambda rows: store.append_metrics(run_id, rows),
                        should_stop=should_stop)
        store.update_run(run_id, {"progress": "probing", "n_params": summary["n_params"],
                                  "dataset_id": summary["dataset_id"]})
        prep = prepare(cfg, log=lambda m: None)
        res = run_suite(cfg, run["seed"], art, prep, log=log)
        store.write_probe_results(run_id, res["rows"])
        for name in ("patching", "equations", "eval"):
            if name in res:
                store.write_run_result(run_id, name, res[name])
        summary["probes"] = res["summary"]
        if "equations" in res:
            eq = res["equations"]
            summary["equations"] = {t: {k: e.get(k) for k in ("equation", "law_recovered", "fit_error_vs_true_law",
                                                               "layer", "error")} for t, e in eq.items()}
        if "patching" in res:
            p = res["patching"]["patched"]
            summary["patching"] = {"top_head": {k: p[0][k] for k in ("layer", "head", "force_r2", "effect")},
                                   "control_head_effect": p[-1]["effect"]}
        if "eval" in res:
            summary["eval"] = [{k: e[k] for k in ("name", "val_mse_ratio", "n_bodies")} |
                               {"force_r2_best": max(e["probe_r2"].get("force", {"0": None}).values() or [None])}
                               for e in res["eval"]]
        hours = (time.time() - t0) / 3600
        log(f"run complete in {hours * 60:.1f} min")
        return store.update_run(run_id, {"status": "completed", "ended_at": store.now(),
                                         "compute_hours": round(hours, 4), "summary": summary, "progress": None})
    except Exception as e:
        log("FAILED: " + "".join(traceback.format_exception(e)))
        hours = (time.time() - t0) / 3600
        return store.update_run(run_id, {"status": "failed", "ended_at": store.now(), "compute_hours": round(hours, 4),
                                         "error": f"{type(e).__name__}: {e}", "progress": None})


if __name__ == "__main__":
    r = execute_run(sys.argv[1])
    sys.exit(0 if r["status"] == "completed" else 1)
