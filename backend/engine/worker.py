"""Queue runner. Polls the experiments table and runs each seed as a subprocess.

    python -m engine.worker            # run forever
    python -m engine.worker --until-idle --max-minutes 50   # used by agent cycles

Lifecycle handled here: approved -> queued (budget admission) -> running -> analyzing | failed.
The agent (or a human) moves analyzing -> complete by recording a finding.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from app import store
from engine.paths import runs_dir

BACKEND = Path(__file__).resolve().parents[1]
TERMINAL = ("completed", "failed", "cancelled")


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class Worker:
    def __init__(self, inline: bool = False):
        self.procs: dict[str, subprocess.Popen] = {}
        self.inline = inline

    # -- recovery
    def recover_orphans(self):
        for r in store.list_runs("running"):
            if r["id"] not in self.procs and not _alive(r.get("pid")):
                store.update_run(r["id"], {"status": "failed", "ended_at": store.now(),
                                           "error": "worker restarted while run was in progress"})

    # -- admission: approved -> queued, within the daily compute budget
    def admit(self):
        if store.is_paused():
            return
        s = store.get_settings()
        used = store.compute_used_today()
        reserved = sum(e["estimate_hours"] for e in store.list_experiments("queued"))
        for e in store.list_experiments("running"):
            done = sum(r.get("compute_hours", 0) for r in store.runs_for(e["id"]))
            reserved += max(0.0, e["estimate_hours"] - done)
        for e in store.list_experiments("approved"):
            if used + reserved + e["estimate_hours"] <= s["daily_budget_hours"]:
                store.transition(e["id"], "queued", "worker", "admitted within daily budget")
                reserved += e["estimate_hours"]

    # -- start runs
    def schedule(self):
        s = store.get_settings()
        running = [r for r in store.list_runs("running")]
        capacity = int(s["max_concurrent_runs"]) - len(running)
        if capacity <= 0:
            return
        # Seeds of experiments already running first, then new experiments (unless paused).
        pending = []
        for e in store.list_experiments("running"):
            pending += [r for r in store.runs_for(e["id"]) if r["status"] == "pending"]
        if not pending and not store.is_paused():
            queued = store.list_experiments("queued")
            if queued:
                e = queued[0]
                for seed in e["config"]["seeds"]:
                    store.create_run(e["id"], seed)
                store.transition(e["id"], "running", "worker")
                pending = [r for r in store.runs_for(e["id"]) if r["status"] == "pending"]
        for r in pending[:capacity]:
            self.start(r["id"])

    def start(self, run_id: str):
        if self.inline:
            from engine.run_job import execute_run

            execute_run(run_id)
            return
        logf = open(store.run_log_path(run_id).with_name("stdout.txt"), "a")
        p = subprocess.Popen([sys.executable, "-m", "engine.run_job", run_id], cwd=BACKEND, stdout=logf,
                             stderr=subprocess.STDOUT, env=os.environ.copy())
        self.procs[run_id] = p
        # Mark as running immediately so capacity accounting is right before the child updates it.
        store.update_run(run_id, {"status": "running", "pid": p.pid, "started_at": store.now()})
        print(f"[worker] started {run_id} (pid {p.pid})", flush=True)

    # -- finish
    def reap(self):
        for rid, p in list(self.procs.items()):
            if p.poll() is not None:
                del self.procs[rid]
                r = store.get_run(rid)
                if r["status"] == "running":  # child died without recording an outcome
                    store.update_run(rid, {"status": "failed", "ended_at": store.now(),
                                           "error": f"process exited with code {p.returncode}"})
                print(f"[worker] {rid} finished: {store.get_run(rid)['status']}", flush=True)
        for e in store.list_experiments("running"):
            runs = store.runs_for(e["id"])
            if runs and all(r["status"] in TERMINAL for r in runs):
                hours = round(sum(r.get("compute_hours", 0) for r in runs), 4)
                store.set_experiment_fields(e["id"], compute_hours=hours)
                ok = any(r["status"] == "completed" for r in runs)
                store.transition(e["id"], "analyzing" if ok else "failed", "worker",
                                 f"{sum(r['status'] == 'completed' for r in runs)}/{len(runs)} runs completed")

    def tick(self):
        self.reap()
        self.admit()
        self.schedule()

    def busy(self) -> bool:
        if self.procs or store.list_runs("running"):
            return True
        if store.list_experiments("running") or store.list_experiments("queued"):
            return True
        return False


def external_worker_online() -> bool:
    try:
        return time.time() - float((runs_dir() / ".worker_heartbeat").read_text().strip()) < 10
    except (FileNotFoundError, ValueError):
        return False


def drain_queue(max_minutes: float) -> int:
    """Run the queue until idle. If a worker process is already running (make dev), wait for it instead,
    so two workers never start the same run."""
    if not external_worker_online():
        return main(["--until-idle", "--max-minutes", str(max_minutes), "--no-rebuild"])
    print("[worker] an external worker is online; waiting for the queue to drain", flush=True)
    t0 = time.time()
    while (time.time() - t0) / 60 < max_minutes:
        busy = (store.list_runs("running") or store.list_experiments("running") or store.list_experiments("queued")
                or (store.list_experiments("approved") and not store.is_paused()))
        if not busy:
            return 0
        time.sleep(5)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--until-idle", action="store_true", help="exit when nothing is queued or running")
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--no-rebuild", action="store_true")
    a = ap.parse_args(argv)
    if not a.no_rebuild:
        store.rebuild_index()
    w = Worker()
    w.recover_orphans()
    t0 = time.time()
    print("[worker] polling queue", flush=True)
    hb = runs_dir() / ".worker_heartbeat"
    hb.parent.mkdir(parents=True, exist_ok=True)
    while True:
        hb.write_text(str(time.time()))
        w.tick()
        if a.until_idle and not w.busy():
            # One more admission pass in case something became admissible.
            w.tick()
            if not w.busy():
                print("[worker] idle, exiting", flush=True)
                return 0
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            print("[worker] time limit reached; leaving running jobs to finish", flush=True)
            for p in w.procs.values():
                p.wait()
            w.reap()
            return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
