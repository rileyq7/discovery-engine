"""The lab store.

Source of truth: plain files under lab/ (committed to git).
  lab/settings.json                    budget, concurrency, auto-approve rules
  lab/PAUSED                           pause flag (present = loop paused)
  lab/experiments/<id>.json
  lab/runs/<run_id>/run.json           run record
  lab/runs/<run_id>/metrics.jsonl      {step, name, value}
  lab/runs/<run_id>/probes.json        [{checkpoint_step, layer, target, r2, control_r2}]
  lab/runs/<run_id>/<name>.json        extra results (equations, patching, eval)
  lab/runs/<run_id>/log.txt
  lab/findings/<id>.json
  lab/questions/<id>.json
  lab/papers/<id>.json + <id>.md
  lab/events/<YYYY-MM-DD>.jsonl        append-only audit log

SQLite (lab/index.sqlite, gitignored) is an index rebuilt from these files.
Every write goes to the file first, then the index.

Guardrails enforced here: findings need completed evidence runs, raw run results
are append-only, `replicated` needs a second agreeing seed, nothing is deleted.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import secrets
import subprocess
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from sqlalchemy import event as sa_event
from sqlmodel import Session, SQLModel, create_engine, delete, select

from app.db.models import Event, Experiment, Finding, Metric, Paper, ProbeResult, Question, Run
from engine.paths import REPO_ROOT, lab_dir, run_artifact_dir

EXPERIMENT_STATES = ("proposed", "approved", "queued", "running", "analyzing", "complete", "failed", "cancelled",
                     "rejected")
TRANSITIONS = {
    "proposed": {"approved", "rejected", "cancelled"},
    "approved": {"queued", "cancelled"},
    "queued": {"running", "cancelled", "approved"},
    "running": {"analyzing", "failed", "cancelled"},
    "analyzing": {"complete", "failed"},
    "complete": set(),
    "failed": set(),
    "cancelled": set(),
    "rejected": set(),
}
VERDICTS = ("supported", "refuted", "inconclusive")
FINDING_STATUSES = ("hypothesis", "supported", "refuted", "inconclusive", "replicated")

DEFAULT_SETTINGS = {
    "daily_budget_hours": 4.0,
    "max_concurrent_runs": 1,
    "per_experiment_cap_hours": 1.0,
    "auto_approve_families": ["E1", "E2", "E3", "E4", "E5"],
    "max_cycles_per_day": 12,
    "throughput": None,  # overrides DEFAULT_THROUGHPUT in compute estimates when set
    "step_overhead_s": None,
    "digest_hour": 7,  # local hour for the daily highlight reel (None disables it)  # overrides DEFAULT_STEP_OVERHEAD_S (seconds per step per layer)
}


class StoreError(ValueError):
    pass


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def today() -> str:
    return dt.date.today().isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


# ---------------------------------------------------------------- file helpers

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(line.rstrip("\n") + "\n")


# ---------------------------------------------------------------- index

_engines: dict[str, Any] = {}
_lock = threading.Lock()


# Bump when a table changes: the index is a cache of lab/ files, so it is dropped and rebuilt.
INDEX_SCHEMA_VERSION = 2


def engine():
    db = lab_dir() / "index.sqlite"
    key = str(db)
    with _lock:
        if key not in _engines:
            db.parent.mkdir(parents=True, exist_ok=True)
            eng = create_engine(f"sqlite:///{db}", connect_args={"check_same_thread": False, "timeout": 30})

            @sa_event.listens_for(eng, "connect")
            def _pragmas(conn, _rec):
                cur = conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.close()

            with eng.begin() as conn:
                ver = conn.exec_driver_sql("PRAGMA user_version").scalar()
            if ver != INDEX_SCHEMA_VERSION:
                SQLModel.metadata.drop_all(eng)
            SQLModel.metadata.create_all(eng)
            if ver != INDEX_SCHEMA_VERSION:  # emptied; every entry point rebuilds the index from lab/ at startup
                with eng.begin() as conn:
                    conn.exec_driver_sql(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
            _engines[key] = eng
        return _engines[key]


def session() -> Session:
    return Session(engine())


def _exp_row(r: dict) -> Experiment:
    return Experiment(id=r["id"], family=r["family"], hypothesis=r["hypothesis"],
                      expected_if_true=r["expected_if_true"], expected_if_false=r["expected_if_false"],
                      config_json=json.dumps(r["config"]), state=r["state"], proposed_by=r["proposed_by"],
                      created_at=r["created_at"], updated_at=r.get("updated_at", ""), verdict=r.get("verdict"),
                      estimate_hours=r.get("estimate_hours", 0.0), compute_hours=r.get("compute_hours", 0.0),
                      replicates_finding_id=r.get("replicates_finding_id"), record_json=json.dumps(r))


def _run_row(r: dict) -> Run:
    return Run(id=r["id"], experiment_id=r["experiment_id"], seed=r["seed"], status=r["status"],
               started_at=r.get("started_at"), ended_at=r.get("ended_at"), compute_hours=r.get("compute_hours", 0.0),
               artifact_dir=r.get("artifact_dir", ""), git_commit=r.get("git_commit"), bad=r.get("bad", False),
               record_json=json.dumps(r))


def _finding_row(r: dict) -> Finding:
    return Finding(id=r["id"], claim=r["claim"], verdict=r["verdict"], status=r["status"],
                   evidence_run_ids=json.dumps(r["evidence_run_ids"]), confidence_note=r["confidence_note"],
                   created_at=r["created_at"], experiment_id=r.get("experiment_id"), record_json=json.dumps(r))


def _question_row(r: dict) -> Question:
    return Question(id=r["id"], text=r["text"], status=r["status"],
                    linked_finding_ids=json.dumps(r.get("linked_finding_ids", [])), created_at=r.get("created_at", ""))


def _paper_row(meta: dict, md: str) -> Paper:
    return Paper(id=meta["id"], title=meta["title"], markdown=md,
                 cited_finding_ids=json.dumps(meta.get("cited_finding_ids", [])), updated_at=meta["updated_at"])


def _event_row(e: dict) -> Event:
    return Event(uid=e["uid"], timestamp=e["timestamp"], actor=e["actor"], type=e["type"],
                 payload_json=json.dumps(e["payload"]))


def rebuild_index() -> dict:
    """Rebuild every SQLite table from lab/ in one transaction."""
    L = lab_dir()
    counts = {}
    with session() as s:
        for tbl in (Experiment, Run, Metric, ProbeResult, Finding, Question, Paper, Event):
            s.exec(delete(tbl))
        exps = [_read_json(p) for p in sorted((L / "experiments").glob("*.json"))]
        s.add_all(_exp_row(r) for r in exps if r)
        counts["experiments"] = len(exps)
        n_runs = 0
        for d in sorted((L / "runs").glob("*/")):
            r = _read_json(d / "run.json")
            if not r:
                continue
            n_runs += 1
            s.add(_run_row(r))
            mf = d / "metrics.jsonl"
            if mf.exists():
                ms = (json.loads(line) for line in mf.read_text().splitlines() if line)
                s.add_all(Metric(run_id=r["id"], **m) for m in ms if math.isfinite(m["value"]))
            for p in _read_json(d / "probes.json", []) or []:
                s.add(ProbeResult(run_id=r["id"], **p))
        counts["runs"] = n_runs
        fs = [_read_json(p) for p in sorted((L / "findings").glob("*.json"))]
        s.add_all(_finding_row(r) for r in fs if r)
        qs = [_read_json(p) for p in sorted((L / "questions").glob("*.json"))]
        s.add_all(_question_row(r) for r in qs if r)
        for p in sorted((L / "papers").glob("*.json")):
            meta = _read_json(p)
            md_path = p.with_suffix(".md")
            s.add(_paper_row(meta, md_path.read_text() if md_path.exists() else ""))
        evs = []
        for p in sorted((L / "events").glob("*.jsonl")):
            evs += [json.loads(x) for x in p.read_text().splitlines() if x]
        evs.sort(key=lambda e: e["timestamp"])
        s.add_all(_event_row(e) for e in evs)
        s.commit()
        counts.update(findings=len(fs), questions=len(qs), events=len(evs))
    return counts


def _upsert(row) -> None:
    with session() as s:
        s.merge(row)
        s.commit()


# ---------------------------------------------------------------- events

def log_event(actor: str, type_: str, payload: dict) -> dict:
    e = {"uid": secrets.token_hex(8), "timestamp": now(), "actor": actor, "type": type_, "payload": payload}
    _append_line(lab_dir() / "events" / f"{today()}.jsonl", json.dumps(e))
    with session() as s:
        s.add(_event_row(e))
        s.commit()
    return e


def list_events(after_id: int = 0, limit: int = 200, types: Optional[Iterable[str]] = None) -> list[dict]:
    with session() as s:
        q = select(Event).where(Event.id > after_id).order_by(Event.id).limit(limit)
        rows = s.exec(q).all()
    out = [{"id": r.id, "timestamp": r.timestamp, "actor": r.actor, "type": r.type,
            "payload": json.loads(r.payload_json)} for r in rows]
    if types:
        out = [e for e in out if e["type"] in set(types)]
    return out


def last_event_id() -> int:
    with session() as s:
        r = s.exec(select(Event.id).order_by(Event.id.desc()).limit(1)).first()
    return r or 0


# ---------------------------------------------------------------- settings / loop

def get_settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    s.update(_read_json(lab_dir() / "settings.json", {}) or {})
    return s


def update_settings(patch: dict, actor: str) -> dict:
    s = get_settings()
    unknown = set(patch) - set(DEFAULT_SETTINGS)
    if unknown:
        raise StoreError(f"unknown settings: {sorted(unknown)}")
    s.update(patch)
    _write_json(lab_dir() / "settings.json", s)
    log_event(actor, "settings.updated", patch)
    return s


def is_paused() -> bool:
    return (lab_dir() / "PAUSED").exists()


def set_paused(paused: bool, actor: str, reason: str = "") -> None:
    f = lab_dir() / "PAUSED"
    if paused:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"paused by {actor} at {now()}\n{reason}\n")
    elif f.exists():
        f.unlink()
    log_event(actor, "loop.paused" if paused else "loop.started", {"reason": reason})


def kill_switch(actor: str) -> list[str]:
    """Pause the loop and cancel queued (not running) work."""
    set_paused(True, actor, "kill switch")
    cancelled = []
    for e in list_experiments():
        if e["state"] in ("queued", "approved"):
            transition(e["id"], "cancelled", actor, "kill switch")
            for r in runs_for(e["id"]):
                if r["status"] == "pending":
                    update_run(r["id"], {"status": "cancelled"}, actor)
            cancelled.append(e["id"])
    return cancelled


# ---------------------------------------------------------------- experiments

def _exp_path(exp_id: str) -> Path:
    return lab_dir() / "experiments" / f"{exp_id}.json"


def create_experiment(config: dict, proposed_by: str, estimate_hours: float,
                      replicates_finding_id: Optional[str] = None, note: str = "") -> dict:
    exp_id = new_id("exp")
    t = now()
    rec = {"id": exp_id, "family": config["family"], "hypothesis": config["hypothesis"],
           "expected_if_true": config["expected_if_true"], "expected_if_false": config["expected_if_false"],
           "config": config, "state": "proposed", "proposed_by": proposed_by, "created_at": t, "updated_at": t,
           "verdict": None, "estimate_hours": round(estimate_hours, 4), "compute_hours": 0.0,
           "replicates_finding_id": replicates_finding_id, "notes": [],
           "history": [{"state": "proposed", "at": t, "actor": proposed_by, "note": note}]}
    _write_json(_exp_path(exp_id), rec)
    _upsert(_exp_row(rec))
    log_event(proposed_by, "experiment.proposed", {"id": exp_id, "family": rec["family"]})
    return rec


def get_experiment(exp_id: str) -> dict:
    rec = _read_json(_exp_path(exp_id))
    if rec is None:
        raise KeyError(exp_id)
    return rec


def list_experiments(state: Optional[str] = None, family: Optional[str] = None) -> list[dict]:
    with session() as s:
        q = select(Experiment)
        if state:
            q = q.where(Experiment.state == state)
        if family:
            q = q.where(Experiment.family == family)
        rows = s.exec(q.order_by(Experiment.created_at)).all()
    return [json.loads(r.record_json) for r in rows]


def _save_experiment(rec: dict) -> None:
    rec["updated_at"] = now()
    _write_json(_exp_path(rec["id"]), rec)
    _upsert(_exp_row(rec))


def transition(exp_id: str, new_state: str, actor: str, note: str = "") -> dict:
    rec = get_experiment(exp_id)
    if new_state not in TRANSITIONS[rec["state"]]:
        raise StoreError(f"illegal transition {rec['state']} -> {new_state} for {exp_id}")
    rec["state"] = new_state
    rec["history"].append({"state": new_state, "at": now(), "actor": actor, "note": note})
    _save_experiment(rec)
    log_event(actor, "experiment.state", {"id": exp_id, "state": new_state, "note": note})
    return rec


def update_experiment_config(exp_id: str, config: dict, estimate_hours: float, actor: str) -> dict:
    rec = get_experiment(exp_id)
    if rec["state"] != "proposed":
        raise StoreError("config can only be edited while proposed")
    rec.update(config=config, family=config["family"], hypothesis=config["hypothesis"],
               expected_if_true=config["expected_if_true"], expected_if_false=config["expected_if_false"],
               estimate_hours=round(estimate_hours, 4))
    rec["history"].append({"state": "proposed", "at": now(), "actor": actor, "note": "config edited"})
    _save_experiment(rec)
    log_event(actor, "experiment.edited", {"id": exp_id})
    return rec


def add_experiment_note(exp_id: str, text: str, actor: str) -> dict:
    rec = get_experiment(exp_id)
    rec.setdefault("notes", []).append({"at": now(), "actor": actor, "text": text})
    _save_experiment(rec)
    return rec


def set_experiment_fields(exp_id: str, **fields) -> dict:
    rec = get_experiment(exp_id)
    rec.update(fields)
    _save_experiment(rec)
    return rec


# ---------------------------------------------------------------- runs

def _run_dir(run_id: str) -> Path:
    return lab_dir() / "runs" / run_id


def create_run(exp_id: str, seed: int, actor: str = "worker") -> dict:
    run_id = new_id("run")
    rec = {"id": run_id, "experiment_id": exp_id, "seed": seed, "status": "pending", "started_at": None,
           "ended_at": None, "compute_hours": 0.0, "artifact_dir": str(run_artifact_dir(run_id)),
           "git_commit": git_commit(), "bad": False, "notes": [], "summary": {}}
    _write_json(_run_dir(run_id) / "run.json", rec)
    _upsert(_run_row(rec))
    log_event(actor, "run.created", {"id": run_id, "experiment_id": exp_id, "seed": seed})
    return rec


def get_run(run_id: str) -> dict:
    rec = _read_json(_run_dir(run_id) / "run.json")
    if rec is None:
        raise KeyError(run_id)
    return rec


RUN_MUTABLE = {"status", "started_at", "ended_at", "compute_hours", "device", "dataset_id", "error", "summary",
               "pid", "progress", "n_params"}


def update_run(run_id: str, patch: dict, actor: str = "worker") -> dict:
    rec = get_run(run_id)
    if rec["status"] in ("completed", "failed", "cancelled") and set(patch) & {"summary", "compute_hours"}:
        raise StoreError("completed runs are immutable")
    bad = set(patch) - RUN_MUTABLE
    if bad:
        raise StoreError(f"cannot set {bad} on a run")
    old_status = rec["status"]
    rec.update(patch)
    _write_json(_run_dir(run_id) / "run.json", rec)
    _upsert(_run_row(rec))
    if "status" in patch and patch["status"] != old_status:
        log_event(actor, "run.state", {"id": run_id, "experiment_id": rec["experiment_id"], "status": patch["status"]})
    return rec


def annotate_run(run_id: str, actor: str, note: Optional[str] = None, bad: Optional[bool] = None) -> dict:
    """Human annotations: notes and the 'bad' flag. They never change results."""
    rec = get_run(run_id)
    if note:
        rec.setdefault("notes", []).append({"at": now(), "actor": actor, "text": note})
    if bad is not None:
        rec["bad"] = bad
    _write_json(_run_dir(run_id) / "run.json", rec)
    _upsert(_run_row(rec))
    log_event(actor, "run.annotated", {"id": run_id, "bad": rec["bad"], "note": note})
    return rec


def runs_for(exp_id: str) -> list[dict]:
    with session() as s:
        rows = s.exec(select(Run).where(Run.experiment_id == exp_id).order_by(Run.id)).all()
    return [json.loads(r.record_json) for r in rows]


def list_runs(status: Optional[str] = None) -> list[dict]:
    with session() as s:
        q = select(Run)
        if status:
            q = q.where(Run.status == status)
        rows = s.exec(q.order_by(Run.id)).all()
    return [json.loads(r.record_json) for r in rows]


def append_metrics(run_id: str, rows: list[dict]) -> None:
    """rows: [{step, name, value}] — append-only."""
    if not rows:
        return
    rows = [{"step": int(r["step"]), "name": r["name"], "value": float(r["value"])} for r in rows]
    with open(_run_dir(run_id) / "metrics.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")  # NaN/inf are kept in the file, where the reviewer looks for them
    with session() as s:
        s.add_all(Metric(run_id=run_id, **r) for r in rows if math.isfinite(r["value"]))
        s.commit()


def get_metrics(run_id: str, names: Optional[list[str]] = None, after_rowid: int = 0) -> list[dict]:
    with session() as s:
        q = select(Metric).where(Metric.run_id == run_id, Metric.rowid > after_rowid)
        if names:
            q = q.where(Metric.name.in_(names))
        rows = s.exec(q.order_by(Metric.step)).all()
    return [{"step": r.step, "name": r.name, "value": r.value} for r in rows]


def metrics_since(after_rowid: int) -> list[dict]:
    with session() as s:
        rows = s.exec(select(Metric).where(Metric.rowid > after_rowid).order_by(Metric.rowid).limit(2000)).all()
    return [{"rowid": r.rowid, "run_id": r.run_id, "step": r.step, "name": r.name, "value": r.value} for r in rows]


def max_metric_rowid() -> int:
    with session() as s:
        r = s.exec(select(Metric.rowid).order_by(Metric.rowid.desc()).limit(1)).first()
    return r or 0


def write_probe_results(run_id: str, rows: list[dict]) -> None:
    path = _run_dir(run_id) / "probes.json"
    if path.exists():
        raise StoreError("probe results are immutable once written")
    _write_json(path, rows)
    with session() as s:
        s.add_all(ProbeResult(run_id=run_id, **r) for r in rows)
        s.commit()


def get_probe_results(run_id: str) -> list[dict]:
    with session() as s:
        rows = s.exec(select(ProbeResult).where(ProbeResult.run_id == run_id)).all()
    return [{"checkpoint_step": r.checkpoint_step, "layer": r.layer, "target": r.target, "r2": r.r2,
             "control_r2": r.control_r2, "control_spread": r.control_spread} for r in rows]


def write_run_result(run_id: str, name: str, obj: Any) -> None:
    path = _run_dir(run_id) / f"{name}.json"
    if path.exists():
        raise StoreError(f"{name} results are immutable once written")
    _write_json(path, obj)


def read_run_result(run_id: str, name: str):
    return _read_json(_run_dir(run_id) / f"{name}.json")


def run_log_path(run_id: str) -> Path:
    return _run_dir(run_id) / "log.txt"


def compute_used_today() -> float:
    d = today()
    total = 0.0
    for r in list_runs():
        if (r.get("started_at") or "").startswith(d):
            total += r.get("compute_hours", 0.0)
    return total


# ---------------------------------------------------------------- findings

def _finding_path(fid: str) -> Path:
    return lab_dir() / "findings" / f"{fid}.json"


def get_finding(fid: str) -> dict:
    rec = _read_json(_finding_path(fid))
    if rec is None:
        raise KeyError(fid)
    return rec


def list_findings(status: Optional[str] = None) -> list[dict]:
    with session() as s:
        q = select(Finding)
        if status:
            q = q.where(Finding.status == status)
        rows = s.exec(q.order_by(Finding.created_at)).all()
    return [json.loads(r.record_json) for r in rows]


def _save_finding(rec: dict) -> None:
    _write_json(_finding_path(rec["id"]), rec)
    _upsert(_finding_row(rec))


# A run cut short by the budget hard stop before this fraction of its planned steps doesn't count toward replication.
MIN_COMPLETION = 0.8


def run_completion(r: dict) -> float:
    """Fraction of planned training steps the run completed (1.0 unless the hard stop cut it short)."""
    s = r.get("summary") or {}
    if not s.get("stopped"):
        return 1.0
    planned = s.get("steps_planned") or get_experiment(r["experiment_id"])["config"]["training"]["steps"]
    done = s.get("steps_completed")
    if done is None:  # runs from before steps were recorded: last logged training step
        done = max((m["step"] for m in get_metrics(r["id"], ["train_loss"])), default=0)
    return min(1.0, done / max(planned, 1))


def _distinct_completed_seeds(run_ids: list[str]) -> set[int]:
    seeds = set()
    for rid in run_ids:
        r = get_run(rid)
        if r["status"] == "completed" and not r.get("bad") and run_completion(r) >= MIN_COMPLETION:
            seeds.add(r["seed"])
    return seeds


def create_finding(data: dict, actor: str) -> dict:
    """data: claim, verdict, evidence_run_ids, confidence_note, [experiment_id, seeds_agree, replicates]."""
    allowed = {"claim", "verdict", "evidence_run_ids", "confidence_note", "experiment_id", "seeds_agree",
               "replicates", "question_ids"}
    unknown = set(data) - allowed
    if unknown:
        raise StoreError(f"unknown finding fields: {sorted(unknown)}")
    claim = (data.get("claim") or "").strip()
    if len(claim) < 5:
        raise StoreError("finding needs a claim")
    verdict = data.get("verdict")
    if verdict not in VERDICTS:
        raise StoreError(f"verdict must be one of {VERDICTS}")
    ev = list(dict.fromkeys(data.get("evidence_run_ids") or []))
    if not ev:
        raise StoreError("a finding must cite at least one completed run ID")
    for rid in ev:
        try:
            r = get_run(rid)
        except KeyError:
            raise StoreError(f"evidence run {rid} does not exist")
        if r["status"] != "completed":
            raise StoreError(f"evidence run {rid} is {r['status']}, not completed")
    note = (data.get("confidence_note") or "").strip()
    if not note:
        raise StoreError("finding needs a confidence_note")
    exp_id = data.get("experiment_id") or get_run(ev[0])["experiment_id"]
    status = verdict
    if data.get("seeds_agree") and verdict != "inconclusive" and len(_distinct_completed_seeds(ev)) >= 2:
        status = "replicated"
    fid = new_id("f")
    rec = {"id": fid, "claim": claim, "verdict": verdict, "status": status, "evidence_run_ids": ev,
           "confidence_note": note, "created_at": now(), "experiment_id": exp_id, "created_by": actor,
           "replicates": data.get("replicates"), "replications": [], "notes": [], "disputed": False,
           "history": [{"status": status, "at": now(), "actor": actor}]}
    _save_finding(rec)
    log_event(actor, "finding.created", {"id": fid, "verdict": verdict, "status": status, "evidence": ev})

    # A replication finding that agrees promotes the original.
    if data.get("replicates"):
        orig = get_finding(data["replicates"])
        agrees = orig["verdict"] == verdict and verdict != "inconclusive"
        new_seeds = _distinct_completed_seeds(ev) - _distinct_completed_seeds(orig["evidence_run_ids"])
        orig["replications"].append({"finding_id": fid, "agrees": agrees, "run_ids": ev, "at": now()})
        if agrees and new_seeds:
            orig["status"] = "replicated"
            orig["history"].append({"status": "replicated", "at": now(), "actor": actor, "via": fid})
        _save_finding(orig)
        log_event(actor, "finding.replication", {"id": orig["id"], "by": fid, "agrees": agrees})

    for q in data.get("question_ids") or []:
        link_question(q, fid, actor)

    # Analysis of an experiment completes it.
    try:
        exp = get_experiment(exp_id)
        if exp["state"] == "analyzing":
            set_experiment_fields(exp_id, verdict=verdict)
            transition(exp_id, "complete", actor, f"finding {fid}")
        elif exp.get("verdict") is None:
            set_experiment_fields(exp_id, verdict=verdict)
    except KeyError:
        pass
    return rec


def promote_finding(fid: str, status: str, actor: str, note: str = "") -> dict:
    rec = get_finding(fid)
    if status not in FINDING_STATUSES:
        raise StoreError(f"status must be one of {FINDING_STATUSES}")
    if status == "replicated":
        if rec["verdict"] == "inconclusive":
            raise StoreError("inconclusive findings cannot be replicated")
        agreeing = [r for r in rec["replications"] if r["agrees"]]
        if len(_distinct_completed_seeds(rec["evidence_run_ids"])) < 2 and not agreeing:
            raise StoreError("replicated needs a second completed run with a different seed that agrees")
    rec["status"] = status
    rec["history"].append({"status": status, "at": now(), "actor": actor, "note": note})
    _save_finding(rec)
    log_event(actor, "finding.status", {"id": fid, "status": status})
    return rec


def annotate_finding(fid: str, actor: str, note: Optional[str] = None, disputed: Optional[bool] = None) -> dict:
    rec = get_finding(fid)
    if note:
        rec["notes"].append({"at": now(), "actor": actor, "text": note})
    if disputed is not None:
        rec["disputed"] = disputed
        rec["history"].append({"status": rec["status"], "at": now(), "actor": actor,
                               "note": "disputed" if disputed else "dispute cleared"})
    _save_finding(rec)
    log_event(actor, "finding.annotated", {"id": fid, "disputed": rec["disputed"]})
    return rec


# ---------------------------------------------------------------- questions

def create_question(text: str, actor: str) -> dict:
    qid = new_id("q")
    rec = {"id": qid, "text": text.strip(), "status": "open", "linked_finding_ids": [], "created_at": now()}
    _write_json(lab_dir() / "questions" / f"{qid}.json", rec)
    _upsert(_question_row(rec))
    log_event(actor, "question.created", {"id": qid})
    return rec


def get_question(qid: str) -> dict:
    rec = _read_json(lab_dir() / "questions" / f"{qid}.json")
    if rec is None:
        raise KeyError(qid)
    return rec


def update_question(qid: str, actor: str, status: Optional[str] = None, finding_id: Optional[str] = None) -> dict:
    rec = get_question(qid)
    if status:
        if status not in ("open", "answered", "dropped"):
            raise StoreError("question status must be open, answered or dropped")
        rec["status"] = status
    if finding_id and finding_id not in rec["linked_finding_ids"]:
        get_finding(finding_id)
        rec["linked_finding_ids"].append(finding_id)
    _write_json(lab_dir() / "questions" / f"{qid}.json", rec)
    _upsert(_question_row(rec))
    log_event(actor, "question.updated", {"id": qid, "status": rec["status"]})
    return rec


def link_question(qid: str, finding_id: str, actor: str) -> dict:
    return update_question(qid, actor, finding_id=finding_id)


def list_questions(status: Optional[str] = None) -> list[dict]:
    with session() as s:
        q = select(Question)
        if status:
            q = q.where(Question.status == status)
        rows = s.exec(q.order_by(Question.created_at)).all()
    return [{"id": r.id, "text": r.text, "status": r.status, "linked_finding_ids": json.loads(r.linked_finding_ids),
             "created_at": r.created_at} for r in rows]


# ---------------------------------------------------------------- papers

def _paper_paths(pid: str) -> tuple[Path, Path]:
    base = lab_dir() / "papers"
    return base / f"{pid}.json", base / f"{pid}.md"


def create_paper(title: str, actor: str, markdown: str = "") -> dict:
    pid = new_id("p")
    meta = {"id": pid, "title": title, "cited_finding_ids": [], "updated_at": now(), "created_at": now()}
    return save_paper(pid, title, markdown or f"# {title}\n", actor, meta=meta)


def save_paper(pid: str, title: str, markdown: str, actor: str, meta: Optional[dict] = None) -> dict:
    jp, mp = _paper_paths(pid)
    meta = meta or _read_json(jp)
    if meta is None:
        raise KeyError(pid)
    import re

    cited = sorted(set(re.findall(r"\bf-\d{8}-\d{6}-[0-9a-f]{4}\b", markdown)))
    meta.update(title=title, cited_finding_ids=cited, updated_at=now())
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(markdown)
    _write_json(jp, meta)
    _upsert(_paper_row(meta, markdown))
    log_event(actor, "paper.updated", {"id": pid})
    return {**meta, "markdown": markdown}


def get_paper(pid: str) -> dict:
    jp, mp = _paper_paths(pid)
    meta = _read_json(jp)
    if meta is None:
        raise KeyError(pid)
    return {**meta, "markdown": mp.read_text() if mp.exists() else ""}


def list_papers() -> list[dict]:
    with session() as s:
        rows = s.exec(select(Paper).order_by(Paper.updated_at.desc())).all()
    return [{"id": r.id, "title": r.title, "updated_at": r.updated_at,
             "cited_finding_ids": json.loads(r.cited_finding_ids)} for r in rows]
