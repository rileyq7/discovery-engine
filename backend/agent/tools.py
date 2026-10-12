"""The agent's command line. Both entry points use it: the scheduled Claude task (following CYCLE.md)
and agent/cycle.py (driving `claude -p`).

    python -m agent.tools check-billing
    python -m agent.tools status                     # pause flag, budget, cycles today
    python -m agent.tools context                    # everything needed to propose (JSON)
    python -m agent.tools schema                     # the experiment config JSON schema
    python -m agent.tools validate <file|->          # config or list of configs
    python -m agent.tools propose <file|->           # 1-3 configs; auto-approve rules apply
    python -m agent.tools run-queue [--max-minutes N]
    python -m agent.tools pending-analysis           # experiments in `analyzing` with their results
    python -m agent.tools results <experiment_id>
    python -m agent.tools finding <file|->           # one finding (or a list)
    python -m agent.tools question add "<text>" | question set <id> <open|answered|dropped> [--finding <fid>]
    python -m agent.tools review-respond <file|->    # answers to open reviewer issues (see context)
    python -m agent.tools cycle-begin | cycle-end [--summary "..."] | cycle-fail --summary "..."
"""
from __future__ import annotations

import argparse
import json
import re
import sys

from app import lab, store
from app.schemas.config import ExperimentConfig

MAX_PROPOSALS = 3


def _load(path: str):
    text = sys.stdin.read() if path == "-" else open(path).read()
    return json.loads(text)


def _out(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _compact_run(r: dict) -> dict:
    s = r.get("summary") or {}
    out = {"run_id": r["id"], "seed": r["seed"], "status": r["status"], "bad": r.get("bad", False),
           "compute_hours": r.get("compute_hours")}
    if r.get("error"):
        out["error"] = r["error"]
    for k in ("final_val_mse", "baseline_cv_mse", "val_mse_ratio", "beats_baseline", "n_params", "stopped",
              "steps_completed", "steps_planned"):
        if k in s:
            out[k] = s[k]
    if s.get("stopped"):
        out["completion"] = round(store.run_completion(r), 3)
    if s.get("probes"):
        out["probes_final"] = s["probes"]["targets"]
        out["control_margin"] = s["probes"]["control_margin"]
    for k in ("equations", "patching", "eval"):
        if k in s:
            out[k] = s[k]
    if (store._run_dir(r["id"]) / "controls.json").exists():  # single-control run, re-judged with backfilled controls
        from engine.probes.recontrol import rejudge

        out["multi_control_rejudge"] = {t: v["new"] for t, v in rejudge(r["id"]).items()}
    return out


TRAJECTORY_POINTS = 80  # enough for every checkpoint of the dense formation runs


def _subsample(items: list, n: int = TRAJECTORY_POINTS) -> list:
    if len(items) <= n:
        return items
    return [items[i] for i in sorted({round(k * (len(items) - 1) / (n - 1)) for k in range(n)})]


def probe_trajectory(run_id: str) -> dict:
    """Per target: (step, best R2 - control at that step, its layer, max raw R2 over layers) at every checkpoint (subsampled only past
    TRAJECTORY_POINTS, which is then stated), for formation timing."""
    rows = store.get_probe_results(run_id)
    out = {}
    for t in sorted({r["target"] for r in rows}):
        all_steps = sorted({r["checkpoint_step"] for r in rows if r["target"] == t})
        steps = _subsample(all_steps)
        if len(steps) < len(all_steps):
            out[f"{t}_note"] = f"subsampled: {len(steps)} of {len(all_steps)} checkpoints shown"
        traj = []
        for s in steps:
            best = max((r for r in rows if r["target"] == t and r["checkpoint_step"] == s),
                       key=lambda r: r["r2"] - (r["control_r2"] or 0))
            top = max((r for r in rows if r["target"] == t and r["checkpoint_step"] == s), key=lambda r: r["r2"])
            traj.append([s, round(best["r2"] - (best["control_r2"] or 0), 3), best["layer"], round(top["r2"], 3)])
        out[t] = traj
    return out


def experiment_results(exp_id: str) -> dict:
    e = store.get_experiment(exp_id)
    runs = store.runs_for(exp_id)
    ratio_traj = {}
    for r in runs:
        ms = _subsample(store.get_metrics(r["id"], ["val_mse_ratio"]))
        ratio_traj[r["id"]] = [[m["step"], round(m["value"], 4)] for m in ms]
    return {"experiment_id": exp_id, "family": e["family"], "state": e["state"], "hypothesis": e["hypothesis"],
            "expected_if_true": e["expected_if_true"], "expected_if_false": e["expected_if_false"],
            "replicates_finding_id": e.get("replicates_finding_id"), "config": e["config"],
            "runs": [_compact_run(r) for r in runs],
            "val_mse_ratio_trajectory": ratio_traj,
            "probe_trajectory": {r["id"]: probe_trajectory(r["id"]) for r in runs if r["status"] == "completed"}}


def related_experiments(exp_id: str, limit: int = 4) -> list[str]:
    """Other arms of a paired design: experiments proposed in the same batch, plus any exp IDs the
    hypothesis or expectations name. Their finished runs are shown to the analyst so paired comparisons
    are read from run records rather than from other findings' prose."""
    e = store.get_experiment(exp_id)
    batch = exp_id.rsplit("-", 1)[0]
    text = " ".join(str(e.get(k) or "") for k in ("hypothesis", "expected_if_true", "expected_if_false"))
    named = re.findall(r"exp-\d{8}-\d{6}-[0-9a-f]{4}", text)
    out = []
    for x in named + [x["id"] for x in store.list_experiments() if x["id"].startswith(batch + "-")]:
        if x != exp_id and x not in out:
            out.append(x)
    return out[:limit]


def related_results(exp_id: str) -> dict:
    out = {}
    for x in related_experiments(exp_id):
        e = store.get_experiment(x)
        runs = store.runs_for(x)
        if not runs:
            continue
        out[x] = {"hypothesis": e["hypothesis"][:600], "state": e["state"],
                  "runs": [{"run_id": r["id"], "seed": r["seed"], "status": r["status"],
                            "completion": _compact_run(r).get("completion")} for r in runs],
                  "probe_trajectory": {r["id"]: probe_trajectory(r["id"]) for r in runs if r["status"] == "completed"}}
    return out


def status() -> dict:
    s = store.get_settings()
    from app.api.loop import cycles_today

    finished = [e for e in cycles_today() if e["type"] == "cycle.finished"]
    used = store.compute_used_today()
    return {"paused": store.is_paused(), "cycles_today": len(finished), "max_cycles_per_day": s["max_cycles_per_day"],
            "compute_used_today": round(used, 4), "daily_budget_hours": s["daily_budget_hours"],
            "budget_left_today": round(max(0.0, s["daily_budget_hours"] - used), 4),
            "per_experiment_cap_hours": s["per_experiment_cap_hours"],
            "auto_approve_families": s["auto_approve_families"]}


def context() -> dict:
    exps = store.list_experiments()
    recent = exps[-15:]
    return {
        "status": status(),
        "findings": [{k: f[k] for k in ("id", "claim", "verdict", "status", "evidence_run_ids", "confidence_note",
                                        "experiment_id", "disputed")} | {"notes": [n["text"] for n in f["notes"]]}
                     for f in store.list_findings()],
        "open_questions": store.list_questions("open"),
        "open_review_issues": _review_issues(),
        "recent_experiments": [
            {"id": e["id"], "family": e["family"], "state": e["state"], "verdict": e.get("verdict"),
             "hypothesis": e["hypothesis"], "proposed_by": e["proposed_by"],
             "notes": [n["text"] for n in e.get("notes", [])],
             "runs": [_compact_run(r) for r in store.runs_for(e["id"])]} for e in recent],
        "in_flight": [{"id": e["id"], "family": e["family"], "state": e["state"], "hypothesis": e["hypothesis"]}
                      for e in exps if e["state"] in ("proposed", "approved", "queued", "running", "analyzing")],
        "seeded_templates": "configs/*.json (E1-E5) and configs/e6/*.json (E6 grid)",
    }


def _review_issues() -> list[dict]:
    from agent import review

    return [{k: i[k] for k in ("review_id", "issue", "severity", "category", "message", "finding_id", "run_id")
             if k in i} for i in review.open_issues()]


def validate_many(raw) -> list[dict]:
    items = raw if isinstance(raw, list) else [raw]
    out = []
    for i, c in enumerate(items):
        cfg, errors, est = lab.check(c)
        out.append({"index": i, "valid": not errors, "errors": errors, "estimate_hours": round(est, 4)})
    return out


def propose_many(raw, actor: str = "agent") -> dict:
    items = raw if isinstance(raw, list) else [raw]
    if len(items) > MAX_PROPOSALS:
        return {"ok": False, "errors": [f"at most {MAX_PROPOSALS} proposals per cycle, got {len(items)}"]}
    checks = validate_many(items)
    if not all(c["valid"] for c in checks):
        return {"ok": False, "results": checks}
    if store.is_paused():
        return {"ok": False, "errors": ["loop is paused; not proposing"]}
    created = []
    for c in items:
        rec = lab.propose(c, actor)
        created.append({"id": rec["id"], "state": rec["state"], "estimate_hours": rec["estimate_hours"],
                        "note": (rec.get("notes") or [{}])[-1].get("text")})
    return {"ok": True, "created": created}


def create_findings(raw, actor: str = "agent") -> dict:
    items = raw if isinstance(raw, list) else [raw]
    created, errors = [], []
    for i, f in enumerate(items):
        f = dict(f)
        q_updates = f.pop("question_updates", None) or []
        new_qs = f.pop("new_questions", None) or []
        try:
            rec = store.create_finding(f, actor)
            created.append({"id": rec["id"], "status": rec["status"]})
            for u in q_updates:
                try:
                    store.update_question(u["id"], actor, u.get("status"), rec["id"])
                except (KeyError, store.StoreError) as e:
                    errors.append(f"finding[{i}] question update {u}: {e}")
            for q in new_qs:
                store.create_question(q, actor)
        except store.StoreError as e:
            errors.append(f"finding[{i}]: {e}")
    return {"ok": not errors, "created": created, "errors": errors}


def cycle_begin() -> dict:
    st = status()
    if st["paused"]:
        store.log_event("agent", "cycle.skipped", {"reason": "paused"})
        return {"ok": False, "reason": "paused"}
    if st["cycles_today"] >= st["max_cycles_per_day"]:
        store.log_event("agent", "cycle.skipped", {"reason": "max cycles per day reached"})
        return {"ok": False, "reason": "max cycles per day reached"}
    if st["budget_left_today"] <= 0:
        store.log_event("agent", "cycle.skipped", {"reason": "daily compute budget used"})
        return {"ok": False, "reason": "daily compute budget used"}
    store.log_event("agent", "cycle.started", {"cycle_today": st["cycles_today"] + 1})
    return {"ok": True, **st}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agent.tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check-billing")
    sub.add_parser("status")
    sub.add_parser("context")
    sub.add_parser("schema")
    for name in ("validate", "propose", "finding", "review-respond"):
        p = sub.add_parser(name)
        p.add_argument("file")
    p = sub.add_parser("run-queue")
    p.add_argument("--max-minutes", type=float, default=45)
    sub.add_parser("pending-analysis")
    p = sub.add_parser("results")
    p.add_argument("experiment_id")
    p = sub.add_parser("question")
    p.add_argument("action", choices=["add", "set"])
    p.add_argument("args", nargs="+")
    p.add_argument("--finding", default=None)
    sub.add_parser("cycle-begin")
    for name in ("cycle-end", "cycle-fail"):
        p = sub.add_parser(name)
        p.add_argument("--summary", default="")
    a = ap.parse_args(argv)

    if a.cmd == "check-billing":
        from agent.claude_cli import BillingError, check_billing

        try:
            check_billing()
        except BillingError as e:
            print(f"BILLING CHECK FAILED: {e}", file=sys.stderr)
            return 1
        print("billing ok: no ANTHROPIC_API_KEY / CLAUDE_CODE_OAUTH_TOKEN in the environment")
        return 0

    store.rebuild_index()
    if a.cmd == "status":
        _out(status())
    elif a.cmd == "context":
        _out(context())
    elif a.cmd == "schema":
        _out(ExperimentConfig.model_json_schema())
    elif a.cmd == "validate":
        res = validate_many(_load(a.file))
        _out(res)
        return 0 if all(r["valid"] for r in res) else 2
    elif a.cmd == "propose":
        res = propose_many(_load(a.file))
        _out(res)
        return 0 if res["ok"] else 2
    elif a.cmd == "run-queue":
        from engine.worker import drain_queue

        return drain_queue(a.max_minutes)
    elif a.cmd == "pending-analysis":
        _out([experiment_results(e["id"]) for e in store.list_experiments("analyzing")])
    elif a.cmd == "results":
        _out(experiment_results(a.experiment_id))
    elif a.cmd == "finding":
        res = create_findings(_load(a.file))
        _out(res)
        return 0 if res["ok"] else 2
    elif a.cmd == "review-respond":
        from agent import review

        errors = review.record_responses(_load(a.file), actor="agent")
        _out({"ok": not errors, "errors": errors, "still_open": len(review.open_issues())})
        return 0 if not errors else 2
    elif a.cmd == "question":
        if a.action == "add":
            _out(store.create_question(" ".join(a.args), "agent"))
        else:
            qid, st = a.args[0], a.args[1]
            _out(store.update_question(qid, "agent", st, a.finding))
    elif a.cmd == "cycle-begin":
        res = cycle_begin()
        _out(res)
        return 0 if res["ok"] else 3
    elif a.cmd == "cycle-end":
        store.log_event("agent", "cycle.finished", {"summary": a.summary})
        _out({"ok": True})
    elif a.cmd == "cycle-fail":
        store.log_event("agent", "cycle.failed", {"summary": a.summary})
        _out({"ok": True})
    return 0


if __name__ == "__main__":
    sys.exit(main())
