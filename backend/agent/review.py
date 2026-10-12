"""Independent reviewer: checks the research agent's work against the raw lab records, every hour.

It is deliberately separate from the research agent. It runs as its own process, each review is a fresh
`claude -p` session with every tool disabled, and it never sees the research agent's reasoning. It is shown
only the findings' text next to raw data that this module reads from lab/ files. It writes lab/reviews/ and
has one power over the lab: it can pause the loop.

    python -m agent.review                 # one review now
    python -m agent.review --forever       # every --interval minutes (first one after --first-delay)
    python -m agent.review --no-llm        # deterministic checks only

Verdicts: OK; FLAG (issues the research agent must answer next cycle, in `review_responses`); PAUSE (writes
lab/PAUSED, commits it and stops the loop until a human looks). PAUSE is reserved for evidence that doesn't
match the raw data, a broken guardrail, or flags the research agent ignored for two cycles.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from app import store
from engine.paths import REPO_ROOT, lab_dir
from engine.probes.controls import counts

ACTOR = "reviewer"
SEVERITY = {"info": 0, "flag": 1, "pause": 2}
VERDICT = {0: "OK", 1: "FLAG", 2: "PAUSE"}
GRAVITY_TARGETS = ("force", "accel_step")
IGNORED_CYCLES_BEFORE_PAUSE = 2
DOWNGRADES = {"supported", "refuted", "inconclusive"}  # statuses the research agent may set when answering a review
PROTECTED_FINDING_FIELDS = ("claim", "verdict", "evidence_run_ids", "confidence_note", "created_at", "experiment_id")
RESULT_FILES = ("probes.json", "patching.json", "equations.json", "eval.json", "controls.json")


def log(msg: str) -> None:
    print(f"[review {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ review records

def reviews_dir() -> Path:
    return lab_dir() / "reviews"


def list_reviews() -> list[dict]:
    """All reviews, oldest first."""
    d = reviews_dir()
    return [json.loads(p.read_text()) for p in sorted(d.glob("r-*.json"))] if d.exists() else []


def _save(rec: dict) -> None:
    store._write_json(reviews_dir() / f"{rec['id']}.json", rec)
    (reviews_dir() / f"{rec['id']}.md").write_text(render_markdown(rec))


def open_issues() -> list[dict]:
    """FLAG/PAUSE issues that the research agent hasn't answered yet."""
    out = []
    for r in list_reviews():
        answered = {x["issue"] for x in r.get("responses", [])}
        for i, iss in enumerate(r["issues"]):
            if SEVERITY[iss["severity"]] >= 1 and i not in answered:
                out.append({"review_id": r["id"], "issue": i, "created_at": r["created_at"], **iss})
    return out


def record_responses(responses: list[dict], actor: str = "agent") -> list[str]:
    """Attach the research agent's answers to review issues. A response that names a finding is also added
    to that finding as a note, since past findings can't be edited."""
    errors = []
    by_id = {r["id"]: r for r in list_reviews()}
    touched = set()
    for resp in responses or []:
        rec = by_id.get(resp.get("review_id"))
        idx = resp.get("issue")
        text = (resp.get("response") or "").strip()
        if not rec or not isinstance(idx, int) or not 0 <= idx < len(rec["issues"]) or not text:
            errors.append(f"bad review response: {resp}")
            continue
        rec.setdefault("responses", [])
        rec["responses"] = [x for x in rec["responses"] if x["issue"] != idx]
        rec["responses"].append({"issue": idx, "response": text, "at": store.now(), "actor": actor})
        fid = resp.get("finding_id") or rec["issues"][idx].get("finding_id")
        if fid:
            try:
                store.annotate_finding(fid, actor, note=f"Response to review {rec['id']} (issue {idx}): {text}")
                new = resp.get("set_status")
                if new:  # the agent may correct an overclaim it accepts, never upgrade
                    if new not in DOWNGRADES:
                        raise store.StoreError(f"set_status must be one of {sorted(DOWNGRADES)}")
                    if store.get_finding(fid)["status"] != new:
                        store.promote_finding(fid, new, actor, note=f"corrected after review {rec['id']} issue {idx}")
            except (KeyError, store.StoreError) as e:
                errors.append(f"{fid}: {e}")
        touched.add(rec["id"])
    for rid in touched:
        rec = by_id[rid]
        answered = {x["issue"] for x in rec["responses"]}
        if all(i in answered for i, iss in enumerate(rec["issues"]) if SEVERITY[iss["severity"]] >= 1):
            rec["status"] = "addressed"
        _save(rec)
        store.log_event(actor, "review.responded", {"id": rid, "status": rec["status"]})
    return errors


# ------------------------------------------------------------------ raw evidence (read from files, not summaries)

def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def raw_run(run_id: str) -> dict:
    """What the run's own files say: metrics, probes per layer, completion. No agent-written text."""
    r = store.get_run(run_id)
    s = r.get("summary") or {}
    metrics = store.get_metrics(run_id, ["val_mse", "baseline_cv_mse", "val_mse_ratio"])
    by_step: dict[int, dict] = {}
    for m in metrics:
        by_step.setdefault(m["step"], {})[m["name"]] = m["value"]
    steps = sorted(by_step)
    rows = store.get_probe_results(run_id)
    final = max((p["checkpoint_step"] for p in rows), default=None)
    probes_final, trajectory = {}, {}
    for t in sorted({p["target"] for p in rows}):
        fin = sorted((p for p in rows if p["target"] == t and p["checkpoint_step"] == final), key=lambda p: p["layer"])
        probes_final[t] = [[p["layer"], round(p["r2"], 4), round(p["control_r2"] or 0, 4),
                            None if p.get("control_spread") is None else round(p["control_spread"], 4)] for p in fin]
        traj = []
        for st in sorted({p["checkpoint_step"] for p in rows if p["target"] == t}):
            at = [p for p in rows if p["target"] == t and p["checkpoint_step"] == st]
            b = max(at, key=lambda p: p["r2"] - (p["control_r2"] or 0))
            top = max(at, key=lambda p: p["r2"])
            traj.append([st, b["layer"], round(b["r2"], 4), round(b["control_r2"] or 0, 4),
                         round(b["r2"] - (b["control_r2"] or 0), 4), top["layer"], round(top["r2"], 4)])
        trajectory[t] = traj
    out = {"run_id": run_id, "experiment_id": r["experiment_id"], "seed": r["seed"], "status": r["status"],
           "bad": r.get("bad", False), "stopped": bool(s.get("stopped")),
           "completion": round(store.run_completion(r), 3) if r["status"] == "completed" else None,
           "final_metrics": by_step[steps[-1]] if steps else None,
           "val_mse_ratio_by_step": [[st, round(by_step[st]["val_mse_ratio"], 5)] for st in steps
                                     if "val_mse_ratio" in by_step[st]],
           "probe_final_checkpoint": final,
           "probes_final [layer, r2, best_control_r2, control_spread]": probes_final,
           "probe_by_step [step, best_margin_layer, its_r2, its_control_r2, margin, max_r2_layer, max_r2]":
               trajectory}
    for k in ("equations", "eval", "patching"):
        if s.get(k):
            out[k] = s[k]
    if (store._run_dir(run_id) / "controls.json").exists():
        from engine.probes.recontrol import rejudge

        out["multi_control_rejudge (old single-control rule vs best-of-3 rule)"] = rejudge(run_id)
    return out


def noise_floor() -> dict:
    """Per target, the largest |R2 - control| at step 0, where the model is itself a random init. Margins
    below this are within random-initialisation noise."""
    out: dict[str, float] = {}
    for r in store.list_runs():
        for p in store.get_probe_results(r["id"]):
            if p["checkpoint_step"] == 0 and p["control_r2"] is not None:
                out[p["target"]] = max(out.get(p["target"], 0.0), abs(p["r2"] - p["control_r2"]))
    return {t: round(v, 3) for t, v in out.items()}


# ------------------------------------------------------------------ deterministic checks

def _issue(severity: str, category: str, message: str, **refs) -> dict:
    return {"severity": severity, "category": category, "message": message, "source": "check",
            **{k: v for k, v in refs.items() if v}}


def _git(*args: str) -> str:
    p = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)  # noqa: S603
    return p.stdout if p.returncode == 0 else ""


def check_evidence(findings: list[dict]) -> list[dict]:
    issues = []
    for f in findings:
        if not f.get("evidence_run_ids"):
            issues.append(_issue("pause", "evidence_mismatch", "finding cites no runs", finding_id=f["id"]))
        for rid in f.get("evidence_run_ids", []):
            try:
                r = store.get_run(rid)
            except KeyError:
                issues.append(_issue("pause", "evidence_mismatch", f"cites run {rid}, which does not exist",
                                     finding_id=f["id"]))
                continue
            if r["status"] != "completed":
                issues.append(_issue("pause", "evidence_mismatch", f"cites run {rid}, which is {r['status']}",
                                     finding_id=f["id"]))
        if f["status"] == "replicated":
            runs = list(f.get("evidence_run_ids", []))
            for rep in f.get("replications", []):
                if rep.get("agrees"):
                    runs += rep.get("run_ids", [])
            valid = store._distinct_completed_seeds([r for r in runs if _exists(r)])
            if len(valid) < 2:
                issues.append(_issue("flag", "bug", f"marked replicated but only {len(valid)} seed(s) are completed, "
                                     "not flagged bad, and at least 80% trained", finding_id=f["id"]))
    return issues


def check_runs(runs: list[dict]) -> list[dict]:
    """Bug signatures in runs that finished since the last review."""
    issues = []
    for r in runs:
        rid = r["id"]
        if r["status"] == "failed":
            issues.append(_issue("flag", "bug", f"run failed: {(r.get('error') or '').strip()[-300:]}", run_id=rid))
            continue
        if r["status"] != "completed":
            continue
        mf = store._run_dir(rid) / "metrics.jsonl"  # the raw file: the index skips non-finite values
        bad = [m for m in (json.loads(x) for x in (mf.read_text().splitlines() if mf.exists() else []) if x.strip())
               if not _finite(m["value"])]
        if bad:
            issues.append(_issue("flag", "bug", f"{len(bad)} non-finite metric values (first: {bad[0]['name']} at "
                                 f"step {bad[0]['step']})", run_id=rid))
        rows = store.get_probe_results(rid)
        if not rows:
            issues.append(_issue("flag", "bug", "completed run has no probe results", run_id=rid))
        if any(not _finite(p["r2"]) for p in rows):
            issues.append(_issue("flag", "bug", "non-finite probe R2", run_id=rid))
        s = r.get("summary") or {}
        for key in ("val_mse_ratio", "final_val_mse"):
            if not _finite(s.get(key)):
                issues.append(_issue("flag", "bug", f"summary.{key} is missing or not finite", run_id=rid))
        margin = store.get_experiment(r["experiment_id"])["config"]["probes"].get("control_margin", 0.1)
        for p in rows:
            # At step 0 the model is itself a random init, so beating the random controls means a leak or a bug.
            # Only force/accel_step: random models agree on them (step-0 excess <= 0.014 in 40 runs), while
            # distance/energy differ between random inits by up to 0.2 (0.45 on tiny probe sets), so a
            # chance excess there is noise, not a bug signature.
            if p["checkpoint_step"] == 0 and p["target"] in GRAVITY_TARGETS \
                    and counts(p["r2"], p["control_r2"], margin, p.get("control_spread")):
                issues.append(_issue("flag", "bug", f"{p['target']} probe beats its random controls by "
                                     f"{p['r2'] - p['control_r2']:.3f} at step 0, before any training (layer "
                                     f"{p['layer']}): leakage or a control bug", run_id=rid))
                break
        if s.get("stopped"):
            c = store.run_completion(r)
            sev = "flag" if c < store.MIN_COMPLETION else "info"
            issues.append(_issue(sev, "bug", f"stopped early by the budget hard stop at {c:.0%} of planned steps",
                                 run_id=rid))
    # identical seeds: two runs of one experiment with the same validation curve
    by_exp: dict[str, list[dict]] = {}
    for r in runs:
        if r["status"] == "completed":
            by_exp.setdefault(r["experiment_id"], []).append(r)
    for exp_id, rs in by_exp.items():
        rs = [x for x in store.runs_for(exp_id) if x["status"] == "completed"]
        curves = {x["id"]: tuple(round(m["value"], 12) for m in store.get_metrics(x["id"], ["val_mse"])) for x in rs}
        ids = list(curves)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if curves[ids[i]] and curves[ids[i]] == curves[ids[j]]:
                    issues.append(_issue("flag", "bug", f"runs {ids[i]} and {ids[j]} have identical val_mse curves "
                                         "despite different seeds", run_id=ids[i]))
    return issues


def check_immutability() -> list[dict]:
    """Raw results are append-only and findings' substance never changes after it is recorded."""
    issues = []
    try:
        lab_rel = lab_dir().resolve().relative_to(Path(REPO_ROOT).resolve()).as_posix()
    except ValueError:
        return []  # the lab is not inside this git repo (e.g. tests); nothing to compare against
    pats = [f"{lab_rel}/runs/*/{n}" for n in RESULT_FILES]
    modified = set(_git("log", "--diff-filter=M", "--format=", "--name-only", "--", *pats).split())
    modified |= set(_git("diff", "--diff-filter=M", "--name-only", "HEAD", "--", *pats).split())
    for path in sorted(modified):
        issues.append(_issue("pause", "guardrail", f"result file {path} was modified after it was written"))
    mpat = f"{lab_rel}/runs/*/metrics.jsonl"
    for out in (_git("log", "--numstat", "--format=", "--", mpat), _git("diff", "--numstat", "HEAD", "--", mpat)):
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[1].isdigit() and int(parts[1]) > 0:
                issues.append(_issue("pause", "guardrail", f"lines were removed from {parts[2]} (metrics are "
                                     "append-only)"))
    for path in sorted((lab_dir() / "findings").glob("*.json")):
        rel = f"{lab_rel}/findings/{path.name}"
        first = _git("log", "--diff-filter=A", "--format=%H", "--", rel).split()
        if not first:
            continue  # not committed yet
        try:
            orig = json.loads(_git("show", f"{first[-1]}:{rel}"))
        except json.JSONDecodeError:
            continue
        cur = json.loads(path.read_text())
        changed = [k for k in PROTECTED_FINDING_FIELDS if orig.get(k) != cur.get(k)]
        if changed:
            issues.append(_issue("pause", "guardrail", f"finding fields {changed} changed after it was recorded",
                                 finding_id=cur["id"]))
    return issues


def check_guardrails() -> list[dict]:
    issues = []
    p = subprocess.run(["make", "-s", "check-billing"], cwd=REPO_ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        issues.append(_issue("pause", "guardrail", f"check-billing failed: {(p.stderr or p.stdout).strip()[-300:]}"))
    s = store.get_settings()
    used, budget = store.compute_used_today(), float(s["daily_budget_hours"])
    if used > budget * 1.25:
        issues.append(_issue("pause", "guardrail", f"compute used today {used:.2f} h is over 125% of the daily "
                             f"budget ({budget} h)"))
    elif used > budget:
        issues.append(_issue("flag", "guardrail", f"compute used today {used:.2f} h exceeds the daily budget "
                             f"({budget} h)"))
    for e in store.list_experiments():
        runs = [r for r in store.runs_for(e["id"]) if r.get("compute_hours")]
        cap = e["config"]["budget"]["max_compute_hours"]
        total = sum(r["compute_hours"] for r in runs)
        if runs and total > cap * 2.5:  # the hard stop is 2x per seed; probes and data add a little
            issues.append(_issue("flag", "guardrail", f"experiment {e['id']} used {total:.3f} h against a "
                                 f"{cap} h budget"))
    return issues


def check_ignored_flags() -> list[dict]:
    """Escalate FLAGs that the research agent left unanswered for IGNORED_CYCLES_BEFORE_PAUSE cycles."""
    issues = []
    finished = [e["timestamp"] for e in _events("cycle.finished")]
    for iss in open_issues():
        n = sum(t > iss["created_at"] for t in finished)
        if n >= IGNORED_CYCLES_BEFORE_PAUSE:
            issues.append(_issue("pause", "guardrail", f"review {iss['review_id']} issue {iss['issue']} has gone "
                                 f"unanswered for {n} cycles: {iss['message'][:200]}"))
    return issues


def _events(type_: str) -> list[dict]:
    out = []
    for f in sorted((lab_dir() / "events").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                if e["type"] == type_:
                    out.append(e)
    return out


# ------------------------------------------------------------------ the independent judgement (claude -p, no tools)

def build_prompt(new_findings: list[dict], new_experiments: list[dict], check_issues: list[dict]) -> str:
    findings = []
    history = _issue_history()
    for f in new_findings:
        e = store.get_experiment(f["experiment_id"])
        findings.append({
            "finding": {k: f[k] for k in ("id", "claim", "verdict", "status", "confidence_note", "evidence_run_ids")},
            "notes_and_status_history": {"notes": [f"{n['actor']}: {n['text'][:600]}" for n in f.get("notes", [])],
                                         "status_history": f.get("history", [])},
            "earlier_review_issues_on_this_finding": history.get(f["id"], []),
            "experiment": {"id": e["id"], "family": e["family"], "hypothesis": e["hypothesis"],
                           "expected_if_true": e["expected_if_true"], "expected_if_false": e["expected_if_false"],
                           "control_margin": e["config"]["probes"].get("control_margin", 0.1)},
            "raw_evidence": [raw_run(r) for r in f["evidence_run_ids"] if _exists(r)],
        })
    recent = store.list_experiments()[-5:]
    payload = {
        "new_or_changed_findings": findings,
        "random_init_noise_floor (max |R2 - control| at step 0, per target)": noise_floor(),
        "last_5_experiments": [{"id": e["id"], "family": e["family"], "state": e["state"],
                                "hypothesis": e["hypothesis"], "proposed_by": e["proposed_by"]} for e in recent],
        "new_experiments_since_last_review": [e["id"] for e in new_experiments],
        "open_questions": [{"id": q["id"], "text": q["text"]} for q in store.list_questions("open")],
        "deterministic_check_results": check_issues,
    }
    return ("Review the research agent's latest work. Everything you need is below; you have no tools.\n\n"
            "```json\n" + json.dumps(payload, indent=1, default=str) + "\n```\n")


def _issue_history() -> dict[str, list[dict]]:
    """Per finding: issues earlier reviews raised about it, with the answers given."""
    out: dict[str, list[dict]] = {}
    for r in list_reviews():
        answers = {x["issue"]: x for x in r.get("responses", [])}
        for i, iss in enumerate(r["issues"]):
            if iss.get("finding_id") and iss["severity"] != "info":
                a = answers.get(i)
                out.setdefault(iss["finding_id"], []).append({
                    "review": r["id"], "category": iss["category"], "message": iss["message"][:500],
                    "answer": f"{a['actor']}: {a['response'][:600]}" if a else None})
    return out


def _exists(run_id: str) -> bool:
    try:
        store.get_run(run_id)
        return True
    except KeyError:
        return False


def llm_review(prompt: str) -> dict:
    from agent.claude_cli import extract_json, run_claude

    system = (Path(__file__).parent / "REVIEW.md").read_text()
    last_err = None
    for _ in range(2):
        text = run_claude(prompt if last_err is None else prompt + f"\n\nYour last reply was not valid ({last_err}). "
                          "Reply with ONLY the JSON object.", system=system, no_tools=True,
                          role="reviewer")
        try:
            out = extract_json(text)
            if out.get("verdict") in ("OK", "FLAG", "PAUSE") and isinstance(out.get("issues"), list):
                return out
            last_err = "missing verdict or issues"
        except Exception as e:  # noqa: BLE001
            last_err = str(e)[:200]
    raise RuntimeError(f"reviewer reply unusable: {last_err}")


def normalise_llm_issues(out: dict) -> list[dict]:
    """The reviewer may PAUSE only for evidence that doesn't match the raw data or a broken guardrail."""
    issues = []
    for iss in out.get("issues", []):
        sev = iss.get("severity") if iss.get("severity") in SEVERITY else "flag"
        cat = iss.get("category") or "other"
        if sev == "pause" and cat not in ("evidence_mismatch", "guardrail"):
            sev = "flag"
        issues.append({"severity": sev, "category": cat, "message": str(iss.get("message", ""))[:2000],
                       "source": "reviewer", **({"finding_id": iss["finding_id"]} if iss.get("finding_id") else {})})
    return issues


# ------------------------------------------------------------------ one review

def run_review(use_llm: bool = True, llm: Optional[Callable[[str], dict]] = None, commit: bool = True,
               push: bool = True) -> dict:
    store.rebuild_index()
    prev = list_reviews()
    since = prev[-1]["created_at"] if prev else ""
    status_changed = {e["payload"].get("id") for e in _events("finding.status") if e["timestamp"] > since}
    new_findings = [f for f in store.list_findings() if f["created_at"] > since or f["id"] in status_changed]
    new_runs = [r for r in store.list_runs() if (r.get("ended_at") or "") > since]
    new_exps = [e for e in store.list_experiments() if e["created_at"] > since]

    issues = (check_evidence(new_findings) + check_runs(new_runs) + check_immutability() + check_guardrails()
              + check_ignored_flags())
    summary, used_llm = "", False
    if use_llm and (new_findings or new_exps):
        try:
            out = (llm or llm_review)(build_prompt(new_findings, new_exps, issues))
            issues += normalise_llm_issues(out)
            summary, used_llm = str(out.get("summary", ""))[:3000], True
        except Exception as e:  # noqa: BLE001
            issues.append(_issue("flag", "bug", f"the reviewer session failed: {str(e)[:300]}"))
    level = max((SEVERITY[i["severity"]] for i in issues), default=0)
    verdict = VERDICT[level]
    if not summary:
        summary = (f"{len(new_findings)} new or changed finding(s), {len(new_runs)} finished run(s), "
                   f"{len(new_exps)} new experiment(s) since the last review.")
    rec = {"id": store.new_id("r"), "created_at": store.now(), "since": since or None, "verdict": verdict,
           "summary": summary, "issues": issues, "status": "open" if level >= 1 else "ok", "responses": [],
           "llm": used_llm, "scope": {"findings": [f["id"] for f in new_findings], "runs": [r["id"] for r in new_runs],
                                      "experiments": [e["id"] for e in new_exps]}}
    _save(rec)
    store.log_event(ACTOR, f"review.{verdict.lower()}", {"id": rec["id"], "issues": len(issues)})
    if verdict == "PAUSE" and not store.is_paused():
        store.set_paused(True, ACTOR, reason=f"review {rec['id']}")
        log("PAUSED the loop")
    log(f"{verdict}: {len(issues)} issue(s) -> lab/reviews/{rec['id']}.md")
    if commit:
        commit_review(rec, push)
    return rec


def commit_review(rec: dict, push: bool) -> None:
    from agent.cycle import git, git_lock, has_remote

    paths = ["lab/reviews"] + (["lab/PAUSED"] if (lab_dir() / "PAUSED").exists() else [])
    with git_lock():
        git("add", "-A", "--", *paths)
        if not git("diff", "--cached", "--quiet", "--", *paths).returncode:
            return
        r = git("commit", "-m", f"review {rec['id']}: {rec['verdict']} ({len(rec['issues'])} issue(s))", "--", *paths)
        if r.returncode:
            log(f"commit failed: {r.stderr.strip()[:300]}")
            return
        if push and has_remote():
            for _ in range(3):
                if not git("push").returncode:
                    break
                git("pull", "--rebase", "--autostash")


def render_markdown(rec: dict) -> str:
    lines = [f"# Review {rec['id']}: {rec['verdict']}", "",
             f"{rec['created_at']} · since {rec['since'] or 'the beginning'} · "
             f"{'independent claude -p session + checks' if rec['llm'] else 'deterministic checks only'}", "",
             rec["summary"], ""]
    if rec["issues"]:
        lines += ["## Issues", ""]
        for i, iss in enumerate(rec["issues"]):
            ref = " ".join(f"`{iss[k]}`" for k in ("finding_id", "run_id") if iss.get(k))
            lines.append(f"{i}. **{iss['severity'].upper()}** ({iss['category']}, {iss['source']}) {ref} {iss['message']}")
    if rec.get("responses"):
        lines += ["", "## Responses from the research agent", ""]
        for x in rec["responses"]:
            lines.append(f"- issue {x['issue']} ({x['actor']}, {x['at']}): {x['response']}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agent.review")
    ap.add_argument("--forever", action="store_true")
    ap.add_argument("--interval", type=float, default=60, help="minutes between reviews")
    ap.add_argument("--first-delay", type=float, default=0, help="minutes before the first review (--forever)")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args(argv)
    if not a.forever:
        run_review(use_llm=not a.no_llm, push=not a.no_push)
        return 0
    log(f"reviewer: every {a.interval} min, first in {a.first_delay} min")
    time.sleep(a.first_delay * 60)
    while True:
        try:
            run_review(use_llm=not a.no_llm, push=not a.no_push)
        except Exception as e:  # noqa: BLE001 - keep reviewing
            log(f"review failed: {e}")
        time.sleep(a.interval * 60)


if __name__ == "__main__":
    sys.exit(main())
