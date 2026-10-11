"""One full agent cycle for local runs, driving headless Claude Code (`claude -p`) on the Max subscription.

    python -m agent.cycle              # one cycle: pull, propose, run, analyze, commit, push
    python -m agent.cycle --forever    # loop: a cycle every --interval minutes while not paused

Steps (SPEC "One cycle"): check billing -> git pull -> analyze anything left in `analyzing` ->
propose (validate, 2 retries with errors fed back) -> auto-approve/gate -> run + probe (worker) ->
analyze -> update questions -> maybe draft paper section -> commit lab/ -> push.
The deterministic parts are code; the model only writes JSON (configs, findings) and prose (papers).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from agent.claude_cli import BillingError, check_billing, extract_json, run_claude
from agent import tools
from app import store
from app.schemas.config import ExperimentConfig
from engine.paths import REPO_ROOT, runs_dir

PROMPTS = Path(__file__).parent / "prompts"
MAX_RETRIES = 2


def log(msg: str) -> None:
    print(f"[cycle {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=check)


def has_remote() -> bool:
    return bool(git("remote").stdout.strip())


def git_pull() -> None:
    if has_remote():
        r = git("pull", "--rebase", "--autostash")
        log(f"git pull: {(r.stdout or r.stderr).strip().splitlines()[-1:] or ['ok']}")


def git_commit_push(message: str, push: bool = True) -> bool:
    # snapshot/ is the read-only copy of the lab that the hosted dashboard (Vercel) serves
    try:
        from app.snapshot import export

        export(REPO_ROOT / "snapshot")
    except Exception as e:
        log(f"snapshot export failed (committing lab/ anyway): {e}")
    paths = ["lab", "snapshot"] if (REPO_ROOT / "snapshot").exists() else ["lab"]
    git("add", "-A", "--", *paths)
    if not git("diff", "--cached", "--quiet", "--", *paths).returncode:
        log("nothing to commit")
        return False
    r = git("commit", "-m", message, "--", *paths)
    if r.returncode:
        log(f"commit failed: {r.stderr.strip()}")
        return False
    if push and has_remote():
        for attempt in range(3):
            p = git("push")
            if not p.returncode:
                break
            log(f"push rejected ({p.stderr.strip()[:200]}); rebasing and retrying")
            git("pull", "--rebase", "--autostash")
        else:
            log("push failed after 3 attempts; commit is kept locally")
    return True


def system_prompt() -> str:
    return (PROMPTS / "system.md").read_text()


def ask_json(prompt: str):
    text = run_claude(prompt, system=system_prompt())
    return extract_json(text)


# ------------------------------------------------------------------ propose

def propose() -> list[dict]:
    st = tools.status()
    ctx = tools.context()
    ref = json.loads((REPO_ROOT / "configs" / "e2.json").read_text())
    prompt = (PROMPTS / "propose.md").read_text().format(
        max_proposals=tools.MAX_PROPOSALS, budget_left=st["budget_left_today"], cap=st["per_experiment_cap_hours"],
        reference_config=json.dumps(ref, indent=2), context=json.dumps(ctx, indent=2, default=str),
        schema=json.dumps(ExperimentConfig.model_json_schema()))
    feedback = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            out = ask_json(prompt + feedback)
        except Exception as e:
            feedback = f"\n\nYour previous reply could not be parsed ({e}). Reply with ONLY the JSON object."
            log(f"proposal attempt {attempt + 1}: unparseable reply")
            continue
        proposals = out.get("proposals") if isinstance(out, dict) else out
        if not isinstance(proposals, list) or not proposals:
            feedback = "\n\nYour previous reply had no `proposals` list. Reply with ONLY the JSON object."
            continue
        proposals = proposals[: tools.MAX_PROPOSALS]
        checks = tools.validate_many(proposals)
        bad = [c for c in checks if not c["valid"]]
        if bad:
            log(f"proposal attempt {attempt + 1}: {len(bad)} invalid config(s): {bad}")
            if attempt < MAX_RETRIES or len(bad) == len(proposals):
                feedback = ("\n\n## Your previous proposals were rejected by the validator\n"
                            f"```json\n{json.dumps(bad, indent=2)}\n```\nFix every error and reply with the full JSON object again.")
                continue
            # last attempt: keep the valid ones rather than losing the whole cycle
            proposals = [p for p, c in zip(proposals, checks) if c["valid"]]
            log(f"dropping {len(bad)} invalid config(s), proposing the {len(proposals)} valid one(s)")
        res = tools.propose_many(proposals)
        if not res["ok"]:
            log(f"proposal not recorded: {res}")
            return []
        for q in (out.get("new_questions") or [])[:2] if isinstance(out, dict) else []:
            store.create_question(q, "agent")
        log(f"proposed {len(res['created'])}: {res['created']} — {out.get('reasoning', '')[:300]}")
        return res["created"]
    store.log_event("agent", "cycle.proposal_failed", {"retries": MAX_RETRIES})
    log("no valid proposal after retries")
    return []


# ------------------------------------------------------------------ analyze

def analyze(exp_id: str) -> dict | None:
    results = tools.experiment_results(exp_id)
    prompt = (PROMPTS / "finding_template.md").read_text().format(
        replicates=results.get("replicates_finding_id") or "null", experiment_id=exp_id,
        questions=json.dumps(store.list_questions("open"), indent=2),
        results=json.dumps(results, indent=2, default=str))
    feedback = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            f = ask_json(prompt + feedback)
        except Exception as e:
            feedback = f"\n\nYour previous reply could not be parsed ({e}). Reply with ONLY the JSON object."
            continue
        if not isinstance(f, dict):
            feedback = "\n\nReply with ONLY one JSON object."
            continue
        f["experiment_id"] = exp_id
        if not f.get("replicates"):
            f.pop("replicates", None)
        res = tools.create_findings(f)
        if res["created"]:
            if res["errors"]:
                log(f"finding recorded with warnings: {res['errors']}")
            log(f"finding {res['created'][0]} for {exp_id}: {f.get('verdict')} — {f.get('claim', '')[:200]}")
            return res["created"][0]
        log(f"finding attempt {attempt + 1} rejected: {res['errors']}")
        feedback = (f"\n\n## Your previous finding was rejected\n{json.dumps(res['errors'])}\n"
                    "Fix it and reply with the full JSON object again.")
    store.add_experiment_note(exp_id, "agent could not produce a valid finding; needs human analysis", "agent")
    return None


# ------------------------------------------------------------------ paper

def maybe_draft_paper() -> None:
    """Keep a running draft: redraft its Results section when the set of citable findings changes."""
    citable = sorted(f["id"] for f in store.list_findings() if f["status"] == "replicated")
    papers = [p for p in store.list_papers() if p["title"] == "Discovery Engine: running draft"]
    if papers:
        pid = papers[0]["id"]
    else:
        pid = store.create_paper("Discovery Engine: running draft", "agent")["id"]
    meta = store.get_paper(pid)
    if meta.get("last_drafted_with") == citable and meta.get("markdown", "").count("## ") > 0:
        return
    from agent.papers import draft_section

    allow = [] if citable else [f["id"] for f in store.list_findings()][-6:]
    section = "Results" if citable else "Preliminary observations (unreplicated)"
    log(f"drafting paper section '{section}' with {len(citable)} replicated finding(s)")
    try:
        draft_section(pid, section, "Summarise what the lab has established so far, including null results.",
                      allow, actor="agent", extra_meta={"last_drafted_with": citable})
    except Exception as e:
        log(f"paper drafting failed: {e}")


# ------------------------------------------------------------------ cycle

def run_cycle(max_minutes: float, push: bool = True, draft_paper: bool = True) -> int:
    try:
        check_billing()
    except BillingError as e:
        log(str(e))
        return 1
    runs_dir().mkdir(parents=True, exist_ok=True)
    lock = open(runs_dir() / ".cycle.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another cycle is running; skipping")
        return 0
    if push:
        git_pull()
    store.rebuild_index()
    begin = tools.cycle_begin()
    if not begin["ok"]:
        log(f"cycle skipped: {begin['reason']}")
        return 0
    n = begin["cycles_today"] + 1
    summary = []
    try:
        # Leftovers first: anything finished but not yet analyzed.
        for e in store.list_experiments("analyzing"):
            fid = analyze(e["id"])
            summary.append(f"analyzed {e['id']} -> {fid}")
        created = propose()
        summary.append(f"proposed {[c['id'] + ':' + c['state'] for c in created]}")
        from engine.worker import drain_queue

        drain_queue(max_minutes)
        for e in store.list_experiments("analyzing"):
            fid = analyze(e["id"])
            summary.append(f"analyzed {e['id']} -> {fid}")
        if draft_paper:
            maybe_draft_paper()
        store.log_event("agent", "cycle.finished", {"summary": summary})
        msg = f"cycle {store.today()} #{n}: " + "; ".join(summary)
    except Exception as e:
        store.log_event("agent", "cycle.failed", {"error": f"{type(e).__name__}: {e}", "summary": summary})
        msg = f"cycle {store.today()} #{n} FAILED: {type(e).__name__}: {e}"
        log(msg)
    git_commit_push(msg[:2000], push=push)
    log(msg)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--forever", action="store_true")
    ap.add_argument("--interval", type=float, default=60, help="minutes between cycle starts (with --forever)")
    ap.add_argument("--max-minutes", type=float, default=45, help="worker time limit per cycle")
    ap.add_argument("--no-push", action="store_true", help="commit locally but do not push/pull")
    ap.add_argument("--no-paper", action="store_true")
    a = ap.parse_args(argv)
    if not a.forever:
        return run_cycle(a.max_minutes, push=not a.no_push, draft_paper=not a.no_paper)
    last = 0.0
    log(f"local loop: a cycle every {a.interval} min while lab/PAUSED is absent")
    while True:
        if not store.is_paused() and time.time() - last >= a.interval * 60:
            last = time.time()
            run_cycle(a.max_minutes, push=not a.no_push, draft_paper=not a.no_paper)
        time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
