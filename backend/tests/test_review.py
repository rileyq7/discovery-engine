"""The independent reviewer: deterministic checks, verdict rules, pausing, and the research agent's answers."""
import json
import subprocess
import time

from tests.conftest import tiny_config


def _completed_experiment(seeds=(0, 1)):
    from app import lab, store
    from engine.worker import Worker

    rec = lab.propose(tiny_config(seeds=list(seeds)), "agent")
    w = Worker(inline=True)
    for _ in range(20):
        w.tick()
        if not w.busy():
            break
    runs = store.runs_for(rec["id"])
    assert all(r["status"] == "completed" for r in runs)
    return rec, runs


def _finding(runs, **over):
    from agent import tools

    f = {"claim": "Tiny model result", "verdict": "refuted", "evidence_run_ids": [r["id"] for r in runs],
         "confidence_note": "tiny", "seeds_agree": len(runs) > 1, **over}
    res = tools.create_findings(f)
    assert res["ok"], res
    return res["created"][0]["id"]


def test_ok_review_sends_raw_evidence_to_a_tool_less_session(fresh_lab):
    from agent import review
    from app import store

    _, runs = _completed_experiment()
    fid = _finding(runs)
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return {"verdict": "OK", "summary": "numbers match", "issues": []}

    rec = review.run_review(llm=fake_llm, commit=False)
    assert rec["verdict"] == "OK" and rec["status"] == "ok" and rec["llm"]
    payload = json.loads(seen["prompt"].split("```json\n", 1)[1].rsplit("```", 1)[0])
    item = payload["new_or_changed_findings"][0]
    assert item["finding"]["id"] == fid and "expected_if_true" in item["experiment"]
    assert {e["run_id"] for e in item["raw_evidence"]} == {r["id"] for r in runs}
    assert item["raw_evidence"][0]["probes_final [layer, r2, best_control_r2, control_spread]"]
    assert (store.lab_dir() / "reviews" / f"{rec['id']}.md").exists()
    # nothing new since: no second claude session
    rec2 = review.run_review(llm=lambda p: (_ for _ in ()).throw(AssertionError("should not call")), commit=False)
    assert rec2["verdict"] == "OK" and not rec2["llm"]


def test_reviewer_may_only_pause_for_evidence_or_guardrails(fresh_lab):
    from agent import review
    from app import store

    _, runs = _completed_experiment()
    _finding(runs)
    rec = review.run_review(llm=lambda p: {"verdict": "PAUSE", "summary": "", "issues": [
        {"severity": "pause", "category": "drift", "message": "wandering"}]}, commit=False)
    assert rec["verdict"] == "FLAG" and not store.is_paused()  # drift is downgraded to a flag
    assert len(review.open_issues()) == 1


def test_evidence_mismatch_pauses_the_loop(fresh_lab):
    from agent import review, tools
    from app import store

    _, runs = _completed_experiment()
    fid = _finding(runs)
    rec = review.run_review(llm=lambda p: {"verdict": "PAUSE", "summary": "", "issues": [
        {"severity": "pause", "category": "evidence_mismatch", "finding_id": fid,
         "message": "quotes R2 0.93, raw is 0.39"}]}, commit=False)
    assert rec["verdict"] == "PAUSE" and store.is_paused()
    assert rec["id"] in (store.lab_dir() / "PAUSED").read_text()
    assert tools.cycle_begin()["reason"] == "paused"


def test_research_agent_answers_flags_and_ignored_flags_escalate(fresh_lab):
    from agent import review
    from app import store

    _, runs = _completed_experiment()
    fid = _finding(runs)
    flag = lambda p: {"verdict": "FLAG", "summary": "", "issues": [  # noqa: E731
        {"severity": "flag", "category": "verdict_mismatch", "finding_id": fid, "message": "criterion not met"}]}
    rec = review.run_review(llm=flag, commit=False)
    assert rec["verdict"] == "FLAG" and len(review.open_issues()) == 1
    # two cycles finish without an answer -> the next review pauses the lab
    time.sleep(1.1)  # timestamps have one-second resolution
    store.log_event("agent", "cycle.finished", {"summary": "x"})
    store.log_event("agent", "cycle.finished", {"summary": "y"})
    rec2 = review.run_review(use_llm=False, commit=False)
    assert rec2["verdict"] == "PAUSE" and store.is_paused()
    assert "unanswered for 2 cycles" in rec2["issues"][-1]["message"]
    # answering closes the issue and leaves a note on the finding
    errors = review.record_responses([{"review_id": rec["id"], "issue": 0, "response": "Agreed; verdict should be "
                                       "inconclusive. Replication proposed."}])
    assert not errors
    assert review.list_reviews()[0]["status"] == "addressed"
    assert not [i for i in review.open_issues() if i["review_id"] == rec["id"]]
    assert "Response to review" in store.get_finding(fid)["notes"][-1]["text"]


def test_bug_signatures_are_flagged(fresh_lab):
    from agent import review
    from app import store

    _, runs = _completed_experiment(seeds=(0,))
    store.append_metrics(runs[0]["id"], [{"step": 999, "name": "val_mse", "value": float("nan")}])
    issues = review.check_runs([store.get_run(runs[0]["id"])])
    assert any("non-finite metric" in i["message"] and i["severity"] == "flag" for i in issues)


def test_finding_edits_after_the_fact_pause(fresh_lab, tmp_path, monkeypatch):
    """The reviewer compares each finding's substance with the version first committed to git."""
    from agent import review
    from app import store

    _, runs = _completed_experiment(seeds=(0,))
    fid = _finding(runs)
    repo = store.lab_dir().parent
    monkeypatch.setattr(review, "REPO_ROOT", repo)
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("add", "-A", "--", store.lab_dir().name)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "lab")
    assert review.check_immutability() == []
    p = store.lab_dir() / "findings" / f"{fid}.json"
    rec = json.loads(p.read_text())
    rec["claim"] = "A much stronger claim"
    p.write_text(json.dumps(rec))
    issues = review.check_immutability()
    assert issues and issues[0]["severity"] == "pause" and "claim" in issues[0]["message"]


def test_tool_less_flags_passed_to_claude(monkeypatch):
    from agent import claude_cli

    seen = {}

    class P:
        returncode, stdout, stderr = 0, json.dumps({"result": "{}"}), ""

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return P()

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    monkeypatch.setattr(claude_cli, "claude_bin", lambda: "claude")
    claude_cli.run_claude("hi", no_tools=True)
    cmd = seen["cmd"]
    assert cmd[cmd.index("--tools") + 1] == "" and "--strict-mcp-config" in cmd


def test_probes_use_several_random_controls(fresh_lab):
    """Each probe is compared with the best of n random-init controls, and the bar rises with their spread."""
    from app import store
    from engine.probes.controls import counts, required_margin

    _, runs = _completed_experiment(seeds=(0,))
    rows = store.get_probe_results(runs[0]["id"])
    assert rows and all(p["control_spread"] is not None and p["control_spread"] >= 0 for p in rows)
    summ = runs[0]["summary"]["probes"]["targets"]
    for t, s in summ.items():
        assert s["required_margin"] == round(required_margin(0.1, s["control_spread"]), 4) >= 0.1
        assert s["beats_control"] == counts(s["r2"], s["control_r2"], 0.1, s["control_spread"])
    assert counts(0.9, 0.7, 0.1, 0.25) is False and counts(0.9, 0.7, 0.1, 0.05) is True


def test_patching_and_eval_run_with_multi_controls(fresh_lab):
    """The patching and eval-set branches share the probe data set up for the controls."""
    from app import lab, store
    from engine.worker import Worker

    ev = {"name": "heavy", "data": {"source": "rebound", "n_bodies": 2, "n_systems": 50, "steps_per_orbit": 32,
                                    "mass_ratio_range": [1.5, 3.0]}}
    rec = lab.propose(tiny_config(probes={"targets": ["force", "accel_step"], "layers": "all", "checkpoints": "final",
                                          "random_control": True, "patching": True}, eval_sets=[ev]), "agent")
    w = Worker(inline=True)
    for _ in range(20):
        w.tick()
        if not w.busy():
            break
    r = store.runs_for(rec["id"])[0]
    assert r["status"] == "completed", r.get("error")
    assert "patching" in r["summary"] and r["summary"]["eval"]
