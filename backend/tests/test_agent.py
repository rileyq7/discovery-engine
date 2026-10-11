"""Agent tooling: billing guard, proposal limits, finding rules, pause flag, paper sections."""
import pytest

from tests.conftest import tiny_config


def test_billing_guard_refuses_api_key(monkeypatch):
    from agent.claude_cli import BillingError, check_billing, run_claude

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with pytest.raises(BillingError):
        check_billing()
    with pytest.raises(BillingError):
        run_claude("hi")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "x")
    with pytest.raises(BillingError):
        check_billing()


def test_tools_check_billing_exit_code(monkeypatch):
    from agent import tools

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert tools.main(["check-billing"]) == 1
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert tools.main(["check-billing"]) == 0


def test_no_anthropic_sdk_anywhere():
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    for p in root.rglob("*.py"):
        if ".venv" in p.parts:
            continue
        assert not re.search(r"^\s*(import anthropic|from anthropic)", p.read_text(), re.M), p


def test_propose_limits_and_auto_approve(fresh_lab):
    from agent import tools
    from app import store

    res = tools.propose_many([tiny_config()] * 4)
    assert not res["ok"]
    bad = tiny_config()
    bad["unknown_field"] = 1
    res = tools.propose_many([bad])
    assert not res["ok"] and "unknown_field" in str(res["results"])
    res = tools.propose_many([tiny_config(), tiny_config(family="E6")])
    assert res["ok"]
    states = [store.get_experiment(c["id"])["state"] for c in res["created"]]
    assert states == ["approved", "proposed"]  # E6 needs a human


def test_pause_flag_stops_cycle(fresh_lab):
    from agent import tools
    from app import store

    store.set_paused(True, "human")
    assert tools.cycle_begin() == {"ok": False, "reason": "paused"}
    assert not tools.propose_many([tiny_config()])["ok"]
    store.set_paused(False, "human")
    assert tools.cycle_begin()["ok"]


def test_findings_need_completed_runs_and_seed_agreement(fresh_lab):
    from agent import tools
    from app import lab, store
    from engine.worker import Worker

    rec = lab.propose(tiny_config(seeds=[0, 1]), "agent")
    w = Worker(inline=True)
    for _ in range(20):
        w.tick()
        if not w.busy():
            break
    runs = store.runs_for(rec["id"])
    assert all(r["status"] == "completed" for r in runs)
    res = tools.create_findings({"claim": "made up run", "verdict": "supported", "evidence_run_ids": ["run-nope"],
                                 "confidence_note": "x"})
    assert not res["ok"]
    q = store.create_question("Does it train?", "human")
    res = tools.create_findings({"claim": "Tiny model fails to beat baseline in 60 steps", "verdict": "refuted",
                                 "evidence_run_ids": [r["id"] for r in runs], "confidence_note": "tiny",
                                 "seeds_agree": True, "question_updates": [{"id": q["id"], "status": "answered"}],
                                 "new_questions": ["What about 600 steps?"]})
    assert res["ok"], res
    f = store.get_finding(res["created"][0]["id"])
    assert f["status"] == "replicated"  # two distinct completed seeds agree
    assert store.get_question(q["id"])["status"] == "answered"
    assert store.get_experiment(rec["id"])["state"] == "complete"
    assert len(store.list_questions("open")) == 1


def test_replicate_creates_new_seed_experiment(fresh_lab):
    from app import lab, store
    from engine.worker import Worker

    rec = lab.propose(tiny_config(), "agent")
    w = Worker(inline=True)
    for _ in range(20):
        w.tick()
        if not w.busy():
            break
    run = store.runs_for(rec["id"])[0]
    f = store.create_finding({"claim": "single seed claim", "verdict": "supported", "evidence_run_ids": [run["id"]],
                              "confidence_note": "one seed"}, "agent")
    assert f["status"] == "supported"
    rep = lab.replicate(f["id"], "human")
    assert rep["config"]["seeds"] == [1] and rep["replicates_finding_id"] == f["id"]
    for _ in range(20):
        w.tick()
        if not w.busy():
            break
    rrun = store.runs_for(rep["id"])[0]
    store.create_finding({"claim": "replication agrees", "verdict": "supported", "evidence_run_ids": [rrun["id"]],
                          "confidence_note": "seed 1", "replicates": f["id"]}, "agent")
    assert store.get_finding(f["id"])["status"] == "replicated"


def test_paper_section_replace_or_append():
    from agent.papers import replace_or_append_section

    md = "# T\n\n## Intro\n\nhello\n\n## Results\n\nold\n\n## Discussion\n\nbye\n"
    out = replace_or_append_section(md, "Results", "## Results\n\nnew results")
    assert "old" not in out and "new results" in out and out.index("Results") < out.index("Discussion")
    out2 = replace_or_append_section(md, "Methods", "We did things.")
    assert out2.rstrip().endswith("## Methods\n\nWe did things.")
