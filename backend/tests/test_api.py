"""Milestone 3 (fast version): queue a config through the API, run it, read results back over the API."""
from fastapi.testclient import TestClient

from tests.conftest import tiny_config


def _client():
    from app.main import app

    return TestClient(app)


def _drain(worker, max_ticks=50):
    for _ in range(max_ticks):
        worker.tick()
        if not worker.busy():
            return
    raise AssertionError("worker never went idle")


def test_unknown_fields_rejected(fresh_lab):
    with _client() as c:
        cfg = tiny_config()
        cfg["model"]["dropout"] = 0.1
        r = c.post("/experiments", json={"config": cfg})
        assert r.status_code == 422
        assert any("dropout" in e for e in r.json()["detail"]["errors"])


def test_budget_rejected(fresh_lab):
    with _client() as c:
        cfg = tiny_config(training={"steps": 400000, "batch_size": 4096}, model={"d_model": 512, "n_heads": 8,
                                                                               "n_layers": 12, "context_len": 512})
        r = c.post("/experiments/validate", json={"config": cfg})
        assert not r.json()["valid"]
        assert any("budget" in e for e in r.json()["errors"])


def test_queue_run_and_read_results(fresh_lab):
    from engine.worker import Worker

    with _client() as c:
        c.post("/loop/start")
        r = c.post("/experiments", json={"config": tiny_config(), "approve": True})
        assert r.status_code == 200, r.text
        exp_id = r.json()["id"]
        assert r.json()["state"] == "approved"
        _drain(Worker(inline=True))
        exp = c.get(f"/experiments/{exp_id}").json()
        assert exp["state"] == "analyzing", exp["history"]
        run_id = exp["runs"][0]["id"]
        run = c.get(f"/runs/{run_id}").json()
        assert run["status"] == "completed"
        metrics = c.get(f"/runs/{run_id}/metrics").json()
        assert {"train_loss", "val_mse", "baseline_cv_mse"} <= {m["name"] for m in metrics}
        probes = c.get(f"/runs/{run_id}/probes").json()
        assert len(probes) == 3 * 3 * 2  # checkpoints {0,30,60} x layers {0,1,2} x targets
        assert all(p["control_r2"] is not None for p in probes)

        # Findings need evidence.
        bad = c.post("/findings", json={"claim": "something holds", "verdict": "supported", "evidence_run_ids": [],
                                        "confidence_note": "n/a"})
        assert bad.status_code == 422
        f = c.post("/findings", json={"claim": "tiny model trains", "verdict": "inconclusive",
                                      "evidence_run_ids": [run_id], "confidence_note": "60 steps only"})
        assert f.status_code == 200, f.text
        assert c.get(f"/experiments/{exp_id}").json()["state"] == "complete"
        # Single seed cannot be promoted to replicated.
        assert c.post(f"/findings/{f.json()['id']}/promote", json={"status": "replicated"}).status_code == 422

        events = c.get("/api/events").json()
        types = {e["type"] for e in events}
        assert {"experiment.proposed", "experiment.state", "run.state", "finding.created"} <= types


def test_kill_switch_cancels_queued_not_running(fresh_lab):
    from app import store

    with _client() as c:
        c.post("/loop/start")
        ids = [c.post("/experiments", json={"config": tiny_config(), "approve": True}).json()["id"] for _ in range(2)]
        store.transition(ids[0], "queued", "test")
        store.transition(ids[0], "running", "test")
        r = c.post("/loop/kill").json()
        assert r["paused"]
        assert ids[1] in r["cancelled"] and ids[0] not in r["cancelled"]
        assert c.get(f"/experiments/{ids[0]}").json()["state"] == "running"


def test_rebuild_index_from_files(fresh_lab):
    from app import store

    with _client() as c:
        c.post("/questions", json={"text": "Does force encoding precede accuracy?"})
        c.post("/experiments", json={"config": tiny_config()})
    counts = store.rebuild_index()
    assert counts["questions"] == 1 and counts["experiments"] == 1
    assert len(store.list_questions("open")) == 1


def test_latex_export(fresh_lab):
    with _client() as c:
        p = c.post("/papers", json={"title": "Test paper"}).json()
        c.put(f"/papers/{p['id']}", json={"title": "Test paper",
                                          "markdown": "# T\n\n## Results\n\nForce $F = m/r^2$ in **layer 3**.\n\n- a_b\n"})
        tex = c.get(f"/papers/{p['id']}/export?format=latex").text
        assert r"\section{Results}" in tex and "$F = m/r^2$" in tex and r"\textbf{layer 3}" in tex
        assert r"a\_b" in tex
