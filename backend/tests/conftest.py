import os
import tempfile

import pytest

_tmp = tempfile.mkdtemp(prefix="de-test-")
os.environ["DE_LAB_DIR"] = os.path.join(_tmp, "lab")
os.environ["DE_RUNS_DIR"] = os.path.join(_tmp, "runs")
os.environ["DE_DEVICE"] = "cpu"


@pytest.fixture
def fresh_lab(tmp_path, monkeypatch):
    monkeypatch.setenv("DE_LAB_DIR", str(tmp_path / "lab"))
    monkeypatch.setenv("DE_RUNS_DIR", str(tmp_path / "runs"))
    from app import store

    store.rebuild_index()
    return tmp_path


def tiny_config(**over):
    cfg = {
        "family": "E1",
        "hypothesis": "tiny test hypothesis",
        "expected_if_true": "ratio < 1",
        "expected_if_false": "ratio >= 1",
        "data": {"source": "rebound", "n_bodies": 2, "n_systems": 120, "steps_per_orbit": 32,
                 "mass_ratio_range": [0.1, 1.0]},
        "model": {"n_layers": 2, "d_model": 32, "n_heads": 2, "context_len": 24},
        "training": {"steps": 60, "batch_size": 32, "lr": 0.003, "checkpoint_every": 30},
        "probes": {"targets": ["force", "distance"], "layers": "all", "checkpoints": "all", "random_control": True},
        "seeds": [0],
        "budget": {"max_compute_hours": 0.5},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg
