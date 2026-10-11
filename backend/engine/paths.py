"""Filesystem layout. `lab/` is the committed source of truth; `runs/` holds heavy, gitignored artifacts."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def lab_dir() -> Path:
    return Path(os.environ.get("DE_LAB_DIR", REPO_ROOT / "lab"))


def runs_dir() -> Path:
    return Path(os.environ.get("DE_RUNS_DIR", REPO_ROOT / "runs"))


def datasets_dir() -> Path:
    return runs_dir() / "_datasets"


def run_artifact_dir(run_id: str) -> Path:
    return runs_dir() / run_id


def device_name() -> str:
    forced = os.environ.get("DE_DEVICE")
    if forced:
        return forced
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
