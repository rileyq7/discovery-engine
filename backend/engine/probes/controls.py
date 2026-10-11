"""Random-model controls: the same probe on a randomly initialised model with the same architecture."""
from __future__ import annotations

import torch

from app.schemas.config import ModelConfig
from engine.models.gpt import OrbitGPT

CONTROL_SEED_OFFSET = 99_991


def random_model(cfg: ModelConfig, seed: int) -> OrbitGPT:
    torch.manual_seed(seed + CONTROL_SEED_OFFSET)
    return OrbitGPT(cfg).eval()


def counts(r2: float, control_r2: float | None, margin: float) -> bool:
    """A probe result only counts if it beats the random-model control by `margin`."""
    if control_r2 is None:
        return False
    return r2 - control_r2 >= margin
