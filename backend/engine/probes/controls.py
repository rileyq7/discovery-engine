"""Random-model controls: the same probe on a randomly initialised model with the same architecture."""
from __future__ import annotations

import torch

from app.schemas.config import ModelConfig
from engine.models.gpt import OrbitGPT

CONTROL_SEED_OFFSET = 99_991


def random_model(cfg: ModelConfig, seed: int) -> OrbitGPT:
    torch.manual_seed(seed + CONTROL_SEED_OFFSET)
    return OrbitGPT(cfg).eval()


def required_margin(margin: float, spread: float | None) -> float:
    """The bar above the best random control: the configured margin, or the spread between the random controls
    when that is larger (random models alone already differ that much on this target and layer)."""
    return max(margin, spread or 0.0)


def counts(r2: float, control_r2: float | None, margin: float, spread: float | None = None) -> bool:
    """A probe result only counts if it beats the (best) random-model control by the required margin."""
    if control_r2 is None:
        return False
    return r2 - control_r2 >= required_margin(margin, spread)
