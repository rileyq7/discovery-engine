"""Experiment config schema. The agent may only emit configs that validate against this.

Unknown fields are rejected (extra="forbid"), not ignored.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FAMILIES = ("E1", "E2", "E3", "E4", "E5", "E6", "custom")
PROBE_TARGETS = ("force", "distance", "velocity", "energy", "accel_step")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(Strict):
    source: Literal["rebound"] = "rebound"
    n_bodies: Literal[2, 3] = 2
    n_systems: int = Field(10000, ge=50, le=200_000)
    steps_per_orbit: int = Field(64, ge=8, le=512)
    mass_ratio_range: tuple[float, float] = (0.1, 1.0)
    eccentricity_range: tuple[float, float] = (0.0, 0.5)
    data_seed: int = 0

    @field_validator("mass_ratio_range")
    @classmethod
    def _mass(cls, v):
        lo, hi = v
        if not (0 < lo <= hi <= 10):
            raise ValueError("mass_ratio_range must satisfy 0 < lo <= hi <= 10")
        return v

    @field_validator("eccentricity_range")
    @classmethod
    def _ecc(cls, v):
        lo, hi = v
        if not (0 <= lo <= hi < 0.95):
            raise ValueError("eccentricity_range must satisfy 0 <= lo <= hi < 0.95")
        return v


class ModelConfig(Strict):
    n_layers: int = Field(4, ge=1, le=12)
    d_model: int = Field(128, ge=16, le=512)
    n_heads: int = Field(4, ge=1, le=16)
    context_len: int = Field(128, ge=8, le=512)

    @model_validator(mode="after")
    def _heads(self):
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


class TrainingConfig(Strict):
    steps: int = Field(20000, ge=1, le=500_000)
    batch_size: int = Field(256, ge=1, le=4096)
    lr: float = Field(3e-4, gt=0, le=1e-1)
    checkpoint_every: int = Field(500, ge=1)


class ProbeConfig(Strict):
    targets: list[Literal["force", "distance", "velocity", "energy", "accel_step"]] = list(PROBE_TARGETS)
    layers: Union[Literal["all"], list[int]] = "all"
    checkpoints: Union[Literal["all", "final"], list[int]] = "all"
    random_control: bool = True
    control_margin: float = Field(0.1, ge=0, le=1)
    patching: bool = False
    symbolic: bool = False


class EvalSet(Strict):
    name: str = Field(min_length=1, max_length=64)
    data: DataConfig


class BudgetConfig(Strict):
    max_compute_hours: float = Field(gt=0, le=48)


class ExperimentConfig(Strict):
    family: Literal["E1", "E2", "E3", "E4", "E5", "E6", "custom"]
    hypothesis: str = Field(min_length=5)
    expected_if_true: str = Field(min_length=3)
    expected_if_false: str = Field(min_length=3)
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    probes: ProbeConfig = ProbeConfig()
    eval_sets: list[EvalSet] = []
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=5)
    budget: BudgetConfig

    @model_validator(mode="after")
    def _checks(self):
        if self.training.checkpoint_every > self.training.steps:
            raise ValueError("training.checkpoint_every must be <= training.steps")
        if isinstance(self.probes.layers, list):
            bad = [l for l in self.probes.layers if not 0 <= l <= self.model.n_layers]
            if bad:
                raise ValueError(f"probes.layers out of range 0..{self.model.n_layers}: {bad}")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        # The model reads tokens of 3 bodies, so a 2-body-trained model can be tested on 3-body sets.
        return self


DATASET_FORMAT = 2  # bump when generated arrays change (2: added accel_step)


def dataset_key(data: DataConfig, n_steps: int) -> str:
    payload = json.dumps({**data.model_dump(mode="json"), "n_steps": n_steps, "format": DATASET_FORMAT}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# Compute-estimate constants, fitted on 23 runs on an M-series laptop (MPS); see DECISIONS.md. At these model sizes a
# training step costs mostly fixed per-layer overhead (kernel launches, optimizer, periodic eval), plus a small term
# proportional to the arithmetic. Values include about 25% headroom for a busy machine.
DEFAULT_THROUGHPUT = 1.4e15  # batch-token-layer-d_model^2 work units per hour
DEFAULT_STEP_OVERHEAD_S = 0.03  # seconds per training step per layer


def estimate_compute_hours(cfg: ExperimentConfig, throughput: float = DEFAULT_THROUGHPUT,
                           step_overhead_s: float = DEFAULT_STEP_OVERHEAD_S) -> float:
    """A-priori estimate of compute hours across all seeds (train + probes + data generation)."""
    m, t = cfg.model, cfg.training
    work = t.steps * t.batch_size * m.context_len * m.n_layers * (12 * m.d_model**2 + 2 * m.context_len * m.d_model)
    train_h = work / throughput + t.steps * m.n_layers * step_overhead_s / 3600
    n_ckpt = t.steps // t.checkpoint_every + 2
    probe_h = 0.0001 * n_ckpt * (m.n_layers + 1) * (1.5 if cfg.probes.patching else 1.0) + 0.01 * len(cfg.eval_sets)
    if cfg.probes.symbolic:
        probe_h += 0.06
    # worst case observed: 10k high-eccentricity systems in about 2 min (cached afterwards)
    data_h = cfg.data.n_systems * 3.5e-6 * (1 + len(cfg.eval_sets))
    return (train_h + probe_h) * len(cfg.seeds) + data_h


def validate_config(raw: dict | str) -> ExperimentConfig:
    if isinstance(raw, str):
        raw = json.loads(raw)
    return ExperimentConfig.model_validate(raw)
