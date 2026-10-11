"""Training loop: next-step displacement prediction on REBOUND orbits, with checkpoints throughout."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from app.schemas.config import ExperimentConfig
from engine.models.gpt import TOKEN_DIM, OrbitGPT
from engine.paths import device_name
from engine.sim.rebound_gen import load_or_generate

VAL_FRACTION = 0.1


@dataclass
class Prepared:
    dataset_id: str
    arrays: dict
    train_idx: np.ndarray
    val_idx: np.ndarray
    pos_scale: float
    delta_scale: float
    mask: np.ndarray  # (6,)


def split_indices(n: int, data_seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(10_000 + data_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * VAL_FRACTION)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def prepare(cfg: ExperimentConfig, log=print) -> Prepared:
    n_steps = cfg.model.context_len + 1
    key, arrays = load_or_generate(cfg.data, n_steps, log=log)
    tr, va = split_indices(len(arrays["pos"]), cfg.data.data_seed)
    mask = np.repeat(arrays["body_mask"], 2).astype(np.float32)
    pos = arrays["pos"][tr].reshape(len(tr), n_steps, TOKEN_DIM)[:, :, mask > 0]
    pos_scale = float(pos.std())
    delta_scale = float(np.diff(pos, axis=1).std())
    return Prepared(key, arrays, tr, va, pos_scale, delta_scale, mask)


def tokens(pos: np.ndarray) -> np.ndarray:
    """(N, T, 3, 2) -> (N, T, 6)."""
    return pos.reshape(pos.shape[0], pos.shape[1], TOKEN_DIM)


def lr_at(step: int, total: int, base: float) -> float:
    warm = min(500, max(1, total // 20))
    if step < warm:
        return base * (step + 1) / warm
    p = (step - warm) / max(1, total - warm)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(model: OrbitGPT, pos: np.ndarray, mask: np.ndarray, pos_scale: float, delta_scale: float,
             device: str, batch: int = 512) -> dict:
    """Prediction MSE (physical units) vs the constant-velocity baseline, over steps t >= 1."""
    model.eval()
    x_all = tokens(pos)  # N, T+1, 6
    m = torch.tensor(mask, device=device)
    se_model = se_cv = 0.0
    count = 0.0
    for i in range(0, len(x_all), batch):
        xb = torch.tensor(x_all[i : i + batch], device=device)
        x, nxt = xb[:, :-1], xb[:, 1:]
        pred = x + model(x / pos_scale) * delta_scale
        prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        cv = 2 * x - prev
        sl = slice(1, None)
        e_model = ((pred - nxt) ** 2 * m)[:, sl]
        e_cv = ((cv - nxt) ** 2 * m)[:, sl]
        se_model += e_model.sum().item()
        se_cv += e_cv.sum().item()
        count += (m.sum() * e_model.shape[0] * e_model.shape[1]).item()
    model.train()
    return {"val_mse": se_model / count, "baseline_cv_mse": se_cv / count,
            "val_mse_ratio": (se_model / count) / max(se_cv / count, 1e-30)}


def checkpoint_steps(total: int, every: int) -> list[int]:
    steps = sorted(set([0] + list(range(every, total + 1, every)) + [total]))
    return steps


def train(cfg: ExperimentConfig, seed: int, artifact_dir: Path, log: Callable[[str], None] = print,
          on_metrics: Callable[[list[dict]], None] = lambda rows: None,
          should_stop: Callable[[], bool] = lambda: False) -> dict:
    t0 = time.time()
    artifact_dir = Path(artifact_dir)
    ck_dir = artifact_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    device = device_name()
    torch.manual_seed(seed)
    np.random.seed(seed)

    prep = prepare(cfg, log=log)
    norm = {"pos_scale": prep.pos_scale, "delta_scale": prep.delta_scale, "dataset_id": prep.dataset_id,
            "mask": prep.mask.tolist()}
    (artifact_dir / "norm.json").write_text(json.dumps(norm))
    (artifact_dir / "config.json").write_text(cfg.model_dump_json(indent=2))

    model = OrbitGPT(cfg.model).to(device)
    log(f"model: {model.n_params():,} params on {device}; dataset {prep.dataset_id}; "
        f"{len(prep.train_idx)} train / {len(prep.val_idx)} val systems")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=0.01, betas=(0.9, 0.98))

    train_pos = torch.tensor(tokens(prep.arrays["pos"][prep.train_idx]), device=device)
    val_pos = prep.arrays["pos"][prep.val_idx][:2000]
    m = torch.tensor(prep.mask, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ck_steps = set(checkpoint_steps(cfg.training.steps, cfg.training.checkpoint_every))
    log_every = max(1, min(50, cfg.training.checkpoint_every // 2))
    last = {}

    def do_checkpoint(step: int):
        nonlocal last
        torch.save(model.state_dict(), ck_dir / f"step_{step:07d}.pt")
        ev = evaluate(model, val_pos, prep.mask, prep.pos_scale, prep.delta_scale, device)
        last = ev
        on_metrics([{"step": step, "name": k, "value": v} for k, v in ev.items()])
        log(f"step {step}: val_mse={ev['val_mse']:.3e} cv_baseline={ev['baseline_cv_mse']:.3e} "
            f"ratio={ev['val_mse_ratio']:.3f}")

    do_checkpoint(0)
    model.train()
    stopped = False
    step = 0
    for step in range(1, cfg.training.steps + 1):
        idx = torch.randint(0, len(train_pos), (cfg.training.batch_size,), generator=gen).to(device)
        xb = train_pos[idx]
        x, y = xb[:, :-1] / prep.pos_scale, (xb[:, 1:] - xb[:, :-1]) / prep.delta_scale
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg.training.steps, cfg.training.lr)
        loss = (((model(x) - y) ** 2) * m).sum() / (m.sum() * y.shape[0] * y.shape[1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            on_metrics([{"step": step, "name": "train_loss", "value": loss.item()},
                        {"step": step, "name": "lr", "value": opt.param_groups[0]["lr"]}])
        if step in ck_steps:
            do_checkpoint(step)
        if should_stop():
            log(f"stop requested at step {step}")
            stopped = True
            break

    if step not in ck_steps:  # always probe the final weights, including after a stop
        do_checkpoint(step)
    hours = (time.time() - t0) / 3600
    summary = {"final_val_mse": last.get("val_mse"), "baseline_cv_mse": last.get("baseline_cv_mse"),
               "val_mse_ratio": last.get("val_mse_ratio"), "beats_baseline": last.get("val_mse_ratio", 9) < 1.0,
               "train_hours": hours, "n_params": model.n_params(), "device": device, "dataset_id": prep.dataset_id,
               "stopped": stopped, "steps_completed": step, "steps_planned": cfg.training.steps}
    log(f"training done in {hours * 60:.1f} min; beats constant-velocity baseline: {summary['beats_baseline']}")
    return summary


def list_checkpoints(artifact_dir: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in (Path(artifact_dir) / "checkpoints").glob("step_*.pt"))


def load_model(cfg: ExperimentConfig, artifact_dir: Path, step: int, device: str = "cpu") -> OrbitGPT:
    model = OrbitGPT(cfg.model)
    sd = torch.load(Path(artifact_dir) / "checkpoints" / f"step_{step:07d}.pt", map_location=device)
    model.load_state_dict(sd)
    return model.to(device).eval()
