"""REBOUND dataset generator for 2-body and hierarchical 3-body systems.

Every system is integrated in the centre-of-mass frame with G = 1 and sampled at
`steps_per_orbit` points per (inner) orbital period. Tokens hold the xy positions of
up to 3 bodies; absent bodies are zero and masked out of the loss.

Ground truth at each step (the probe targets):
  force    (2,)  total gravitational force on body 2 (index 1)
  distance ()    |r2 - r1|
  velocity (2,)  v2 - v1
  energy   ()    total energy of the system (conserved, so constant per system)
  accel_step (2,) acceleration of body 2 in the model's own time units: (F / m2) * dt^2,
                  i.e. displacement change per snapshot. Because sampling is a fixed number of
                  snapshots per orbit, dt grows with orbit size (dt^2 ~ a^3), so this differs from
                  the physical force by an orbit-dependent factor. Probing both tells apart
                  "learned physical gravity" from "learned a per-step version of gravity".
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import rebound

from app.schemas.config import DataConfig, dataset_key
from engine.paths import datasets_dir

MAX_BODIES = 3


def _simulate_one(rng: np.random.Generator, data: DataConfig, n_steps: int):
    lo, hi = data.mass_ratio_range
    elo, ehi = data.eccentricity_range
    m1 = 1.0
    m2 = float(rng.uniform(lo, hi))
    a = float(math.exp(rng.uniform(math.log(0.5), math.log(2.0))))
    e = float(rng.uniform(elo, ehi))
    omega = float(rng.uniform(0, 2 * math.pi))
    f0 = float(rng.uniform(0, 2 * math.pi))

    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.add(m=m1)
    sim.add(m=m2, a=a, e=e, omega=omega, f=f0)
    masses = [m1, m2, 0.0]
    if data.n_bodies == 3:
        m3 = float(rng.uniform(lo, hi))
        a3 = a * float(rng.uniform(3.5, 6.0))
        sim.add(m=m3, a=a3, e=float(rng.uniform(0, 0.2)), omega=float(rng.uniform(0, 2 * math.pi)),
                f=float(rng.uniform(0, 2 * math.pi)))
        masses[2] = m3
    sim.move_to_com()
    period = 2 * math.pi * math.sqrt(a**3 / (m1 + m2))
    dt = period / data.steps_per_orbit

    nb = data.n_bodies
    pos = np.zeros((n_steps, MAX_BODIES, 2), dtype=np.float32)
    force = np.zeros((n_steps, 2), dtype=np.float32)
    dist = np.zeros(n_steps, dtype=np.float32)
    vel = np.zeros((n_steps, 2), dtype=np.float32)
    energy = np.zeros(n_steps, dtype=np.float32)
    accel = np.zeros((n_steps, 2), dtype=np.float32)
    for t in range(n_steps):
        sim.integrate(t * dt)
        ps = sim.particles
        for b in range(nb):
            pos[t, b] = (ps[b].x, ps[b].y)
        p1, p2 = ps[0], ps[1]
        dx, dy = p2.x - p1.x, p2.y - p1.y
        r = math.hypot(dx, dy)
        dist[t] = r
        vel[t] = (p2.vx - p1.vx, p2.vy - p1.vy)
        fx = -m1 * m2 * dx / r**3
        fy = -m1 * m2 * dy / r**3
        if nb == 3:
            p3 = ps[2]
            dx3, dy3 = p2.x - p3.x, p2.y - p3.y
            r3 = math.hypot(dx3, dy3)
            fx += -masses[2] * m2 * dx3 / r3**3
            fy += -masses[2] * m2 * dy3 / r3**3
        force[t] = (fx, fy)
        accel[t] = (fx / m2 * dt**2, fy / m2 * dt**2)
        energy[t] = sim.energy()
    return dict(dt=dt, pos=pos, force=force, distance=dist, velocity=vel, energy=energy, accel_step=accel,
                masses=np.array(masses, dtype=np.float32), period=period, a=a, e=e)


def generate(data: DataConfig, n_steps: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(data.data_seed)
    keys = ["pos", "force", "distance", "velocity", "energy", "accel_step", "masses"]
    out: dict[str, list] = {k: [] for k in keys + ["period", "a", "e", "dt"]}
    for _ in range(data.n_systems):
        s = _simulate_one(rng, data, n_steps)
        for k in out:
            out[k].append(s[k])
    arrays = {k: np.stack(v) if k in keys else np.array(v, dtype=np.float32) for k, v in out.items()}
    mask = np.zeros(MAX_BODIES, dtype=np.float32)
    mask[: data.n_bodies] = 1
    arrays["body_mask"] = mask
    return arrays


def load_or_generate(data: DataConfig, n_steps: int, log=print) -> tuple[str, dict[str, np.ndarray]]:
    """Return (dataset_id, arrays). Datasets are cached by config hash, so they regenerate deterministically."""
    key = dataset_key(data, n_steps)
    d = datasets_dir() / key
    f = d / "data.npz"
    if f.exists():
        return key, _load(f)
    d.mkdir(parents=True, exist_ok=True)
    # Concurrent runs can need the same dataset: one process generates it under the lock, the others wait and load.
    with open(d / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if f.exists():
            return key, _load(f)
        t0 = time.time()
        arrays = generate(data, n_steps)
        tmp = d / f"data.tmp-{os.getpid()}.npz"
        np.savez_compressed(tmp, **arrays)
        tmp.replace(f)
        meta = {"id": key, "config": data.model_dump(mode="json"), "n_steps": n_steps,
                "generated_seconds": round(time.time() - t0, 2), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"generated dataset {key}: {data.n_systems} systems x {n_steps} steps in {meta['generated_seconds']}s")
    return key, arrays


def _load(f: Path) -> dict[str, np.ndarray]:
    z = np.load(f)
    return {k: z[k] for k in z.files}


def list_datasets() -> list[dict]:
    out = []
    root = datasets_dir()
    if not root.exists():
        return out
    for d in sorted(root.iterdir()):
        m = d / "meta.json"
        if m.exists():
            out.append(json.loads(m.read_text()))
    return out


def dataset_path(key: str) -> Path:
    return datasets_dir() / key / "data.npz"
