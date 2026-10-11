"""The probe suite, run on every checkpoint listed in the config.

Produces:
  probe rows      [{checkpoint_step, layer, target, r2, control_r2}]
  patching        head scores + patch effects (if probes.patching)
  equations       PySR result on probe-predicted |F| (if probes.symbolic)
  eval            out-of-distribution evaluation (if eval_sets)
  summary         compact numbers the agent reads
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from app.schemas.config import ExperimentConfig
from engine.models.gpt import collect_activations
from engine.probes.controls import counts, random_model
from engine.probes.linear import fit_probe, target_array
from engine.sim.rebound_gen import load_or_generate
from engine.train import Prepared, evaluate, list_checkpoints, load_model, tokens

PROBE_SYSTEMS = 300
T0 = 8  # skip the first steps, where the model has too little context


def _probe_inputs(prep: Prepared, pos: np.ndarray):
    x = tokens(pos[:, :-1]) / prep.pos_scale
    T = x.shape[1]
    steps = np.arange(T0, T, 2)
    return torch.tensor(x, dtype=torch.float32), steps


def _layers(cfg: ExperimentConfig) -> list[int]:
    return list(range(cfg.model.n_layers + 1)) if cfg.probes.layers == "all" else sorted(cfg.probes.layers)


def _select_checkpoints(cfg: ExperimentConfig, available: list[int]) -> list[int]:
    c = cfg.probes.checkpoints
    if c == "all":
        return available
    if c == "final":
        return available[-1:]
    want = set(c)
    chosen = [s for s in available if s in want]
    return chosen or available[-1:]


def run_suite(cfg: ExperimentConfig, seed: int, artifact_dir: Path, prep: Prepared,
              log: Callable[[str], None] = print) -> dict:
    artifact_dir = Path(artifact_dir)
    layers = _layers(cfg)
    targets = list(cfg.probes.targets)
    probe_sys = prep.val_idx[:PROBE_SYSTEMS]
    pos = prep.arrays["pos"][probe_sys]
    x, steps = _probe_inputs(prep, pos)
    groups = np.repeat(np.arange(len(probe_sys)), len(steps))
    Y = {t: target_array(prep.arrays, t, probe_sys, steps) for t in targets}

    def acts_for(model):
        a = collect_activations(model, x, layers)["resid"]
        return {l: a[l][:, steps].reshape(-1, a[l].shape[-1]) for l in layers}

    control = {}
    if cfg.probes.random_control:
        ra = acts_for(random_model(cfg.model, seed))
        for l in layers:
            for t in targets:
                control[(l, t)] = fit_probe(ra[l], Y[t], groups, seed)["r2"]
        log("random-model control: " + ", ".join(
            f"{t}=" + "/".join(f"{control[(l, t)]:.2f}" for l in layers) for t in targets))

    ckpts = _select_checkpoints(cfg, list_checkpoints(artifact_dir))
    rows = []
    final_fits = {}
    for i, step in enumerate(ckpts):
        model = load_model(cfg, artifact_dir, step)
        a = acts_for(model)
        for l in layers:
            for t in targets:
                fit = fit_probe(a[l], Y[t], groups, seed)
                rows.append({"checkpoint_step": step, "layer": l, "target": t, "r2": round(fit["r2"], 5),
                             "control_r2": None if not control else round(control[(l, t)], 5)})
                if step == ckpts[-1]:
                    final_fits[(l, t)] = fit
        if step == ckpts[-1]:
            np.savez_compressed(artifact_dir / "activations_final.npz",
                                **{f"layer_{l}": a[l].astype(np.float16) for l in layers})
        if i % 5 == 0 or step == ckpts[-1]:
            best = max((r for r in rows if r["checkpoint_step"] == step and r["target"] == targets[0]),
                       key=lambda r: r["r2"])
            log(f"probes @ step {step}: best {targets[0]} R2={best['r2']:.3f} (layer {best['layer']})")

    out = {"rows": rows}
    final_model = load_model(cfg, artifact_dir, ckpts[-1])

    if cfg.probes.patching:
        y = (tokens(pos[:, 1:]) - tokens(pos[:, :-1])) / prep.delta_scale
        n = min(200, len(probe_sys))
        out["patching"] = _run_patching(final_model, x[:n].numpy(), y[:n], prep.arrays["force"][probe_sys][:, :-1][:n],
                                        prep.mask, seed)
        top = out["patching"]["patched"][0]
        log(f"patching: top force head L{top['layer']}H{top['head']} (R2={top['force_r2']:.2f}) "
            f"patch effect={top['effect']:+.2f}")

    sym_targets = [t for t in ("force", "accel_step") if t in targets] if cfg.probes.symbolic else []
    if sym_targets:
        from engine.probes.symbolic import laws, run_pysr

        law_defs = laws(cfg.data.steps_per_orbit)
        out["equations"] = {}
        for t in sym_targets:
            best_l = max(layers, key=lambda l: final_fits[(l, t)]["r2"])
            fit = final_fits[(best_l, t)]
            te = fit["test_mask"]
            n_steps_ = len(steps)
            inputs = {
                "r": target_array(prep.arrays, "distance", probe_sys, steps)[te, 0],
                "m2": np.repeat(prep.arrays["masses"][probe_sys][:, 1], n_steps_)[te],
                "a": np.repeat(prep.arrays["a"][probe_sys], n_steps_)[te],
            }
            mag = np.linalg.norm(fit["pred_test"], axis=1)
            log(f"PySR on |{t}| probe outputs from layer {best_l} ({len(mag)} points)...")
            try:
                eq = run_pysr(inputs, mag, law_defs[t], seed=seed)
                eq.update(layer=best_l, probe_r2=fit["r2"])
                log(f"PySR: |{t}| = {eq['equation']}  (law recovered: {eq['law_recovered']}, "
                    f"error vs true law {eq['fit_error_vs_true_law']:.3f})")
            except Exception as e:  # PySR/Julia failures are recorded, not fatal
                eq = {"error": f"{type(e).__name__}: {e}", "layer": best_l}
                log(f"PySR failed for {t}: {e}")
            out["equations"][t] = eq

    if cfg.eval_sets:
        out["eval"] = []
        for es in cfg.eval_sets:
            _, arr = load_or_generate(es.data, cfg.model.context_len + 1, log=log)
            ev = evaluate(final_model, arr["pos"][:2000], np.repeat(arr["body_mask"], 2).astype(np.float32),
                          prep.pos_scale, prep.delta_scale, "cpu")
            sys_idx = np.arange(min(PROBE_SYSTEMS, len(arr["pos"])))
            xo, st = _probe_inputs(prep, arr["pos"][sys_idx])
            ao = collect_activations(final_model, xo, layers)["resid"]
            g = np.repeat(np.arange(len(sys_idx)), len(st))
            pr = {}
            for t in targets:
                Yo = target_array(arr, t, sys_idx, st)
                pr[t] = {str(l): round(fit_probe(ao[l][:, st].reshape(-1, ao[l].shape[-1]), Yo, g, seed)["r2"], 4)
                         for l in layers}
            out["eval"].append({"name": es.name, **ev, "probe_r2": pr, "n_bodies": es.data.n_bodies,
                                "mass_ratio_range": list(es.data.mass_ratio_range)})
            log(f"eval[{es.name}]: mse ratio vs CV baseline {ev['val_mse_ratio']:.3f}; force R2 best layer "
                f"{max(pr.get('force', {'0': 0}).values()):.3f}")

    out["summary"] = summarize(rows, targets, cfg.probes.control_margin)
    return out


def _run_patching(model, x, y, force, mask, seed):
    from engine.probes.patching import run_patching

    return run_patching(model, x, y.astype(np.float32), force, mask, seed=seed)


def summarize(rows: list[dict], targets: list[str], margin: float, metrics: list[dict] | None = None) -> dict:
    if not rows:
        return {}
    final_step = max(r["checkpoint_step"] for r in rows)
    s = {"final_checkpoint": final_step, "control_margin": margin, "targets": {}}
    for t in targets:
        fin = [r for r in rows if r["target"] == t and r["checkpoint_step"] == final_step]
        if not fin:
            continue
        best = max(fin, key=lambda r: r["r2"] - (r["control_r2"] or 0))
        # Formation: first checkpoint after which the best layer stays above control + margin.
        series = sorted((r for r in rows if r["target"] == t and r["layer"] == best["layer"]),
                        key=lambda r: r["checkpoint_step"])
        formed_at = None
        for i, r in enumerate(series):
            if all(counts(q["r2"], q["control_r2"], margin) for q in series[i:]):
                formed_at = r["checkpoint_step"]
                break
        s["targets"][t] = {"best_layer": best["layer"], "r2": best["r2"], "control_r2": best["control_r2"],
                           "beats_control": counts(best["r2"], best["control_r2"], margin),
                           "formed_at_step": formed_at}
    return s
