"""Milestone 2 check: a force planted in fake activations is recovered; the random control scores near zero."""
import numpy as np
import torch

from app.schemas.config import DataConfig, ModelConfig
from engine.models.gpt import OrbitGPT, collect_activations
from engine.probes.controls import counts, random_model
from engine.probes.linear import fit_probe, probe_r2
from engine.sim.rebound_gen import generate


def _physics(n_sys=150, T=40):
    arr = generate(DataConfig(n_systems=n_sys, steps_per_orbit=32), T)
    steps = np.arange(4, T)
    force = arr["force"][:, steps].reshape(-1, 2).astype(np.float64)
    groups = np.repeat(np.arange(n_sys), len(steps))
    return arr, steps, force, groups


def test_planted_force_is_recovered():
    _, _, force, groups = _physics()
    rng = np.random.default_rng(0)
    d = 128
    latent = np.concatenate([force, rng.normal(size=(len(force), d - 2))], axis=1)
    mix = np.linalg.qr(rng.normal(size=(d, d)))[0]
    acts = latent @ mix + 0.01 * rng.normal(size=latent.shape)
    r2 = probe_r2(acts, force, groups)
    print(f"planted force R2 = {r2:.4f}")
    assert r2 > 0.95


def test_random_control_scores_near_zero():
    """Same architecture, random init, inputs that carry no information about the planted force."""
    _, steps, force, groups = _physics()
    n_sys, T = 150, 40
    rng = np.random.default_rng(1)
    x = torch.tensor(rng.normal(size=(n_sys, T, 6)), dtype=torch.float32)
    model = random_model(ModelConfig(n_layers=2, d_model=64, n_heads=4, context_len=T), seed=0)
    acts = collect_activations(model, x, layers=[0, 1, 2])["resid"]
    for layer, a in acts.items():
        r2 = probe_r2(a[:, steps].reshape(-1, a.shape[-1]), force, groups)
        print(f"control layer {layer}: R2 = {r2:.4f}")
        assert r2 < 0.05


def test_counts_needs_margin():
    assert counts(0.9, 0.2, 0.1)
    assert not counts(0.25, 0.2, 0.1)
    assert not counts(0.9, None, 0.1)


def test_probe_generalises_across_systems_not_rows():
    """Test rows come from systems never seen in training."""
    _, _, force, groups = _physics(n_sys=60)
    fit = fit_probe(np.random.default_rng(0).normal(size=(len(force), 8)), force, groups)
    te_sys = set(groups[fit["test_mask"]])
    tr_sys = set(groups[~fit["test_mask"]])
    assert te_sys and tr_sys and not te_sys & tr_sys


def test_patching_runs_on_tiny_model():
    from engine.probes.patching import run_patching

    arr, _, _, _ = _physics(n_sys=50, T=25)
    pos = arr["pos"].reshape(50, 25, 6)
    x = (pos[:, :-1] / pos.std()).astype(np.float32)
    y = (np.diff(pos, axis=1) / np.diff(pos, axis=1).std()).astype(np.float32)
    model = OrbitGPT(ModelConfig(n_layers=2, d_model=32, n_heads=2, context_len=24))
    res = run_patching(model, x, y, arr["force"][:, :-1], np.repeat(arr["body_mask"], 2), top_k=2)
    assert len(res["head_scores"]) == 4
    assert len(res["patched"]) == 3 and res["patched"][-1]["control"]


def test_symbolic_law_matcher():
    from engine.probes.symbolic import laws, matches_law

    L = laws(64)
    assert matches_law("m2/r**2", L["force"]["law"])
    assert matches_law("1.05*m2/(r*r)", L["force"]["law"])
    assert not matches_law("m2/r", L["force"]["law"])
    assert not matches_law("m2/r**2 + r", L["force"]["law"])
    k = (2 * np.pi / 64) ** 2
    assert matches_law(f"{k}*a**3/((m2 + 1)*r**2)", L["accel_step"]["law"])
    assert not matches_law("m2/r**2", L["accel_step"]["law"])


def test_accel_step_is_what_next_step_prediction_needs():
    """accel_step should match the second difference of body 2's position (per-snapshot curvature)."""
    arr = generate(DataConfig(n_systems=50, steps_per_orbit=64), 40)
    p2 = arr["pos"][:, :, 1]  # body 2 (index 1)
    second_diff = p2[:, 2:] - 2 * p2[:, 1:-1] + p2[:, :-2]
    m1, m2 = arr["masses"][:, 0], arr["masses"][:, 1]
    # body 2's acceleration is m1/(m1+m2) of the relative acceleration in the COM frame, and F/m2 is
    # already the acceleration of body 2, so second_diff ~= accel_step at the middle point.
    acc = arr["accel_step"][:, 1:-1]
    rel = np.abs(second_diff - acc).mean() / np.abs(acc).mean()
    assert rel < 0.01, rel
    # and it is NOT proportional to the physical force across systems of different sizes
    ratio = np.linalg.norm(acc, axis=-1).mean(1) / np.linalg.norm(arr["force"][:, 1:-1], axis=-1).mean(1)
    assert ratio.std() / ratio.mean() > 0.3


def test_planted_accel_step_is_recovered():
    arr = generate(DataConfig(n_systems=150, steps_per_orbit=32), 40)
    steps = np.arange(4, 40)
    acc = arr["accel_step"][:, steps].reshape(-1, 2).astype(np.float64)
    groups = np.repeat(np.arange(150), len(steps))
    rng = np.random.default_rng(0)
    latent = np.concatenate([acc / acc.std(), rng.normal(size=(len(acc), 62))], axis=1)
    acts = latent @ np.linalg.qr(rng.normal(size=(64, 64)))[0]
    assert probe_r2(acts, acc, groups) > 0.95


def test_concurrent_dataset_generation_is_safe(fresh_lab):
    """Two processes asking for the same uncached dataset: one generates it, the other waits and loads it."""
    import multiprocessing as mp

    from engine.sim.rebound_gen import dataset_path
    from tests.conftest import tiny_config
    from app.schemas.config import ExperimentConfig, dataset_key

    cfg = ExperimentConfig.model_validate(tiny_config())
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_gen_worker, args=(cfg.data.model_dump(mode="json"), 24)) for _ in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
    assert all(p.exitcode == 0 for p in procs)
    assert dataset_path(dataset_key(cfg.data, 24)).exists()


def _gen_worker(data: dict, n_steps: int):
    from app.schemas.config import DataConfig
    from engine.sim.rebound_gen import load_or_generate

    load_or_generate(DataConfig.model_validate(data), n_steps, log=lambda m: None)
