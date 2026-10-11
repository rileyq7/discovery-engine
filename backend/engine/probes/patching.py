"""Activation patching on the attention heads whose outputs best encode the force.

1. Fit a force probe on each head's contribution to the residual stream (final checkpoint).
2. For the top-k heads (and the lowest-scoring head as a control), replace that head's
   output with its output on a *different* system, and measure how much next-step
   prediction error grows. A head that carries the force should hurt prediction when
   patched; the control head should not.
"""
from __future__ import annotations

import numpy as np
import torch

from engine.models.gpt import OrbitGPT, collect_activations
from engine.probes.linear import probe_r2


@torch.no_grad()
def _mse(model: OrbitGPT, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, t0: int) -> float:
    pred = model(x)
    return (((pred - y) ** 2) * mask)[:, t0:].sum().item() / (mask.sum().item() * x.shape[0] * (x.shape[1] - t0))


def run_patching(model: OrbitGPT, x: np.ndarray, y: np.ndarray, force: np.ndarray, mask: np.ndarray,
                 t0: int = 8, top_k: int = 3, seed: int = 0) -> dict:
    """x, y: normalised inputs/targets (N, T, 6); force: (N, T, 2) ground truth."""
    model = model.cpu().eval()
    N, T, _ = x.shape
    xt, yt, mt = torch.tensor(x), torch.tensor(y), torch.tensor(mask)
    acts = collect_activations(model, xt, layers=[], heads=True)["heads"]
    steps = np.arange(t0, T, 2)
    groups = np.repeat(np.arange(N), len(steps))
    Y = force[:, steps].reshape(-1, 2)
    head_scores = []
    for layer, a in acts.items():  # a: N, T, H, C
        for h in range(a.shape[2]):
            X = a[:, steps, h].reshape(-1, a.shape[-1])
            head_scores.append({"layer": layer + 1, "head": h, "force_r2": probe_r2(X, Y, groups, seed)})
    head_scores.sort(key=lambda d: d["force_r2"], reverse=True)

    rng = np.random.default_rng(seed)
    perm = torch.tensor(rng.permutation(N))
    clean = _mse(model, xt, yt, mt, t0)
    selected = head_scores[:top_k] + [dict(head_scores[-1], control=True)]
    results = []
    for hs in selected:
        blk = model.blocks[hs["layer"] - 1].attn
        h = hs["head"]
        # Capture this head's output on corrupted inputs (another system).
        cache = {}

        def grab(_m, _i, out):
            cache["z"] = out.detach().clone()

        hd = blk.head_hook.register_forward_hook(grab)
        model(xt[perm])
        hd.remove()
        zc = cache["z"]

        def patch(z, zc=zc, h=h):
            z = z.clone()
            z[:, :, h] = zc[:, :, h]
            return z

        blk.patch_fn = patch
        patched = _mse(model, xt, yt, mt, t0)
        blk.patch_fn = None
        results.append({**hs, "clean_mse": clean, "patched_mse": patched,
                        "effect": patched / max(clean, 1e-30) - 1.0, "control": hs.get("control", False)})
    return {"head_scores": head_scores, "patched": results, "clean_mse": clean}
