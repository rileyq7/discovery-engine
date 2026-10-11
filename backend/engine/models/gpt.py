"""Small causal GPT over continuous tokens, with forward hooks for activations.

Input token at step t: xy positions of up to 3 bodies (6 numbers, normalised).
Output at step t: predicted displacement to step t+1 (6 numbers, normalised).

`collect_activations` returns the residual stream at layer 0 (embeddings) through
layer n_layers (after each block), and optionally each head's contribution to the
residual stream (head output projected through its slice of W_O).
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.schemas.config import ModelConfig

TOKEN_DIM = 6


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        # Hook point: per-head outputs before projection, shape (B, T, H, d_head).
        self.head_hook = nn.Identity()
        # Patching: optional fn(z) -> z applied to per-head outputs.
        self.patch_fn = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        z = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        z = z.transpose(1, 2)  # B, T, H, d_head
        if self.patch_fn is not None:
            z = self.patch_fn(z)
        z = self.head_hook(z)
        return self.proj(z.reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class OrbitGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(TOKEN_DIM, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.context_len, cfg.d_model))
        self.blocks = nn.ModuleList(Block(cfg.d_model, cfg.n_heads) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, TOKEN_DIM)
        self.apply(self._init)
        nn.init.normal_(self.pos, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        T = x.shape[1]
        h = self.embed(x) + self.pos[:, :T]
        h = self.resid_hook(h, 0)
        for i, blk in enumerate(self.blocks):
            h = self.resid_hook(blk(h), i + 1)
        return self.head(self.ln_f(h))

    # Residual hook: identity unless a capture is active.
    _capture: dict | None = None

    def resid_hook(self, h, layer: int):
        if self._capture is not None and layer in self._capture["layers"]:
            self._capture["resid"][layer].append(h.detach().float().cpu())
        return h

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@contextmanager
def capture(model: OrbitGPT, layers: list[int], heads: bool = False):
    """Capture residual stream (and optionally per-head contributions) during forward passes."""
    store = {"layers": set(layers), "resid": {l: [] for l in layers}, "heads": {}}
    handles = []
    if heads:
        for i, blk in enumerate(model.blocks):
            store["heads"][i] = []
            W = blk.attn.proj.weight  # (C, C); column slice per head

            def hook(_m, _inp, out, i=i, W=W, attn=blk.attn):
                B, T, H, dh = out.shape
                # Head contribution to residual: z_h @ W[:, h*dh:(h+1)*dh].T  -> (B, T, H, C)
                Wh = W.view(W.shape[0], H, dh)  # C, H, dh
                contrib = torch.einsum("bthd,chd->bthc", out, Wh)
                store["heads"][i].append(contrib.detach().float().cpu())

            handles.append(blk.attn.head_hook.register_forward_hook(hook))
    model._capture = store
    try:
        yield store
    finally:
        model._capture = None
        for h in handles:
            h.remove()


@torch.no_grad()
def collect_activations(model: OrbitGPT, x: torch.Tensor, layers: list[int], heads: bool = False,
                        batch_size: int = 256) -> dict:
    model.eval()
    dev = next(model.parameters()).device
    with capture(model, layers, heads) as st:
        for i in range(0, len(x), batch_size):
            model(x[i : i + batch_size].to(dev))
    out = {"resid": {l: torch.cat(v).numpy() for l, v in st["resid"].items()}}
    if heads:
        out["heads"] = {l: torch.cat(v).numpy() for l, v in st["heads"].items()}
    return out
