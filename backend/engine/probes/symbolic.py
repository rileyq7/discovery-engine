"""Symbolic regression (PySR) on probe outputs.

Units: G = 1 and the primary mass M = 1 in every simulated system.

Two laws are tested, one per probe target:
  force       |F| = m2 / r^2                                  (physical gravity)
  accel_step  |a_step| = (2*pi/N)^2 * a^3 / ((1 + m2) * r^2)  (gravity in the model's own time units:
              N snapshots per orbit, so dt = P / N with P = 2*pi*sqrt(a^3 / (1 + m2)))
Inputs are ground-truth variables: r (separation), m2 (secondary mass), and a (semi-major axis) for accel_step.
"""
from __future__ import annotations

import math

import numpy as np


def laws(steps_per_orbit: int) -> dict:
    k = (2 * math.pi / steps_per_orbit) ** 2
    return {
        "force": {"vars": ["r", "m2"], "law": "m2 / r**2", "fn": lambda v: v["m2"] / v["r"] ** 2},
        "accel_step": {"vars": ["r", "m2", "a"], "law": f"{k:.6g} * a**3 / ((1 + m2) * r**2)",
                       "fn": lambda v: k * v["a"] ** 3 / ((1 + v["m2"]) * v["r"] ** 2)},
    }


def run_pysr(inputs: dict[str, np.ndarray], y: np.ndarray, law: dict, seed: int = 0, n_points: int = 1500,
             niterations: int = 40, timeout_s: int = 600) -> dict:
    names = law["vars"]
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n_points, n), replace=False)
    hold = np.setdiff1d(np.arange(n), idx)[:5000]
    if len(hold) == 0:
        hold = idx
    X = np.stack([inputs[k] for k in names], axis=1)

    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=niterations, binary_operators=["+", "-", "*", "/"], unary_operators=["square", "cube"],
        maxsize=20, populations=8, deterministic=True, parallelism="serial", random_state=seed,
        timeout_in_seconds=timeout_s, progress=False, verbosity=0, temp_equation_file=True,
        model_selection="best",
    )
    model.fit(X[idx], y[idx], variable_names=names)
    best = model.get_best()
    pred = model.predict(X[hold])
    truth = law["fn"]({k: inputs[k][hold] for k in names})
    probe = y[hold]

    def rel_rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b**2)) + 1e-30))

    return {
        "equation": str(best["sympy_format"]),
        "complexity": int(best["complexity"]),
        "loss": float(best["loss"]),
        "fit_error_vs_probe": rel_rmse(pred, probe),
        "fit_error_vs_true_law": rel_rmse(pred, truth),
        "true_law_error_vs_probe": rel_rmse(truth, probe),
        "law_recovered": bool(matches_law(best["sympy_format"], law["law"])),
        "true_law": law["law"],
        "variables": names,
        "pareto": [{"complexity": int(e.complexity), "loss": float(e.loss), "equation": str(e.sympy_format)}
                   for e in model.equations_.itertuples()],
    }


def matches_law(expr, law: str, tol: float = 0.1) -> bool:
    """True if expr / law is a constant within `tol` of 1."""
    import sympy as sp

    syms = {s: sp.Symbol(s) for s in ("r", "m2", "a")}
    try:
        e = sp.sympify(str(expr), locals=syms)
        ratio = sp.simplify(e / sp.sympify(law, locals=syms))
        if ratio.free_symbols:
            return False
        return abs(float(ratio) - 1.0) < tol
    except Exception:
        return False
