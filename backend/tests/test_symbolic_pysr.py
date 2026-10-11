"""PySR recovers both laws from clean data (slow: starts Julia)."""
import numpy as np
import pytest


@pytest.mark.slow
@pytest.mark.parametrize("target", ["force", "accel_step"])
def test_pysr_recovers_law(target):
    from engine.probes.symbolic import laws, run_pysr

    law = laws(64)[target]
    rng = np.random.default_rng(0)
    n = 3000
    v = {"r": rng.uniform(0.3, 3.0, n), "m2": rng.uniform(0.1, 1.0, n), "a": rng.uniform(0.5, 2.0, n)}
    y = law["fn"](v) * (1 + 0.01 * rng.normal(size=n))
    res = run_pysr(v, y, law, niterations=40, n_points=1000)
    print(target, res["equation"], res["fit_error_vs_true_law"])
    assert res["fit_error_vs_true_law"] < 0.05, res["equation"]
