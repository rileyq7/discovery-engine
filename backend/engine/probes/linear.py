"""Linear (ridge) probes for physical quantities, with train/test split by system."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-3, 4, 8)
TARGET_KEYS = {"force": "force", "distance": "distance", "velocity": "velocity", "energy": "energy",
               "accel_step": "accel_step"}


def target_array(arrays: dict, target: str, sys_idx: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Ground truth for `target` at the given systems and steps -> (N_sys * N_steps, d)."""
    a = arrays[TARGET_KEYS[target]][sys_idx][:, steps]
    return a.reshape(len(sys_idx) * len(steps), -1).astype(np.float64)


def system_split(n_sys: int, seed: int = 0, test_frac: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sys)
    n_test = max(1, int(round(n_sys * test_frac)))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def fit_probe(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, seed: int = 0) -> dict:
    """Ridge probe X -> Y. `groups` gives the system index of each row; test rows are unseen systems.

    Returns R^2 on held-out systems (uniform average across output dims) plus the fitted pieces.
    """
    uniq = np.unique(groups)
    tr_sys, te_sys = system_split(len(uniq), seed)
    tr = np.isin(groups, uniq[tr_sys])
    te = ~tr
    xs = StandardScaler().fit(X[tr])
    ys = StandardScaler().fit(Y[tr])
    model = RidgeCV(alphas=ALPHAS).fit(xs.transform(X[tr]), ys.transform(Y[tr]))
    def predict(Xn):
        return ys.inverse_transform(model.predict(xs.transform(Xn)).reshape(len(Xn), -1))

    pred = predict(X[te])
    r2 = float(r2_score(Y[te], pred, multioutput="uniform_average"))
    return {"r2": max(r2, -1.0), "alpha": float(model.alpha_), "test_mask": te, "pred_test": pred,
            "predict": predict}


def probe_r2(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, seed: int = 0) -> float:
    return fit_probe(X, Y, groups, seed)["r2"]
