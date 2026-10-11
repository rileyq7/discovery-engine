"""Backfill multi-control probe baselines for runs recorded with a single random control.

Probe results are immutable, so this writes a new result file per run, `controls.json`, with k random-init
control R2s per (layer, target) on the run's own probe data. No checkpoints are needed. It then re-judges
each run's final-checkpoint probes and formation steps under the multi-control rule
(beat the best control by max(margin, spread)).

    python -m engine.probes.recontrol [--k 3] [--run RUN_ID ...]
"""
from __future__ import annotations

import argparse
import json

from app import store
from app.schemas.config import ExperimentConfig
from engine.probes.controls import counts
from engine.probes.suite import _layers, fit_controls, probe_data
from engine.train import prepare


def backfill(run_id: str, k: int = 3) -> dict:
    r = store.get_run(run_id)
    cfg = ExperimentConfig.model_validate(store.get_experiment(r["experiment_id"])["config"])
    prep = prepare(cfg, log=lambda m: None)
    layers = _layers(cfg)
    targets, _x, _s, groups, Y, acts_for, _ps, _pos = probe_data(cfg, prep, layers)
    _best, _spread, vals = fit_controls(cfg, r["seed"], acts_for, Y, groups, layers, targets, k)
    rows = [{"layer": l, "target": t, "control_r2s": [round(v, 5) for v in vals[(l, t)]]} for (l, t) in vals]
    store.write_run_result(run_id, "controls", {"k": k, "rows": rows,
                                                "note": "backfilled random-init controls; control 0 is the "
                                                        "original single control"})
    return rejudge(run_id)


def _controls(run_id: str) -> dict | None:
    f = store._run_dir(run_id) / "controls.json"
    if not f.exists():
        return None
    return {(x["layer"], x["target"]): x["control_r2s"] for x in json.loads(f.read_text())["rows"]}


def rejudge(run_id: str) -> dict:
    """Per target: the original verdict (single control, margin) and the multi-control verdict, at the final
    checkpoint's best layer, plus formation steps under both rules."""
    ctl = _controls(run_id)
    r = store.get_run(run_id)
    margin = store.get_experiment(r["experiment_id"])["config"]["probes"].get("control_margin", 0.1)
    rows = store.get_probe_results(run_id)
    final = max(p["checkpoint_step"] for p in rows)
    out = {}
    for t in sorted({p["target"] for p in rows}):
        def multi(p):
            c = ctl[(p["layer"], t)]
            return max(c), max(c) - min(c)

        fin = [p for p in rows if p["target"] == t and p["checkpoint_step"] == final]
        old_best = max(fin, key=lambda p: p["r2"] - (p["control_r2"] or 0))
        new_best = max(fin, key=lambda p: p["r2"] - multi(p)[0] - max(margin, multi(p)[1]))
        bc, sp = multi(new_best)

        def formed(rule):
            series = sorted((p for p in rows if p["target"] == t and p["layer"] == layer), key=lambda p: p["checkpoint_step"])
            for i, p in enumerate(series):
                if all(rule(q) for q in series[i:]):
                    return p["checkpoint_step"]
            return None

        layer = old_best["layer"]
        old_formed = formed(lambda q: counts(q["r2"], q["control_r2"], margin))
        layer = new_best["layer"]
        new_formed = formed(lambda q: counts(q["r2"], multi(q)[0], margin, multi(q)[1]))
        out[t] = {"old": {"layer": old_best["layer"], "margin": round(old_best["r2"] - old_best["control_r2"], 3),
                          "counts": counts(old_best["r2"], old_best["control_r2"], margin), "formed_at": old_formed},
                  "new": {"layer": new_best["layer"], "margin": round(new_best["r2"] - bc, 3),
                          "bar": round(max(margin, sp), 3), "counts": counts(new_best["r2"], bc, margin, sp),
                          "formed_at": new_formed}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="engine.probes.recontrol")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--run", nargs="*")
    a = ap.parse_args(argv)
    store.rebuild_index()
    ids = a.run or [r["id"] for r in store.list_runs("completed")]
    for rid in ids:
        r = store.get_run(rid)
        if any(p.get("control_spread") is not None for p in store.get_probe_results(rid)[:1]):
            continue  # recorded with multi-control probes already
        res = rejudge(rid) if _controls(rid) else backfill(rid, a.k)
        print(json.dumps({"run_id": rid, "seed": r["seed"], "targets": res}), flush=True)


if __name__ == "__main__":
    main()
