"""Lab operations shared by the API, the CLI and the agent tools: propose, validate, approve, replicate."""
from __future__ import annotations

import copy
import json
import os
from typing import Optional

from pydantic import ValidationError

from app import store
from app.schemas.config import (DEFAULT_STEP_OVERHEAD_S, DEFAULT_THROUGHPUT, ExperimentConfig, estimate_compute_hours,
                               validate_config)


class ProposalError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _format_errors(e: ValidationError) -> list[str]:
    out = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "<root>"
        out.append(f"{loc}: {err['msg']}")
    return out


def estimate(cfg: ExperimentConfig) -> float:
    # env (e.g. a CPU-only machine) > lab/settings.json > the MPS-calibrated defaults
    s = store.get_settings()
    tp = os.environ.get("DE_THROUGHPUT") or s.get("throughput") or DEFAULT_THROUGHPUT
    oh = os.environ.get("DE_STEP_OVERHEAD") or s.get("step_overhead_s") or DEFAULT_STEP_OVERHEAD_S
    return estimate_compute_hours(cfg, float(tp), float(oh))


def check(raw: dict | str) -> tuple[Optional[ExperimentConfig], list[str], float]:
    """Validate schema and budget. Returns (config, errors, estimated hours)."""
    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
        cfg = validate_config(raw)
    except json.JSONDecodeError as e:
        return None, [f"invalid JSON: {e}"], 0.0
    except ValidationError as e:
        return None, _format_errors(e), 0.0
    est = estimate(cfg)
    s = store.get_settings()
    errors = []
    if est > cfg.budget.max_compute_hours:
        errors.append(f"budget: estimated {est:.3f} compute-hours exceeds budget.max_compute_hours "
                      f"{cfg.budget.max_compute_hours}; reduce steps/batch/model size/seeds or raise the budget")
    if est > s["daily_budget_hours"]:
        errors.append(f"budget: estimated {est:.3f} compute-hours exceeds the daily budget "
                      f"{s['daily_budget_hours']}")
    return cfg, errors, est


def auto_approvable(cfg: ExperimentConfig, est: float) -> tuple[bool, str]:
    s = store.get_settings()
    if cfg.family not in s["auto_approve_families"]:
        return False, f"family {cfg.family} needs human approval"
    if est > s["per_experiment_cap_hours"]:
        return False, f"estimate {est:.2f}h is over the per-experiment cap {s['per_experiment_cap_hours']}h"
    return True, "under cap and in an allowed family"


def propose(raw: dict | str, actor: str, auto_approve: bool = True, replicates_finding_id: Optional[str] = None,
            note: str = "") -> dict:
    cfg, errors, est = check(raw)
    if errors:
        raise ProposalError(errors)
    rec = store.create_experiment(cfg.model_dump(mode="json"), actor, est, replicates_finding_id, note)
    if auto_approve:
        ok, why = auto_approvable(cfg, est)
        if ok:
            rec = store.transition(rec["id"], "approved", "auto-approve", why)
        else:
            store.add_experiment_note(rec["id"], f"waiting for human approval: {why}", "auto-approve")
            rec = store.get_experiment(rec["id"])
    return rec


def edit_config(exp_id: str, raw: dict, actor: str) -> dict:
    cfg, errors, est = check(raw)
    if errors:
        raise ProposalError(errors)
    return store.update_experiment_config(exp_id, cfg.model_dump(mode="json"), est, actor)


def approve(exp_id: str, actor: str, note: str = "") -> dict:
    exp = store.get_experiment(exp_id)
    _, errors, _ = check(exp["config"])
    if errors:
        raise ProposalError(errors)
    return store.transition(exp_id, "approved", actor, note)


def reject(exp_id: str, actor: str, note: str = "") -> dict:
    return store.transition(exp_id, "rejected", actor, note)


def replicate(finding_id: str, actor: str) -> dict:
    """Start a replication: the evidence experiment's config with fresh seeds."""
    f = store.get_finding(finding_id)
    run_seeds = {store.get_run(r)["seed"] for r in f["evidence_run_ids"]}
    exp = store.get_experiment(f["experiment_id"])
    cfg = copy.deepcopy(exp["config"])
    used = set(run_seeds) | set(cfg["seeds"])
    new_seed = max(used) + 1
    cfg["seeds"] = [new_seed]
    cfg["hypothesis"] = f"Replication of {finding_id}: {f['claim']}"[:2000]
    n_old = max(1, len(exp["config"]["seeds"]))
    share = max(cfg["budget"]["max_compute_hours"] / n_old, 0.01) * 1.5
    cfg["budget"]["max_compute_hours"] = round(max(share, estimate(ExperimentConfig.model_validate(cfg)) * 1.25), 3)
    return propose(cfg, actor, replicates_finding_id=finding_id, note=f"replication of {finding_id}")
