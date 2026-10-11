# CYCLE.md — one Discovery Engine cycle

You are the scheduled research agent. Each run starts from a fresh clone of this repo. Run exactly
**one cycle** by following these steps in order, then stop. All state lives in `lab/` (committed);
heavy artifacts in `runs/` are gitignored and regenerate from configs + seeds.

Read `backend/agent/prompts/system.md` first: it defines the research question, what the probes
measure (always compare `force` with `accel_step`), and the rules you must follow.

All commands run from the repo root. `T` below means `PYTHONPATH=backend backend/.venv/bin/python -m agent.tools`.

## 0. Setup and guards
1. `make check-billing`. **If it fails, stop immediately** and report the failure. Never set or use
   `ANTHROPIC_API_KEY`, never install or import the `anthropic` SDK.
2. If `backend/.venv` is missing: `make setup` (installs Python deps, frontend deps, and Julia for PySR).
   In a cloud environment without a GPU, export `DE_DEVICE=cpu`.
3. `git pull --rebase`.
4. `T cycle-begin`. If it exits non-zero (paused via `lab/PAUSED`, daily cycle cap, or daily budget
   used), **stop here without committing**. A committed `lab/PAUSED` file always stops the cycle.

## 1. Analyze leftovers
`T pending-analysis` lists experiments in `analyzing` with their results. For each one, write a
finding (step 5) before proposing anything new.

## 2. Propose (1 to 3 experiments)
1. `T context` — findings, open questions, recent experiments with results, budget left today.
2. `T schema` — the config JSON schema. Start from a seeded config in `configs/` when possible.
3. Write your proposals to `/tmp/proposals.json` (a JSON list of configs). Each needs a falsifiable
   `hypothesis`, `expected_if_true`, `expected_if_false`, and a `budget.max_compute_hours` that
   covers the estimate. Keep them CPU-friendly when running in the cloud.
4. `T validate /tmp/proposals.json`. Fix every error. You get **2 retries**; if still invalid, skip
   proposing this cycle (note why in the cycle summary).
5. `T propose /tmp/proposals.json`. Experiments under the per-experiment cap in an auto-approve
   family are approved automatically; the rest wait for a human in the UI. Do not try to approve
   anything yourself.

## 3. Run and probe
`T run-queue --max-minutes 45` runs approved experiments (training, checkpoints, the probe suite)
until the queue is empty or time runs out. Unfinished runs are fine; the next cycle continues.

## 4. Analyze
`T pending-analysis` again. For each experiment:
- Compare measured results with `expected_if_true` / `expected_if_false`.
- Verdict: `supported`, `refuted`, or `inconclusive`. Null and negative results are findings too.

## 5. Record findings
Write one finding per analyzed experiment to `/tmp/finding.json`, using the template in
`backend/agent/prompts/finding_template.md`:
```json
{"claim": "...with the key numbers...", "verdict": "supported|refuted|inconclusive",
 "evidence_run_ids": ["run-..."], "confidence_note": "...", "experiment_id": "exp-...",
 "seeds_agree": false, "replicates": null,
 "question_updates": [{"id": "q-...", "status": "answered"}], "new_questions": ["..."]}
```
`T finding /tmp/finding.json`. It rejects findings without completed evidence runs; fix them and retry
(2 retries). Use only run IDs shown by the tools.

## 6. Next
- Update open questions: `T question add "<text>"`, `T question set <q-id> answered --finding <f-id>`.
- Optional: if there are newly `replicated` findings, update the running draft paper
  (`PYTHONPATH=backend backend/.venv/bin/python -c "from agent.cycle import maybe_draft_paper; maybe_draft_paper()"`
  uses `claude -p`; alternatively edit `lab/papers/<id>.md` yourself, citing only replicated findings as `[f-...]`).

## 7. Commit and push
1. `T cycle-end --summary "<one line: what was proposed, run, found>"`
2. `PYTHONPATH=backend backend/.venv/bin/python -m app.snapshot` (refreshes `snapshot/`, which the hosted dashboard serves)
3. `git add lab snapshot && git commit -m "cycle <date> #<n>: <summary>" && git push`
   (on a push conflict: `git pull --rebase` and push again). **Exactly one commit per cycle**, touching only `lab/` and `snapshot/`.

## Never
- Edit or delete anything under `lab/runs/` or `lab/findings/` by hand; the tools only append.
- Delete null or negative results.
- Access the network from runs, except the dataset allowlist (Gaia, MAST/TESS, GWOSC/LIGO) in phase 2.
- Submit or publish anything outside this repo.
