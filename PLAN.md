# Build plan

Built milestone by milestone, following SPEC.md. Each milestone has a check that has to pass before the next one starts.

## Milestone 1: Engine core
- `engine/sim/rebound_gen.py`: REBOUND 2-body (and hierarchical 3-body) generator. Each system is sampled at
  `steps_per_orbit` points per inner orbit, in the centre-of-mass frame. The generator also writes ground-truth
  force, distance, relative velocity and energy for every step. Datasets are cached by a config hash under `runs/_datasets/`.
- `engine/models/gpt.py`: a small causal GPT over continuous tokens (xy positions of up to 3 bodies). Forward hooks
  capture the residual stream after each block and the per-head attention outputs.
- `engine/train.py`: AdamW with warmup and cosine decay. It predicts the next-step displacement, saves checkpoints at
  step 0, every `checkpoint_every` steps and at the end, and logs train loss, val MSE and the constant-velocity
  baseline MSE.
- `engine/cli.py`: `python -m engine.cli run configs/e1.json` runs an experiment end to end without the server.
- Check: E1 trains in under 30 min on this laptop, and the val MSE beats the constant-velocity baseline.

## Milestone 2: Probes
- `engine/probes/linear.py`: ridge probes per layer and checkpoint for force, distance, velocity and energy. Train
  and test sets are split by system.
- `engine/probes/controls.py`: the same probe on a randomly initialised model with the same architecture. A result
  only counts if it beats the control by a set margin.
- `engine/probes/patching.py`: ranks heads by per-head force-probe R², then patches each top head's output from a
  different system and measures the damage.
- `engine/probes/symbolic.py`: PySR on probe-predicted |F| against (r, m1, m2).
- Check: in a unit test, a force planted in fake activations is recovered with R² > 0.95, and the random control
  scores near zero.

## Milestone 3: Backend and storage
- `lab/` file store (the source of truth) plus a SQLite index rebuilt on startup, using the SQLModel tables from the spec.
- FastAPI endpoints from the spec, a worker that polls the queue and runs one subprocess per run, and an SSE stream.
- Check: queue E1 through the API, watch it run, and read the metrics and probe results back over the API.

## Milestone 4: UI
- Vite, React, TS, Tailwind, Recharts and TanStack Query. Six pages, a probe heatmap and a formation chart.
- Check: propose, approve, watch and read the results of an experiment from the browser alone.

## Milestone 5: Agent loop and papers
- `agent/tools.py` (the CLI the agent uses), `agent/CYCLE.md`, `agent/cycle.py` (via `claude -p`), `make check-billing`,
  the pause flag, and the paper editor with draft-section.
- Check: 5 cycles from a fresh clone, one commit each, every finding cites run IDs, and the pause flag stops the next cycle.

## Check results (2026-10-11)
- **M1 passed.** E1 trained in 11.1 min on MPS (Apple Silicon). Held-out MSE 1.31e-7 vs constant-velocity 2.55e-5 (ratio 0.005).
- **M2 passed.** A force planted in fake activations is recovered with R² 0.9999, and the random-model control gives −0.001
  at every layer. PySR recovers `m2/r**2` exactly. Added the `accel_step` target, tested against the per-step second
  difference of positions (within 1%).
- **M3 passed.** E1 (seed 1) was queued with `POST /experiments`, run by the worker, and streamed over `GET /stream` (SSE).
  Metrics and probe results were read back from `/runs/{id}/metrics` and `/runs/{id}/probes`.
- **M4 passed.** In the browser: proposed an E6 cell from a template, saw it held at the gate (E6 is not auto-approved),
  approved it, watched the live loss curve on the dashboard, and read the run detail (heatmap, formation chart, stats).
- **M5 passed.** Five cycles ran through `claude -p` from a fresh clone of a bare git remote (standing in for GitHub).
  There was one commit per cycle (5 commits, 19:53–21:37) and `check-billing` passed before each one. The cycles
  produced 10 findings, and every one cites completed run IDs: 5 replicated, 2 supported (single seed),
  2 inconclusive, 1 refuted. Null and refuted results were kept. 11 experiments completed. One E6 cell is waiting
  for human approval, and one is still queued (`--max-minutes 15` left only enough time for one run). The running
  draft paper's Results section cites only replicated findings. Pause test: `lab/PAUSED` was committed from a
  second clone, and the next cycle pulled it, logged `cycle skipped: paused`, and made no commit.
  Fix after the run: in cycle 5, all three proposal attempts had one over-budget config, so the agent proposed
  nothing. Now the final attempt proposes the valid subset.
