# Decisions

Where the spec left something open, these notes record what was picked (usually the simplest option) and why.

## Science and engine
- **Per-snapshot acceleration probe (`accel_step`).** Every orbit has `steps_per_orbit` snapshots, so the time between
  snapshots is dt = P/N and grows with orbit size (dt² ∝ a³). The model never sees physical time. To predict the
  next position it needs the per-snapshot acceleration (F/m₂)·dt² = (2π/N)²·a³ / ((1+m₂)·r²), not the physical force
  m₂/r². A weak `force` probe alone could therefore mean "learned a step-based gravity" rather than "learned no
  gravity". `accel_step` is a fifth probe target next to the spec's four. It is on by default, has random-model
  controls, and appears in the heatmap and formation charts. PySR fits its law as well (inputs r, m₂, a). A test
  checks that it equals the second difference of positions (within 1%) and that it is *not* proportional to the
  physical force across orbit sizes. The agent's system prompt tells it to always compare the two.
- **Units.** G = 1 and the primary mass is M = 1 in every system, so the force law is |F| = m₂/r², with m₂ = the mass
  ratio. Absolute mass and time scale cannot be identified from positions sampled at fixed phase intervals, so
  varying M would add nothing.
- **Tokens.** Each step is the xy positions of up to 3 bodies (6 numbers) in the centre-of-mass frame, with absent
  bodies set to zero. One token format lets a 2-body-trained model be evaluated on 3-body sets (E4). 3-body systems
  are hierarchical: the third body sits at 3.5–6× the inner semi-major axis.
- **Prediction target.** The model predicts the normalised displacement to the next step. Reported MSE is in
  physical position units over steps t ≥ 1, compared with constant-velocity extrapolation (2xₜ − xₜ₋₁) on the
  same held-out systems.
- **Default E1 size.** The spec's example config (20k steps × batch 256) takes about 80 min on MPS, which breaks the
  "under 30 minutes" check. The seeded E1 uses 5k steps × batch 128 at lr 1e-3 and trains in 11 min to an MSE
  ratio of 0.005 (about 200× better than the baseline).
- **Probe protocol.** Probes run on 300 held-out systems at every other step from t = 8, with a 70/30
  train/test split *by system* (no leakage across rows from the same orbit). Ridge uses RidgeCV over
  α ∈ 10⁻³…10⁴ on standardised inputs and outputs, and R² is averaged across output dimensions. The control is
  the same probe on a randomly initialised model of the same architecture (one per layer, since it does not change
  over checkpoints). The default margin is 0.1 (`probes.control_margin`).
- **Activations to disk.** Saving all layers × all checkpoints would be about 2–3 GB per run. Only the final
  checkpoint's probe activations are saved (float16, `activations_final.npz`). Earlier checkpoints are
  recomputed from the saved weights when needed.
- **Activation patching.** Heads are ranked by a force probe on each head's contribution to the residual stream.
  The top 3 heads, plus the lowest-ranked head as a control, are then patched with that head's output on a
  different system, and the relative increase in next-step MSE is reported.
- **Extra config fields.** Beyond the spec example the schema has `data.eccentricity_range`, `data.data_seed`,
  `probes.control_margin`, `probes.patching`, `probes.symbolic` and `eval_sets` (for E4). Unknown fields are still
  rejected.
- **E6 sweep.** Each grid cell is its own experiment (12 configs in `configs/e6/`), so state, budget and findings
  stay per experiment. The schema has no sweep construct.
- **Budget hard stop.** Training stops at 1.2× its share of `budget.max_compute_hours` (the run is still recorded,
  with `summary.stopped = true`).
- **Compute estimate.** steps × batch × context × layers × (12d² + 2·ctx·d) / throughput, calibrated on this
  M-series laptop (MPS). It can be overridden with `throughput` in `lab/settings.json` for slower machines or cloud CPUs.

## Lab state and workflow
- **Repo root = this directory** (the spec's `discovery-engine/`). `lab/` sits at the repo root and is committed.
  `runs/` (checkpoints, datasets, logs from processes) is gitignored. `lab/index.sqlite` is gitignored and rebuilt
  from files on API, worker and agent-tool startup.
- **Write path.** File first (atomic temp+rename, or append for jsonl), then a SQLite upsert. SQLite runs in WAL mode
  with a busy timeout so the API, worker and run subprocesses can share it.
- **Pause flag.** The file `lab/PAUSED` (committed) is the loop flag. If it is absent, the loop is running. The UI's
  Pause/Start buttons write or remove it. While paused, the worker starts no new experiments and agent cycles are
  skipped. The kill switch pauses and also cancels `approved` and `queued` experiments; running ones finish.
- **States.** The spec's states, plus `rejected` for a human reject at the approval gate. The worker owns
  approved → queued (daily-budget admission) → running → analyzing | failed. Recording a finding moves
  analyzing → complete.
- **Replicated.** A finding becomes `replicated` (a) automatically, when the agent sets `seeds_agree` and the
  evidence includes ≥ 2 completed, not-bad runs with distinct seeds, (b) when a replication finding (`replicates`)
  agrees and adds a new seed, or (c) by a human Promote, which enforces the same evidence rule. Inconclusive
  findings cannot be replicated.
- **Re-run with new seed** creates a new experiment with the same config and one fresh seed. Runs are never
  mutated.
- **Run immutability.** Metrics are append-only. Probe, patching, equation and eval files cannot be overwritten.
  Humans can add notes and a `bad` flag, which excludes the run from seed-agreement counts.
- **Agent design.** `agent/cycle.py` does the deterministic orchestration (validation, retries, approval, worker,
  git), and `claude -p` only writes JSON (configs, findings) and paper prose. The scheduled task follows
  `agent/CYCLE.md` and uses the same `agent/tools.py` CLI, so both entry points hit the same validators.
- **Local loop in `make dev`.** `make dev` runs the local loop (one cycle per hour while not paused). Use
  `LOOP=off make dev` when the hourly scheduled task runs cycles instead, so two agents don't push concurrently.
  A lock file (`runs/.cycle.lock`) stops two local cycles from overlapping.
- **Running draft paper.** Each cycle the agent updates "Discovery Engine: running draft". When the set of
  replicated findings changes, it redrafts the Results section. If nothing is replicated yet, it writes a clearly
  labelled "Preliminary observations (unreplicated)" section using the citation override, so there is something
  to read after a night of running.
- **Paper citations.** The draft-section prompt only offers replicated findings plus per-claim overrides. Disallowed
  IDs in the output trigger a rewrite, then get marked "(NOT CITABLE)". The editor shows citation warnings.

## Frontend
- **shadcn/ui.** Components are written in the shadcn pattern (cva + tailwind-merge + clsx, lucide icons) in
  `src/components/ui/index.tsx` rather than generated with the CLI. Radix isn't needed for the few primitives used.
- **Routing.** Hash routing (`#/runs/<id>`) with no router dependency. The FastAPI app also serves the built UI,
  and API paths like `/experiments` would otherwise collide with page paths.
- **Extra packages implied by the spec:** `remark-math`, `rehype-katex` and `katex` (KaTeX in react-markdown),
  `remark-gfm` (tables), the `@codemirror/*` packages that make up CodeMirror 6, `@tailwindcss/vite` (Tailwind v4),
  and `clsx`, `tailwind-merge`, `class-variance-authority`, `lucide-react` (shadcn/ui's own dependencies).
- **Backend packages implied by the spec:** `uvicorn` (to serve FastAPI), `numpy`, `httpx` (FastAPI's TestClient), `pytest`.
- **API paths** are mounted both at the spec paths (`/experiments`, …) and under `/api` (used by the UI via the Vite proxy).
- **Figures in papers** are references such as `![caption](figure:heatmap/<run_id>/force)`, rendered live in the
  preview. LaTeX export leaves them as plain text.
- **Where the hourly loop runs.** On CPU the E1 model trains at about 1.4e13 units/h (benchmark: 3.8 s/step, about 30× slower
  than MPS), so E1 would take about 5 h per seed. The hourly loop therefore runs on the laptop (`make loop` /
  `make dev`), pushing to GitHub each cycle. A cloud routine would need `DE_DEVICE=cpu` and `DE_THROUGHPUT=1.4e13`
  (env override for the compute estimate), and would only fit tiny experiments.
