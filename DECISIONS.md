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
- **Several random controls per probe.** Every probe is also fitted on `probes.n_controls` (default 3) randomly
  initialised models. `control_r2` is the best of them and `control_spread` is max − min. A result counts only if
  R² − control_r2 ≥ max(`control_margin`, `control_spread`). Why: at step 0, where the model is itself a random
  init, single-control margins reached 0.22 on `distance` and 0.18 on `energy` (but ≤ 0.014 on `force` /
  `accel_step`). Random models genuinely differ on those targets, so a fixed 0.1 bar counted noise. The spread
  raises the bar exactly where random models disagree. Runs recorded with one control got a backfilled
  `controls.json` (k = 3, written once; `engine.probes.recontrol`). Its first control reproduces the stored one to
  5e-6, and the agent and reviewer see the re-judged verdicts as `multi_control_rejudge`.
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
- **Budget hard stop.** A runaway guard stops training at 2× its seed's share of `budget.max_compute_hours`. The clock
  starts at the first training step, so dataset generation doesn't count. A stopped run gets a checkpoint and probes
  at its stop step, and records `steps_completed` / `steps_planned`. *History:* the first version stopped at 1.2×,
  counted data generation, and probed the last regular checkpoint. Combined with a low estimate, it cut 6 of 23
  early runs short; one (seed 14) trained only 55% of its steps.
- **Truncated runs and replication.** A run that completed less than 80% of its planned steps doesn't count as an
  agreeing seed (`store.MIN_COMPLETION`). The agent sees `stopped` / `completion` for every run and must mention it.
- **Compute estimate.** Per training step: n_layers × 0.03 s of fixed overhead, plus work/1.4e15 per hour, where
  work = batch × context × layers × (12d² + 2·ctx·d). Probes, symbolic regression, eval sets and worst-case data
  generation are added on top. Fitted on 23 runs on this laptop (MPS): at these sizes per-step overhead dominates, so
  the old d²-only formula was 1.3–2.4× too low for d = 64. Every past run finishes within 0.94× of its new estimate.
  `throughput` and `step_overhead_s` in `lab/settings.json`, or `DE_THROUGHPUT` / `DE_STEP_OVERHEAD`, override it
  for other machines.
- **Dataset cache under concurrency.** Generation takes a per-dataset file lock (`fcntl`). Concurrent runs that need
  the same new dataset wait for one generator instead of racing on a shared temp file, which once failed a run.

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
- **Independent reviewer** (`agent/review.py`, `agent/REVIEW.md`). It runs hourly as a separate process, 30 min
  after the research loop starts. Deterministic checks cover: evidence exists and is completed; replicated
  findings have ≥ 2 valid seeds; NaNs in raw `metrics.jsonl`; failed runs; early stops; identical seed curves;
  force/`accel_step` beating their controls at step 0 (leak signature); result files or metrics lines modified in
  git; findings' substance changed since first committed; billing; budget. For judgement it starts a fresh
  `claude -p` session with all tools disabled (`--tools ""`, `--strict-mcp-config`, no session persistence). That
  session sees each new finding next to raw run data read from files, the pre-registered expectations, the
  random-init noise floor, the last 5 experiments and the open questions. It never sees the research agent's
  reasoning. Verdicts:
  - **OK:** nothing to act on.
  - **FLAG:** the research agent must answer in `review_responses` next cycle. A response naming a finding becomes
    a note on it.
  - **PAUSE:** writes and commits `lab/PAUSED`.

  The model may pause only for `evidence_mismatch` or `guardrail`; anything else is downgraded to FLAG. Flags left
  unanswered for 2 completed cycles escalate to PAUSE. Reviews live in `lab/reviews/` (JSON + Markdown), file-only
  (no index table). A shared `runs/.git.lock` serialises the reviewer's and the loop's git operations.
- **Agent transcripts.** Every `claude -p` call (research propose/analyze, paper, reviewer, digest) uses
  stream-json output, and its full reply, any thinking blocks, timing and model are appended to
  `lab/transcripts/<date>.jsonl`. This is the record of *why* the lab did what it did. Prompts aren't stored, since
  they are large and derivable from lab state.
- **Lab tutor chat** ("Ask the lab"). Each thread is a `claude -p` session, resumed on follow-ups, streamed to the UI
  over SSE. Tools: Read/Grep/Glob inside the repo plus `scripts/lab` (read-only status, context, results, raw runs,
  transcripts). It cannot edit. `--setting-sources project` keeps user-level permission rules out, which was tested:
  other Bash commands are denied. Web search is off unless enabled for a message, because queries leave the
  machine. The system prompt (`agent/prompts/tutor.md`) is written for an ML-fluent, physics-new owner: it bridges
  from ML, teaches with this lab's runs, and includes a physics primer and related work. Each message carries a live
  `<lab_state>` snapshot. Threads live in `runs/chats/` (local, gitignored): conversations are personal, not lab
  records. If a session can't be resumed, the thread is replayed into a fresh one.
- **Daily highlight reel.** At `settings.digest_hour` (default 07:00) the research loop writes
  `lab/digests/<date>.md`, even when paused. It covers everything since the previous digest (the first covers 12 h):
  cycles, the agent's proposing reasoning (from transcripts), experiments, runs, findings, reviews and responses,
  things needing a human, code changes from git (so fixed bugs aren't reported as open), and a physics concept of the
  day tied to the night's numbers. It is written by a tool-less `claude -p` in the tutor's voice from facts gathered
  in code, and shown on the Highlights page plus a dashboard banner.
- **API caching.** `/api/*` responses carry `Cache-Control: no-store` and the UI fetches with `cache: "no-store"`.
  Unknown `/api/...` paths return a JSON 404 rather than the SPA's `index.html`; once, a browser cached that HTML for
  a new endpoint.
- **Index schema changes.** `lab/index.sqlite` carries `PRAGMA user_version`. On a mismatch, the tables are dropped
  and recreated, and the startup `rebuild_index()` refills them from `lab/`. Non-finite metric values stay in
  `metrics.jsonl` but are skipped in the index, where SQLite would reject them.
- **Paper drafting.** The draft prompt includes computed counts (findings by verdict and status, evidence runs, runs
  cut short), so the model doesn't count. It also includes the "Science and engine" section of this file as the
  documented protocol for Methods. Sections are inserted in paper order (Abstract, Introduction, Methods, Results, …).
  When the replicated set changes, the running draft redrafts Results, adds Methods if missing, and refreshes the
  Abstract.
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
