# Discovery Engine: Build Spec

Oct 11, 2026 · @Riley

## Overview

Discovery Engine is a self-running lab that trains small transformers on astrophysical data, probes their internals to test whether they learned the real physics or shortcuts, and turns the results into tracked findings and paper drafts.

**Research question.** When a transformer learns to predict orbits and light curves well, does its internal representation contain the actual physical law (Newtonian force, Kepler's laws), or task-specific heuristics that happen to work? And at what point in training does the real structure form, if ever?

**Why it's open.** Vafa et al. (2025, "What Has a Foundation Model Found?") showed a transformer trained on orbits predicted trajectories well but did not recover Newtonian gravity. Cranmer's symbolic regression work showed laws can sometimes be extracted from networks. This system sits between the two: continuous, automated experiments that check prediction, internal representation, and extractable equations together, across training checkpoints.

**What it does, end to end:**

1. Generates or loads data with a known ground truth (REBOUND N-body simulations first; Gaia, TESS and LIGO open data later).
2. Trains small transformers and saves checkpoints throughout training.
3. Runs a probe suite on every checkpoint: linear probes for physical quantities, activation patching, symbolic regression on probe outputs.
4. An agent loop reads results, proposes the next experiment within a budget, queues it, and records findings with links to the exact runs that support or refute them. Null results are recorded as findings, not dropped.
5. A web UI tracks the queue, runs, findings and paper drafts, and lets you pause, steer or approve at any point.

**Non-goals for v0:** large models, multi-GPU training, real-time telescope ingestion, auto-submitting papers anywhere. Nothing leaves the machine without you approving it.

## Architecture

&#91;embedded content: Discovery Engine architecture · 7 components, 1 loop\]

The loop closes through the agent: results land in an append-only store, the agent turns them into findings and the next proposal, and the approval gate decides what reaches the queue. You steer from the web UI at the gate, never by editing past results.

## Experiment loop

The loop runs forever until paused: each cycle takes one experiment from proposal to a recorded finding, then uses that finding to propose the next.

**Experiment states:** `proposed` → `approved` → `queued` → `running` → `analyzing` → `complete` (or `failed`, `cancelled`). Every transition is logged with a timestamp and actor (agent or human).

**One cycle:**

1. **Propose.** The agent reads the findings log, open questions, and recent results, then writes 1 to 3 experiment configs as JSON, each with a hypothesis, the expected result if true, and the expected result if false.
2. **Validate.** Configs are checked against the schema and budget. Invalid configs go back to the agent with the error; the agent gets 2 retries.
3. **Approve.** Auto-approved if under the per-experiment compute cap and in an allowed experiment family. Anything else waits for you in the UI.
4. **Run.** A worker trains the model and saves checkpoints, metrics and activations to disk.
5. **Probe.** The probe suite runs on the checkpoints listed in the config.
6. **Analyze.** The agent reads the metrics and plots, then writes a finding: claim, verdict (`supported`, `refuted`, `inconclusive`), evidence run IDs, and a confidence note.
7. **Next.** The agent updates the open questions list and proposes the next experiment.

**Guardrails:**

- Daily compute budget in GPU-hours (or CPU-hours) and a max of N concurrent runs, both set in the UI.
- Every finding must cite at least one completed run ID. A finding with no evidence is rejected.
- A finding moves to `replicated` only after a second run with a different seed agrees.
- Null and negative results are kept and shown, never deleted.
- The agent cannot edit past results, only append. Raw run data is immutable.
- No network access from runs except an allowlist of dataset sources (Gaia, MAST for TESS, GWOSC for LIGO).
- Paper drafts can only cite `replicated` findings by default; you can override per claim.
- A kill switch in the UI pauses the loop and cancels queued (not running) jobs.

**Agent backend.** Everything runs on the Max subscription; there is no API key anywhere in the system. Two entry points, both on the subscription: a scheduled Claude task fires roughly hourly and runs one full cycle by following agent/CYCLE.md, and for manual or local runs, agent/cycle.py calls headless Claude Code (claude -p) logged in with the Max account.

**Subscription-only rules:**

- No `anthropic` SDK dependency and no `ANTHROPIC_API_KEY` in code, `.env`, CI, or the scheduled task environment. If that variable is set, Claude Code bills the API instead of the subscription.
- `make check-billing` fails loudly if `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` is set, and runs before every cycle.
- State lives in the GitHub repo, because each scheduled run starts from a fresh clone: pull, run one cycle, commit, push.
- Usage limits are shared with your normal Claude and Claude Code use, so the daily budget in the UI caps cycles per day and leaves headroom for your own work.
- Runs are CPU-friendly by default so a cloud-run cycle can finish. GPU sweeps and real-data pulls (Gaia, TESS, LIGO) run on your own computer instead.

## Experiment catalog

The system ships with six seeded experiments; E1 and E2 must work before the agent loop is switched on.

| ID | Name | Question | Data | Main output |
| --- | --- | --- | --- | --- |
| E1 | Baseline orbit prediction | Can a small transformer predict the next position of a two-body orbit? | REBOUND, 2-body, 10k systems | Prediction MSE vs held-out systems |
| E2 | Force probe | Do the residual stream activations linearly encode the gravitational force vector? | E1 checkpoints | Probe R² per layer per checkpoint |
| E3 | Law extraction | Does symbolic regression on probe outputs recover F = GMm/r²? | E2 probe outputs | Recovered equation + fit error |
| E4 | Transfer test | Does the model generalize to unseen mass ratios and to 3-body systems? | REBOUND, out-of-distribution sets | Prediction error vs probe R² on OOD data |
| E5 | Formation timing | At which training step does force encoding appear, and does it come before or after prediction accuracy? | Dense checkpoints from E1 | Probe R² and loss curves on one time axis |
| E6 | Scaling sweep | Does force encoding improve with model size, data size, or neither? | E1 setup, 4 model sizes × 3 data sizes | Grid of probe R² and loss |

**Probes to implement:** linear probes (ridge regression) per layer for force vector, distance r, relative velocity, and total energy; activation patching on the attention heads with the highest probe scores; PySR symbolic regression on probe predictions.

**Control for every probe:** the same probe trained on a randomly initialized model with the same architecture. A probe result only counts if it beats the random-model control by a set margin.

**Phase 2 (after v0 works):** TESS light curves for transit periods (lightkurve), Gaia DR3 binary star orbits (astroquery), LIGO strain data for chirp mass (gwosc). Same probe-then-extract pattern, real data as the verifier.

## Data model and config schema

The source of truth is plain files committed to the repo under lab/ (experiments, runs, metrics, probe results, findings, questions, papers as JSON or Markdown), and SQLite is a local index rebuilt from those files on startup so the UI stays fast. Heavy data (datasets, checkpoints, activations) is gitignored and regenerated from configs and seeds when needed; it lives under `runs/<run_id>/` and is referenced by path.

| Table | Key fields |
| --- | --- |
| `experiments` | id, family (E1 to E6 or custom), hypothesis, expected\_if\_true, expected\_if\_false, config\_json, state, proposed\_by, created\_at |
| `runs` | id, experiment\_id, seed, status, started\_at, ended\_at, compute\_hours, artifact\_dir, git\_commit |
| `metrics` | run\_id, step, name, value |
| `probe_results` | run\_id, checkpoint\_step, layer, target, r2, control\_r2 |
| `findings` | id, claim, verdict, status (hypothesis, supported, refuted, inconclusive, replicated), evidence\_run\_ids, confidence\_note, created\_at |
| `questions` | id, text, status (open, answered, dropped), linked\_finding\_ids |
| `papers` | id, title, markdown, cited\_finding\_ids, updated\_at |
| `events` | id, timestamp, actor, type, payload\_json (audit log of every state change) |

**Experiment config (JSON, validated with Pydantic):**

```json
{
  "family": "E2",
  "hypothesis": "Layer 3 residual stream linearly encodes the force vector after 50% of training",
  "expected_if_true": "Probe R2 > 0.8 at layer 3 from step 5000 on, control R2 < 0.2",
  "expected_if_false": "Probe R2 stays near control at all layers",
  "data": {"source": "rebound", "n_bodies": 2, "n_systems": 10000, "steps_per_orbit": 64, "mass_ratio_range": [0.1, 1.0]},
  "model": {"n_layers": 4, "d_model": 128, "n_heads": 4, "context_len": 128},
  "training": {"steps": 20000, "batch_size": 256, "lr": 0.0003, "checkpoint_every": 500},
  "probes": {"targets": ["force", "distance", "velocity", "energy"], "layers": "all", "checkpoints": "all", "random_control": true},
  "seeds": [0, 1],
  "budget": {"max_compute_hours": 2}
}
```

The agent may only emit configs matching this schema. Unknown fields are rejected, not ignored.

## UI

A single-page web app with a left sidebar and six pages, dark mode by default, live-updating while the loop runs.

| Page | What it shows | Actions |
| --- | --- | --- |
| Dashboard | Loop status (running or paused), today's compute used vs budget, runs in progress with live loss curves, latest 5 findings, open questions count | Start/pause loop, kill switch, set budget |
| Experiments | Table of all experiments: family, hypothesis, state, verdict, compute, date. Filter by family, state, verdict | Approve or reject proposed experiments, edit a config before approving, propose one by hand |
| Run detail | Config, loss curves, probe R² heatmap (layer × checkpoint) with control overlay, extracted equations, logs, artifact links | Re-run with new seed, mark as bad, add a note |
| Findings | Claim, verdict badge, status, linked evidence runs, replication count. Timeline view of how findings evolved | Promote, dispute, add a note, start a replication |
| Papers | Markdown editor with live preview on the right, a findings sidebar to drag in claims and figures, auto-generated references to run IDs | Ask agent to draft a section, export Markdown and LaTeX |
| Data | Dataset browser: sample orbits plotted in 2D, distribution of masses and periods, dataset versions | Generate a new dataset, compare two datasets |

**Key visuals to get right:**

- **Probe heatmap:** layers on the y axis, training step on the x axis, color = probe R² minus control R². This one picture answers "when and where does the physics appear."
- **Formation timing chart:** loss and probe R² on one shared step axis, so you can see whether understanding comes before or after accuracy.
- **Findings timeline:** each finding as a card on a vertical timeline, color-coded by verdict, so the research story reads top to bottom.

**Live updates:** the backend pushes run progress and new findings over Server-Sent Events; no manual refresh.

## API, repo structure, tech stack

Python backend for everything that touches models, a TypeScript frontend for the UI, one command to start both.

**Stack:**

| Layer | Choice | Why |
| --- | --- | --- |
| Simulation | REBOUND | Fast, accurate N-body with a Python API |
| Models | PyTorch, custom small GPT with forward hooks | Full control of activations; no framework lock-in |
| Probes | scikit-learn (ridge), PySR (symbolic regression) | Simple, standard, inspectable |
| Real data (phase 2) | astroquery, lightkurve, gwosc | Official access libraries for Gaia, TESS, LIGO |
| Backend | FastAPI, SQLModel on SQLite, Pydantic | One process, typed schemas shared with the agent |
| Job runner | A worker process polling the queue table | No Redis or Celery needed for one machine |
| Agent | Scheduled Claude tasks + headless Claude Code (claude -p), Max subscription only, configs validated against the schema | Structured, auditable proposals |
| Frontend | Vite, React, TypeScript, Tailwind, shadcn/ui, Recharts, TanStack Query | Fast to build, looks good by default |
| Paper editor | CodeMirror 6 with Markdown, react-markdown with KaTeX for preview | Equations render inline |
| Hardware | Apple Silicon (MPS) or CUDA, falls back to CPU | v0 models are small enough for a laptop |

**Repo layout:**

```
discovery-engine/
  backend/
    app/main.py            # FastAPI app, SSE stream
    app/api/               # experiments, runs, findings, papers, loop, data
    app/db/models.py       # SQLModel tables
    app/schemas/config.py  # Pydantic experiment config
    engine/sim/            # REBOUND dataset generators
    engine/models/         # small GPT + hooks
    engine/train.py
    engine/probes/         # linear, patching, pysr, controls
    engine/worker.py       # queue runner
    agent/CYCLE.md         # step-by-step instructions every scheduled run follows
    agent/cycle.py         # same cycle for local runs via claude -p
    agent/prompts/         # system prompt, finding template
    tests/
  frontend/
    src/pages/             # Dashboard, Experiments, RunDetail, Findings, Papers, Data
    src/components/        # ProbeHeatmap, FormationChart, FindingCard
  runs/                    # artifacts (gitignored)
  Makefile                 # make dev, make worker, make loop, make test
```

**Core endpoints:**

- `GET/POST /experiments`, `POST /experiments/{id}/approve`, `POST /experiments/{id}/reject`
- `GET /runs/{id}`, `GET /runs/{id}/metrics`, `GET /runs/{id}/probes`
- `GET/POST /findings`, `POST /findings/{id}/replicate`
- `GET/PUT /papers/{id}`, `POST /papers/{id}/draft-section`, `GET /papers/{id}/export?format=latex`
- `POST /loop/start`, `POST /loop/pause`, `PUT /loop/budget`
- `GET /stream` (Server-Sent Events for live updates)

## Build milestones and kickoff prompt

Build in five milestones inside one Claude Code session; each must pass its check before the next starts, so "one shot" means one session, not one untested dump of code.

1. **Engine core.** REBOUND generator, small GPT, training with checkpoints, E1 runs from the CLI.
   - Check: E1 trains on a laptop in under 30 minutes and prediction MSE beats a constant-velocity baseline.
2. **Probes.** Linear probes with random-model controls, PySR extraction, E2 and E3 from the CLI.
   - Check: a unit test where the force is planted directly in a fake activation recovers R² > 0.95; the random control scores near zero.
3. **Backend and storage.** SQLite schema, FastAPI endpoints, worker process, SSE stream.
   - Check: queue an E1 config through the API, watch it run, see metrics and probe results come back over the API.
4. **UI.** All six pages wired to the API, probe heatmap and formation chart working on real E1/E2 runs.
   - Check: from the browser alone you can propose, approve, watch, and read the results of an experiment.
5. **Agent loop and papers.** CYCLE.md and cycle.py, make check-billing, the lab/ file store committed to GitHub, paper editor with draft-section.
   - Check: 5 cycles run end to end via claude -p from a fresh clone, each pushes one commit, every finding cites run IDs, and a pause flag committed to the repo stops the next cycle.

**Done means:** `make dev` starts everything, the dashboard shows the loop running, and after a night of running there are runs, findings (including nulls) and a draft paper section you can read.

**Kickoff prompt for Claude Code** (paste this with the spec attached as `SPEC.md` in an empty repo):

```
Read SPEC.md fully before writing code. Build Discovery Engine exactly as specified,
milestone by milestone. Before each milestone, write a short plan in PLAN.md. After
each milestone, run its check and show me the result. Do not start the next
milestone until the check passes. Write tests as you go. Keep models small enough
to train on my laptop. Ask me before adding any dependency not listed in the spec.
Never use the Anthropic API, the anthropic SDK, or an API key: every model call goes
through claude -p on my Max subscription. If something in the spec is ambiguous,
pick the simplest option, note it in
DECISIONS.md, and keep going.
```

**To get SPEC.md:** export this doc as Markdown and drop it in the repo root.

**Turning on the loop:** once milestone 5 passes, push the repo to GitHub and tell Claude in chat the repo name. Claude sets up an hourly scheduled task that clones the repo, runs `make check-billing`, follows `agent/CYCLE.md` for one cycle, and pushes. It can run in the cloud (CPU, simulations only) or on your computer (your hardware, real-data pulls allowed).
