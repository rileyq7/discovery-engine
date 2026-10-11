# Discovery Engine

A self-running lab that trains small transformers on simulated orbits, probes their internals to see
whether they learned real gravity or shortcuts, and turns the results into tracked findings and paper drafts.
The full spec is in [SPEC.md](SPEC.md), the build plan in [PLAN.md](PLAN.md), and the choices made where the
spec was open in [DECISIONS.md](DECISIONS.md).

## Quick start

```bash
make setup        # Python venv (uv), frontend deps, Julia for PySR (first run takes a few minutes)
make dev          # API :8000 + worker + UI http://localhost:5173 + local agent loop
LOOP=off make dev # same, without the local agent loop (use when the scheduled task runs cycles)
```

Other targets: `make e1` (run E1 from the CLI), `make worker`, `make cycle` (one agent cycle),
`make loop`, `make review` (one independent review), `make review-loop`, `make test` (fast), `make test-all` (includes PySR), `make check-billing`.

## How it fits together

```
configs/ (E1-E6 seeds) ──► agent proposes ──► validate (schema + budget) ──► approval gate (auto or you, in the UI)
                                                                                     │
lab/ (committed source of truth) ◄── findings ◄── agent analyzes ◄── probes ◄── worker trains (runs/ artifacts)
     └─ SQLite index rebuilt on startup ─► FastAPI ─► SSE ─► React UI (pause, steer, approve)
```

- **Engine** (`backend/engine/`): REBOUND generator, a small GPT with residual and per-head hooks, training with
  checkpoints, and the probe suite. The probes are ridge regressions per layer and checkpoint for `force`,
  `accel_step`, `distance`, `velocity` and `energy`, each with a random-model control, plus activation patching
  and PySR law extraction.
- **Why `accel_step`:** orbits are sampled at a fixed number of snapshots per lap, so the model only ever sees
  "one snapshot to the next". What it needs is acceleration per snapshot, (F/m₂)·dt², which grows with orbit
  size. The probes measure that alongside physical force, so the lab can tell "learned gravity" from "learned a
  per-step version of gravity".
- **Backend** (`backend/app/`): file store in `lab/` with a SQLite index, the spec's REST endpoints (also under
  `/api`), and `GET /stream` for live updates over SSE.
- **Agent** (`backend/agent/`): `CYCLE.md` for the hourly scheduled task, and `cycle.py` for local cycles through
  `claude -p`. Both use the same `agent/tools.py` CLI and validators. No API key and no `anthropic` SDK anywhere.
- **Tutor and highlights:** "Ask the lab" is a read-only chat that sees the whole project, the live lab state and the
  agents' reasoning (`lab/transcripts/`), and teaches the physics from an ML starting point. Every morning the loop
  writes a highlight reel to `lab/digests/` (Highlights page).
- **UI** (`frontend/`): Dashboard, Experiments, Run detail (probe heatmap with control overlay, formation chart,
  equations), Findings timeline, Papers (CodeMirror + KaTeX preview, drag-in findings and figures, LaTeX export)
  and Data.

## Guardrails

- A daily compute budget, max concurrent runs, an auto-approve cap per experiment and a max number of cycles per
  day, all set from the dashboard.
- Every finding cites completed run IDs. `replicated` needs a second agreeing seed. Null results are kept.
- Raw run results are append-only. The agent cannot edit the past.
- `lab/PAUSED` (committed) pauses everything, and the kill switch also cancels queued work.
- Paper drafts cite only replicated findings unless you allow one per claim.
- An independent reviewer (a separate, tool-less `claude -p` session) checks each new finding against the raw run
  files every hour. It can flag issues the research agent must answer, or pause the lab. Reports are in
  `lab/reviews/` and on the dashboard.
- Probe results count only if they beat the best of 3 random-init controls by max(0.1, the controls' spread).
- `make check-billing` fails if `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` is set. It runs before every cycle.

## Turning on the hourly loop

Push this repo to GitHub, then tell Claude in chat the repo name. It sets up an hourly scheduled task that clones
the repo, runs `make check-billing`, follows `backend/agent/CYCLE.md` for one cycle, and pushes one commit.
In the cloud it uses CPU and simulations only (`DE_DEVICE=cpu`, and consider lowering `throughput` in
`lab/settings.json` so estimates match). On your own machine it can use MPS/CUDA and real-data pulls.
