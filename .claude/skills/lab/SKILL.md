---
name: lab
description: Start, stop or check Discovery Engine (API, worker, UI, research loop, reviewer). Use when the user says "start the lab", "restart discovery", "stop the lab", "is the lab running", or after a reboot or crash.
argument-hint: "[start | stop | status | restart] [--no-loop]"
---

# Run Discovery Engine

Everything goes through `scripts/lab-ctl`, which runs the services in the background with logs in
`runs/*.log`. It is idempotent: `start` only starts what isn't already running.

Argument given: `$ARGUMENTS` (default: `start`).

## Steps

1. **Status first.** Run `scripts/lab-ctl status` to see what's up, free disk and whether `lab/PAUSED` exists.
2. **Act on the argument.**
   - `start` (default): `scripts/lab-ctl start`. Add `--no-loop` to bring up only the API, worker and
     UI, with no research agent or reviewer (for browsing results without spending compute).
   - `stop`: `scripts/lab-ctl stop`. This also kills a training run in progress; it will be marked
     failed on the next start and the agent can rerun that seed. Mention that to the user.
   - `restart`: `scripts/lab-ctl stop` then `scripts/lab-ctl start`. Use it after code changes so
     the loop and reviewer pick them up.
   - `status`: just report.
3. **If start refuses:**
   - `check-billing` failed: an `ANTHROPIC_API_KEY` or `CLAUDE_CODE_OAUTH_TOKEN` is set. Tell the user to
     unset it. Never work around it: the lab must only use `claude -p` on the Max subscription.
   - Not enough disk: report `df -h` and the size of `runs/` (`du -sh runs`). Checkpoints under
     `runs/run-*/checkpoints` are scratch, because results live in `lab/`. Still, only delete them if the
     user says yes.
4. **After a crash or reboot**, check the tail of `runs/worker.log` and `runs/loop.log` for errors, and
   list runs that were marked `failed` by orphan recovery
   (`cd backend && .venv/bin/python -c "from app import store; print([r['id'] for r in store.list_runs('failed')][-5:])"`).
5. **If `lab/PAUSED` exists**, say so and quote the note. The loop won't run cycles until the user
   resumes from the Dashboard. Don't remove the file yourself.

## Report

Keep it short: which services are running, the UI link http://localhost:5173, free disk, whether the
lab is paused, and when the first cycle starts (about 2 min after start; the reviewer about 30 min).
This laptop is a 16 GB M3. Keep `max_concurrent_runs` at 1 unless the user asks, because heavy
parallel load crashed it before.
