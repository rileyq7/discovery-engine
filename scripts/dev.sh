#!/usr/bin/env bash
# Start API, worker, UI and (unless LOOP=off) the local agent loop; Ctrl-C stops all of them.
set -euo pipefail
cd "$(dirname "$0")/.."
make -s check-billing
pids=()
cleanup() { trap - INT TERM EXIT; kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true; }
trap cleanup INT TERM EXIT
prefix() { sed -u "s/^/[$1] /"; }
(cd backend && exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 2>&1 | prefix api) & pids+=($!)
sleep 2
(cd backend && exec .venv/bin/python -m engine.worker --no-rebuild 2>&1 | prefix worker) & pids+=($!)
(cd frontend && exec npx vite --port 5173 --strictPort 2>&1 | prefix ui) & pids+=($!)
if [ "${LOOP:-local}" != "off" ]; then
  (cd backend && exec .venv/bin/python -m agent.cycle --forever 2>&1 | prefix loop) & pids+=($!)
  (cd backend && exec .venv/bin/python -m agent.review --forever --first-delay 30 2>&1 | prefix review) & pids+=($!)
fi
echo "Discovery Engine: UI http://localhost:5173  ·  API http://127.0.0.1:8000  ·  loop ${LOOP:-local}"
wait
