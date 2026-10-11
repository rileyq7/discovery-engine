# Discovery Engine. `make setup` once, then `make dev`.
SHELL := /bin/bash
PY := backend/.venv/bin/python
export PYTHONPATH := $(CURDIR)/backend

.PHONY: setup dev api worker ui loop cycle check-billing test test-all build e1

setup:
	cd backend && uv venv -q --python 3.12 .venv && uv pip install -q --python .venv/bin/python -r requirements.txt
	cd frontend && npm install --silent
	$(PY) -c "import pysr" >/dev/null  # installs Julia for PySR on first run

# API (:8000) + worker + UI (:5173) + local agent loop. LOOP=off to skip the loop
# (e.g. when the hourly scheduled task runs cycles instead).
dev:
	@bash scripts/dev.sh

api:
	cd backend && .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000

worker:
	cd backend && .venv/bin/python -m engine.worker

ui:
	cd frontend && npx vite

loop: check-billing
	cd backend && .venv/bin/python -m agent.cycle --forever

cycle: check-billing
	cd backend && .venv/bin/python -m agent.cycle

check-billing:
	@if [ -n "$$ANTHROPIC_API_KEY" ]; then echo "check-billing FAILED: ANTHROPIC_API_KEY is set. Discovery Engine runs on the Max subscription only; unset it." >&2; exit 1; fi
	@if [ -n "$$CLAUDE_CODE_OAUTH_TOKEN" ]; then echo "check-billing FAILED: CLAUDE_CODE_OAUTH_TOKEN is set. Unset it so claude -p uses the logged-in Max account." >&2; exit 1; fi
	@if grep -rqsE '^\s*(ANTHROPIC_API_KEY|CLAUDE_CODE_OAUTH_TOKEN)\s*=' .env backend/.env frontend/.env 2>/dev/null; then echo "check-billing FAILED: an API key is set in a .env file" >&2; exit 1; fi
	@if grep -rqsE '^\s*(import anthropic|from anthropic)' backend --include=*.py --exclude-dir=.venv; then echo "check-billing FAILED: the anthropic SDK is imported" >&2; exit 1; fi
	@echo "check-billing ok: Max subscription only (no API key, no anthropic SDK)"

test:
	cd backend && .venv/bin/python -m pytest -q -m "not slow"

test-all:
	cd backend && .venv/bin/python -m pytest -q

build:
	cd frontend && npm run build

e1:
	cd backend && .venv/bin/python -m engine.cli run ../configs/e1.json
