"""Read-only helpers behind `scripts/lab run` and `scripts/lab transcripts` (used by the chat tutor)."""
from __future__ import annotations

import json
import sys

from app import store
from engine.paths import lab_dir


def run(run_id: str) -> None:
    from agent.review import raw_run

    store.rebuild_index()
    print(json.dumps(raw_run(run_id), indent=1, default=str))


def transcripts(n: int = 5) -> None:
    recs = []
    for f in sorted((lab_dir() / "transcripts").glob("*.jsonl"))[-3:]:
        recs += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    if not recs:
        print("no transcripts yet (they are recorded from now on)")
    for r in recs[-n:]:
        print(f"=== {r['at']} {r['role']} ({r['seconds']}s): {r['task']}")
        for t in r.get("thinking") or []:
            print("[thinking]", t[:4000])
        print(r["reply"][:6000])
        print()


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["?"]
    if cmd == "run":
        run(args[0])
    elif cmd == "transcripts":
        transcripts(int(args[0]) if args else 5)
    else:
        sys.exit("usage: labcli run <run_id> | transcripts [N]")
