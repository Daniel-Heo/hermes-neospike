"""Daily Neospike maintenance (`hermes cron create --no-agent`).

Only the pref-reconcile stage calls an LLM (Hermes's auxiliary model; off with "pref_reconcile_enabled": false
in <home>/neospike/config.json); every other stage is plain code.

Stages run in order and a failed stage does not stop the rest. The stages run for every home of
common.target_homes (root first, then the Neospike profiles); a failed home does not stop the others.
Success prints nothing (Hermes then delivers nothing). Any failure prints one line per home ("<home>: ok", or
"<home>: " and its failed stages joined by " | ", a backup failure always first) and exits 1, so Hermes sends
it to the job's failure delivery.
The batch start time T is taken once and passed to every home's conv-import (--until T): messages
after T are the next day's. conv-import / conv-index keep their own time budgets (the CLI defaults) and leave
what is left for the next day; their progress and results go to <home>/neospike/logs/conv-YYYYMM.jsonl.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neospike_cron_common as common  # noqa: E402  (copied next to this script)

STAGE_TIMEOUT_SECONDS = 1800  # conv-index embeds new chunks; Hermes's whole-script timeout is 3600 s
EXIT_FAILED = 1
BACKUP_STAGE = "backup"
HOME_OK = "ok"
STAGE_SEPARATOR = " | "


def stages(home: Path, until: float | None = None) -> list[tuple[str, list[str]]]:
    """until: the batch start time T for conv-import (None: the CLI takes its own now)."""
    boundary = [] if until is None else ["--until", repr(until)]
    return [
        (BACKUP_STAGE, ["backup"]),  # incremental by default
        # Connection rows whose memory/note is gone (code only; the counts are the stage JSON).
        ("link-prune", ["link-prune"]),
        ("conv-import", ["conv-import", "--state-db", str(home / "state.db"), *boundary]),
        ("conv-index", ["conv-index"]),
        # Memories saved while the embedder failed get their vector (and save-time dedupe) before the
        # first stage that reads memory vectors. It ends with the save follow-up a turn runs itself:
        # note-absorb (cos >= 0.8 to an automatic note chunk) then topic-assign, for what no turn handled.
        ("embed-pending", ["embed-pending"]),
        # Similar preference/constraint pairs (cos >= 0.8, not the save-time duplicates): an LLM decides
        # keep / update / delete per pair, a few calls a day; rewritten rows are embedded the next day.
        ("pref-reconcile", ["pref-reconcile"]),
        # A linked group of more than 5 memories without a topic becomes a topic (applied at once).
        ("topic-discover", ["topic-discover"]),
        # A topic with more than 10 live members becomes an automatic note (memories, topic deleted).
        ("topic-to-note", ["topic-to-note"]),
    ]


def run_stage(home: Path, name: str, args: list[str]) -> str | None:
    """None on success, else the failure line."""
    try:
        done = common.run_cli(home, args, STAGE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return f"FAILED {name}: timeout after {STAGE_TIMEOUT_SECONDS} s"
    except OSError as exc:
        return f"FAILED {name}: {exc}"
    if done.returncode == 0:
        return None
    return f"FAILED {name} (exit {done.returncode}): {common.last_line(done.stderr, done.stdout)}"


def run_home(home: Path, until: float | None = None) -> list[str]:
    """Failure lines of one home's stages (empty = all passed), backup first, the rest in stage order."""
    failures = []
    for name, args in stages(home, until):
        line = run_stage(home, name, args)
        if line is not None:
            failures.append((name != BACKUP_STAGE, line))
    return [line for _, line in sorted(failures, key=_backup_first)]


def main() -> int:
    until = time.time()  # boundary T of this batch, the same for every home
    results = [(home, run_home(home, until)) for home in common.target_homes(common.home_dir())]
    if not any(failures for _, failures in results):
        return 0
    for home, failures in results:
        print(f"{home}: {STAGE_SEPARATOR.join(failures) or HOME_OK}")
    return EXIT_FAILED


def _backup_first(failure: tuple[bool, str]) -> bool:
    return failure[0]  # stable sort: backup (False) first, the rest in stage order


if __name__ == "__main__":
    raise SystemExit(main())
