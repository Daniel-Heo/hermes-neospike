"""Per-prefetch injection log: one JSON line per turn, no query text.

<HERMES_HOME>/neospike/logs/recall-YYYYMM.jsonl. Files older than the configured
number of months are deleted when a line is written.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

LOG_VERSION = 1
LOG_DIRECTORY = "logs"
QUERY_HASH_CHARS = 16
_FILE_PATTERN = re.compile(r"^recall-(\d{4})(\d{2})\.jsonl$")
_FILE_GLOB = "recall-*.jsonl"

LANE_MEMORY = "memory"
LANE_NOTE = "note"
LANE_CONVERSATION = "conversation"
# Gold/stale ids in question sets are written "<prefix>:<id>".
LANE_PREFIX = {LANE_MEMORY: "mem", LANE_NOTE: "note", LANE_CONVERSATION: "conv"}

# Reasons the current code can observe.
REASON_FUSION_LIMIT = "fusion_limit"  # a lane returned it, but RRF ranked it below the prefetch limit
REASON_SUPERSEDED = "superseded"  # an invalid memory scored; the active end of its chain took the slot
# Its recall line is already in the live context (state.db rows or this prefetch's profile block), or it
# was saved in this conversation after the last compaction; the slot is not refilled.
REASON_IN_CONTEXT = "in_context"
REASON_SOURCE_CAP = "source_cap"  # the recall block already holds the per-source maximum of lines
REASON_CHAR_LIMIT = "char_limit"  # the line does not fit the recall block's character budget
# An overflow item may carry its own reason under this key (default REASON_FUSION_LIMIT).
DROP_REASON_KEY = "drop_reason"
# Reasons the current code cannot observe yet; kept in every line so readers see the gap.
# below_threshold: filtered inside store/notes search before results return.
# duplicate: not recorded yet.
UNTRACKED_REASONS = ("below_threshold", "duplicate")


def query_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:QUERY_HASH_CHARS]


def lane_of(item: dict[str, Any]) -> str:
    kind = item.get("kind")
    if kind == "note":
        return LANE_NOTE
    if kind == "conversation":
        return LANE_CONVERSATION
    return LANE_MEMORY


def item_ref(item: dict[str, Any]) -> str:
    """Question-set id of a search result, e.g. 'mem:12', 'note:3', 'img:5'."""
    return f"{LANE_PREFIX[lane_of(item)]}:{item.get('id')}"


def _score(item: dict[str, Any]) -> float | None:
    value = item.get("score")
    return round(float(value), 6) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _entry_item(item: dict[str, Any], rank: int) -> dict[str, Any]:
    return {"lane": lane_of(item), "id": item.get("id"), "score": _score(item), "rank": rank}


def build_entry(
    *,
    now: float,
    session_id: str,
    query: str,
    injected: Iterable[dict[str, Any]],
    overflow: Iterable[dict[str, Any]],
    unavailable_lanes: Iterable[str],
    lane_errors: dict[str, str] | None = None,
    profile: dict[str, int] | None = None,
    situation: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """injected: the items whose lines the recall block carries. profile (when a <neospike_profile> block went
    out with this prefetch): {"lines": shown, "over_limit": left out by the size cap}. situation (when a
    <neospike_situation> block went out): [{"pref": id, "score": cosine}] of its lines (to measure the cutoff)."""
    injected = list(injected)
    dropped = []
    for rank, item in enumerate(overflow, len(injected) + 1):
        entry = _entry_item(item, rank)
        entry["reason"] = item.get(DROP_REASON_KEY) or REASON_FUSION_LIMIT
        dropped.append(entry)
    extra = {"profile": dict(profile)} if profile is not None else {}
    if situation:
        extra["situation"] = [dict(entry) for entry in situation]
    return {
        "v": LOG_VERSION,
        "ts": datetime.fromtimestamp(now).astimezone().isoformat(timespec="seconds"),
        "session": session_id,
        "query_hash": query_hash(query),
        "injected": [_entry_item(item, rank) for rank, item in enumerate(injected, 1)],
        "dropped": dropped,
        "unavailable_lanes": list(unavailable_lanes),
        # why each unavailable lane failed ("ExceptionType: message", first FAILURE_CAUSE_CHARS chars).
        "lane_errors": dict(lane_errors or {}),
        "untracked_reasons": list(UNTRACKED_REASONS),
        **extra,
    }


NEWLINE = chr(10)
FAILURE_CAUSE_CHARS = 300
FAILURE_FILE_PREFIX = "embedding-failures"  # logs/embedding-failures-YYYYMM.jsonl 


def failure_cause(error: BaseException) -> str:
    """'ExceptionType: message', clipped to FAILURE_CAUSE_CHARS (recall lane errors, embedding failures)."""
    return f"{type(error).__name__}: {error}"[:FAILURE_CAUSE_CHARS]


def append_failure(directory: Path, operation: str, cause: str, *, now: float | None = None) -> Path:
    """One line {"v", "ts", "operation", "cause"} in directory/embedding-failures-YYYYMM.jsonl.
    Callers own error handling (OSError)."""
    moment = time.time() if now is None else now
    local = time.localtime(moment)
    path = Path(directory) / f"{FAILURE_FILE_PREFIX}-{local.tm_year:04d}{local.tm_mon:02d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"v": LOG_VERSION, "ts": datetime.fromtimestamp(moment).astimezone().isoformat(timespec="seconds"),
             "operation": operation, "cause": cause[:FAILURE_CAUSE_CHARS]}
    with path.open("a", encoding="utf-8", newline=NEWLINE) as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + NEWLINE)
    return path


def _month_index(year: int, month: int) -> int:
    return year * 12 + (month - 1)


class RecallLog:
    """Append-only monthly JSONL writer. Callers own error handling (OSError)."""

    def __init__(self, directory: Path, *, months: int, clock: Callable[[], float] = time.time) -> None:
        if isinstance(months, bool) or not isinstance(months, int) or months < 1:
            raise ValueError("months must be a positive integer")
        self.directory = Path(directory)
        self.months = months
        self.clock = clock
        self._pruned_month: int | None = None

    def path_for(self, now: float) -> Path:
        local = time.localtime(now)
        return self.directory / f"recall-{local.tm_year:04d}{local.tm_mon:02d}.jsonl"

    def record(self, **fields: Any) -> Path:
        now = self.clock()
        entry = build_entry(now=now, **fields)
        self.directory.mkdir(parents=True, exist_ok=True)
        local = time.localtime(now)
        current = _month_index(local.tm_year, local.tm_mon)
        if self._pruned_month != current:
            self.prune(current)
            self._pruned_month = current
        path = self.path_for(now)
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
        return path

    def prune(self, current_month: int) -> list[Path]:
        removed = []
        for path in self.directory.glob(_FILE_GLOB):
            match = _FILE_PATTERN.match(path.name)
            if not match:
                continue
            age = current_month - _month_index(int(match.group(1)), int(match.group(2)))
            if age > self.months:
                path.unlink(missing_ok=True)
                removed.append(path)
        return removed
