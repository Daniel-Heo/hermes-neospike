"""Turn-local, opt-in Neospike diagnostics.

No transcript or provider request data is kept: only the already-rendered recall and the
outcomes seen during a single turn.
"""
from __future__ import annotations

import html
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from .policy import AdmitVerdict, screen_text


_RECALL_CAP = 1_800
_OPEN_TURN_CAP = 32
_SEARCH_CAP = 900
_SEARCH_SHOWN = 5
_ENTRY_COUNT_CAP = 12
_ENTRY_CHARS_CAP = 2800
_HIDDEN_RECALL = "Context hidden by safety filter."
_HIDDEN_EXCERPT = "content hidden by safety filter"
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MARKUP_RE = re.compile(r"([\\`*_{}\[\]#|!])")


def _inert(value: object, cap: int = 240) -> str:
    """Bounded text with control characters and Markdown/HTML syntax neutralised."""
    text = str(value).replace("\x00", "").replace("\r", " ").replace("\n", " ")
    text = html.escape(_CONTROL_RE.sub(" ", text)[:cap], quote=False)
    return _MARKUP_RE.sub(r"\\\1", text)


@dataclass
class _TurnTrace:
    recall: str = ""
    changes: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    unsure_outcomes: list[str] = field(default_factory=list)
    write_attempted: bool = False
    clipped: bool = False
    search: str = ""

    def add(self, bucket: list[str], line: str) -> None:
        shown = self.changes + self.failures + self.unsure_outcomes
        if len(shown) >= _ENTRY_COUNT_CAP or sum(map(len, shown)) + len(line) > _ENTRY_CHARS_CAP:
            self.clipped = True
            return
        bucket.append(line)


class TurnTraceLog:
    """Bounded per-session trace store owned by one provider."""

    def __init__(self) -> None:
        self._open_turns: OrderedDict[str, _TurnTrace] = OrderedDict()
        self._lock = threading.RLock()

    def _trace(self, key: str) -> _TurnTrace | None:
        return self._open_turns.get(key)

    def start(self, key: str) -> None:
        if not key:
            return
        with self._lock:
            self._open_turns.pop(key, None)  # an interrupted turn never got finalized
            self._open_turns[key] = _TurnTrace()
            while len(self._open_turns) > _OPEN_TURN_CAP:
                self._open_turns.popitem(last=False)

    def recall(self, key: str, rendered: str) -> None:
        with self._lock:
            trace = self._trace(key)
            if not trace:
                return
            # Old stored records predate the ingest policy: screen the whole payload before splitting/clipping.
            if screen_text(rendered) is AdmitVerdict.BLOCK:
                trace.recall = _HIDDEN_RECALL
                return
            quoted = "\n".join("> " + _inert(row, 600) for row in rendered.splitlines() if row.strip())
            trace.recall = quoted[:_RECALL_CAP]
            trace.clipped |= len(quoted) > _RECALL_CAP

    def change(self, key: str, fact: object, scope: object, *, action: str = "Saved") -> None:
        with self._lock:
            trace = self._trace(key)
            if trace:
                trace.write_attempted = True
                trace.add(trace.changes, f"• {action}: {_inert(fact)} — scope: {_inert(scope, 80)}")

    def search(self, key: str, results: list[dict]) -> None:
        with self._lock:
            trace = self._trace(key)
            if not trace:
                return
            rows = []
            for hit in results[:_SEARCH_SHOWN]:
                excerpt = hit.get('text') or ''
                if screen_text(str(excerpt)) is AdmitVerdict.BLOCK:
                    excerpt = _HIDDEN_EXCERPT
                rows.append(f"• #{_inert(hit.get('id', '?'), 30)}: {_inert(excerpt)}")
            body = '\n'.join(rows) if rows else 'No matches.'
            trace.search = body[:_SEARCH_CAP]
            trace.clipped |= len(body) > _SEARCH_CAP or len(results) > _SEARCH_SHOWN

    def record_deletion(self, key: str, memory_id: int | str, *, kind: str = "record") -> None:
        with self._lock:
            trace = self._trace(key)
            if trace:
                # A deleted record must not survive as a fresh copy in the chat.
                trace.recall = "Context hidden after record deletion."
                trace.search = ""
                trace.changes.clear()
                self.change(key, f"{kind} #{memory_id}", "current accessible scope", action="Deleted")

    def failure(self, key: str, label: str, *, attempted_write: bool = True) -> None:
        with self._lock:
            trace = self._trace(key)
            if trace:
                trace.write_attempted |= attempted_write
                trace.add(trace.failures, f"• {_inert(label, 120)}")

    def record_unsure_outcome(self, key: str, label: str) -> None:
        """An outcome whose effect on storage cannot be inferred."""
        with self._lock:
            trace = self._trace(key)
            if trace:
                trace.add(trace.unsure_outcomes, f"• {_inert(label, 180)}")

    def append(self, key: str, response: str, *, enabled: bool) -> str:
        with self._lock:
            trace = self._open_turns.pop(key, None)
        if not enabled or trace is None:
            return response
        parts = ["**Neospike: diagnostics**"]
        parts += ["\nRetrieved context:", trace.recall] if trace.recall else ["\nRetrieved context: none."]
        if trace.search:
            parts += ["\nExplicit search — passed to agent:", trace.search]
        if trace.changes:
            parts += ["\nMemory changes:", *trace.changes]
        if trace.unsure_outcomes:
            parts += ["\nObserved results:", *trace.unsure_outcomes]
        if not trace.changes and not trace.unsure_outcomes:
            parts.append("\nNo saves: the operation did not change memory." if trace.write_attempted
                         else "\nNo saves: Neospike received no successful write operation.")
        if trace.failures:
            parts += ["\nRejections/failures:", *trace.failures]
        if trace.clipped:
            parts.append("\nDiagnostics truncated.")
        return response.rstrip() + "\n\n---\n\n" + "\n".join(parts)
