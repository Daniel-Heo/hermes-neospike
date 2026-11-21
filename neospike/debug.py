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
_DELETED_RECALL = "Context hidden after record deletion."
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MARKUP_RE = re.compile(r"([\\`*_{}\[\]#|!])")

_TITLE = "**Neospike: diagnostics**"
_SEPARATOR = "---"
# A literal "> " row marker: a real blockquote would swallow the following lines (lazy continuation).
_QUOTE_MARK = "\\> "
# CommonMark hard line break; visible, so it survives trailing-whitespace stripping.
_LINE_BREAK = "\\\n"
_RECALL_HEAD = "Retrieved context:"
_RECALL_NONE = "Retrieved context: none."
_SEARCH_HEAD = "Explicit search — passed to agent:"
_SEARCH_NONE = "No matches."
_CHANGES_HEAD = "Memory changes:"
_OBSERVED_HEAD = "Observed results:"
_NO_SAVE_UNCHANGED = "No saves: the operation did not change memory."
_NO_SAVE_NO_WRITE = "No saves: Neospike received no successful write operation."
_FAILURES_HEAD = "Rejections/failures:"
_TRUNCATED = "Diagnostics truncated."
# Lines a diagnostics block (current or older blank-line format) can consist of, for spotting copies.
_BLOCK_LINES = frozenset({
    "", _SEPARATOR, _TITLE, _RECALL_HEAD, _RECALL_NONE, _HIDDEN_RECALL, _DELETED_RECALL, _SEARCH_HEAD,
    _SEARCH_NONE, _CHANGES_HEAD, _OBSERVED_HEAD, _FAILURES_HEAD, _TRUNCATED,
})
_BLOCK_LINE_PREFIXES = ("• ", "- ", "* ", "> ", _QUOTE_MARK)
_NO_SAVE_PREFIX = "No saves:"
# Section headings whose following lines (model-written copies may drop the bullets) belong to the block.
_SECTION_HEADS = (_RECALL_HEAD, "Explicit search", _CHANGES_HEAD, _OBSERVED_HEAD, _FAILURES_HEAD)


def _inert(value: object, cap: int = 240) -> str:
    """Bounded text with control characters and Markdown/HTML syntax neutralised."""
    text = str(value).replace("\x00", "").replace("\r", " ").replace("\n", " ")
    text = html.escape(_CONTROL_RE.sub(" ", text)[:cap], quote=False)
    return _MARKUP_RE.sub(r"\\\1", text)


def _clip(text: str, cap: int) -> str:
    """Cut to cap without leaving half of a backslash escape at the end."""
    clipped = text[:cap]
    trailing = len(clipped) - len(clipped.rstrip("\\"))
    return clipped[:-1] if trailing % 2 else clipped


def _block_line(line: str) -> str:
    text = line.strip()
    return text[:-1].rstrip() if text.endswith("\\") else text


def _is_block_tail(lines: list[str], start: int) -> bool:
    """Whether lines[start:] read as diagnostics blocks only: fixed lines, bullet rows, section text."""
    in_section = False
    for line in lines[start:]:
        text = _block_line(line)
        if text.endswith(":") and text.startswith(_SECTION_HEADS):
            in_section = True
        elif text in _BLOCK_LINES or text.startswith(_NO_SAVE_PREFIX):
            in_section = False
        elif not in_section and not text.startswith(_BLOCK_LINE_PREFIXES):
            return False
    return True


def strip_block_copies(response: str) -> str:
    """Remove diagnostics blocks the model copied onto the end of its reply.

    The cut starts at the first title line from which everything to the end reads as diagnostics blocks,
    together with the separator above it; text above that title (lists included) and a title quoted
    inside the reply are left alone.
    """
    lines = response.splitlines(keepends=True)
    for cut, line in enumerate(lines):
        if _block_line(line) == _TITLE and _is_block_tail(lines, cut):
            break
    else:
        return response
    above = cut
    while above > 0 and not _block_line(lines[above - 1]):
        above -= 1
    if above > 0 and _block_line(lines[above - 1]) == _SEPARATOR:
        cut = above - 1
    return "".join(lines[:cut])


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
            quoted = "\n".join(_QUOTE_MARK + _inert(row, 600) for row in rendered.splitlines() if row.strip())
            trace.recall = _clip(quoted, _RECALL_CAP)
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
            body = '\n'.join(rows) if rows else _SEARCH_NONE
            trace.search = _clip(body, _SEARCH_CAP)
            trace.clipped |= len(body) > _SEARCH_CAP or len(results) > _SEARCH_SHOWN

    def record_deletion(self, key: str, memory_id: int | str, *, kind: str = "record") -> None:
        with self._lock:
            trace = self._trace(key)
            if trace:
                # A deleted record must not survive as a fresh copy in the chat.
                trace.recall = _DELETED_RECALL
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
        parts = [_TITLE]
        parts += [_RECALL_HEAD, *trace.recall.split("\n")] if trace.recall else [_RECALL_NONE]
        if trace.search:
            parts += [_SEARCH_HEAD, *trace.search.split("\n")]
        if trace.changes:
            parts += [_CHANGES_HEAD, *trace.changes]
        if trace.unsure_outcomes:
            parts += [_OBSERVED_HEAD, *trace.unsure_outcomes]
        if not trace.changes and not trace.unsure_outcomes:
            parts.append(_NO_SAVE_UNCHANGED if trace.write_attempted else _NO_SAVE_NO_WRITE)
        if trace.failures:
            parts += [_FAILURES_HEAD, *trace.failures]
        if trace.clipped:
            parts.append(_TRUNCATED)
        # The blank line keeps "---" a separator: right under a paragraph it would turn it into a heading.
        body = strip_block_copies(response).rstrip()
        return body + "\n\n" + _SEPARATOR + "\n" + _LINE_BREAK.join(parts)
