"""Conversation transcript family.

- Capture: sync_turn copies user/assistant messages into conversation_messages in one short write
  transaction. Tool output, system/control messages and empty messages are not conversation.
- Masking: never refused. Secrets become [REDACTED], injection lines [BLOCKED LINE]; the count is
  kept in masked_count. Only masked text is ever written (patterns of policy.py and this module).
- Indexing (off the turn path): ConversationIndexer chunks dirty conversations at message
  boundaries (chunking.pack_conversation_chunks), then fills NULL embeddings outside any write
  transaction. Leftovers are picked up at the next start or by the CLI `conv-index`.
- Search: hybrid score like note chunks; adjacent hits of one conversation merge into one result.
- Split messages: the parts of one message over the chunk budget form a forward chain
  (conversation_chunks.next_chunk_id, part k -> k+1, last NULL), re-linked in the transaction that
  rewrites the chunks. conv-read and the search merge follow it.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
import sqlite3
import threading
import time
from contextlib import closing
from operator import itemgetter
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import policy
from .chunking import (MESSAGE_JOINER, CONTEXT_TOKEN_BUDGET, ConversationChunk, pack_conversation_chunks,
                       format_message)
from .notes import CHUNK_OVERLAP_TOKENS, NOTE_IMPORTANCE, TokenCounter, default_token_counter, note_chunk_max_tokens
from .store import (CHUNK_CHAIN_FIELD, SOURCE_LAST_ID_FIELD, SYNCED_UPTO_FIELD, MemoryStore, dot_similarities, freshness,
                    blend_score, chain_message_chunks, normalize_rows)

KIND = "conversation"
SOURCE_PREFIX = "conv"
CAPTURE_ROLES = ("user", "assistant")
REDACTED = "[REDACTED]"
BLOCKED_LINE = "[BLOCKED LINE]"
TITLE_MAX_CHARS = 80
EMBED_BATCH = 8  # chunks per saved batch of the CLI conv-index (a failure loses at most this many)
LIVE_EMBED_BATCH = 1  # the gateway's background indexer saves every vector at once
# Daily time budgets (seconds, per home): 3 homes x (240 + 540) = 39 min, inside Hermes's 3600 s script limit.
IMPORT_BUDGET_SECONDS = 240.0
INDEX_BUDGET_SECONDS = 540.0
STALE_RESTART = -(2 ** 62)  # _rechunk restart below every (negated) stale chunk seq
PROGRESS_EVERY = 10  # progress(done, total) every this many conversations (and at the end)
WORKER_JOIN_SECONDS = 5.0
CONVERSATION_LIST_LIMIT = 200
_SQL_BATCH = 500
_PRIVATE_KEY_MARKER = "PRIVATE KEY"
# Extra secret patterns and host control markers.
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:ghp|github_pat|xox[baprs]|AKIA)[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"-----BEGIN [^-]+ PRIVATE KEY-----.*?-----END [^-]+ PRIVATE KEY-----", re.DOTALL),
)
_CONTROL_MARKERS = ("[CONTEXT COMPACTION", "[ASYNC DELEGATION", "[IMPORTANT: Background process")


class ConversationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

    def as_dict(self) -> dict[str, str]:
        return {"error": self.code, "message": str(self)}


# --- masking ---------------------------------------------------------------------------------

def _mask_patterns() -> tuple:
    """policy.py + this module's secret patterns, ordered so no pattern strands part of a secret:
    whole private-key blocks first (policy matches only the BEGIN line), then policy's header
    patterns (a folded `Authorization:` / `Bearer x` keeps its token otherwise), then this module's."""
    blocks = tuple(pattern for pattern in _SECRET_PATTERNS if _PRIVATE_KEY_MARKER in pattern.pattern)
    rest = tuple(pattern for pattern in _SECRET_PATTERNS if _PRIVATE_KEY_MARKER not in pattern.pattern)
    return blocks + tuple(policy._CREDENTIAL_RES) + rest


_SECRET_MASKS = _mask_patterns()
_INJECTION_MASKS = tuple(policy._PROMPT_ATTACK_RES)


def _is_injection_line(line: str) -> bool:
    cleaned = " ".join(line.split())
    return any(pattern.search(cleaned) for pattern in _INJECTION_MASKS)


def mask_text(text: str) -> tuple[str, int]:
    """(masked text, number of masks). Injection lines are replaced whole, then secrets."""
    count = 0
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if _is_injection_line(line):
            lines[index] = BLOCKED_LINE
            count += 1
    masked = "\n".join(lines)
    for pattern in _SECRET_MASKS:
        masked, hits = pattern.subn(REDACTED, masked)
        count += hits
    return masked, count


def is_capturable(role: Any, content: Any) -> bool:
    """User/assistant text only: no tool output, system or control-marker messages, no empty text."""
    if role not in CAPTURE_ROLES or not isinstance(content, str) or not content.strip():
        return False
    return not any(marker in content for marker in _CONTROL_MARKERS)


def capture_rows(messages: Iterable[tuple[Any, Any, float]]) -> list[tuple[str, str, int, float]]:
    """(role, masked content, mask count, timestamp) of the capturable (role, content, timestamp)
    messages: the single capture rule shared by sync_turn and conv-import."""
    rows: list[tuple[str, str, int, float]] = []
    for role, content, timestamp in messages:
        if is_capturable(role, content):
            masked, count = mask_text(content.strip())
            rows.append((role, masked, count, timestamp))
    return rows


def _title(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            line = " ".join(line.split())
            return line if len(line) <= TITLE_MAX_CHARS else line[:TITLE_MAX_CHARS - 1] + "…"
    return ""


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- writes ----------------------------------------------------------------------------------

class ConversationStore:
    """Writes of the conversation family on a MemoryStore connection (short transactions only)."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def append(self, session_id: str, turns: Iterable[tuple[str, str]]) -> int:
        """Masked INSERT of the capturable (role, content) turns; returns the number stored."""
        if not isinstance(session_id, str) or not session_id:
            return 0
        rows = [(role, masked, count) for role, masked, count, _ in capture_rows((role, content, 0.0) for role, content in turns)]
        if not rows:
            return 0
        first_user = next((content for role, content, _ in rows if role == "user"), "")
        title = _title(first_user)
        connection = self.store._conn
        with self.store._lock:
            try:
                connection.execute("BEGIN IMMEDIATE")
                now = time.time()
                row = connection.execute("SELECT id FROM conversations WHERE session_id=?", (session_id,)).fetchone()
                if row is None:
                    conversation_id = int(connection.execute(
                        "INSERT INTO conversations(session_id,started_at,last_message_at,title) VALUES (?,?,?,?)",
                        (session_id, now, now, title),
                    ).lastrowid)
                else:
                    conversation_id = int(row[0])
                    # A resumed session is open again; its title comes from the first user message.
                    connection.execute(
                        "UPDATE conversations SET last_message_at=?,ended_at=NULL,"
                        "title=CASE WHEN title='' THEN ? ELSE title END WHERE id=?",
                        (now, title, conversation_id),
                    )
                seq = int(connection.execute(
                    "SELECT COALESCE(MAX(seq),0) FROM conversation_messages WHERE conversation_id=?", (conversation_id,),
                ).fetchone()[0])
                connection.executemany(
                    "INSERT INTO conversation_messages(conversation_id,seq,role,content,created_at,masked_count) "
                    "VALUES (?,?,?,?,?,?)",
                    [(conversation_id, seq + offset, role, content, now, count)
                     for offset, (role, content, count) in enumerate(rows, 1)],
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return len(rows)

    def sync_session(self, session_id: str, started_at: float, title: str,
                     rows: Sequence[tuple[str, str, int, float]], until: float,
                     source_max_id: int | None = None) -> "SyncOutcome":
        """conv-import of one session (capture_rows output, messages <= until) in one short transaction:
        a new session becomes an ended conversation; an existing one gets its missing rows (plan_merge).
        Either way synced_through becomes until."""
        connection = self.store._conn
        with self.store._lock:
            try:
                connection.execute("BEGIN IMMEDIATE")
                writer = _SyncWriter(connection)
                row = connection.execute("SELECT id FROM conversations WHERE session_id=?", (session_id,)).fetchone()
                if row is not None:
                    outcome = writer.merge(int(row[0]), rows, until)
                elif rows:
                    outcome = writer.create(session_id, started_at, title, rows, until)
                else:
                    outcome = SyncOutcome(False, 0, 0, 0, False)
                if source_max_id is not None:
                    connection.execute(
                        f"UPDATE conversations SET {SOURCE_LAST_ID_FIELD}=MAX(COALESCE({SOURCE_LAST_ID_FIELD},0),?) "
                        "WHERE session_id=?", (source_max_id, session_id))
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return outcome

    def end_session(self, session_id: str) -> bool:
        if not session_id:
            return False
        with self.store._lock, self.store._conn:
            cursor = self.store._conn.execute(
                "UPDATE conversations SET ended_at=? WHERE session_id=? AND ended_at IS NULL", (time.time(), session_id),
            )
        return cursor.rowcount == 1

# --- import / incremental sync of Hermes sessions (CLI conv-import) -----------------
#
# Boundary T (until) = the daily batch start: only source messages with timestamp <= T are matched;
# later ones belong to the next run. conversations.synced_through records the T a conversation was
# last matched at, so a run picks only sessions with a message in (synced_through, T] (new: NULL).

_STATE_MESSAGE_COLUMNS = ("session_id", "role", "content", "timestamp")
# Hermes shows rows with active=1 OR compacted=1 (idx_messages_display_page); active=0 AND
# compacted=0 rows were rewound away. _compressed_summary=1 rows are synthetic compaction summaries.
_STATE_VISIBLE_COLUMNS = ("active", "compacted")
_STATE_SUMMARY_COLUMN = "_compressed_summary"
# Compaction re-stamps the messages it keeps (timestamp = compaction time) and may keep a compacted
# original next to an active copy with the same display_identity. Hermes shows a session in display_order
# (the first id of the identity), one row per identity; the reader does the same when the columns exist.
_STATE_ORDER_COLUMN = "display_order"
_STATE_IDENTITY_COLUMN = "display_identity"


def open_state_db(path: str | Path) -> sqlite3.Connection:
    """Hermes state.db opened read-only (mode=ro URI): nothing is ever written to the source."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ConversationError("not_found", f"state.db not found: {source}")
    return sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


class StateDbReader:
    """Reads (session_id, started_at, title, messages) out of a Hermes state.db, one session at a time."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        columns = _columns(connection, "messages")
        missing = [name for name in _STATE_MESSAGE_COLUMNS if name not in columns]
        if missing:
            raise ConversationError("invalid_source", f"state.db messages lacks columns: {', '.join(missing)}")
        where = ["role IN ('user','assistant')"]
        if all(name in columns for name in _STATE_VISIBLE_COLUMNS):
            where.append("(active=1 OR compacted=1)")
        if _STATE_SUMMARY_COLUMN in columns:
            where.append(f"COALESCE({_STATE_SUMMARY_COLUMN},0)=0")
        self._visible = " AND ".join(where)
        order = f"COALESCE({_STATE_ORDER_COLUMN},id),id" if _STATE_ORDER_COLUMN in columns else "timestamp,id"
        identity = _STATE_IDENTITY_COLUMN if _STATE_IDENTITY_COLUMN in columns else "NULL"
        self._message_sql = (f"SELECT role,content,timestamp,{identity} FROM messages WHERE session_id=? AND timestamp<=? AND "
                             + self._visible + f" ORDER BY {order}")
        session_columns = _columns(connection, "sessions")
        self._session_sql = None
        if {"id", "started_at"} <= session_columns:
            title = "title" if "title" in session_columns else "NULL"
            self._session_sql = f"SELECT started_at,{title} FROM sessions WHERE id=?"

    def session_spans(self, until: float, limit: int | None = None) -> list[tuple[str, float, float | None, int | None]]:
        """(session_id, first message time, last visible user/assistant message time or None, largest visible
        message id or None) of the sessions with a message at or before until, oldest first; limit caps
        the number of sessions."""
        sql = (f"SELECT session_id,MIN(timestamp),MAX(CASE WHEN {self._visible} THEN timestamp END),"
               f"MAX(CASE WHEN {self._visible} THEN id END) FROM messages "
               "WHERE timestamp<=? GROUP BY session_id ORDER BY 2,1")
        params: tuple = (until,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (until, limit)
        return [(str(row[0]), float(row[1]), None if row[2] is None else float(row[2]),
                 None if row[3] is None else int(row[3]))
                for row in self.connection.execute(sql, params) if row[0]]

    def session(self, session_id: str, first_seen: float,
                until: float = math.inf) -> tuple[float, str, list[tuple[Any, Any, float]]]:
        started_at, title = first_seen, ""
        if self._session_sql is not None:
            row = self.connection.execute(self._session_sql, (session_id,)).fetchone()
            if row is not None:
                started_at = first_seen if row[0] is None else float(row[0])
                title = row[1] if isinstance(row[1], str) else ""
        messages: list[tuple[Any, Any, float]] = []
        identities: set = set()
        for row in self.connection.execute(self._message_sql, (session_id, until)):
            if row[3] is not None:
                if row[3] in identities:
                    continue  # the same displayed message kept twice (compacted original + active copy)
                identities.add(row[3])
            messages.append((row[0], row[1], float(row[2])))
        return started_at, title, messages


class MergePlan:
    """Where the source rows go among the stored rows of one conversation.

    order: final message order, (stored message id, None) or (None, source row). A stored row equal to
    a source row (role + masked text) is kept as it is; only source rows with no stored copy are new.
    reorder: the stored rows change order or a new row lands before one (seq renumbered, re-chunked)."""

    __slots__ = ("order", "inserted", "masked", "ours_only", "reorder")

    def __init__(self, order: list, stored_ids: Sequence[int], ours_only: int) -> None:
        self.order = order
        new_rows = [row for local_id, row in order if local_id is None]
        self.inserted = len(new_rows)
        self.masked = sum(row[2] for row in new_rows)
        self.ours_only = ours_only
        kept = [local_id for local_id, _ in order if local_id is not None]
        appended = order[:len(kept)] == [(local_id, None) for local_id in kept]
        self.reorder = kept != list(stored_ids) or not appended


# Hermes folds a message typed during a running turn into that turn's user text (hermes-agent
# agent/turn_iteration_prep.py:375, original + "\n\nUser correction during the turn: " + text), so one live
# row can hold several source user messages; it is matched part by part.
TURN_CORRECTION_SEPARATOR = "\n\nUser correction during the turn: "


def _stored_parts(stored: Sequence[tuple[int, str, str, float]]) -> list[tuple[int, str, str, float]]:
    """stored rows with a folded user row split into its parts (each part keeps the row's id)."""
    parts: list[tuple[int, str, str, float]] = []
    for row in stored:
        if row[1] == "user" and TURN_CORRECTION_SEPARATOR in row[2]:
            parts.extend((row[0], row[1], part.strip(), row[3])
                         for part in row[2].split(TURN_CORRECTION_SEPARATOR) if part.strip())
        else:
            parts.append(row)
    return parts


def _match_parts(parts: Sequence[tuple[int, str, str, float]], source: Sequence[tuple[str, str, int, float]]) -> list:
    """source index -> stored part index (or None). Equal rows in order first (difflib longest matching
    runs), then each source row left takes the first unmatched equal stored part anywhere: rows stored in
    another order (an import that sorted by timestamp, which compaction rewrites) are found
    and moved, never stored twice."""
    stored_keys = [(row[1], row[2]) for row in parts]
    source_keys = [(row[0], row[1]) for row in source]
    match: list = [None] * len(source)
    used = [False] * len(parts)
    for block in difflib.SequenceMatcher(None, stored_keys, source_keys, autojunk=False).get_matching_blocks():
        for offset in range(block.size):
            match[block.b + offset] = block.a + offset
            used[block.a + offset] = True
    free: dict[tuple[str, str], list[int]] = {}
    for index, key in enumerate(stored_keys):
        if not used[index]:
            free.setdefault(key, []).append(index)
    for index, key in enumerate(source_keys):
        if match[index] is None and free.get(key):
            match[index] = free[key].pop(0)
    return match


def plan_merge(stored: Sequence[tuple[int, str, str, float]], source: Sequence[tuple[str, str, int, float]],
               until: float) -> MergePlan:
    """stored: (id, role, content, created_at) in seq order; source: capture_rows output (<= until).

    The final order is the source order (_match_parts maps each source row to its stored copy, a folded
    live user row by its parts). A stored row with no source copy stays right after the stored row it
    followed; among the new rows that come next it goes before the first one whose message time is
    later than its created_at (live rows carry their turn-end time). ours_only counts those rows
    created at or before until (rows after until are the next run's to match)."""
    parts = _stored_parts(stored)
    match = _match_parts(parts, source)
    matched = {parts[index][0] for index in match if index is not None}
    following: dict[int | None, list[tuple[int, str, str, float]]] = {}
    anchor: int | None = None
    for row in stored:
        if row[0] in matched:
            anchor = row[0]
        else:
            following.setdefault(anchor, []).append(row)
    order: list = []
    emitted: set[int] = set()
    pending = list(following.get(None, ()))
    for source_index, row in enumerate(source):
        part = match[source_index]
        if part is None:
            while pending and pending[0][3] <= row[3]:
                order.append((pending.pop(0)[0], None))
            order.append((None, row))
            continue
        local_id = parts[part][0]
        if local_id in emitted:
            continue  # a later part of a folded row already placed
        order.extend((item[0], None) for item in pending)
        order.append((local_id, None))
        emitted.add(local_id)
        pending = list(following.get(local_id, ()))
    order.extend((item[0], None) for item in pending)
    ours_only = sum(1 for row in stored if row[0] not in matched and row[3] <= until)
    return MergePlan(order, [row[0] for row in stored], ours_only)


class SyncOutcome:
    __slots__ = ("created", "inserted", "masked", "ours_only", "reordered")

    def __init__(self, created: bool, inserted: int, masked: int, ours_only: int, reordered: bool) -> None:
        self.created = created
        self.inserted = inserted
        self.masked = masked
        self.ours_only = ours_only
        self.reordered = reordered

    @classmethod
    def new(cls, rows: Sequence[tuple[str, str, int, float]]) -> "SyncOutcome":
        return cls(True, len(rows), sum(row[2] for row in rows), 0, False)

    @classmethod
    def of(cls, plan: MergePlan) -> "SyncOutcome":
        return cls(False, plan.inserted, plan.masked, plan.ours_only, plan.reorder)


_MESSAGE_INSERT_SQL = ("INSERT INTO conversation_messages(conversation_id,seq,role,content,created_at,masked_count) "
                       "VALUES (?,?,?,?,?,?)")
_STORED_ROWS_SQL = "SELECT id,role,content,created_at FROM conversation_messages WHERE conversation_id=? ORDER BY seq"


def _stored_rows(connection: sqlite3.Connection, conversation_id: int) -> list[tuple[int, str, str, float]]:
    return [(int(row[0]), str(row[1]), str(row[2]), float(row[3]))
            for row in connection.execute(_STORED_ROWS_SQL, (conversation_id,))]


class _SyncWriter:
    """ConversationStore.sync_session's writes, inside the caller's transaction."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def create(self, session_id: str, started_at: float, title: str, rows: Sequence[tuple[str, str, int, float]],
               until: float) -> SyncOutcome:
        last = rows[-1][3]
        conversation_id = int(self.connection.execute(
            f"INSERT INTO conversations(session_id,started_at,ended_at,last_message_at,title,{SYNCED_UPTO_FIELD}) "
            "VALUES (?,?,?,?,?,?)", (session_id, min(started_at, rows[0][3]), last, last, title, until),
        ).lastrowid)
        self.connection.executemany(_MESSAGE_INSERT_SQL, [
            (conversation_id, seq, role, content, timestamp, count)
            for seq, (role, content, count, timestamp) in enumerate(rows, 1)])
        return SyncOutcome.new(rows)

    def merge(self, conversation_id: int, rows: Sequence[tuple[str, str, int, float]], until: float) -> SyncOutcome:
        plan = plan_merge(_stored_rows(self.connection, conversation_id), rows, until)
        if plan.reorder:
            self._renumber(conversation_id, plan.order)
        elif plan.inserted:
            self._append(conversation_id, plan.order)
        if plan.inserted:
            last = max(row[3] for _, row in plan.order if row is not None)
            self.connection.execute(
                "UPDATE conversations SET last_message_at=MAX(last_message_at,?),"
                "ended_at=CASE WHEN ended_at IS NULL THEN NULL ELSE MAX(ended_at,?) END WHERE id=?",
                (last, last, conversation_id))
        self.connection.execute(
            f"UPDATE conversations SET {SYNCED_UPTO_FIELD}=MAX(COALESCE({SYNCED_UPTO_FIELD},?),?) WHERE id=?",
            (until, until, conversation_id))
        return SyncOutcome.of(plan)

    def _append(self, conversation_id: int, order: list) -> None:
        base = int(self.connection.execute(
            "SELECT COALESCE(MAX(seq),0) FROM conversation_messages WHERE conversation_id=?", (conversation_id,),
        ).fetchone()[0])
        new_rows = [row for local_id, row in order if local_id is None]
        self.connection.executemany(_MESSAGE_INSERT_SQL, [
            (conversation_id, base + offset, role, content, timestamp, count)
            for offset, (role, content, count, timestamp) in enumerate(new_rows, 1)])

    def _renumber(self, conversation_id: int, order: list) -> None:
        """seq 1..n in plan order (negated first: UNIQUE(conversation_id, seq)); the chunks are marked stale and
        indexed_seq reset, so conv-index chunks this conversation from the start (only new texts are embedded)."""
        connection = self.connection
        connection.execute("UPDATE conversation_messages SET seq=-seq WHERE conversation_id=?", (conversation_id,))
        connection.executemany("UPDATE conversation_messages SET seq=? WHERE id=?", [
            (seq, local_id) for seq, (local_id, _) in enumerate(order, 1) if local_id is not None])
        connection.executemany(_MESSAGE_INSERT_SQL, [
            (conversation_id, seq, row[0], row[1], row[3], row[2])
            for seq, (local_id, row) in enumerate(order, 1) if local_id is None])
        # Chunk seqs are negated (stale): _rechunk then rebuilds the whole conversation, and a chunk whose
        # text is unchanged takes its old vector by text_hash before the stale rows are deleted.
        connection.execute("UPDATE conversation_chunks SET seq_start=-seq_start,seq_end=-seq_end "
                           "WHERE conversation_id=? AND seq_start>0", (conversation_id,))
        connection.execute("UPDATE conversations SET indexed_seq=0 WHERE id=?", (conversation_id,))


class _LocalConversations:
    """conversations.synced_through per session_id, and the merge plan of a dry run: on the live store
    (under its lock) or on a mode=ro connection of the memory DB (no table yet: nothing stored)."""

    def __init__(self, store: MemoryStore | None, read_connection: sqlite3.Connection | None) -> None:
        self.store = store
        self.connection = store._conn if store is not None else read_connection
        self.has_table = self.connection is not None and self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conversations'").fetchone() is not None

    def synced(self) -> dict[str, tuple[float | None, int | None]]:
        if not self.has_table:
            return {}
        if self.store is None:
            return self._synced()
        with self.store._lock:
            return self._synced()

    def _synced(self) -> dict[str, tuple[float | None, int | None]]:
        present = _columns(self.connection, "conversations")
        through = SYNCED_UPTO_FIELD if SYNCED_UPTO_FIELD in present else "NULL"
        max_id = SOURCE_LAST_ID_FIELD if SOURCE_LAST_ID_FIELD in present else "NULL"
        return {str(row[0]): (None if row[1] is None else float(row[1]), None if row[2] is None else int(row[2]))
                for row in self.connection.execute(f"SELECT session_id,{through},{max_id} FROM conversations")}

    def dry_merge(self, session_id: str, rows: Sequence[tuple[str, str, int, float]], until: float) -> SyncOutcome:
        found = None if not self.has_table else self.connection.execute(
            "SELECT id FROM conversations WHERE session_id=?", (session_id,)).fetchone()
        if found is None:
            return SyncOutcome.new(rows)
        return SyncOutcome.of(plan_merge(_stored_rows(self.connection, int(found[0])), rows, until))


class Deadline:
    """Time budget of one stage; seconds <= 0 or None = no budget."""

    def __init__(self, seconds: float | None) -> None:
        self.at = None if seconds is None or seconds <= 0 else time.monotonic() + seconds

    def passed(self) -> bool:
        return self.at is not None and time.monotonic() >= self.at


def _report_progress(progress: Any, done: int, total: int, result: dict) -> None:
    if progress is not None and (done == total or done % PROGRESS_EVERY == 0):
        progress(done, total, result)


CONVERSATION_LOG_DIRECTORY = "logs"
_PROGRESS_LOG_NAME = "conv-%Y%m.jsonl"


class ProgressLog:
    """progress callback of conv-import / conv-index: one JSON line per report and one for the stage
    result in <neospike dir>/logs/conv-YYYYMM.jsonl ("done"/"total" = conversations, N/M)."""

    def __init__(self, data_dir: str | Path, stage: str) -> None:
        self.directory = Path(data_dir) / CONVERSATION_LOG_DIRECTORY
        self.stage = stage

    def __call__(self, done: int, total: int, result: dict) -> None:
        line = {"stage": self.stage, "progress": f"{done}/{total}", "done": done, "total": total}
        if "until" in result:
            line["until"] = result["until"]
        self._write(line)

    def finish(self, result: dict) -> None:
        self._write({"stage": self.stage, "result": result})

    def _write(self, line: dict) -> None:
        now = time.time()
        line = {"time": round(now, 3), **line}
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / time.strftime(_PROGRESS_LOG_NAME, time.localtime(now))
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        except OSError:
            pass  # the log is a record only; the stage result is still printed


def import_state_db(state_db: str | Path, *, store: MemoryStore | None = None,
                    read_connection: sqlite3.Connection | None = None, limit: int | None = None,
                    until: float | None = None, budget_seconds: float | None = None, progress: Any = None) -> dict:
    """Match Hermes sessions (messages <= until, default now) into the conversation tables.

    A session is processed when it is new or has a message after its conversation's synced_through:
    a new one becomes an ended conversation, an existing one gets only its missing rows (plan_merge;
    same capture and masking rules as sync_turn, live rows are never doubled, rows only we hold are
    kept and counted in ours_only); then synced_through = until. One transaction per conversation:
    an error or the time budget stops the run and the next run picks up what is left (remaining).
    store=None is a dry run against read_connection (mode=ro; MemoryStore is not opened, since
    opening it rewrites the file). progress(done, total, result) runs every PROGRESS_EVERY
    conversations and at the end. Chunks and embeddings are left to ConversationIndexer (conv-index)."""
    if limit is not None and limit < 1:
        raise ConversationError("invalid", "--limit must be >= 1")
    until = time.time() if until is None else float(until)
    if not math.isfinite(until):
        raise ConversationError("invalid", "--until must be a finite epoch time")
    dry_run = store is None
    local = _LocalConversations(store, read_connection)
    deadline = Deadline(budget_seconds)
    result = {"until": until, "sessions_seen": 0, "candidates": 0, "processed": 0, "remaining": 0, "imported": 0,
              "updated": 0, "unchanged": 0, "skipped_existing": 0, "skipped_empty": 0, "messages": 0, "masked": 0,
              "reordered": 0, "ours_only": 0, "stopped": None, "dry_run": dry_run}
    with closing(open_state_db(state_db)) as source:
        handled = 0
        try:
            reader = StateDbReader(source)
            synced = local.synced()
            candidates = []
            for session_id, first_seen, last, max_id in reader.session_spans(until, limit):
                result["sessions_seen"] += 1
                if last is None:
                    result["skipped_empty"] += 1
                elif _up_to_date(synced.get(session_id), last, max_id):
                    result["skipped_existing"] += 1
                else:
                    candidates.append((session_id, first_seen, max_id))
            result["candidates"] = result["remaining"] = len(candidates)
            for session_id, first_seen, max_id in candidates:
                if deadline.passed():
                    result["stopped"] = "budget"
                    break
                _sync_one(reader, local, store, session_id, first_seen, until, max_id, session_id in synced, result)
                handled += 1
                result["remaining"] = len(candidates) - handled
                _report_progress(progress, handled, len(candidates), result)
        except sqlite3.Error as exc:
            # Conversations committed before the error stay matched; the next run continues with the rest.
            result["stopped"] = "error"
            result["error"] = "sqlite_error"
            result["message"] = f"conv-import stopped: {type(exc).__name__}: {exc}"
    return result


def _up_to_date(synced: tuple[float | None, int | None] | None, last: float, max_id: int | None) -> bool:
    """Nothing to match: no message after synced_through and no message id above source_max_id (a late
    message written with a timestamp before the previous boundary still has a new, larger id)."""
    if synced is None or synced[0] is None or last > synced[0]:
        return False
    return max_id is None or (synced[1] is not None and max_id <= synced[1])


def _sync_one(reader: StateDbReader, local: _LocalConversations, store: MemoryStore | None, session_id: str,
              first_seen: float, until: float, max_id: int | None, known: bool, result: dict) -> None:
    started_at, source_title, messages = reader.session(session_id, first_seen, until)
    rows = capture_rows(messages)
    if not rows and not known:
        result["skipped_empty"] += 1  # nothing to store, so nothing records the check: seen again next run
        return
    if store is None:
        outcome = local.dry_merge(session_id, rows, until)
    else:
        title_text = source_title.strip() or next((content for role, content, _, _ in rows if role == "user"), "")
        outcome = ConversationStore(store).sync_session(session_id, started_at, _title(mask_text(title_text)[0]),
                                                        rows, until, max_id)
    result["processed"] += 1
    if outcome.created:
        result["imported"] += 1
    elif outcome.inserted:
        result["updated"] += 1
    else:
        result["unchanged"] += 1
    result["messages"] += outcome.inserted
    result["masked"] += outcome.masked
    result["reordered"] += int(outcome.reordered)
    result["ours_only"] += outcome.ours_only


# --- indexing (background) -------------------------------------------------------------------

class ConversationIndexer:
    """Chunks dirty conversations and embeds NULL chunks; never embeds inside a write transaction.

    index_pending() is synchronous (CLI conv-index). start()/notify() run it on a daemon thread so
    sync_turn returns without waiting for the embedder.
    """

    def __init__(self, store: MemoryStore, embedder: Any, *, token_counter: TokenCounter | None = None,
                 max_tokens: int | None = None, context_tokens: int = CONTEXT_TOKEN_BUDGET,
                 embed_batch: int = LIVE_EMBED_BATCH) -> None:
        self.store = store
        self.embed_batch = max(1, int(embed_batch))
        self.context_tokens = context_tokens
        self.embedder = embedder
        self.token_counter = token_counter or default_token_counter()
        self.max_tokens = max_tokens or note_chunk_max_tokens()
        self.last_error: str | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._thread: threading.Thread | None = None

    # -- synchronous pass ---------------------------------------------------------------------

    def index_pending(self, *, stop: threading.Event | None = None, budget_seconds: float | None = None,
                      progress: Any = None) -> dict:
        """Chunk the dirty conversations, then embed NULL chunks in saved batches of EMBED_BATCH.

        budget_seconds (None/<=0: none) stops between conversations or chunks; what is left stays dirty
        or NULL for the next run (remaining). progress(done, total, result) as in import_state_db."""
        started = time.perf_counter()
        deadline = Deadline(budget_seconds)
        dirty = self._dirty_conversations()
        result = {"conversations": 0, "chunks_written": 0, "embedded": 0, "pending": 0, "processed": 0,
                  "remaining": 0, "remaining_conversations": len(dirty), "stopped": None}
        for conversation_id in dirty:
            if stop is not None and stop.is_set():
                result["stopped"] = "stop"
                break
            if deadline.passed():
                result["stopped"] = "budget"
                break
            changed = self._rechunk(conversation_id)
            if changed is not None:
                result["conversations"] += 1
                result["chunks_written"] += changed
            result["processed"] += 1
            result["remaining_conversations"] = len(dirty) - result["processed"]
            _report_progress(progress, result["processed"], len(dirty), result)
        try:
            if result["stopped"] is None:
                result["embedded"] = self._embed_pending(stop, deadline, result)
        finally:
            result["pending"] = self.pending_embeddings()
            result["remaining"] = result["remaining_conversations"] + result["pending"]
            result["seconds"] = round(time.perf_counter() - started, 4)
        return result

    def pending_embeddings(self) -> int:
        with self.store._lock:
            return int(self.store._conn.execute(
                "SELECT COUNT(*) FROM conversation_chunks WHERE embedding IS NULL").fetchone()[0])

    def _dirty_conversations(self) -> list[int]:
        with self.store._lock:
            return [int(row[0]) for row in self.store._conn.execute(
                "SELECT c.id FROM conversations c JOIN conversation_messages m ON m.conversation_id=c.id "
                "GROUP BY c.id HAVING MAX(m.seq)>c.indexed_seq ORDER BY c.id"
            )]

    def _rechunk(self, conversation_id: int) -> int | None:
        """Re-chunk from the last chunk's first message (chunking is prefix-stable); rows written,
        or None when a concurrent indexer changed the tail first."""
        connection = self.store._conn
        with self.store._lock:
            restart = int(connection.execute(
                "SELECT COALESCE(MAX(seq_start),0) FROM conversation_chunks WHERE conversation_id=?", (conversation_id,),
            ).fetchone()[0])
            if restart < 0 or connection.execute(
                    "SELECT 1 FROM conversation_chunks WHERE conversation_id=? AND seq_start<0 LIMIT 1",
                    (conversation_id,)).fetchone() is not None:
                restart = STALE_RESTART  # renumbered by conv-import, rebuilt from the first message
            existing = self._tail(conversation_id, restart)
            messages = [(int(row[0]), str(row[1]), str(row[2])) for row in connection.execute(
                "SELECT seq,role,content FROM conversation_messages WHERE conversation_id=? AND seq>=? ORDER BY seq",
                (conversation_id, restart),
            )]
            context_row = connection.execute(
                "SELECT content FROM conversation_messages WHERE conversation_id=? AND seq<? AND role='user' "
                "ORDER BY seq DESC LIMIT 1", (conversation_id, restart),
            ).fetchone()
        if not messages:
            return 0
        pieces = pack_conversation_chunks(
            messages, self.token_counter.count, max_tokens=self.max_tokens, context_tokens=self.context_tokens,
            overlap_tokens=CHUNK_OVERLAP_TOKENS,
            cut_tokens=self.token_counter.cut, context=None if context_row is None else str(context_row[0]),
        )
        wanted = {(piece.seq_start, piece.part): piece for piece in pieces}
        hashes = {key: _text_hash(piece.text) for key, piece in wanted.items()}
        with self.store._lock:
            try:
                connection.execute("BEGIN IMMEDIATE")
                if self._tail(conversation_id, restart) != existing:
                    connection.rollback()
                    return None
                written = self._write_tail(conversation_id, existing, wanted, hashes)
                chain_message_chunks(connection, conversation_id, restart)
                connection.execute(
                    "UPDATE conversations SET indexed_seq=? WHERE id=? AND indexed_seq<?",
                    (messages[-1][0], conversation_id, messages[-1][0]),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return written

    def _tail(self, conversation_id: int, restart: int) -> dict[tuple[int, int], tuple[int, int, str]]:
        """(seq_start, part) -> (id, seq_end, text_hash) of the chunks from restart on."""
        return {(int(row[1]), int(row[2])): (int(row[0]), int(row[3]), str(row[4])) for row in self.store._conn.execute(
            "SELECT id,seq_start,part,seq_end,text_hash FROM conversation_chunks WHERE conversation_id=? AND seq_start>=?",
            (conversation_id, restart),
        )}

    def _write_tail(self, conversation_id: int, existing: dict, wanted: dict[tuple[int, int], ConversationChunk],
                    hashes: dict[tuple[int, int], str]) -> int:
        connection = self.store._conn
        written = 0
        for key, piece in wanted.items():
            digest = hashes[key]
            current = existing.get(key)
            if current is not None and current[1] == piece.seq_end and current[2] == digest:
                continue
            reused = connection.execute(
                "SELECT embedding FROM conversation_chunks WHERE text_hash=? AND embedding IS NOT NULL LIMIT 1", (digest,),
            ).fetchone()
            embedding = None if reused is None else reused[0]
            if current is None:
                connection.execute(
                    "INSERT INTO conversation_chunks(conversation_id,seq_start,seq_end,part,text,prefix_chars,text_hash,embedding) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (conversation_id, piece.seq_start, piece.seq_end, piece.part, piece.text, piece.prefix_chars, digest, embedding),
                )
            else:
                # In place (no delete): a growing last chunk keeps its id and leaves no tombstone.
                connection.execute(
                    "UPDATE conversation_chunks SET seq_end=?,text=?,prefix_chars=?,text_hash=?,embedding=? WHERE id=?",
                    (piece.seq_end, piece.text, piece.prefix_chars, digest, embedding, current[0]),
                )
            written += 1
        # Deleted last: a stale chunk's vector stays reusable by text_hash while the new ones are written.
        for key, (chunk_id, _, _) in existing.items():
            if key not in wanted:
                connection.execute("DELETE FROM conversation_chunks WHERE id=?", (chunk_id,))
                written += 1
        return written

    def _embed_pending(self, stop: threading.Event | None, deadline: "Deadline | None" = None,
                       result: dict | None = None) -> int:
        """Embeds NULL chunks embed_batch at a time, each batch saved in its own short transaction; an
        embedder error (e.g. TimeoutError) saves the vectors done so far before it propagates."""
        connection = self.store._conn
        embedded = 0
        while not self._halted(stop, deadline, result):
            with self.store._lock:
                rows = connection.execute(
                    "SELECT id,text,text_hash FROM conversation_chunks WHERE embedding IS NULL ORDER BY id LIMIT ?",
                    (self.embed_batch,),
                ).fetchall()
            if not rows:
                break
            updates: list[tuple[bytes, int, str]] = []
            try:
                for row in rows:
                    if self._halted(stop, deadline, result):
                        break
                    updates.append((self._pack(self.embedder.embed_document(str(row[1]))), int(row[0]), str(row[2])))
            finally:
                embedded += self._save_embeddings(updates)
            if len(updates) < len(rows):
                break
        return embedded

    @staticmethod
    def _halted(stop: threading.Event | None, deadline: "Deadline | None", result: dict | None) -> bool:
        if stop is not None and stop.is_set():
            if result is not None:
                result["stopped"] = "stop"
            return True
        if deadline is not None and deadline.passed():
            if result is not None:
                result["stopped"] = "budget"
            return True
        return False

    def _save_embeddings(self, updates: Sequence[tuple[bytes, int, str]]) -> int:
        if not updates:
            return 0
        saved = 0
        with self.store._lock, self.store._conn as connection:
            for update in updates:
                saved += connection.execute(
                    "UPDATE conversation_chunks SET embedding=? WHERE id=? AND text_hash=? AND embedding IS NULL", update,
                ).rowcount
        return saved

    def _pack(self, embedding: Any) -> bytes:
        vector = normalize_rows(embedding)
        if len(vector) != self.store.dimensions or not all(math.isfinite(value) for value in vector) or not any(vector):
            raise RuntimeError("Invalid conversation chunk embedding")
        return vector.tobytes()

    # -- background thread --------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="neospike-conversation-indexer", daemon=True)
        self._thread.start()

    def notify(self) -> None:
        self._idle.clear()
        self._wake.set()

    def wait_idle(self, timeout: float) -> bool:
        return self._idle.wait(timeout)

    def stop(self, timeout: float = WORKER_JOIN_SECONDS) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            if self._stop.is_set():
                break
            self._wake.clear()
            try:
                self.index_pending(stop=self._stop)
                self.last_error = None
            except Exception as exc:  # Indexing is retried on the next turn, start, or conv-index.
                self.last_error = f"{type(exc).__name__}: {exc}"
            if not self._wake.is_set():
                self._idle.set()
        self._idle.set()


# --- search (read-only on any connection) ----------------------------------------------------

def has_searchable(connection: sqlite3.Connection) -> bool:
    found = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conversation_chunks'").fetchone()
    return found is not None and connection.execute(
        "SELECT 1 FROM conversation_chunks WHERE embedding IS NOT NULL LIMIT 1").fetchone() is not None


class ConversationHits(list):
    """search_conversation_chunks results; `omitted` counts the excerpts left out as still in the live context
    (merged like the results). A plain list to every caller that does not look."""

    def __init__(self, items: Iterable[dict] = (), omitted: int = 0) -> None:
        super().__init__(items)
        self.omitted = omitted


def live_messages(rows: Iterable[tuple[Any, Any]]) -> frozenset[tuple[str, str]]:
    """(role, masked content) of the live conversation's active state.db rows (role, content), by the
    capture rule (capture_rows), so they compare equal to the stored conversation_messages."""
    return frozenset((role, masked) for role, masked, _, _ in capture_rows((role, content, 0.0) for role, content in rows))


def _live_chunk_ids(connection: sqlite3.Connection, rows: Sequence, live: frozenset[tuple[str, str]]) -> set[int]:
    """Ids of the rows (chunk id, conversation id, seq_start, seq_end) every message of which is in
    live (the context Hermes resends). A chunk with a message not there (summarized away) or missing stays."""
    spans: dict[int, list[int]] = {}
    for row in rows:
        span = spans.setdefault(int(row[1]), [int(row[2]), int(row[3])])
        span[0], span[1] = min(span[0], int(row[2])), max(span[1], int(row[3]))
    present: dict[tuple[int, int], bool] = {}
    for conversation_id, (low, high) in spans.items():
        for seq, role, content in connection.execute(
                "SELECT seq,role,content FROM conversation_messages WHERE conversation_id=? AND seq BETWEEN ? AND ?",
                (conversation_id, low, high)):
            present[(conversation_id, int(seq))] = (role, content) in live
    return {int(row[0]) for row in rows if int(row[2]) > 0 and all(
        present.get((int(row[1]), seq), False) for seq in range(int(row[2]), int(row[3]) + 1))}


def search_conversation_chunks(connection: sqlite3.Connection, query: str, query_embedding: Any, *, dimensions: int,
                               semantic_threshold: float, limit: int = 5, live_session: str = "",
                               live: frozenset[tuple[str, str]] | None = None) -> ConversationHits:
    """Cosine-gated chunk search (note-chunk score); adjacent hits of one conversation become one result.
    live_session with live (live_messages of its active state.db rows): its chunks whose every message is
    in live are left out before merging and the limit; `omitted` on the result counts them as merged excerpts.
    live None: nothing is left out (state.db was unreadable, or the rows were not read)."""
    qvec = list(query_embedding)
    if len(qvec) != dimensions:
        raise ValueError(f"Expected {dimensions} dimensions, got {len(qvec)}")
    now = time.time()
    # Score every vector first (id + embedding only, one numpy product), then read text and
    # conversation fields of the gated chunks only.
    vectors = connection.execute("SELECT id,embedding FROM conversation_chunks WHERE embedding IS NOT NULL").fetchall()
    gated = {int(row[0]): semantic for row, semantic in zip(vectors, dot_similarities(qvec, [row[1] for row in vectors]))
             if math.isfinite(semantic) and semantic >= semantic_threshold}
    if not gated:
        return ConversationHits()
    rows = connection.execute(
        "SELECT ch.id,ch.conversation_id,ch.seq_start,ch.seq_end,ch.part,ch.text,ch.prefix_chars,"
        "c.last_message_at,c.session_id,c.title FROM conversation_chunks ch "
        "JOIN conversations c ON c.id=ch.conversation_id WHERE ch.id IN (SELECT value FROM json_each(?)) "
        "ORDER BY ch.id", (json.dumps(sorted(gated)),)
    ).fetchall()
    in_context: set[int] = set()
    if live_session and live is not None:
        in_context = _live_chunk_ids(connection, [row for row in rows if row[8] == live_session], live)
    hits, held = [], []
    for row in rows:
        semantic = gated[int(row[0])]
        score = blend_score(semantic, NOTE_IMPORTANCE, freshness(now, float(row[7])))
        hit = {"chunk_id": int(row[0]), "conversation_id": int(row[1]), "seq_start": int(row[2]),
               "seq_end": int(row[3]), "part": int(row[4]), "text": str(row[5]), "prefix_chars": int(row[6]),
               "score": score, "session_id": row[8], "title": row[9]}
        (held if int(row[0]) in in_context else hits).append(hit)
    merged = merge_adjacent(hits, split_chains(connection, hits))
    merged.sort(key=itemgetter("score"), reverse=True)
    omitted = len(merge_adjacent(held, split_chains(connection, held))) if held else 0
    return ConversationHits(merged[:limit], omitted)


def message_chain(connection: sqlite3.Connection, conversation_id: int, seq: int) -> list:
    """Chunks of one message in chain order, rows (id, part, text, prefix_chars, next_chunk_id).

    One row for an unsplit single-message chunk, [] when no chunk covers exactly this message. The
    head is the part no other part's next points to. ConversationError when the chain is broken
    (not one head, a cycle, or parts left unreached)."""
    rows = connection.execute(
        f"SELECT id,part,text,prefix_chars,{CHUNK_CHAIN_FIELD} FROM conversation_chunks "
        "WHERE conversation_id=? AND seq_start=? AND seq_end=?", (conversation_id, seq, seq),
    ).fetchall()
    if len(rows) <= 1:
        return list(rows)
    by_id = {int(row[0]): row for row in rows}
    targets = {int(row[4]) for row in rows if row[4] is not None}
    heads = [chunk_id for chunk_id in by_id if chunk_id not in targets]
    if len(heads) != 1:
        raise ConversationError("broken_chain", f"Conversation {conversation_id} seq {seq}: {len(heads)} chain heads")
    ordered: list = []
    cursor: int | None = heads[0]
    while cursor is not None and cursor in by_id and len(ordered) < len(rows):
        ordered.append(by_id[cursor])
        following = by_id[cursor][4]
        cursor = None if following is None else int(following)
    if len(ordered) != len(rows) or cursor is not None:
        raise ConversationError("broken_chain", f"Conversation {conversation_id} seq {seq}: chain does not cover its parts")
    return ordered


def part_overlaps(block: str, bodies: Sequence[str]) -> list[int] | None:
    """Characters each body repeats from the previous one, found by aligning the bodies (chain order)
    on the message block: body 0 starts the block, the last body ends it, and every body starts
    inside the previous one. Exact for repetitive text too; None when the chunker's whitespace
    normalisation (stripped paragraph/line/sentence atoms) leaves no alignment."""
    if not bodies or not block.startswith(bodies[0]):
        return None
    starts = _align(block, bodies, 1, [0])
    if starts is None:
        return None
    return [0] + [starts[index - 1] + len(bodies[index - 1]) - starts[index] for index in range(1, len(bodies))]


def _align(block: str, bodies: Sequence[str], index: int, starts: list[int]) -> list[int] | None:
    """Depth-first alignment; the latest start (smallest overlap) is tried first."""
    if index == len(bodies):
        return starts if starts[-1] + len(bodies[-1]) == len(block) else None
    previous_start = starts[-1]
    previous_end = previous_start + len(bodies[index - 1])
    body = bodies[index]
    if index == len(bodies) - 1:  # the last body ends the block: one candidate start
        start = len(block) - len(body)
        fits = previous_start < start <= previous_end and block.endswith(body)
        return [*starts, start] if fits else None
    candidates: list[int] = []
    position = block.find(body, previous_start + 1, previous_end + len(body))
    while position != -1:
        candidates.append(position)
        position = block.find(body, position + 1, previous_end + len(body))
    for start in reversed(candidates):
        found = _align(block, bodies, index + 1, [*starts, start])
        if found is not None:
            return found
    return None


def join_overlap(left: str, right: str) -> str:
    """Fallback join when no alignment exists: drop the longest prefix of right that ends left at a
    whitespace boundary (may over-trim repetitive text; read_message reports matches_message)."""
    for size in range(min(len(left), len(right)), 0, -1):
        if (size == len(right) or right[size].isspace()) and left.endswith(right[:size]):
            return left + right[size:]
    return left + MESSAGE_JOINER + right


def _join_bodies(bodies: Sequence[str], overlaps: Sequence[int] | None) -> str:
    if overlaps is not None:
        return "".join(body[overlap:] for body, overlap in zip(bodies, overlaps))
    text = ""
    for index, body in enumerate(bodies):
        text = body if index == 0 else join_overlap(text, body)
    return text


class SplitChains:
    """Split messages among search hits: (conversation_id, seq) -> chunk ids in chain order, and the
    overlap characters of every aligned part (chunk id -> chars repeated from the previous part)."""

    def __init__(self) -> None:
        self.chains: dict[tuple[int, int], list[int]] = {}
        self.overlaps: dict[int, int] = {}

    def chain(self, hit: dict) -> list[int] | None:
        if hit["seq_start"] != hit["seq_end"]:
            return None
        return self.chains.get((hit["conversation_id"], hit["seq_start"]))


def _message_row(connection: sqlite3.Connection, conversation_id: int, seq: int) -> tuple[str, str] | None:
    row = connection.execute(
        "SELECT role,content FROM conversation_messages WHERE conversation_id=? AND seq=?", (conversation_id, seq),
    ).fetchone()
    return None if row is None else (str(row[0]), str(row[1]))


def split_chains(connection: sqlite3.Connection, hits: Sequence[dict]) -> SplitChains:
    found = SplitChains()
    seen: set[tuple[int, int]] = set()
    for hit in hits:
        key = (hit["conversation_id"], hit["seq_start"])
        if hit["seq_start"] != hit["seq_end"] or key in seen:
            continue
        seen.add(key)
        chain = message_chain(connection, *key)
        if len(chain) <= 1:
            continue
        found.chains[key] = [int(row[0]) for row in chain]
        message = _message_row(connection, *key)
        bodies = [str(row[2])[int(row[3]):] for row in chain]
        overlaps = None if message is None else part_overlaps(format_message(*message), bodies)
        if overlaps is not None:
            found.overlaps.update(zip(found.chains[key], overlaps))
    return found


class _HitOrder:
    """Sort key of hits: conversation, first message, then position in the message's chunk chain."""

    def __init__(self, chains: SplitChains) -> None:
        self.chains = chains

    def __call__(self, hit: dict) -> tuple[int, int, int]:
        chain = self.chains.chain(hit)
        position = chain.index(hit["chunk_id"]) if chain and hit["chunk_id"] in chain else hit["part"]
        return (hit["conversation_id"], hit["seq_start"], position)


def _chain_next(chain: list[int] | None, previous: dict | None, hit: dict) -> bool:
    """hit is the part right after previous in one split message's chain."""
    return (chain is not None and previous is not None and previous["seq_start"] == hit["seq_start"]
            and previous["chunk_id"] in chain and hit["chunk_id"] in chain
            and chain.index(hit["chunk_id"]) == chain.index(previous["chunk_id"]) + 1)


def merge_adjacent(hits: Sequence[dict], chains: SplitChains | None = None) -> list[dict]:
    """One result per run of hits of one conversation whose seq ranges touch or overlap.

    Hits on parts of one split message always merge (same seq), in chain order; consecutive parts
    join without the overlap the chunker repeats. Such a result carries message_parts, part (first
    hit part), parts and read, the conv-read command that returns the whole message."""
    chains = chains or SplitChains()
    ordered = sorted(hits, key=_HitOrder(chains))
    results: list[dict] = []
    current: dict | None = None
    previous: dict | None = None
    for hit in ordered:
        chain = chains.chain(hit)
        if current is not None and hit["conversation_id"] == current["id"] and hit["seq_start"] <= current["seq_end"] + 1:
            body = hit["text"][hit["prefix_chars"]:]
            if not _chain_next(chain, previous, hit):
                current["text"] += MESSAGE_JOINER + body
            elif hit["chunk_id"] in chains.overlaps:
                current["text"] += body[chains.overlaps[hit["chunk_id"]]:]
            else:
                current["text"] = join_overlap(current["text"], body)
            current["seq_end"] = max(current["seq_end"], hit["seq_end"])
            current["score"] = max(current["score"], hit["score"])
            current["chunk_ids"].append(hit["chunk_id"])
        else:
            if current is not None:
                results.append(_finish(current))
            current = {"id": hit["conversation_id"], "kind": KIND, "session_id": hit["session_id"], "title": hit["title"],
                       "chunk_ids": [hit["chunk_id"]], "seq_start": hit["seq_start"], "seq_end": hit["seq_end"],
                       "text": hit["text"], "score": hit["score"]}
        _note_part(current, hit, chain)
        previous = hit
    if current is not None:
        results.append(_finish(current))
    return results


def _note_part(result: dict, hit: dict, chain: list[int] | None) -> None:
    """Split-message fields of a result (for the first split message it touches)."""
    if not chain:
        return
    if "message_parts" not in result:
        result.update({"message_parts": len(chain), "part": hit["part"], "parts": [], "message_seq": hit["seq_start"],
                       "read": f"conv-read --conversation {hit['conversation_id']} --seq {hit['seq_start']}"})
    if result["message_seq"] == hit["seq_start"]:
        result["parts"].append(hit["part"])


def _finish(result: dict) -> dict:
    result["origin"] = f"{SOURCE_PREFIX}:{result['id']} #{result['seq_start']}-{result['seq_end']}"
    return result


def read_message(connection: sqlite3.Connection, conversation_id: int, seq: int) -> dict:
    """CLI conv-read: one message, its chunks joined in chain order without the repeated overlap.

    matches_message is False when the parts do not align on the stored (masked) message, e.g. after
    the chunker's whitespace normalisation; the text is then the fallback join."""
    message = _message_row(connection, conversation_id, seq)
    if message is None:
        raise ConversationError("not_found", f"Conversation {conversation_id} has no message seq {seq}")
    role, content = message
    chain = message_chain(connection, conversation_id, seq)
    chunk_ids = [int(item[0]) for item in chain]
    if len(chain) <= 1:
        return {"conversation_id": conversation_id, "seq": seq, "role": role, "message_parts": 1,
                "chunk_ids": chunk_ids, "text": content, "matches_message": True}
    bodies = [str(item[2])[int(item[3]):] for item in chain]
    text = _join_bodies(bodies, part_overlaps(format_message(role, content), bodies))
    label = format_message(role, "")
    if text.startswith(label):
        text = text[len(label):]
    return {"conversation_id": conversation_id, "seq": seq, "role": role, "message_parts": len(chain),
            "chunk_ids": chunk_ids, "text": text, "matches_message": text == content}


def search(store: MemoryStore, query: str, query_embedding: Any, *, limit: int = 5, live_session: str = "",
           live: frozenset[tuple[str, str]] | None = None) -> ConversationHits:
    with store._lock:
        return search_conversation_chunks(store._conn, query, query_embedding, dimensions=store.dimensions,
                                          semantic_threshold=store.semantic_threshold, limit=limit,
                                          live_session=live_session, live=live)


# --- read-only listings (dashboard mode=ro) --------------------------------------------

_SUMMARY_COLUMNS = (
    "c.id,c.session_id,c.started_at,c.ended_at,c.last_message_at,c.title,c.summary_note_id,"
    "(SELECT COUNT(*) FROM conversation_messages m WHERE m.conversation_id=c.id) messages,"
    "(SELECT COALESCE(SUM(masked_count),0) FROM conversation_messages m WHERE m.conversation_id=c.id) masked"
)


def _summary(row: sqlite3.Row | tuple) -> dict:
    keys = ("id", "session_id", "started_at", "ended_at", "last_message_at", "title", "summary_note_id", "messages", "masked")
    return dict(zip(keys, tuple(row)))


def list_conversations(connection: sqlite3.Connection, *, limit: int = CONVERSATION_LIST_LIMIT) -> list[dict]:
    return [_summary(row) for row in connection.execute(
        f"SELECT {_SUMMARY_COLUMNS} FROM conversations c ORDER BY c.last_message_at DESC, c.id DESC LIMIT ?", (limit,),
    )]


def messages_of(connection: sqlite3.Connection, conversation_id: int) -> list[dict]:
    return [{"seq": int(row[0]), "role": row[1], "content": row[2], "created_at": row[3], "masked_count": int(row[4])}
            for row in connection.execute(
                "SELECT seq,role,content,created_at,masked_count FROM conversation_messages "
                "WHERE conversation_id=? ORDER BY seq", (conversation_id,),
            )]


def conversation_record(connection: sqlite3.Connection, conversation_id: int) -> dict:
    row = connection.execute(f"SELECT {_SUMMARY_COLUMNS} FROM conversations c WHERE c.id=?", (conversation_id,)).fetchone()
    if row is None:
        raise ConversationError("not_found", f"Conversation {conversation_id} not found")
    record = _summary(row)
    record["transcript"] = messages_of(connection, conversation_id)
    return record
