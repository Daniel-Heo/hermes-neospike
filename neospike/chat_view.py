"""Read-only conversation view of the Hermes chat database (state.db) for the dashboard.

Sessions, transcripts and search hits come straight from state.db, opened only through
chat_history.open_readonly, and are shown as stored there (no masking). The messages shown follow the selection of
is_capturable (user and assistant text, no tool output, system or control-marker messages, no empty
text) and Hermes' display rule (visible rows, one row per display identity,
display order). That rule is re-read here because the read-only authorizer of chat_history does not allow the
PRAGMA table_info(...) column checks.
"""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any

from . import chat_history
from .chat_history import ChatHistoryError

CONVERSATION_LIST_LIMIT = 200
TITLE_MAX_CHARS = 80
CAPTURE_ROLES = ("user", "assistant")
# Host control markers: a message holding one is not conversation.
_CONTROL_MARKERS = ("[CONTEXT COMPACTION", "[ASYNC DELEGATION", "[IMPORTANT: Background process")
SEARCH_LIMIT = 20
# Hits asked of chat_history.search before the selection and display rules drop some of them.
SEARCH_FETCH = chat_history.MAX_SEARCH_RESULTS
_SQL_BATCH = 500
# Hermes' display columns: rows with active=1 OR compacted=1 are shown,
# _compressed_summary=1 rows are synthetic compaction summaries, display_order / display_identity give the order and
# one row per displayed message.
_REQUIRED_COLUMNS = ("session_id", "role", "content", "timestamp")
_VISIBLE_COLUMNS = ("active", "compacted")
_SUMMARY_COLUMN = "_compressed_summary"
_ORDER_COLUMN = "display_order"
_IDENTITY_COLUMN = "display_identity"


def is_capturable(role: Any, content: Any) -> bool:
    """User/assistant text only: no tool output, system or control-marker messages, no empty text."""
    if role not in CAPTURE_ROLES or not isinstance(content, str) or not content.strip():
        return False
    return not any(marker in content for marker in _CONTROL_MARKERS)


def _title(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            line = " ".join(line.split())
            return line if len(line) <= TITLE_MAX_CHARS else line[:TITLE_MAX_CHARS - 1] + "…"
    return ""


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    """Column names of table ({} when it does not exist), read without PRAGMA table_info."""
    if not chat_history._has_table(connection, table):
        return set()
    return {column[0] for column in connection.execute(f"SELECT * FROM {table} LIMIT 0").description}


class _DisplayReader:
    """The user/assistant rows Hermes shows of a session, in display order (Hermes' display rule)."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        columns = _columns(connection, "messages")
        missing = [name for name in _REQUIRED_COLUMNS if name not in columns]
        if missing:
            raise ChatHistoryError("unavailable", f"Hermes chat database messages lacks columns: {', '.join(missing)}")
        where = ["role IN ('user','assistant')"]
        if all(name in columns for name in _VISIBLE_COLUMNS):
            where.append("(active=1 OR compacted=1)")
        if _SUMMARY_COLUMN in columns:
            where.append(f"COALESCE({_SUMMARY_COLUMN},0)=0")
        self.visible = " AND ".join(where)
        order = f"COALESCE({_ORDER_COLUMN},id),id" if _ORDER_COLUMN in columns else "timestamp,id"
        identity = _IDENTITY_COLUMN if _IDENTITY_COLUMN in columns else "NULL"
        self._message_sql = (f"SELECT role,content,timestamp,{identity} FROM messages WHERE session_id=? AND "
                             f"{self.visible} ORDER BY {order}")
        session_columns = _columns(connection, "sessions")
        self._session_sql = None
        if {"id", "started_at"} <= session_columns:
            title = "title" if "title" in session_columns else "NULL"
            self._session_sql = f"SELECT started_at,{title} FROM sessions WHERE id=?"
        elif {"id", "title"} <= session_columns:
            self._session_sql = "SELECT NULL,title FROM sessions WHERE id=?"

    def session_spans(self) -> list[tuple[str, float, float | None]]:
        """(session_id, first message time, last shown user/assistant message time or None) of every session."""
        sql = (f"SELECT session_id,MIN(timestamp),MAX(CASE WHEN {self.visible} THEN timestamp END) FROM messages "
               "GROUP BY session_id")
        return [(str(row[0]), float(row[1]), None if row[2] is None else float(row[2]))
                for row in self.connection.execute(sql) if row[0]]

    def session_head(self, session_id: str, first_seen: float) -> tuple[float, str]:
        """(sessions.started_at or first_seen, sessions.title or "") of one session."""
        started_at, title = first_seen, ""
        if self._session_sql is not None:
            row = self.connection.execute(self._session_sql, (session_id,)).fetchone()
            if row is not None:
                started_at = first_seen if row[0] is None else float(row[0])
                title = row[1] if isinstance(row[1], str) else ""
        return started_at, title

    def shown_rows(self, session_id: str) -> list[tuple[str, str, float]]:
        """(role, stripped content, timestamp) of the shown messages that pass is_capturable, in display order."""
        rows: list[tuple[str, str, float]] = []
        identities: set = set()
        for role, content, timestamp, identity in self.connection.execute(self._message_sql, (session_id,)):
            if identity is not None:
                if identity in identities:
                    continue  # the same displayed message kept twice (compacted original + active copy)
                identities.add(identity)
            if is_capturable(role, content):
                rows.append((role, content.strip(), float(timestamp)))
        return rows


def _session_title(source_title: str, rows: list[tuple[str, str, float]]) -> str:
    """sessions.title, else the first user message, as one line of at most TITLE_MAX_CHARS."""
    first_user = next((content for role, content, _ in rows if role == "user"), "")
    return _title(source_title.strip() or first_user)


class _Session:
    """One session as the dashboard shows it: its shown messages and its summary."""

    __slots__ = ("session_id", "started_at", "source_title", "rows")

    def __init__(self, reader: _DisplayReader, session_id: str, first_seen: float) -> None:
        self.session_id = session_id
        self.started_at, self.source_title = reader.session_head(session_id, first_seen)
        self.rows = reader.shown_rows(session_id)

    def summary(self) -> dict:
        rows = self.rows
        return {"session_id": self.session_id, "title": _session_title(self.source_title, rows),
                "started_at": min(self.started_at, rows[0][2]) if rows else self.started_at,
                "last_message_at": max(row[2] for row in rows) if rows else None, "messages": len(rows)}

    def transcript(self) -> list[dict]:
        return [{"seq": seq, "role": role, "content": content, "created_at": timestamp}
                for seq, (role, content, timestamp) in enumerate(self.rows, start=1)]


def database_path(home: Path) -> Path:
    """state.db of home; unavailable (not not_found, which means a missing session here) when it does not exist."""
    path = chat_history.chat_database_path(home)
    if not path.is_file():
        raise ChatHistoryError("unavailable", f"Hermes chat database does not exist: {chat_history.STATE_DB_NAME}")
    return path


class _Database:
    """`with _Database(home) as connection`: state.db of home opened read-only, closed on exit."""

    def __init__(self, home: Path) -> None:
        self.path = database_path(home)
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self.connection = chat_history.open_readonly(self.path)
        return self.connection

    def __exit__(self, *exc_info: Any) -> None:
        if self.connection is not None:
            self.connection.close()


def _limit(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ChatHistoryError("invalid", "limit must be a positive integer")
    return value


def _span_last(span: tuple) -> float:
    return span[2]


def _summary_order(summary: dict) -> tuple:
    return summary["last_message_at"], summary["session_id"]


def list_sessions(home: Path, limit: Any = None) -> list[dict]:
    """Summaries (session_id, title, started_at, last_message_at, messages) of the sessions with a shown message,
    latest last_message_at first, at most limit (CONVERSATION_LIST_LIMIT). Sessions are read in order of their last
    visible message time, an upper bound of their last shown message time, and reading stops once no further
    session can enter the list."""
    limit = _limit(limit, CONVERSATION_LIST_LIMIT)
    try:
        with _Database(home) as connection:
            reader = _DisplayReader(connection)
            spans = [span for span in reader.session_spans() if span[2] is not None]
            spans.sort(key=_span_last, reverse=True)
            summaries: list[dict] = []
            floor = -math.inf  # the limit-th latest last_message_at once the list is full
            for session_id, first_seen, last_visible in spans:
                if len(summaries) >= limit and last_visible < floor:
                    break
                session = _Session(reader, session_id, first_seen)
                if not session.rows:
                    continue
                summaries.append(session.summary())
                if len(summaries) >= limit:
                    summaries.sort(key=_summary_order, reverse=True)
                    del summaries[limit:]
                    floor = summaries[-1]["last_message_at"]
            summaries.sort(key=_summary_order, reverse=True)
            return summaries[:limit]
    except sqlite3.DatabaseError as exc:
        raise ChatHistoryError("unavailable", f"Hermes chat database read failed: {exc}") from None


def _first_seen(connection: sqlite3.Connection, session_id: str) -> float | None:
    row = connection.execute("SELECT MIN(timestamp) FROM messages WHERE session_id = ?", (session_id,)).fetchone()
    return None if row is None or row[0] is None else float(row[0])


def session_record(home: Path, session_id: Any) -> dict:
    """Summary of one session plus its transcript (seq from 1, role, content, created_at); not_found when the
    session has no shown message."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ChatHistoryError("invalid", "session_id must be a non-empty string")
    try:
        with _Database(home) as connection:
            reader = _DisplayReader(connection)
            first_seen = _first_seen(connection, session_id)
            session = None if first_seen is None else _Session(reader, session_id, first_seen)
            if session is None or not session.rows:
                raise ChatHistoryError("not_found", f"Conversation {session_id} not found")
            record = session.summary()
            record["transcript"] = session.transcript()
            return record
    except sqlite3.DatabaseError as exc:
        raise ChatHistoryError("unavailable", f"Hermes chat database read failed: {exc}") from None


def _shown_contents(connection: sqlite3.Connection, reader: _DisplayReader, message_ids: list[int]) -> dict:
    """message id -> content of the ids that pass the display rule (visible user/assistant rows)."""
    contents: dict[int, Any] = {}
    for start in range(0, len(message_ids), _SQL_BATCH):
        batch = message_ids[start:start + _SQL_BATCH]
        placeholders = ", ".join("?" for _ in batch)
        sql = f"SELECT id, content FROM messages WHERE id IN ({placeholders}) AND {reader.visible}"
        for row in connection.execute(sql, batch):
            contents[int(row[0])] = row[1]
    return contents


def _search_title(reader: _DisplayReader, connection: sqlite3.Connection, session_id: str) -> str:
    first_seen = _first_seen(connection, session_id)
    if first_seen is None:
        return ""
    _, source_title = reader.session_head(session_id, first_seen)
    rows = [] if source_title.strip() else reader.shown_rows(session_id)  # the messages only when no title is set
    return _session_title(source_title, rows)


def search_sessions(home: Path, query: Any, limit: Any = None) -> list[dict]:
    """Messages matching query (chat_history.search, newest first) that the transcript shows, one item per message
    (a session matched twice gives two items): session_id, title, message_id, role, created_at and the
    chat_history.search excerpt (text). At most limit (SEARCH_LIMIT) items."""
    limit = _limit(limit, SEARCH_LIMIT)
    found = chat_history.search(database_path(home), query, limit=SEARCH_FETCH)["results"]
    if not found:
        return []
    try:
        with _Database(home) as connection:
            reader = _DisplayReader(connection)
            contents = _shown_contents(connection, reader, [int(hit["message_id"]) for hit in found])
            titles: dict[str, str] = {}
            results: list[dict] = []
            for hit in found:
                if not is_capturable(hit["role"], contents.get(int(hit["message_id"]))):
                    continue
                session_id = hit["session_id"]
                if session_id not in titles:
                    titles[session_id] = _search_title(reader, connection, session_id)
                results.append({"session_id": session_id, "title": titles[session_id],
                                "message_id": int(hit["message_id"]), "role": hit["role"],
                                "created_at": hit["timestamp"], "text": hit["excerpt"]})
                if len(results) >= limit:
                    break
            return results
    except sqlite3.DatabaseError as exc:
        raise ChatHistoryError("unavailable", f"Hermes chat database read failed: {exc}") from None
