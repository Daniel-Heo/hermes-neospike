"""chat_period reads messages through the sessions table so every lookup uses messages(session_id, timestamp):
same rows and conversation summaries as the plain time-range query, and no full scan of messages."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from neospike import chat_history

HOUR = 3_600.0
DAY = 86_400.0
T = datetime(2026, 11, 20, 12, 0).timestamp()
SINCE, UNTIL = T, T + DAY

# Hermes names and columns (persistence/sqlite/_schema_ddl.py), reduced to what chat_period reads.
_SCHEMA = (
    "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, title TEXT, parent_session_id TEXT, "
    "started_at REAL NOT NULL, ended_at REAL, end_reason TEXT, message_count INTEGER DEFAULT 0)",
    "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id), "
    "role TEXT NOT NULL, content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, "
    "timestamp REAL NOT NULL, active INTEGER DEFAULT 1)",
    "CREATE INDEX idx_sessions_started ON sessions(started_at DESC)",
    "CREATE INDEX idx_messages_session ON messages(session_id, timestamp)",
    "CREATE INDEX idx_messages_session_id ON messages(session_id, id)",
)

S_OPEN = "20261120_100000_open0001"      # ended_at NULL
S_CROSS = "20261119_120000_cross001"     # started the day before, runs into the period
S_AFTER = "20261121_130000_after001"     # starts after the period
S_EARLY_END = "20261120_090000_early001"  # ended_at earlier than its last message in the period
S_CHILD = "20261121_150000_child001"     # compression child started after the period, carries parent timestamps
S_BEFORE = "20261115_120000_before01"    # wholly before the period

_SESSIONS = (
    (S_OPEN, "cli", "open", None, T - 2 * HOUR, None, None),
    (S_CROSS, "cli", "cross", None, T - DAY, T + 2 * HOUR, "user_exit"),
    (S_AFTER, "cli", None, None, T + DAY + HOUR, None, None),
    (S_EARLY_END, "cli", "early end", None, T - 3 * HOUR, T + HOUR, "user_exit"),
    (S_CHILD, "cli", "child", S_EARLY_END, T + DAY + 3 * HOUR, None, None),
    (S_BEFORE, "cli", "before", None, T - 5 * DAY, T - 4 * DAY, "user_exit"),
)

_MESSAGES = (
    # session, role, content, timestamp
    (S_OPEN, "user", "open before", T - HOUR),
    (S_OPEN, "user", "open one", T + HOUR),
    (S_OPEN, "assistant", "open two", T + 3 * HOUR),
    (S_CROSS, "user", "cross before", T - DAY + 60),
    (S_CROSS, "assistant", "cross inside", T + 90 * 60),
    (S_AFTER, "user", "after", T + DAY + 2 * HOUR),
    (S_EARLY_END, "user", "early inside", T + 30 * 60),
    (S_EARLY_END, "assistant", "after its ended_at", T + 5 * HOUR),
    (S_CHILD, "assistant", "carried parent row", T + 5 * HOUR),
    (S_CHILD, "user", "child own row", T + DAY + 4 * HOUR),
    (S_BEFORE, "user", "long ago", T - 5 * DAY + 60),
    (S_OPEN, "user", "at the until bound", UNTIL),
    (S_CROSS, "user", "at the since bound", SINCE),
)

# The plain time-range queries chat_period ran before it went through sessions, kept here verbatim.
_OLD_ROWS = ("SELECT messages.id FROM messages WHERE messages.timestamp >= ? AND messages.timestamp < ? "
             "ORDER BY messages.timestamp DESC, messages.id DESC LIMIT ?")
_OLD_ROWS_ROLE = ("SELECT messages.id FROM messages WHERE messages.role = ? AND messages.timestamp >= ? "
                  "AND messages.timestamp < ? ORDER BY messages.timestamp DESC, messages.id DESC LIMIT ?")
_OLD_ROWS_SESSION = ("SELECT messages.id FROM messages WHERE (messages.session_id = ? OR messages.session_id LIKE "
                     "'%' || ? ESCAPE '\\') AND messages.timestamp >= ? AND messages.timestamp < ? "
                     "ORDER BY messages.timestamp ASC, messages.id ASC LIMIT ?")
_OLD_SPANS = ("SELECT messages.session_id, MIN(messages.timestamp), MAX(messages.timestamp), COUNT(*) FROM messages "
              "WHERE messages.timestamp >= ? AND messages.timestamp < ? GROUP BY messages.session_id")


@pytest.fixture()
def database(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    connection = sqlite3.connect(path)
    try:
        for statement in _SCHEMA:
            connection.execute(statement)
        connection.executemany("INSERT INTO sessions (id, source, title, parent_session_id, started_at, ended_at, "
                               "end_reason) VALUES (?, ?, ?, ?, ?, ?, ?)", _SESSIONS)
        connection.executemany("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                               _MESSAGES)
        connection.commit()
    finally:
        connection.close()
    return path


def _old(path: Path, sql: str, parameters: tuple) -> list:
    connection = sqlite3.connect(path)
    try:
        return [tuple(row) if len(row) > 1 else row[0] for row in connection.execute(sql, parameters)]
    finally:
        connection.close()


def _ids(payload: dict) -> list[int]:
    return [item["message_id"] for item in payload["results"]]


def _spans(payload: dict) -> set[tuple]:
    return {(item["session_id"], item["first_timestamp"], item["last_timestamp"], item["message_count"])
            for item in payload["sessions"]}


@pytest.mark.parametrize("limit", [50, 3])
def test_same_rows_and_summaries_as_the_time_range_query(database: Path, limit: int) -> None:
    payload = chat_history.period(database, SINCE, UNTIL, limit=limit, exclude_notifications=False)
    old_ids = _old(database, _OLD_ROWS, (SINCE, UNTIL, limit + 1))
    assert _ids(payload) == old_ids[:limit] and payload["truncated"] == (len(old_ids) > limit)
    assert _spans(payload) == set(_old(database, _OLD_SPANS, (SINCE, UNTIL)))
    in_period = {item["session_id"] for item in payload["sessions"]}
    assert {S_OPEN, S_CROSS, S_EARLY_END, S_CHILD} == in_period  # incl. after ended_at and the carried row


def test_same_rows_with_role_and_session(database: Path) -> None:
    by_role = chat_history.period(database, SINCE, UNTIL, role="assistant", limit=50, exclude_notifications=False)
    assert _ids(by_role) == _old(database, _OLD_ROWS_ROLE, ("assistant", SINCE, UNTIL, 51))
    suffix = S_EARLY_END[-8:]
    by_session = chat_history.period(database, SINCE, UNTIL, session=suffix, order="oldest", limit=50,
                                     exclude_notifications=False)
    assert _ids(by_session) == _old(database, _OLD_ROWS_SESSION, (suffix, suffix, SINCE, UNTIL, 51))
    assert [item["session_id"] for item in by_session["sessions"]] == [S_EARLY_END]


def _plan(path: Path, sql: str, parameters: list) -> list[str]:
    connection = sqlite3.connect(path)
    try:
        return [row[3] for row in connection.execute("EXPLAIN QUERY PLAN " + sql, parameters)]
    finally:
        connection.close()


@pytest.mark.parametrize(("session", "role", "exclude_notifications"),
                         [(None, None, False), (None, "user", True), (S_OPEN[-8:], None, False)])
def test_plans_use_the_session_index_and_never_scan_messages(database: Path, session, role,
                                                             exclude_notifications: bool) -> None:
    search_filter = chat_history._build_filter(session, role, None, SINCE, UNTIL, exclude_notifications,
                                               via_sessions=True)
    for sql in (chat_history.period_rows_sql(search_filter, chat_history.ORDER_NEWEST),
                chat_history.session_spans_sql(search_filter)):
        plan = _plan(database, sql, search_filter.parameters)
        assert not any(line.startswith("SCAN messages") for line in plan), plan
        assert any("INDEX idx_messages_session (session_id=? AND timestamp>?" in line for line in plan), plan


def test_without_a_sessions_table_the_time_range_query_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "bare.db"
    connection = sqlite3.connect(path)
    with connection:
        connection.execute(_SCHEMA[1])
        connection.executemany("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                               _MESSAGES)
    connection.close()
    payload = chat_history.period(path, SINCE, UNTIL, limit=50, exclude_notifications=False)
    assert _ids(payload) == _old(path, _OLD_ROWS, (SINCE, UNTIL, 51))
