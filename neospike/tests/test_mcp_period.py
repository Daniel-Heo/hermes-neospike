"""MCP chat_period / mem_period: listings of a time range without a query, and the relative times of parse_time."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pytest

from neospike import chat_history
from neospike.config import NeospikeConfig
from neospike.database import canonical_database_path
from neospike.mcp_server import NeospikeTools, ToolError
from neospike.store import MemoryStore
from neospike.tests.test_mcp_chat_history import (COMPACTION_ID, KEPT_IDS, KEPT_USER_IDS, NOTIFICATION_IDS, NOTIFY_SESSION,
                                                   _SCHEMA, _calls, build_notification_db)
from neospike.tests.test_mcp_memory_search import DIMS, _note, _vector

NS = "period-test"
HOUR = 3_600.0
DAY = 86_400.0
T = datetime(2026, 11, 20, 12, 0).timestamp()  # local time
SINCE, UNTIL = T, T + DAY
OTHER_PROJECT = "D:/somewhere/else"
SESSION_A = "20261120_120000_aaaaaaaa"
SESSION_B = "20261120_130000_bbbbbbbb"
SESSION_C = "20261119_090000_cccccccc"
TITLE_A = "Period work A"
KOREAN_TEXT = "어제 한 일"


def _error(call, *args, **kwargs) -> dict:
    with pytest.raises(ToolError) as caught:
        call(*args, **kwargs)
    return json.loads(str(caught.value))


# -- parse_time ------------------------------------------------------------------------------------

@pytest.mark.parametrize(("text", "seconds"), [("24h", DAY), ("7d", 7 * DAY), ("2w", 14 * DAY), ("1H", HOUR)])
def test_relative_times(text: str, seconds: float) -> None:
    now = 1_800_000_000.0
    assert chat_history.parse_time("since", text, now) == now - seconds


@pytest.mark.parametrize("text", ["0d", "x", "-1d", "1y", "1.5d", "d", "9" * 400 + "d"])
def test_bad_relative_times_are_invalid(text: str) -> None:
    with pytest.raises(chat_history.ChatHistoryError) as caught:
        chat_history.parse_time("since", text)
    assert caught.value.code == "invalid"


def test_period_needs_since_before_until() -> None:
    for since, until in ((None, None), (T, T), (T + 1, T), ("1h", "2h")):
        with pytest.raises(chat_history.ChatHistoryError) as caught:
            chat_history.parse_period(since, until)
        assert caught.value.code == "invalid"
    start, end = chat_history.parse_period("2h", None)
    assert end - start == pytest.approx(2 * HOUR, abs=1.0)


# -- chat_period -----------------------------------------------------------------------------------

_CHAT_ROWS = (
    # id, session, role, content, tool_call_id, tool_calls, tool_name, timestamp, active
    (1, SESSION_C, "user", "before the period", None, None, None, T - HOUR, 1),
    (2, SESSION_A, "user", "first question " + KOREAN_TEXT, None, None, None, T, 1),
    (3, SESSION_A, "assistant", "", None, _calls(("call_1", "note_write", {"text": KOREAN_TEXT})), None, T + HOUR, 1),
    (4, SESSION_A, "tool", '{"ok": true}', "call_1", None, "note_write", T + 2 * HOUR, 1),
    (5, SESSION_B, "user", "second question", None, None, None, T + 3 * HOUR, 1),
    (6, SESSION_B, "assistant", "compacted answer", None, None, None, T + 4 * HOUR, 0),
    (7, SESSION_B, "user", "deleted message", None, None, None, T + 5 * HOUR, 1),
    (8, SESSION_A, "user", "x" * 1_000, None, None, None, T + 6 * HOUR, 1),
    (9, SESSION_A, "user", "at the until bound", None, None, None, UNTIL, 1),
)


def _build_chat_db(path: Path) -> Path:
    connection = sqlite3.connect(path)
    try:
        for statement in _SCHEMA:
            connection.execute(statement)
        connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT)")
        connection.executemany("INSERT INTO sessions VALUES (?, ?)", ((SESSION_A, TITLE_A), (SESSION_B, None)))
        connection.executemany("INSERT INTO messages (id, session_id, role, content, tool_call_id, tool_calls, "
                               "tool_name, timestamp, active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", _CHAT_ROWS)
        connection.execute("DELETE FROM messages WHERE id = 7")  # a deleted message: Hermes removes the row
        connection.commit()
    finally:
        connection.close()
    return path


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    result = tmp_path / "home"
    result.mkdir()
    NeospikeConfig().save(result)
    _build_chat_db(result / "state.db")
    return result


def _chat(tools: NeospikeTools, **options) -> dict:
    return json.loads(tools.chat_period(options.pop("since", SINCE), options.pop("until", UNTIL), **options))


def _ids(payload: dict) -> list[int]:
    return [item["message_id"] for item in payload["results"]]


def test_chat_period_range_newest_first_like_search_results(home: Path) -> None:
    payload = _chat(NeospikeTools(home, NS))
    assert _ids(payload) == [8, 6, 5, 4, 3, 2]  # since <= t < until; the deleted row is gone
    assert payload["truncated"] is False and payload["count"] == 6 and list(payload)[-1] == "truncated"
    by_id = {item["message_id"]: item for item in payload["results"]}
    assert by_id[6]["active"] == 0  # inactive rows stay, as in chat_search
    assert by_id[8]["excerpt"] == "x" * chat_history.EXCERPT_CHARS
    assert by_id[3]["tool_calls"] == [{"id": "call_1", "name": "note_write"}]
    assert KOREAN_TEXT in by_id[3]["excerpt"] and "\\u" not in by_id[3]["excerpt"]
    assert by_id[2]["timestamp"] == T and by_id[2]["time"] == datetime.fromtimestamp(T).isoformat(timespec="seconds")


def test_chat_period_session_summary(home: Path) -> None:
    payload = _chat(NeospikeTools(home, NS), limit=1)
    assert list(payload)[:3] == ["since", "until", "sessions"]
    assert payload["truncated"] is True and _ids(payload) == [8]
    [a, b] = payload["sessions"]  # the whole period whatever the limit, most recent first
    assert (a["session_id"], a["title"], a["message_count"]) == (SESSION_A, TITLE_A, 4)
    assert (a["first_timestamp"], a["last_timestamp"]) == (T, T + 6 * HOUR)
    assert (b["session_id"], b["title"], b["message_count"]) == (SESSION_B, None, 2)
    assert payload["sessions_truncated"] is False
    users = _chat(NeospikeTools(home, NS), role="user")
    assert _ids(users) == [8, 5, 2]
    assert [(s["session_id"], s["message_count"]) for s in users["sessions"]] == [(SESSION_A, 2), (SESSION_B, 1)]


def test_chat_period_filters_order_and_limit(home: Path) -> None:
    tools = NeospikeTools(home, NS)
    assert _ids(_chat(tools, session=SESSION_B[-8:])) == [6, 5]
    assert _ids(_chat(tools, session=SESSION_A)) == [8, 4, 3, 2]
    assert _ids(_chat(tools, tool_name="note_write")) == [4, 3]
    assert [s["message_count"] for s in _chat(tools, tool_name="note_write")["sessions"]] == [2]
    oldest = _chat(tools, order="oldest", limit=2)
    assert _ids(oldest) == [2, 3] and oldest["truncated"] is True
    following = _chat(tools, since=oldest["results"][-1]["timestamp"], order="oldest", limit=2)
    assert _ids(following) == [3, 4]  # since is inclusive: the last row comes again
    newest = _chat(tools, limit=2)
    assert _ids(_chat(tools, until=newest["results"][-1]["timestamp"], limit=2)) == [5, 4]
    assert _ids(_chat(tools, since=datetime.fromtimestamp(T).isoformat(), until=T + HOUR)) == [2]
    for bad in ({"limit": 0}, {"limit": 51}, {"order": "recent"}, {"role": "robot"}, {"session": "abc"},
                {"since": "soon"}, {"since": UNTIL}):
        assert _error(_chat, tools, **bad)["error"] == "invalid"


def test_chat_period_relative_since(home: Path) -> None:
    database = home / "state.db"
    connection = sqlite3.connect(database)
    with connection:
        connection.execute("INSERT INTO messages (id, session_id, role, content, timestamp) VALUES "
                           "(20, ?, 'user', 'just now', ?)", (SESSION_C, time.time() - 60))
    connection.close()
    payload = json.loads(NeospikeTools(home, NS).chat_period("1h"))
    assert _ids(payload) == [20] and payload["sessions"][0]["title"] is None


def test_chat_period_leaves_the_file_unchanged(home: Path) -> None:
    database = home / "state.db"
    before, mtime = database.read_bytes(), os.stat(database).st_mtime_ns
    tools = NeospikeTools(home, NS)
    _chat(tools)
    _chat(tools, tool_name="note_write", order="oldest", limit=1)
    assert database.read_bytes() == before and os.stat(database).st_mtime_ns == mtime


def test_chat_period_works_while_paused_and_mem_period_is_refused(home: Path) -> None:
    tools = NeospikeTools(home, NS)
    NeospikeConfig(enabled=False).save(home)
    assert _chat(tools)["count"] == 6
    assert _error(tools.mem_period, "1d")["error"] == "paused"


@pytest.fixture()
def notify_tools(tmp_path: Path) -> NeospikeTools:
    result = tmp_path / "notify"
    result.mkdir()
    NeospikeConfig().save(result)
    build_notification_db(result / "state.db", T)
    return NeospikeTools(result, NS)


def _counts(payload: dict) -> list[tuple[str, int]]:
    return [(s["session_id"], s["message_count"]) for s in payload["sessions"]]


def test_chat_period_drops_notifications_by_default(notify_tools: NeospikeTools) -> None:
    payload = _chat(notify_tools, order="oldest", limit=50)
    assert _ids(payload) == list(KEPT_IDS) and _counts(payload) == [(NOTIFY_SESSION, len(KEPT_IDS))]
    users = _chat(notify_tools, role="user", limit=50)
    assert _ids(users) == sorted(KEPT_USER_IDS, reverse=True)
    assert _counts(users) == [(NOTIFY_SESSION, len(KEPT_USER_IDS))]


def test_chat_period_can_keep_notifications(notify_tools: NeospikeTools) -> None:
    payload = _chat(notify_tools, order="oldest", limit=50, exclude_notifications=False)
    assert _ids(payload) == sorted(KEPT_IDS + NOTIFICATION_IDS)
    assert _counts(payload) == [(NOTIFY_SESSION, len(KEPT_IDS) + len(NOTIFICATION_IDS))]


def test_chat_period_limit_counts_messages_left(notify_tools: NeospikeTools) -> None:
    payload = _chat(notify_tools, order="oldest", limit=3)
    assert _ids(payload) == [101, 105, 106] and payload["count"] == 3 and payload["truncated"] is True
    last = _chat(notify_tools, order="oldest", limit=len(KEPT_IDS))
    assert payload["truncated"] is True and last["truncated"] is False
    kept = _chat(notify_tools, order="oldest", limit=3, exclude_notifications=False)
    assert _ids(kept) == [101, 102, 103]


def test_chat_period_drops_context_compaction_summaries(notify_tools: NeospikeTools) -> None:
    assert COMPACTION_ID not in _ids(_chat(notify_tools, role="user", limit=50))
    assert COMPACTION_ID in _ids(_chat(notify_tools, role="user", limit=50, exclude_notifications=False))


def test_chat_period_exclude_notifications_must_be_bool(notify_tools: NeospikeTools) -> None:
    assert _error(_chat, notify_tools, exclude_notifications=1)["error"] == "invalid"


def test_chat_period_without_database(tmp_path: Path) -> None:
    NeospikeConfig().save(tmp_path)
    assert _error(NeospikeTools(tmp_path, NS).chat_period, "1d")["error"] == "not_found"


# -- mem_period ------------------------------------------------------------------------------------

class _Memories:
    """Ids of the memories the mem_period fixture saves."""
    created_only = updated = no_vector = superseded = other_project = tagged = outside = 0


@pytest.fixture()
def mem_tools(tmp_path: Path) -> NeospikeTools:
    home = tmp_path / "home"
    NeospikeConfig().save(home)
    store = MemoryStore(canonical_database_path(home / "neospike", "memory"), dimensions=DIMS)
    try:
        def remember(text: str, **metadata) -> int:
            return store.insert(NS, text, _vector(0.8), source="agent:s", kind="durable",
                                project=metadata.pop("project", ""), metadata={"scope": "global", **metadata})

        def stamp(item_id: int, created: float, updated: float | None) -> None:
            store._conn.execute("UPDATE items SET created_at=?, updated_at=? WHERE id=?", (created, updated, item_id))

        ids = _Memories()
        ids.outside = remember("memory outside")
        ids.created_only = remember("memory created only")
        ids.updated = remember("memory updated later")
        ids.no_vector = remember("memory not embedded yet")
        ids.superseded = remember("memory superseded")
        ids.other_project = remember("memory of another project", scope="project", project=OTHER_PROJECT)
        ids.tagged = remember("memory tagged")
        with store._conn:
            # Changes first, stamps after: the update trigger moves updated_at of a change that leaves it alone.
            store._conn.execute("UPDATE items SET embedding=X'' WHERE id=?", (ids.no_vector,))
            store._conn.execute("UPDATE items SET invalid_at=?, superseded_by=? WHERE id=?",
                                (T + 4 * HOUR, ids.created_only, ids.superseded))
            store._conn.execute("UPDATE items SET metadata_json=json_set(metadata_json,'$.tags',json('[\"task\"]')) "
                                "WHERE id=?", (ids.tagged,))
            stamp(ids.outside, T - 3 * DAY, T - 3 * DAY)
            stamp(ids.created_only, T + 1 * HOUR, None)  # no updated_at: created_at counts
            stamp(ids.updated, T - 10 * DAY, T + 2 * HOUR)
            stamp(ids.no_vector, T + 3 * HOUR, T + 3 * HOUR)
            stamp(ids.superseded, T + 4 * HOUR, T + 4 * HOUR)
            stamp(ids.other_project, T + 5 * HOUR, T + 5 * HOUR)
            stamp(ids.tagged, T + 6 * HOUR, T + 6 * HOUR)
            documents = (("agent plan", "agent", '["task"]', False, 7), ("human plan", "human", "[]", False, 8),
                         ("TOPIC: period", "agent", '["topic"]', False, 9), ("trashed note", "agent", "[]", True, 10))
            for title, author, tags_json, trashed, hours in documents:
                note_id = _note(store, title, author, [(f"{title} body", 0.8)], trashed=trashed, tags_json=tags_json)
                stamp(note_id, T + hours * HOUR, T + hours * HOUR)
    finally:
        store.close()
    tools = NeospikeTools(home, NS)
    tools.ids = ids
    return tools


def _mem(tools: NeospikeTools, **options) -> dict:
    return json.loads(tools.mem_period(options.pop("since", SINCE), options.pop("until", UNTIL), **options))


def _labels(payload: dict) -> list[str]:
    return [item.get("title") or item["text"] for item in payload["results"]]


def test_mem_period_lists_memories_notes_and_docs_newest_first(mem_tools: NeospikeTools) -> None:
    payload = _mem(mem_tools)
    assert _labels(payload) == ["human plan", "agent plan", "memory tagged", "memory not embedded yet",
                                "memory updated later", "memory created only"]  # no topic page, trash, other project
    assert payload["truncated"] is False and list(payload)[-1] == "truncated"
    doc, note, memory = payload["results"][0], payload["results"][1], payload["results"][-1]
    assert doc["kind"] == "doc" and note["kind"] == "note" and note["tags"] == ["task"] and note["rev"] == 1
    assert "body" not in note and "text" not in note
    assert memory == {"kind": "memory", "id": mem_tools.ids.created_only, "text": "memory created only",
                      "origin": memory["origin"], "item_kind": "durable", "confidence": 1.0,
                      "time": datetime.fromtimestamp(T + HOUR).isoformat(timespec="seconds"),
                      "timestamp": T + HOUR, "created_at": T + HOUR, "updated_at": None}
    updated = payload["results"][-2]
    assert updated["created_at"] == T - 10 * DAY and updated["timestamp"] == T + 2 * HOUR


def test_mem_period_kind_tag_history_order_and_limit(mem_tools: NeospikeTools) -> None:
    assert _labels(_mem(mem_tools, kind="doc")) == ["human plan"]
    assert _labels(_mem(mem_tools, kind="note")) == ["agent plan"]
    assert [item["kind"] for item in _mem(mem_tools, kind="memory")["results"]] == ["memory"] * 4
    assert _labels(_mem(mem_tools, tag="task")) == ["agent plan", "memory tagged"]
    history = _mem(mem_tools, kind="memory", include_history=True)["results"]
    [old] = [item for item in history if item["text"] == "memory superseded"]
    assert old["superseded_by"] == mem_tools.ids.created_only
    oldest = _mem(mem_tools, order="oldest", limit=2)
    assert _labels(oldest) == ["memory created only", "memory updated later"] and oldest["truncated"] is True
    newest = _mem(mem_tools, limit=2)
    assert _labels(newest) == ["human plan", "agent plan"] and newest["truncated"] is True
    rest = _mem(mem_tools, until=newest["results"][-1]["timestamp"], limit=20)
    assert _labels(rest) == ["memory tagged", "memory not embedded yet", "memory updated later",
                             "memory created only"]
    assert _labels(_mem(mem_tools, since=T - 4 * DAY, until=T)) == ["memory outside"]
    for bad in ({"limit": 0}, {"limit": 21}, {"kind": "chunk"}, {"order": "rank"}, {"include_history": 1},
                {"since": "x"}, {"since": UNTIL}):
        assert _error(_mem, mem_tools, **bad)["error"] == "invalid"


def test_mem_period_relative_since_and_database_unchanged(mem_tools: NeospikeTools) -> None:
    database = mem_tools.database
    before, mtime = database.read_bytes(), os.stat(database).st_mtime_ns
    relative = json.loads(mem_tools.mem_period("2w"))
    span = datetime.fromisoformat(relative["until"]) - datetime.fromisoformat(relative["since"])
    assert span.total_seconds() == pytest.approx(14 * DAY, abs=1.0)
    _mem(mem_tools, include_history=True, tag="task")
    assert database.read_bytes() == before and os.stat(database).st_mtime_ns == mtime
