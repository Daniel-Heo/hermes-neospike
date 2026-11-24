"""Dashboard conversation view read straight from a Hermes chat database (state.db) built in tmp_path."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pytest

from neospike import chat_history, chat_view
from neospike.chat_history import ChatHistoryError
from neospike.dashboard import plugin_api as api
from neospike.store import MemoryStore
from neospike.tests.test_chat_search_trigram import _REBUILD, _SCHEMA
from neospike.tests.test_dashboard_notes import DASHBOARD, client, home  # noqa: F401 (pytest fixtures)
from neospike.tests.test_dashboard_notes import _db as _memory_db
from neospike.tests.test_notes import DIMENSIONS

SESSION_KEY = "20261122_040000_aaaaaaaa"
SESSION_NOTE = "20261121_090000_bbbbbbbb"
SESSION_EMPTY = "20261123_010000_cccccccc"
T0 = datetime(2026, 11, 22, 4, 0).timestamp()
DAY = 86_400.0
NOTIFICATION = "[IMPORTANT: Background process finished] flowerpot watered"
WORD = "flowerpot"
LONG_TEXT = "y " * 250 + WORD + "x " * 91 + " end of note"

_SESSIONS = ((SESSION_KEY, "cli", "{}", ""), (SESSION_NOTE, "telegram", "{}", "Long note"),
             (SESSION_EMPTY, "cli", "{}", None))
_ROWS = (
    (1, SESSION_KEY, "user", "  Where is the spare key?  ", None, None, None, T0),
    (2, SESSION_KEY, "assistant", "Under the blue flowerpot by the door.", None, None, None, T0 + 10),
    (3, SESSION_KEY, "tool", "flowerpot tool output", "call_1", None, "terminal", T0 + 20),
    (4, SESSION_KEY, "user", NOTIFICATION, None, None, None, T0 + 30),
    (5, SESSION_KEY, "system", "flowerpot system prompt", None, None, None, T0 + 40),
    (6, SESSION_KEY, "user", "   ", None, None, None, T0 + 50),
    (7, SESSION_KEY, "assistant", "[CONTEXT COMPACTION] flowerpot summary", None, None, None, T0 + 60),
    (8, SESSION_NOTE, "user", LONG_TEXT, None, None, None, T0 - DAY),
    (9, SESSION_NOTE, "assistant", "한국어 백업 정리를 끝냈다 flowerpot", None, None, None, T0 - DAY + 10),
    (10, SESSION_EMPTY, "tool", "flowerpot", "call_2", None, "terminal", T0 + 100),
    (11, SESSION_EMPTY, "user", NOTIFICATION, None, None, None, T0 + 110),
)
SHOWN_IDS = {1, 2, 8, 9}


def build_state_db(path: Path, *, with_trigram: bool = True) -> Path:
    with closing(sqlite3.connect(path)) as connection:
        for statement in _SCHEMA if with_trigram else _SCHEMA[:2]:
            connection.execute(statement)
        connection.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?)", _SESSIONS)
        connection.executemany("INSERT INTO messages (id, session_id, role, content, tool_call_id, tool_calls, "
                               "tool_name, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", _ROWS)
        if with_trigram:
            connection.execute(_REBUILD)
        connection.commit()
    return path


def _row_timestamp(row: tuple) -> float:
    return row[7]


def _expected_rows(session_id: str) -> list[tuple]:
    """(role, stripped content, timestamp) of the session's rows that conversations.is_capturable selects."""
    rows = sorted((row for row in _ROWS if row[1] == session_id), key=_row_timestamp)
    return [(row[2], row[3].strip(), row[7]) for row in rows if chat_view.is_capturable(row[2], row[3])]


@pytest.fixture()
def state_home(tmp_path: Path) -> Path:
    home = tmp_path / "state-home"
    home.mkdir()
    build_state_db(home / "state.db")
    return home


def test_list_order_and_fields(state_home: Path) -> None:
    listed = chat_view.list_sessions(state_home)
    assert [item["session_id"] for item in listed] == [SESSION_KEY, SESSION_NOTE]  # SESSION_EMPTY shows nothing
    assert listed[0] == {"session_id": SESSION_KEY, "title": "Where is the spare key?", "started_at": T0,
                         "last_message_at": T0 + 10, "messages": 2}
    assert listed[1] == {"session_id": SESSION_NOTE, "title": "Long note", "started_at": T0 - DAY,
                         "last_message_at": T0 - DAY + 10, "messages": 2}
    assert [item["session_id"] for item in chat_view.list_sessions(state_home, 1)] == [SESSION_KEY]


@pytest.mark.parametrize("session_id", [SESSION_KEY, SESSION_NOTE])
def test_transcript_follows_the_selection_rule(state_home: Path, session_id: str) -> None:
    record = chat_view.session_record(state_home, session_id)
    expected = _expected_rows(session_id)
    assert [(entry["role"], entry["content"], entry["created_at"]) for entry in record["transcript"]] == expected
    assert [entry["seq"] for entry in record["transcript"]] == list(range(1, len(expected) + 1))
    assert record["messages"] == len(expected)
    assert {entry["role"] for entry in record["transcript"]} <= {"user", "assistant"}
    assert all(NOTIFICATION != entry["content"] for entry in record["transcript"])


def test_search_keeps_shown_messages_one_item_each(state_home: Path) -> None:
    found = chat_view.search_sessions(state_home, WORD)
    # tool, system, notification, control-marker and empty-session rows are left out; SESSION_NOTE matches twice.
    assert [(hit["message_id"], hit["session_id"]) for hit in found] == [
        (2, SESSION_KEY), (9, SESSION_NOTE), (8, SESSION_NOTE)]
    assert {hit["message_id"] for hit in found} <= SHOWN_IDS
    assert found[0]["title"] == "Where is the spare key?" and found[1]["title"] == "Long note"
    assert found[0]["created_at"] == T0 + 10 and found[0]["role"] == "assistant" and WORD in found[0]["text"]
    assert [hit["message_id"] for hit in chat_view.search_sessions(state_home, "정리를")] == [9]
    assert len(chat_view.search_sessions(state_home, WORD, 1)) == 1


def test_search_text_is_the_chat_search_excerpt(state_home: Path) -> None:
    raw = chat_history.search(chat_history.chat_database_path(state_home), WORD, session=SESSION_NOTE,
                              role="user")["results"][0]["excerpt"]
    hit = next(item for item in chat_view.search_sessions(state_home, WORD) if item["message_id"] == 8)
    assert hit["text"] == raw and WORD in raw and len(raw) == chat_history.EXCERPT_CHARS


def test_missing_session_and_missing_database(state_home: Path, tmp_path: Path) -> None:
    for session_id in ("20261101_000000_dddddddd", SESSION_EMPTY):
        with pytest.raises(ChatHistoryError) as missing:
            chat_view.session_record(state_home, session_id)
        assert missing.value.code == "not_found"
    empty_home = tmp_path / "no-state"
    empty_home.mkdir()
    for call, arguments in ((chat_view.list_sessions, ()), (chat_view.session_record, (SESSION_KEY,)),
                            (chat_view.search_sessions, (WORD,))):
        with pytest.raises(ChatHistoryError) as absent:
            call(empty_home, *arguments)
        assert absent.value.code == "unavailable" and "does not exist" in absent.value.message


def test_search_without_trigram_index_is_an_error(tmp_path: Path) -> None:
    home = tmp_path / "old-hermes"
    home.mkdir()
    build_state_db(home / "state.db", with_trigram=False)
    with pytest.raises(ChatHistoryError) as missing:
        chat_view.search_sessions(home, WORD)
    assert missing.value.code == "unavailable" and chat_history.FTS_TABLE in missing.value.message
    assert [item["session_id"] for item in chat_view.list_sessions(home)] == [SESSION_KEY, SESSION_NOTE]


# dashboard (read-only tab) ------------------------------------------------------------------------
@pytest.fixture()
def dashboard_home(home: Path) -> Path:
    """Hermes home with state.db and a memory.db (which holds no conversation tables)."""
    MemoryStore(_memory_db(home), dimensions=DIMENSIONS).close()
    with closing(sqlite3.connect(_memory_db(home))) as connection:
        assert not [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE name LIKE 'conversation%'")]
    build_state_db(home / "state.db")
    return home


def test_dashboard_routes_read_state_db_only(dashboard_home: Path, client) -> None:
    listed = client.get("/conversations").json()["conversations"]
    assert [(item["session_id"], item["title"], item["messages"]) for item in listed] == [
        (SESSION_KEY, "Where is the spare key?", 2), (SESSION_NOTE, "Long note", 2)]
    record = client.get(f"/conversations/{SESSION_KEY}").json()
    assert [entry["seq"] for entry in record["transcript"]] == [1, 2]
    assert client.get("/conversations/20261101_000000_dddddddd").status_code == 404
    found = client.get("/conversations/search", params={"q": WORD}).json()["results"]
    assert [hit["message_id"] for hit in found] == [2, 9, 8]
    assert client.get("/conversations/search", params={"q": " "}).json() == {"results": []}
    assert client.post("/conversations").status_code == 405
    assert client.delete(f"/conversations/{SESSION_KEY}").status_code == 405
    targets = sqlite3.connect.targets  # the read-only connect guard of the client fixture
    assert targets and all(target.split("?")[0].endswith("/state.db") for target in targets)
    assert not api.subprocess.run.commands  # the search runs in the dashboard process, not the CLI
    script = (DASHBOARD / "dist" / "index.js").read_text(encoding="utf-8")
    assert '"tab.conversations": "Conversations"' in script and '"tab.conversations": "대화"' in script
    assert "/conversations/search?q=" in script and "showDialog(item.session_id)" in script


def test_dashboard_without_state_db_is_a_clear_error(home: Path, client) -> None:
    for path in ("/conversations", f"/conversations/{SESSION_KEY}", f"/conversations/search?q={WORD}"):
        response = client.get(path)
        assert response.status_code == 503 and "does not exist" in response.json()["detail"]
