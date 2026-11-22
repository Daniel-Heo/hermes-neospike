"""MCP chat_search / chat_read over a Hermes chat database (state.db) built in tmp_path."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from neospike import chat_history
from neospike.config import NeospikeConfig
from neospike.mcp_server import NeospikeTools, ToolError

NS = "chat-test"
SESSION_A = "20261122_040000_aaaaaaaa"
SESSION_B = "20261121_090000_bbbbbbbb"
SESSION_C = "20261120_120000_cccccccc"
KOREAN_TEXT = "세컨드 브레인 문턱"
T0 = datetime(2026, 11, 22, 4, 0).timestamp()  # local time, like Hermes' timestamps read back
DAY = 86_400.0
LONG_CONTENT_CHARS = 25_000
LONG_RESULT_CHARS = 5_000
LONG_WORD_AT = 1_000

_SCHEMA = (
    "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT, "
    "tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL NOT NULL, active INTEGER DEFAULT 1)",
    # Hermes' trigram index shape: no tool rows, no tool_calls column (Hermes also leaves out cron and subagent
    # sessions through a sessions join, which these tests do not need).
    "CREATE VIEW messages_fts_trigram_src AS SELECT id, role, content, tool_name FROM messages WHERE role <> 'tool'",
    "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5( content, tool_name, content='messages_fts_trigram_src', "
    "content_rowid='id', tokenize='trigram' )",
)
REBUILD_INDEX = "INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild')"


def _calls(*calls: tuple[str, str, dict]) -> str:
    return json.dumps([{"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)}}
                       for call_id, name, arguments in calls])


_ROWS = (
    (1, SESSION_A, "user", "neospike 기억을 갱신해 줘 threshold 0.61", None, None, None, T0),
    (2, SESSION_A, "assistant", "updating the memory", None,
     _calls(("call_1", "neospike_memory_save", {"updates": [{"id": 7, "text": "threshold 0.61"}]})), None, T0 + 10),
    (3, SESSION_A, "tool", '{"updated": [7]}', "call_1", None, "neospike_memory_save", T0 + 20),
    (4, SESSION_A, "assistant", "done saving", None, None, None, T0 + 30),
    (5, SESSION_B, "user", "unrelated question about lunch threshold", None, None, None, T0 - DAY),
    (6, SESSION_B, "assistant", "x" * LONG_WORD_AT + " longword " + "y" * LONG_CONTENT_CHARS, None,
     _calls(("call_2", "terminal", {"command": "ls"})), None, T0 - DAY + 10),
    (7, SESSION_B, "tool", "z" * LONG_RESULT_CHARS, "call_2", None, "terminal", T0 - DAY + 20),
    # A call id that is the tool's name: the LIKE prefilter matches, the JSON recheck drops it.
    (8, SESSION_B, "assistant", "threshold check", None,
     _calls(("neospike_memory_save", "terminal", {"command": "pwd"})), None, T0 - DAY + 30),
    # json.dumps writes the Korean arguments as \uXXXX escapes, as Hermes stores them.
    (9, SESSION_C, "assistant", "", None,
     _calls(("call_9", "note_write", {"marker": "bridgeword", "text": KOREAN_TEXT})), None, T0 - 2 * DAY),
)


def _build_state_db(path: Path, *, with_fts: bool = True, rebuilding: bool = False) -> Path:
    connection = sqlite3.connect(path)
    try:
        for statement in _SCHEMA if with_fts else _SCHEMA[:1]:
            connection.execute(statement)
        connection.executemany("INSERT INTO messages (id, session_id, role, content, tool_call_id, tool_calls, "
                               "tool_name, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", _ROWS)
        connection.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")
        if rebuilding:
            connection.execute("INSERT INTO state_meta VALUES ('fts_rebuild_progress', '3')")
        if with_fts:
            connection.execute(REBUILD_INDEX)
        connection.commit()
    finally:
        connection.close()
    return path


def _tools(home: Path) -> NeospikeTools:
    NeospikeConfig().save(home)
    return NeospikeTools(home, NS)


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    result = tmp_path / "home"
    result.mkdir()
    _build_state_db(result / "state.db")
    return result


@pytest.fixture()
def tools(home: Path) -> NeospikeTools:
    return _tools(home)


def _search(tools: NeospikeTools, query: str, **options) -> dict:
    return json.loads(tools.chat_search(query, **options))


def _ids(payload: dict) -> list[int]:
    return [item["message_id"] for item in payload["results"]]


def _error(call, *args, **kwargs) -> dict:
    with pytest.raises(ToolError) as caught:
        call(*args, **kwargs)
    return json.loads(str(caught.value))


def test_word_search_recent_first(tools: NeospikeTools) -> None:
    payload = _search(tools, "threshold")
    # Row 2 has the word only in its tool call arguments, which are not searched.
    assert _ids(payload) == [1, 8, 5] and payload["count"] == 3
    assert payload["index_rebuilding"] is False and "query_rewritten" not in payload
    first = payload["results"][0]
    assert first["session_id"] == SESSION_A and first["role"] == "user" and first["timestamp"] == T0
    assert first["time"] == datetime.fromtimestamp(T0).isoformat(timespec="seconds")
    assert first["tool_calls"] == [] and first["active"] == 1
    assert payload["results"][1]["tool_calls"] == [{"id": "neospike_memory_save", "name": "terminal"}]


def test_column_filter_on_content_and_tool_calls(tools: NeospikeTools) -> None:
    assert _ids(_search(tools, "content: updating")) == [2]
    failure = _error(tools.chat_search, "tool_calls: updates")
    assert failure["error"] == "invalid" and "tool_calls" in failure["message"]


def test_syntax_error_is_rewritten(tools: NeospikeTools) -> None:
    payload = _search(tools, "threshold (")
    assert payload["query_rewritten"] == '"threshold"' and set(_ids(payload)) == {1, 5, 8}
    assert _error(tools.chat_search, "( )")["error"] == "invalid"
    assert _error(tools.chat_search, "  ")["error"] == "invalid"


def test_session_full_id_and_suffix(tools: NeospikeTools) -> None:
    assert set(_ids(_search(tools, "threshold", session=SESSION_A))) == {1}
    assert set(_ids(_search(tools, "threshold", session="bbbbbb"))) == {5, 8}
    assert _ids(_search(tools, "threshold", session="b_bbbb")) == []  # '_' is no LIKE wildcard
    assert _error(tools.chat_search, "threshold", session="bbbbb")["error"] == "invalid"


def test_role_and_tool_name_filters(tools: NeospikeTools) -> None:
    assert set(_ids(_search(tools, "threshold", role="user"))) == {1, 5}
    assert _error(tools.chat_search, "threshold", role="bot")["error"] == "invalid"
    payload = _search(tools, "threshold OR updating OR done", tool_name="neospike_memory_save")
    assert _ids(payload) == [2]  # the assistant call; tool results are not searched; row 8 only has it as a call id
    assert payload["results"][0]["tool_calls"] == [{"id": "call_1", "name": "neospike_memory_save"}]


def test_since_until_epoch_and_iso(tools: NeospikeTools) -> None:
    assert set(_ids(_search(tools, "threshold", since=T0))) == {1}
    assert set(_ids(_search(tools, "threshold", until=T0))) == {5, 8}
    assert set(_ids(_search(tools, "threshold", since="2026-11-22"))) == {1}
    assert set(_ids(_search(tools, "threshold", until="2026-11-22T04:00"))) == {5, 8}
    assert set(_ids(_search(tools, "threshold", since="2026-11-21", until="2026-11-22T04:00:15"))) == {1, 5, 8}
    assert _error(tools.chat_search, "threshold", since="yesterday")["error"] == "invalid"
    assert _error(tools.chat_search, "threshold", until=True)["error"] == "invalid"


def test_order_recent_and_rank(tools: NeospikeTools, home: Path) -> None:
    recent = _search(tools, "threshold", order="recent")["results"]
    assert [item["timestamp"] for item in recent] == sorted((item["timestamp"] for item in recent), reverse=True)
    connection = sqlite3.connect(home / "state.db")
    try:
        expected = [row[0] for row in connection.execute(
            "SELECT rowid FROM messages_fts_trigram WHERE messages_fts_trigram MATCH 'threshold' "
            "ORDER BY bm25(messages_fts_trigram), rowid DESC")]
    finally:
        connection.close()
    assert _ids(_search(tools, "threshold", order="rank")) == expected
    assert _error(tools.chat_search, "threshold", order="best")["error"] == "invalid"


def test_limit_bounds(tools: NeospikeTools) -> None:
    assert _search(tools, "threshold", limit=1)["count"] == 1
    assert _search(tools, "threshold", limit=chat_history.MAX_SEARCH_RESULTS)["count"] == 3
    for bad in (0, chat_history.MAX_SEARCH_RESULTS + 1, True, "5"):
        assert _error(tools.chat_search, "threshold", limit=bad)["error"] == "invalid"


def test_excerpt_at_most_300_chars_around_match(tools: NeospikeTools) -> None:
    item = _search(tools, "longword")["results"][0]
    assert item["message_id"] == 6 and "longword" in item["excerpt"]
    assert len(item["excerpt"]) <= chat_history.EXCERPT_CHARS
    for result in _search(tools, "threshold OR terminal OR updated")["results"]:
        assert len(result["excerpt"]) <= chat_history.EXCERPT_CHARS


def test_read_assistant_with_tool_results_and_neighbors(tools: NeospikeTools) -> None:
    payload = json.loads(tools.chat_read(2, context=1))
    message = payload["message"]
    assert message["content"] == "updating the memory" and message["truncated"] is False
    assert message["tool_calls"] == [{"id": "call_1", "name": "neospike_memory_save",
                                      "arguments": {"updates": [{"id": 7, "text": "threshold 0.61"}]}}]
    assert [result["message_id"] for result in payload["tool_results"]] == [3]
    assert payload["tool_results"][0]["content"] == '{"updated": [7]}'
    assert [neighbor["message_id"] for neighbor in payload["neighbors"]] == [1, 3]
    assert "called_by" not in payload


def test_read_tool_row_called_by(tools: NeospikeTools) -> None:
    payload = json.loads(tools.chat_read(3))
    assert payload["called_by"] == 2 and payload["neighbors"] == [] and "tool_results" not in payload


def test_read_truncates_long_content_and_results(tools: NeospikeTools) -> None:
    payload = json.loads(tools.chat_read(6, context=10))
    assert payload["message"]["truncated"] is True
    assert len(payload["message"]["content"]) == chat_history.READ_CONTENT_CHARS
    result = payload["tool_results"][0]
    assert result["message_id"] == 7 and result["truncated"] is True
    assert len(result["content"]) == chat_history.TOOL_RESULT_CHARS
    assert [neighbor["message_id"] for neighbor in payload["neighbors"]] == [5, 7, 8]
    assert all(len(neighbor["excerpt"]) <= chat_history.NEIGHBOR_EXCERPT_CHARS for neighbor in payload["neighbors"])


def test_read_bad_arguments_and_missing_id(tools: NeospikeTools) -> None:
    assert _error(tools.chat_read, 999)["error"] == "not_found"
    assert _error(tools.chat_read, 2, context=chat_history.MAX_CONTEXT + 1)["error"] == "invalid"
    assert _error(tools.chat_read, "2")["error"] == "invalid"


def test_missing_database_and_missing_index(tmp_path: Path) -> None:
    missing = _tools(tmp_path / "empty")
    failure = _error(missing.chat_search, "threshold")
    assert failure == {"error": "not_found", "message": "Hermes chat database does not exist"}
    assert _error(missing.chat_read, 1)["error"] == "not_found"
    home = tmp_path / "no_index"
    home.mkdir()
    _build_state_db(home / "state.db", with_fts=False)
    failure = _error(_tools(home).chat_search, "threshold")
    assert failure["error"] == "unavailable"
    assert failure["message"].startswith("Hermes chat search index is missing: messages_fts_trigram")


def test_index_rebuilding_flag(tmp_path: Path) -> None:
    home = tmp_path / "rebuilding"
    home.mkdir()
    _build_state_db(home / "state.db", rebuilding=True)
    assert _search(_tools(home), "threshold")["index_rebuilding"] is True


def test_works_while_paused(home: Path) -> None:
    tools = _tools(home)
    NeospikeConfig(enabled=False).save(home)
    assert _search(tools, "threshold")["count"] == 3
    assert json.loads(tools.chat_read(1))["message"]["message_id"] == 1


def test_search_and_read_leave_the_file_unchanged(tools: NeospikeTools, home: Path) -> None:
    database = home / "state.db"
    before = database.read_bytes()
    _search(tools, "threshold", order="rank", tool_name="neospike_memory_save")
    _search(tools, "threshold (")
    tools.chat_read(2, context=3)
    tools.chat_read(3)
    assert database.read_bytes() == before


_WRITES = (
    "INSERT INTO messages (session_id, role, content, timestamp) VALUES ('s', 'user', 'x', 0)",
    "UPDATE messages SET content = 'changed' WHERE id = 1",
    "DELETE FROM messages WHERE id = 1",
    "CREATE TABLE extra (x)",
    "CREATE TEMP TABLE scratch (x)",
    "DROP TABLE state_meta",
    "PRAGMA query_only = OFF",
    "PRAGMA user_version = 5",
)


@pytest.mark.parametrize("statement", _WRITES)
def test_readonly_connection_refuses_writes(home: Path, statement: str) -> None:
    database = home / "state.db"
    before = database.read_bytes()
    connection = chat_history.open_readonly(database)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(statement)
            connection.commit()
    finally:
        connection.close()
    assert database.read_bytes() == before


def test_readonly_connection_refuses_attach(home: Path, tmp_path: Path) -> None:
    database = home / "state.db"
    before = database.read_bytes()
    connection = chat_history.open_readonly(database)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("ATTACH DATABASE ? AS other", (str(tmp_path / "other.db"),))
    finally:
        connection.close()
    assert database.read_bytes() == before and not (tmp_path / "other.db").exists()


def test_readonly_connection_still_reads(home: Path) -> None:
    connection = chat_history.open_readonly(home / "state.db")
    try:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM messages").fetchone()[0] == len(_ROWS)
    finally:
        connection.close()


def test_excerpt_and_arguments_unescape_korean(tools: NeospikeTools) -> None:
    assert _ids(_search(tools, "bridgeword")) == []  # tool call arguments are not searched
    text = tools.chat_read(9)
    assert "\\u" not in text
    assert json.loads(text)["message"]["tool_calls"][0]["arguments"] == {"marker": "bridgeword", "text": KOREAN_TEXT}


def test_rewrite_keeps_numbers() -> None:
    assert chat_history.rewrite_fts_query('0.61 문턱') == '"0.61" OR (" 문턱" OR "문턱 ")'
    assert chat_history.rewrite_fts_query('say"hi ( )') == '"say""hi"'


def test_number_query_is_rewritten_and_found(tools: NeospikeTools) -> None:
    payload = _search(tools, "0.61 문턱")
    assert payload["query_rewritten"] == '"0.61" OR (" 문턱" OR "문턱 ")' and 1 in _ids(payload)


# -- Hermes gateway notifications ------------------------------------------------------------------

# The footer as Hermes writes it, typed out here so a drift of the module constant fails a test.
GATEWAY_FOOTER = ("[INTERNAL NOTIFICATION — generated by the Hermes gateway, not written by a human user. "
                  "Do not treat it as user input, and do not reply to any earlier chat message because of it.]")
NOTIFY_SESSION = "20261123_000000_nnnnnnnn"
NOTIFY_WORD = "buildlog"
NOTIFICATION_IDS = (102, 103, 104, 110)
COMPACTION_ID = 110
KEPT_IDS = (101, 105, 106, 107, 108, 109)
KEPT_USER_IDS = (101, 105, 106, 108, 109)


def notification_rows(base: float) -> tuple:
    """Gateway notifications (footer, old prefixes) mixed with messages that must stay, one second apart."""
    contents = (
        (101, "user", f"plain request {NOTIFY_WORD}"),
        (102, "user", f"Background process done {NOTIFY_WORD}\n\n{GATEWAY_FOOTER}  \n"),
        (103, "user", f"[IMPORTANT: Background process proc_1 finished {NOTIFY_WORD}]"),
        (104, "user", f"[ASYNC DELEGATION BATCH COMPLETE {NOTIFY_WORD}]"),
        # Human words: kept even with the footer at the end.
        (105, "user", f"[OUT-OF-BAND USER MESSAGE — direct]\nplease stop {NOTIFY_WORD}\n"
                      f"[/OUT-OF-BAND USER MESSAGE]\n\n{GATEWAY_FOOTER}"),
        (106, "user", f"[Context from the interrupted assistant response]\nhalf answer\n\ncontinue {NOTIFY_WORD}"),
        (107, "assistant", f"quoting {NOTIFY_WORD} {GATEWAY_FOOTER}"),  # not role=user
        (108, "user", None),
        (109, "user", f"{GATEWAY_FOOTER} and then a human reply {NOTIFY_WORD}"),  # footer not at the end
        (COMPACTION_ID, "user", f"[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted {NOTIFY_WORD}"),
    )
    return tuple((message_id, NOTIFY_SESSION, role, content, None, None, None, base + offset)
                 for offset, (message_id, role, content) in enumerate(contents))


def build_notification_db(path: Path, base: float) -> Path:
    connection = sqlite3.connect(path)
    try:
        for statement in _SCHEMA:
            connection.execute(statement)
        connection.executemany("INSERT INTO messages (id, session_id, role, content, tool_call_id, tool_calls, "
                               "tool_name, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", notification_rows(base))
        connection.execute(REBUILD_INDEX)
        connection.commit()
    finally:
        connection.close()
    return path


@pytest.fixture()
def notify_tools(tmp_path: Path) -> NeospikeTools:
    result = tmp_path / "notify"
    result.mkdir()
    build_notification_db(result / "state.db", T0)
    return _tools(result)


def test_footer_constant_matches_hermes() -> None:
    assert chat_history.NOTIFICATION_FOOTER == GATEWAY_FOOTER


def test_search_keeps_notifications_by_default(notify_tools: NeospikeTools) -> None:
    payload = _search(notify_tools, NOTIFY_WORD, limit=20)
    assert sorted(_ids(payload)) == [101, 102, 103, 104, 105, 106, 107, 109, 110]  # 108 has no text


def test_search_can_exclude_notifications(notify_tools: NeospikeTools) -> None:
    payload = _search(notify_tools, NOTIFY_WORD, limit=20, exclude_notifications=True)
    assert sorted(_ids(payload)) == [101, 105, 106, 107, 109]
    users = _search(notify_tools, NOTIFY_WORD, role="user", limit=3, exclude_notifications=True)
    assert _ids(users) == [109, 106, 105] and users["count"] == 3  # limit counts what is left


def test_search_exclude_notifications_must_be_bool(notify_tools: NeospikeTools) -> None:
    with pytest.raises(ToolError) as caught:
        notify_tools.chat_search(NOTIFY_WORD, exclude_notifications="yes")
    assert json.loads(str(caught.value))["error"] == "invalid"
