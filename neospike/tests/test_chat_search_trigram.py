"""chat_search over the trigram index of a Hermes chat database (state.db) built in tmp_path."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from neospike import chat_history
from neospike.config import NeospikeConfig
from neospike.mcp_server import NeospikeTools, ToolError

NS = "trigram-test"
SESSION_A = "20261122_040000_aaaaaaaa"
SESSION_B = "20261121_090000_bbbbbbbb"
SESSION_CRON = "20261121_100000_cccccccc"
T0 = datetime(2026, 11, 22, 4, 0).timestamp()
DAY = 86_400.0

# Hermes' tables and its trigram index: the source view leaves out tool rows and cron/subagent sessions and has no
# tool_calls column.
_SCHEMA = (
    "CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT NOT NULL, model_config TEXT, title TEXT)",
    "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT, "
    "tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL NOT NULL, active INTEGER DEFAULT 1)",
    "CREATE VIEW messages_fts_trigram_src AS SELECT m.id, m.role, m.content, m.tool_name FROM messages AS m "
    "JOIN sessions AS s ON s.id = m.session_id WHERE m.role <> 'tool' AND s.source NOT IN ('cron', 'subagent') "
    "AND json_extract(s.model_config, '$._delegate_from') IS NULL",
    "CREATE VIRTUAL TABLE messages_fts_trigram USING fts5( content, tool_name, content='messages_fts_trigram_src', "
    "content_rowid='id', tokenize='trigram' )",
)
_REBUILD = "INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild')"
_SESSIONS = ((SESSION_A, "cli", "{}", "A"), (SESSION_B, "telegram", "{}", "B"), (SESSION_CRON, "cron", "{}", "C"))

NOTIFICATION = "[IMPORTANT: Background process finished] 증분 정리 완료"


def _calls(name: str, arguments: dict) -> str:
    return json.dumps([{"id": "call_1", "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}])


_ROWS = (
    (1, SESSION_A, "user", "인덱스를 만들었다", None, None, None, T0),
    (2, SESSION_A, "assistant", "증분 백업을 고쳤다", None, None, None, T0 + 10),
    (3, SESSION_A, "user", "백업 고침으로 끝냈다", None, None, None, T0 + 20),
    (4, SESSION_A, "assistant", "검색이 느려서 다시 본다", None, None, None, T0 + 30),
    (5, SESSION_A, "user", "백업", None, None, None, T0 + 40),
    (6, SESSION_B, "assistant", "running the backup now", None,
     _calls("terminal", {"command": "zipcopy 인덱스를 백업"}), None, T0 - DAY),
    (7, SESSION_B, "tool", "toolresultword 인덱스를 저장 백업 완료", "call_1", None, "terminal", T0 - DAY + 10),
    (8, SESSION_B, "user", "The Backup failed twice", None, None, None, T0 - DAY + 20),
    (9, SESSION_B, "user", NOTIFICATION, None, None, None, T0 - DAY + 30),
    (10, SESSION_CRON, "user", "cron 인덱스를 돌렸다", None, None, None, T0 - DAY + 40),
)


def _build(path: Path, *, with_trigram: bool = True) -> Path:
    connection = sqlite3.connect(path)
    try:
        for statement in _SCHEMA if with_trigram else _SCHEMA[:2]:
            connection.execute(statement)
        connection.executemany("INSERT INTO sessions VALUES (?, ?, ?, ?)", _SESSIONS)
        connection.executemany("INSERT INTO messages (id, session_id, role, content, tool_call_id, tool_calls, "
                               "tool_name, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", _ROWS)
        if with_trigram:
            connection.execute(_REBUILD)
        connection.commit()
    finally:
        connection.close()
    return path


def _tools(home: Path) -> NeospikeTools:
    NeospikeConfig().save(home)
    return NeospikeTools(home, NS)


@pytest.fixture()
def tools(tmp_path: Path) -> NeospikeTools:
    home = tmp_path / "home"
    home.mkdir()
    _build(home / "state.db")
    return _tools(home)


def _ids(tools: NeospikeTools, query: str, **options) -> list[int]:
    return [item["message_id"] for item in json.loads(tools.chat_search(query, **options))["results"]]


def _error(tools: NeospikeTools, query: str) -> dict:
    with pytest.raises(ToolError) as caught:
        tools.chat_search(query)
    return json.loads(str(caught.value))


def test_korean_three_characters_find_a_word_with_a_particle(tools: NeospikeTools) -> None:
    assert _ids(tools, "인덱스") == [1]  # rows 6 (call arguments), 7 (tool result) and 10 (cron) are not indexed


def test_korean_two_characters_pad_front_and_back(tools: NeospikeTools) -> None:
    # " 백업" finds row 2, "백업 " finds row 3 at the very start of its text.
    assert _ids(tools, "백업") == [3, 2]


def test_korean_two_characters_limits(tools: NeospikeTools) -> None:
    assert _ids(tools, "검색") == []  # at the start of the text and followed by a particle
    assert 5 not in _ids(tools, "백업")  # the whole text is the one word


def test_padding_of_terms() -> None:
    assert chat_history.fts_term("백업") == '(" 백업" OR "백업 ")'
    assert chat_history.fts_term("a") == '("  a" OR "a  ")'
    assert chat_history.fts_term("인덱스") == '"인덱스"'
    assert chat_history.pad_fts_expression("백업 OR 인덱스") == '(" 백업" OR "백업 ") OR 인덱스'
    assert chat_history.pad_fts_expression('content: "ab"') == 'content: (" ab" OR "ab ")'
    assert chat_history.pad_fts_expression("ab*") == "ab*"
    assert chat_history.rewrite_fts_query("0.61 백업") == '"0.61" OR (" 백업" OR "백업 ")'


def test_english_search_is_case_insensitive_substring(tools: NeospikeTools) -> None:
    assert _ids(tools, "backup") == [8, 6]  # "The Backup" and "the backup now"
    assert _ids(tools, "failed twice") == [8]  # implicit AND of two terms


def test_tool_results_and_tool_call_arguments_are_not_found(tools: NeospikeTools) -> None:
    assert _ids(tools, "toolresultword") == []
    assert _ids(tools, "zipcopy") == []


def test_tool_name_filter_uses_the_messages_columns(tools: NeospikeTools) -> None:
    assert _ids(tools, "running", tool_name="terminal") == [6]
    assert _ids(tools, "backup", tool_name="terminal") == [6]
    assert _ids(tools, "backup", tool_name="note_write") == []


def test_since_until_and_order(tools: NeospikeTools) -> None:
    assert _ids(tools, "백업", since=T0 + 15) == [3]
    assert _ids(tools, "백업", until=T0 + 15) == [2]
    assert _ids(tools, "백업", order="recent") == [3, 2]
    assert sorted(_ids(tools, "백업", order="rank")) == [2, 3]


def test_exclude_notifications(tools: NeospikeTools) -> None:
    assert _ids(tools, "증분") == [2, 9]
    assert _ids(tools, "증분", exclude_notifications=True) == [2]


def test_tool_calls_column_filter_is_an_error(tools: NeospikeTools) -> None:
    for query in ("tool_calls: zipcopy", "{content tool_calls}: zipcopy", "-tool_calls: zipcopy"):
        failure = _error(tools, query)
        assert failure["error"] == "invalid" and "tool_calls" in failure["message"], query
    assert _ids(tools, "content: 백업") == [3, 2]


def test_rejected_query_is_retried_with_padding(tools: NeospikeTools) -> None:
    payload = json.loads(tools.chat_search("백업 ("))
    assert payload["query_rewritten"] == '(" 백업" OR "백업 ")'
    assert [item["message_id"] for item in payload["results"]] == [3, 2]


def test_missing_trigram_index_is_an_error(tmp_path: Path) -> None:
    home = tmp_path / "old"
    home.mkdir()
    _build(home / "state.db", with_trigram=False)
    failure = _error(_tools(home), "인덱스")
    assert failure["error"] == "unavailable" and "messages_fts_trigram" in failure["message"]

