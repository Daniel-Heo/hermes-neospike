"""The review inbox does not exist: no table, no daily stage, no tool, no proposal."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

from neospike import topics
from neospike.notes import ORIGIN_AUTOMATION, NotePermissionError, NoteService
from neospike.provider import NeospikeProvider
from neospike.store import SCHEMA_VERSION, MemoryStore
from neospike.tests.support_provider import ThreeTopicEmbedder as FakeEmbedder
from neospike.tests.test_notes import DIMENSIONS, BowEmbedder
from neospike.tests.test_topics import NS, _mix, _remember, _store

GONE_TABLES = ("candidates", "inbox_decisions", "lint_state")
GONE_STAGES = ("inbox-maintain", "lint", "inbox-review-code")
GONE_TOOLS = ("neospike_review", "neospike_approve", "neospike_reject")
DAILY_SCRIPT = Path(__file__).resolve().parents[1] / "cron" / "neospike_daily.py"
V20 = 20
# The pre-v21 shape of the three tables (an older store.py) with one row each.
V20_TABLES = """
CREATE TABLE candidates (id INTEGER PRIMARY KEY, owner TEXT NOT NULL, text TEXT NOT NULL, kind TEXT NOT NULL,
    confidence REAL NOT NULL, origin TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', created_at REAL NOT NULL,
    updated_at REAL, UNIQUE(owner, text, status));
CREATE TABLE inbox_decisions (id INTEGER PRIMARY KEY, owner TEXT NOT NULL, candidate_id INTEGER NOT NULL,
    kind TEXT NOT NULL, decision TEXT NOT NULL, decided_by TEXT NOT NULL, decided_at REAL NOT NULL);
CREATE TABLE lint_state (note_id INTEGER PRIMARY KEY, checked_hash TEXT NOT NULL, checked_at REAL NOT NULL);
INSERT INTO candidates(owner,text,kind,confidence,origin,created_at) VALUES ('ns','orphan','lint_orphan',1,'lint:note:1',1);
INSERT INTO inbox_decisions(owner,candidate_id,kind,decision,decided_by,decided_at) VALUES ('ns',1,'topic_new','auto_applied','code',1);
INSERT INTO lint_state VALUES (1,'h',1);
INSERT INTO tombstones(table_name,row_key,deleted_at) VALUES ('candidates','9',1),('lint_state','9',1);
"""


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _row_counts(connection: sqlite3.Connection, tables: set[str]) -> dict[str, int]:
    return {table: int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]) for table in sorted(tables)}


# (a) ------------------------------------------------------------------------------------------------------
def test_migration_drops_the_three_tables_and_keeps_other_rows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _remember(store, "kept memory", _mix(0.9, 0, 1))
    path, dimensions = store.path, store.dimensions
    store.close()
    with closing(sqlite3.connect(path)) as connection, connection:
        for table in GONE_TABLES:
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        connection.executescript(V20_TABLES)
        connection.execute("UPDATE meta_entries SET value=? WHERE key='schema_version'", (str(V20),))
    with closing(sqlite3.connect(path)) as connection:
        before = _row_counts(connection, _tables(connection) - set(GONE_TABLES))
        gone_tombstones = int(connection.execute(
            "SELECT COUNT(*) FROM tombstones WHERE table_name IN ('candidates','lint_state')").fetchone()[0])
    MemoryStore(path, dimensions=dimensions).close()
    with closing(sqlite3.connect(path)) as connection:
        assert not _tables(connection) & set(GONE_TABLES)
        after = _row_counts(connection, _tables(connection))
        version = connection.execute("SELECT value FROM meta_entries WHERE key='schema_version'").fetchone()[0]
    assert SCHEMA_VERSION == 23 and version == "23"  # v23 (the preferences table) is the current schema
    # Only the tombstones of the dropped tables go (their rows can never be restored again).
    assert after == {**before, "tombstones": before["tombstones"] - gone_tombstones}
    assert before["items"] == 1


# (b) ------------------------------------------------------------------------------------------------------
def test_daily_stages_have_no_inbox_or_lint_stage(tmp_path: Path) -> None:
    sys.path.insert(0, str(DAILY_SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("daily_no_inbox", DAILY_SCRIPT)
        daily = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(daily)
    finally:
        sys.path.remove(str(DAILY_SCRIPT.parent))
    names = [name for name, _ in daily.stages(tmp_path)]
    commands = [args[0] for _, args in daily.stages(tmp_path)]
    assert names and not set(GONE_STAGES) & set(names)
    assert not {"inbox-maintain", "lint", "inbox-review-input"} & set(commands)


# (c) ------------------------------------------------------------------------------------------------------
def test_automation_edit_of_a_human_note_is_refused(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db", dimensions=DIMENSIONS)
    try:
        notes = NoteService(store, BowEmbedder())
        human = notes.save("Plan", "# Plan\nWritten by a person.", author="human")
        before = notes.read(note_id=human["id"])
        with pytest.raises(NotePermissionError) as refused:
            notes.save("Plan", "# Plan\nAutomation rewrite.", author="agent", origin=ORIGIN_AUTOMATION,
                       expected_rev=human["rev"], namespace=NS)
        assert refused.value.as_dict()["error"] == "forbidden"
        after = notes.read(note_id=human["id"])
        assert (after["rev"], after["body"]) == (before["rev"], before["body"])
        assert "candidates" not in _tables(store._conn)
    finally:
        store.close()


# (d) ------------------------------------------------------------------------------------------------------
def test_topic_discover_creates_topics_without_inbox_rows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        ids = [_remember(store, f"otter {axis}", _mix(0.95, 0, axis), subject="otter") for axis in range(1, 7)]
        result = topics.discover(store, NS)
        assert result["topic_new"] == 1 and len(result["topic_ids"]) == 1
        members = [int(row[0]) for row in store._conn.execute(
            "SELECT src_id FROM item_links WHERE relation='topic' AND label=? ORDER BY src_id", (result["topic_ids"][0],))]
        assert members == ids
        assert not _tables(store._conn) & set(GONE_TABLES)
    finally:
        store.close()


# (e) ------------------------------------------------------------------------------------------------------
def test_provider_has_no_review_tools() -> None:
    names = [schema["name"] for schema in NeospikeProvider(embedder=FakeEmbedder()).get_tool_schemas()]
    assert not set(GONE_TOOLS) & set(names)
    assert len(names) == 9  # the memory, note, doc and maintenance tools; no image tools
