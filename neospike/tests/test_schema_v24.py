"""Schema v24: the v23 -> v24 migration drops the copied conversation tables (in place on open and through
schema-migrate), keeps every other table's rows, rolls back on a failed check, and VACUUMs the file afterwards."""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from neospike import cli, schema_v22, schema_v24
from neospike import folders as folder_family
from neospike.store import SCHEMA_VERSION, MemoryStore

DIM = 4
OWNER = "default:local"
CONVERSATION_ROWS = {"conversations": 2, "conversation_messages": 3, "conversation_chunks": 2}
CONVERSATION_TOMBSTONES = 2
# Bulk transcript rows for the file-size checks: about 4 MB of text the VACUUM must give back.
BULK_MESSAGES, BULK_CHARS = 2000, 2000
# The conversation tables as v6..v23 created them (columns, indexes and the FTS table the migration must drop).
V23_CONVERSATION_DDL = """
CREATE TABLE conversations (id INTEGER PRIMARY KEY, session_id TEXT NOT NULL UNIQUE, started_at REAL NOT NULL,
    ended_at REAL, last_message_at REAL NOT NULL, title TEXT NOT NULL DEFAULT '', summary_note_id INTEGER,
    indexed_seq INTEGER NOT NULL DEFAULT 0, updated_at REAL, synced_through REAL, source_max_id INTEGER);
CREATE TABLE conversation_messages (id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    seq INTEGER NOT NULL, role TEXT NOT NULL CHECK (role IN ('user','assistant')), content TEXT NOT NULL,
    created_at REAL NOT NULL, masked_count INTEGER NOT NULL DEFAULT 0, updated_at REAL, UNIQUE(conversation_id, seq));
CREATE TABLE conversation_chunks (id INTEGER PRIMARY KEY, conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    seq_start INTEGER NOT NULL, seq_end INTEGER NOT NULL, part INTEGER NOT NULL DEFAULT 0, text TEXT NOT NULL,
    prefix_chars INTEGER NOT NULL DEFAULT 0, text_hash TEXT NOT NULL, embedding BLOB, updated_at REAL, next_id INTEGER,
    UNIQUE(conversation_id, seq_start, part));
CREATE INDEX conversation_chunks_hash ON conversation_chunks(text_hash);
CREATE INDEX conversation_chunks_pending ON conversation_chunks(id) WHERE embedding IS NULL;
CREATE INDEX conversations_last_message ON conversations(last_message_at);
CREATE VIRTUAL TABLE conversation_chunks_fts USING fts5(
    text, content='conversation_chunks', content_rowid='id', tokenize='unicode61');
"""


def _v23(path: Path) -> None:
    """A v23 database: memories, a folder, a deleted memory's tombstone, and the conversation copy."""
    store = MemoryStore(path, dimensions=DIM)
    try:
        kept = store.insert(OWNER, "Falcon deploys on Sunday.", [1.0, 0.0, 0.0, 0.0], source="agent:s1")
        store.insert(OWNER, "Heron ships on Monday.", [0.0, 1.0, 0.0, 0.0], source="agent:s1")
        gone = store.insert(OWNER, "Ibis is retired.", [0.0, 0.0, 1.0, 0.0], source="agent:s1")
        folder = folder_family.create_folder(store._conn, "Sea")
        folder_family.move_items(store._conn, folder["id"], [("memory", kept)])
        with store._lock, store._conn:
            store._conn.execute("DELETE FROM items WHERE id=?", (gone,))  # the trigger writes an items tombstone
    finally:
        store.close()
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.executescript(V23_CONVERSATION_DDL + schema_v24.LEGACY_CONVERSATION_TRIGGERS_DDL)
            connection.executemany("INSERT INTO conversations(id,session_id,started_at,last_message_at,title) "
                                   "VALUES (?,?,1.0,2.0,'t')", [(1, "s1"), (2, "s2")])
            connection.executemany("INSERT INTO conversation_messages(conversation_id,seq,role,content,created_at) "
                                   "VALUES (?,?,?,?,1.0)", [(1, 1, "user", "q"), (1, 2, "assistant", "a"),
                                                            (2, 1, "user", "q2")])
            connection.executemany("INSERT INTO conversation_chunks(conversation_id,seq_start,seq_end,text,text_hash) "
                                   "VALUES (?,?,?,?,?)", [(1, 1, 2, "user: q assistant: a", "h1"),
                                                          (2, 1, 1, "user: q2", "h2")])
            connection.executemany("INSERT INTO tombstones(table_name,row_key,deleted_at) VALUES (?,?,1.0)",
                                   [("conversations", "3"), ("conversation_messages", "9")])
            connection.execute("UPDATE meta_entries SET value='23' WHERE key='schema_version'")
        connection.execute("PRAGMA user_version=23")


def _add_bulk_messages(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executemany("INSERT INTO conversation_messages(conversation_id,seq,role,content,created_at) "
                               "VALUES (2,?,'assistant',?,1.0)",
                               [(seq, f"{seq:06d}" + "x" * BULK_CHARS) for seq in range(10, 10 + BULK_MESSAGES)])
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # the rows sit in the main file before the migration


def _file_size(path: Path) -> int:
    """Main file plus its -wal (what the data takes on disk), read with no connection open."""
    wal = path.with_name(path.name + "-wal")
    return path.stat().st_size + (wal.stat().st_size if wal.exists() else 0)


def _counts(path: Path) -> dict[str, int]:
    with closing(sqlite3.connect(path)) as connection:
        return schema_v22.table_counts(connection)


def _names(path: Path) -> set[str]:
    with closing(sqlite3.connect(path)) as connection:
        return {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master")}


def _versions(path: Path) -> tuple[str, int]:
    with closing(sqlite3.connect(path)) as connection:
        meta = connection.execute("SELECT value FROM meta_entries WHERE key='schema_version'").fetchone()[0]
        return str(meta), int(connection.execute("PRAGMA user_version").fetchone()[0])


def _expected_after(before: dict[str, int]) -> dict[str, int]:
    expected = {table: count for table, count in before.items() if table not in schema_v24.DROPPED_NAMES}
    expected["tombstones"] -= CONVERSATION_TOMBSTONES
    return expected


def _no_conversation_names(path: Path) -> bool:
    return not [name for name in _names(path) if name.startswith("conversation")]


@pytest.fixture
def bulky_v23_db(tmp_path: Path) -> Path:
    path = tmp_path / "bulky" / "neospike" / "memory.db"
    path.parent.mkdir(parents=True)
    _v23(path)
    _add_bulk_messages(path)
    return path


@pytest.fixture
def v23_db(tmp_path: Path) -> Path:
    path = tmp_path / "home" / "neospike" / "memory.db"
    path.parent.mkdir(parents=True)
    _v23(path)
    return path


def test_fixture_is_a_v23_database_with_the_conversation_copy(v23_db: Path) -> None:
    before = _counts(v23_db)
    assert {table: before[table] for table in CONVERSATION_ROWS} == CONVERSATION_ROWS
    assert before["items"] == 2 and before["folder_items"] == 1 and before["tombstones"] == 1 + CONVERSATION_TOMBSTONES
    assert {"conversation_chunks_fts", "conversation_chunks_fts_ai", "conversations_last_message"} <= _names(v23_db)
    assert _versions(v23_db) == ("23", 23)


def test_open_in_place_drops_the_conversation_tables_and_keeps_every_other_row(v23_db: Path) -> None:
    before = _counts(v23_db)
    store = MemoryStore(v23_db, dimensions=DIM)
    try:
        assert [hit.id for hit in store.search(OWNER, "x", [1.0, 0.0, 0.0, 0.0], limit=1)] == [1]
    finally:
        store.close()
    assert _no_conversation_names(v23_db)
    assert _counts(v23_db) == _expected_after(before)
    assert _versions(v23_db) == (str(SCHEMA_VERSION), schema_v24.SCHEMA_V24) == ("24", 24)
    snapshot = v23_db.with_name("memory.pre-v24.db")  # the step snapshot of the open path keeps the old copy
    assert _counts(snapshot) == before and _versions(snapshot) == ("23", 23)
    MemoryStore(v23_db, dimensions=DIM).close()  # opening a v24 database again changes nothing
    assert _counts(v23_db) == _expected_after(before)


def _run(home: Path, *arguments: str, capsys) -> tuple[int, dict]:
    code = cli.main(["--home", str(home), *arguments])
    return code, json.loads(capsys.readouterr().out)


def test_schema_migrate_dry_run_then_apply(v23_db: Path, capsys) -> None:
    home = v23_db.parent.parent
    before, raw = _counts(v23_db), v23_db.read_bytes()
    code, dry = _run(home, "schema-migrate", "--dry-run", capsys=capsys)
    assert code == 0 and dry["ok"] and dry["dry_run"] and v23_db.read_bytes() == raw  # read only
    assert dry["v24"]["dropped_rows"] == CONVERSATION_ROWS
    assert dry["v24"]["tombstones_dropped"] == CONVERSATION_TOMBSTONES
    assert dry["v24"]["rows_after"] == _expected_after(before)
    code, applied = _run(home, "schema-migrate", "--apply", capsys=capsys)
    assert code == 0 and applied["applied"] and applied["backup"] == "memory.pre-v24.db"
    assert applied["v24"]["migrated"] and applied["v24"]["quick_check"] == "ok"
    assert applied["v24"]["vacuum"] == schema_v24.VACUUM_DONE
    assert applied["v24"]["fts_integrity"] == "ok"
    assert _no_conversation_names(v23_db) and _counts(v23_db) == _expected_after(before)
    assert _versions(v23_db) == ("24", 24)
    assert _counts(v23_db.with_name("memory.pre-v24.db")) == before
    assert _run(home, "schema-migrate", "--apply", capsys=capsys) == (0, {"applied": False, "reason": "already v24"})


def test_open_in_place_vacuums_and_the_file_shrinks(bulky_v23_db: Path) -> None:
    before = _file_size(bulky_v23_db)
    assert before > BULK_MESSAGES * BULK_CHARS
    MemoryStore(bulky_v23_db, dimensions=DIM).close()
    after = _file_size(bulky_v23_db)
    assert _no_conversation_names(bulky_v23_db) and after < before - BULK_MESSAGES * BULK_CHARS // 2, (before, after)


def test_schema_migrate_apply_vacuums_and_the_file_shrinks(bulky_v23_db: Path, capsys) -> None:
    before = _file_size(bulky_v23_db)
    code, applied = _run(bulky_v23_db.parent.parent, "schema-migrate", "--apply", capsys=capsys)
    assert code == 0 and applied["v24"]["vacuum"] == schema_v24.VACUUM_DONE
    after = _file_size(bulky_v23_db)
    assert _no_conversation_names(bulky_v23_db) and after < before - BULK_MESSAGES * BULK_CHARS // 2, (before, after)


class _NoVacuumConnection:
    """A connection whose VACUUM fails (every other call goes to the real connection)."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def execute(self, sql: str, *parameters):
        if sql.strip().upper() == "VACUUM":
            raise sqlite3.OperationalError("database is locked")
        return self._connection.execute(sql, *parameters)

    def __getattr__(self, name: str):
        return getattr(self._connection, name)


def test_a_failed_vacuum_keeps_the_migration_and_warns(v23_db: Path, caplog) -> None:
    with closing(sqlite3.connect(v23_db)) as connection, caplog.at_level(logging.WARNING):
        report = schema_v24.migrate_v24(_NoVacuumConnection(connection))
    assert report["migrated"] and report["vacuum"].startswith("failed: OperationalError")
    assert "VACUUM after the v24 migration failed" in caplog.text
    assert _no_conversation_names(v23_db) and _versions(v23_db) == ("24", 24)


class _OffByOneCounts:
    """table_counts that reports one items row too many after the drop (a failed check)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, connection: sqlite3.Connection) -> dict[str, int]:
        self.calls += 1
        counts = schema_v22.table_counts(connection)
        if self.calls == 2:
            counts["items"] += 1
        return counts


def test_a_failed_check_rolls_everything_back(v23_db: Path, monkeypatch) -> None:
    before = _counts(v23_db)
    monkeypatch.setattr(schema_v24, "table_counts", _OffByOneCounts())
    with closing(sqlite3.connect(v23_db)) as connection:
        with pytest.raises(schema_v22.SchemaError, match="row counts"):
            schema_v24.migrate_v24(connection)
        assert not connection.in_transaction
    assert _counts(v23_db) == before and _versions(v23_db) == ("23", 23)
    assert "conversation_chunks_fts" in _names(v23_db)


def test_a_v22_database_takes_v23_then_v24(v23_db: Path) -> None:
    with closing(sqlite3.connect(v23_db)) as connection:
        with connection:
            connection.execute("UPDATE meta_entries SET value='22' WHERE key='schema_version'")
        report = schema_v24.migrate_to_v24(connection)
    assert report["v23"]["migrated"] and report["v24"]["migrated"]
    assert _no_conversation_names(v23_db) and _versions(v23_db) == ("24", 24)


def test_a_new_database_is_created_at_v24_without_conversation_tables(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    MemoryStore(path, dimensions=DIM).close()
    assert _no_conversation_names(path) and _versions(path) == ("24", 24)
