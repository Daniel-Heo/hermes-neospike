"""Backup increments: items rows changed after the watermark come from the updated_at index, read-only hits stay out."""
from __future__ import annotations

import json
import sqlite3
import zipfile
from contextlib import closing
from pathlib import Path

import pytest

from neospike import backup as db_backup
from neospike.store import ITEMS_UPDATED_INDEX, MemoryStore
from neospike.tests.test_backup import NAMESPACE, _backup, _vector
from neospike.tests.test_notes import DIMENSIONS

ITEMS_FROM = 'FROM "items"'


@pytest.fixture
def env(tmp_path: Path):
    store = MemoryStore(tmp_path / "live" / "memory.db", dimensions=DIMENSIONS)
    yield store, None, tmp_path / "backups"
    store.close()


class _StatementTrace:
    """sqlite3 trace callback that keeps every expanded statement."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    def __call__(self, statement: str) -> None:
        self.statements.append(statement)


def _increment_plan(db: Path, since: float) -> list[str]:
    """EXPLAIN QUERY PLAN details of the statement the increment runs for items (taken from the trace)."""
    with closing(sqlite3.connect(db)) as connection:
        tracer = _StatementTrace()
        connection.set_trace_callback(tracer)
        db_backup.changed_rows(connection, "items", since).fetchall()
        connection.set_trace_callback(None)
        [statement] = [sql for sql in tracer.statements if sql.lstrip().upper().startswith("SELECT") and ITEMS_FROM in sql]
        assert "ORDER BY" not in statement.upper(), statement
        return [row[3] for row in connection.execute(f"EXPLAIN QUERY PLAN {statement}")]


def _last_at(store) -> float:
    return float(store._conn.execute("SELECT value FROM meta_entries WHERE key=?", (db_backup.META_LAST_AT,)).fetchone()[0])


def test_items_increment_query_uses_updated_index(env) -> None:
    store, _, directory = env
    for seed in range(5):
        store.insert(NAMESPACE, f"The user keeps plant number {seed}.", _vector(seed), source="agent:s1")
    assert _backup(store, directory)["kind"] == "full"
    details = _increment_plan(store.path, _last_at(store))
    assert any(ITEMS_UPDATED_INDEX in detail for detail in details), details
    assert not any(detail.startswith("SCAN items") for detail in details), details


def test_read_only_change_stays_out_edit_goes_in(env) -> None:
    store, _, directory = env
    read_id = store.insert(NAMESPACE, "The user waters plants at dawn.", _vector(1), source="agent:s1")
    edited_id = store.insert(NAMESPACE, "The user dislikes slugs.", _vector(2), source="agent:s1")
    assert _backup(store, directory)["kind"] == "full"
    with store._lock, store._conn:  # read bookkeeping after the watermark, updated_at untouched
        store._conn.execute("UPDATE items SET accessed_at=?, access_count=access_count+1 WHERE id=?",
                            (_last_at(store) + 1.0, read_id))
    assert _backup(store, directory) == {"created": False}
    with store._lock, store._conn:  # an edit: the tracking trigger bumps updated_at
        store._conn.execute("UPDATE items SET importance=0.9 WHERE id=?", (edited_id,))
    increment = _backup(store, directory)
    assert increment["kind"] == "incremental" and increment["rows"]["items"] == 1
    with zipfile.ZipFile(increment["path"]) as archive:
        ids = [json.loads(line)["id"] for line in archive.read("items.jsonl").decode("utf-8").splitlines() if line]
    assert ids == [edited_id]
