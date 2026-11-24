"""The newest-first dashboard memory list reads an index in order instead of sorting."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from neospike.dashboard import plugin_api
from neospike.store import ITEM_MEMORY, MEMORY_CREATED_INDEX, SCHEMA_VERSION, MemoryStore

DIMENSIONS = 8
LIMIT = 200
TEMP_SORT = "USE TEMP B-TREE FOR ORDER BY"

# The dashboard memory list query, as plugin_api runs it.
_KINDS = ",".join("?" * len(plugin_api.LISTED_ITEM_KINDS))
MEMORY_LIST_SQL = (f"SELECT {plugin_api._MEMORY_COLS} FROM items m WHERE m.type='{ITEM_MEMORY}' "
                   f"AND m.kind IN ({_KINDS}) ORDER BY m.created_at DESC, m.id DESC LIMIT ?")


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "memory.db"
    MemoryStore(path, dimensions=DIMENSIONS).close()
    return path


def _plan(path: Path, sql: str, parameters: tuple) -> list[str]:
    connection = sqlite3.connect(path)
    try:
        return [str(row[3]) for row in connection.execute("EXPLAIN QUERY PLAN " + sql, parameters)]
    finally:
        connection.close()


def _indexes(path: Path) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        return {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    finally:
        connection.close()


def test_memory_list_reads_the_created_index_without_a_sort(database: Path) -> None:
    plan = _plan(database, MEMORY_LIST_SQL, (*plugin_api.LISTED_ITEM_KINDS, LIMIT))
    assert not any(TEMP_SORT in line for line in plan), plan
    assert any(f"INDEX {MEMORY_CREATED_INDEX}" in line for line in plan), plan


def test_opening_an_existing_database_adds_the_index(database: Path) -> None:
    connection = sqlite3.connect(database)
    with connection:  # a database written before the index existed
        connection.execute(f"DROP INDEX {MEMORY_CREATED_INDEX}")
    connection.close()
    assert MEMORY_CREATED_INDEX not in _indexes(database)
    MemoryStore(database, dimensions=DIMENSIONS).close()
    assert MEMORY_CREATED_INDEX in _indexes(database)
    connection = sqlite3.connect(database)
    try:
        version = connection.execute("SELECT value FROM meta_entries WHERE key='schema_version'").fetchone()[0]
    finally:
        connection.close()
    assert version == str(SCHEMA_VERSION)
