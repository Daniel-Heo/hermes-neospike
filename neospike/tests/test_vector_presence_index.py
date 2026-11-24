"""Partial indexes on items for stored vectors and embedding-pending rows (synthetic vectors only)."""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
import pytest

from neospike import vector_index
from neospike.store import PENDING_EMBED_MARK, MemoryStore, vector_pending_sql

DIMENSIONS = 32
WIDTH = DIMENSIONS * 4
OTHER_WIDTH = 2 * WIDTH  # a vector of another model, written past the store's checks
NAMESPACE = "owner"
ROWS = 40
SEED = 517
VECTOR_INDEX = "items_vector_updated"
PENDING_INDEX = "items_embed_pending"
OLD_STAMP_SECONDS = 3600.0


@pytest.fixture(autouse=True)
def _fresh_registry():
    vector_index.drop_all()
    yield
    vector_index.drop_all()


@pytest.fixture
def store(tmp_path: Path):
    opened = MemoryStore(tmp_path / "memory.db", dimensions=DIMENSIONS)
    yield opened
    opened.close()


def _vectors(count: int) -> np.ndarray:
    rows = np.random.default_rng(SEED).normal(size=(count, DIMENSIONS)).astype(np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _insert(store: MemoryStore, label: str, blob: bytes, stamp: float) -> int:
    with store._write_scope():
        cur = store._conn.execute(
            "INSERT INTO items(type,owner,text_hash,text,origin,embedding,created_at,updated_at) "
            "VALUES ('memory',?,?,?,'agent:s',?,?,?)",
            (NAMESPACE, hashlib.sha256(label.encode()).hexdigest(), label, blob, stamp, stamp))
    return int(cur.lastrowid)


def _plan(store: MemoryStore, sql: str, args: tuple = ()) -> list[str]:
    return [str(row[3]) for row in store._conn.execute("EXPLAIN QUERY PLAN " + sql, args)]


def _indexed_ids(store: MemoryStore) -> set[int]:
    state = vector_index._registry[(vector_index._database_file(store._conn), "items")]
    return set(faiss.vector_to_array(state.index.id_map).tolist())


def _search(store: MemoryStore, ids: list[int]) -> list[float]:
    return vector_index.similarities(store._conn, "items", ids, _vectors(1)[0].tolist(), limit=5)


def _uses(plan: list[str], index: str) -> bool:
    return any(index in line for line in plan) and not any(line.startswith("SCAN items") and index not in line
                                                           for line in plan)


def test_queries_take_the_partial_indexes(store: MemoryStore) -> None:
    vector_rows = vector_index._vector_rows_sql()
    count = _plan(store, f"SELECT count(*) FROM items WHERE {vector_rows}", (WIDTH,))
    assert any(f"COVERING INDEX {VECTOR_INDEX}" in line for line in count), count
    changed = _plan(store, vector_index._CHANGED_SQL["items"].format(vector_rows=vector_rows), (0.0, WIDTH))
    assert _uses(changed, VECTOR_INDEX), changed
    pending = (f"SELECT id FROM items WHERE type='memory' AND {vector_pending_sql()} ORDER BY id",
               f"SELECT id FROM items WHERE type IN ('note','doc') AND title IS NULL AND {vector_pending_sql()} "
               "ORDER BY id")
    for sql in pending:
        plan = _plan(store, sql)
        assert _uses(plan, PENDING_INDEX), (sql, plan)


def test_pending_row_enters_the_index_once_embedded(store: MemoryStore) -> None:
    vectors = _vectors(ROWS + 1)
    stamp = time.time() - OLD_STAMP_SECONDS
    ids = [_insert(store, f"m{index}", vectors[index].tobytes(), stamp) for index in range(ROWS)]
    pending_id = _insert(store, "pending", PENDING_EMBED_MARK, stamp)
    assert [row["id"] for row in store.items_missing_embedding()] == [pending_id]
    _search(store, ids + [pending_id])
    assert _indexed_ids(store) == set(ids)
    assert store.write_missing_embedding(pending_id, vectors[ROWS].tolist())
    assert store.items_missing_embedding() == []
    _search(store, ids)  # not among the allowed ids: found through updated_at alone
    assert _indexed_ids(store) == set(ids) | {pending_id}


def test_rows_of_another_width_are_skipped(store: MemoryStore) -> None:
    vectors = _vectors(ROWS)
    stamp = time.time() - OLD_STAMP_SECONDS
    ids = [_insert(store, f"m{index}", vectors[index].tobytes(), stamp) for index in range(ROWS)]
    wide_old = _insert(store, "wide-old", b"\x00" * OTHER_WIDTH, stamp)
    _search(store, ids + [wide_old])  # build
    assert _indexed_ids(store) == set(ids)
    wide_new = _insert(store, "wide-new", b"\x00" * OTHER_WIDTH, time.time())
    sims = _search(store, ids + [wide_old, wide_new])  # sync: changed rows and allowed ids the index lacks
    assert _indexed_ids(store) == set(ids)
    assert np.isnan(sims[-1]) and np.isnan(sims[-2])


def test_existing_database_gains_the_indexes_on_open(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    MemoryStore(path, dimensions=DIMENSIONS).close()
    connection = sqlite3.connect(path)
    try:
        for name in (VECTOR_INDEX, PENDING_INDEX):
            connection.execute(f"DROP INDEX {name}")
        connection.commit()
        assert connection.execute("SELECT value FROM meta_entries WHERE key='schema_version'").fetchone()[0] == "24"
        assert not connection.execute("SELECT name FROM sqlite_master WHERE type='index' AND name IN (?,?)",
                                      (VECTOR_INDEX, PENDING_INDEX)).fetchall()
    finally:
        connection.close()
    reopened = MemoryStore(path, dimensions=DIMENSIONS)
    try:
        names = {str(row[0]) for row in reopened._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name IN (?,?)", (VECTOR_INDEX, PENDING_INDEX))}
    finally:
        reopened.close()
    assert names == {VECTOR_INDEX, PENDING_INDEX}
