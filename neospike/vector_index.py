"""faiss HNSW candidates for items vector search (memories and document parts; v22 one table).

SQLite stays the source. Per process and per (database file, table) one IndexIDMap2(IndexHNSWFlat) holds only
the vectors and their row ids (items.id); it is filled from the database on first use and
never written to disk. similarities() is a drop-in for dot_similarities(): the caller still picks the allowed rows
with its own SQL; for a database file, however few they are, only the HNSW candidates among them (every row at or
above the caller's cosine floor, or its top `limit`) get a cosine (exact, from their stored blob) and every other
row gets NaN, which every caller already skips (math.isfinite). An in-memory database (no file path) has no index:
every allowed row is compared.

Changes by any process: a row id the index lacks is added on the next search; a vector changed in place is found
by SQL (items.updated_at) and added again under the same id
(HNSW cannot remove: the old copy stays, its hits are rescored with the current blob). The index is built again
when the copies no longer in the database exceed DEAD_RATIO of it or meta embedding_model/dimensions change.
faiss and numpy are imported inside the functions (plain store readers do not load them).
"""
from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
from collections.abc import Sequence

EXACT_LIMIT = 0  # allowed rows at or below this: full comparison; 0 = a database file always takes HNSW
DEAD_RATIO = 0.3  # stale copies / index size above this: build again
_HNSW_M = 32
_EF_CONSTRUCTION = 200
_EF_SEARCH = 512
_RADIUS_MARGIN = 1e-4  # float32 scores vs the callers' float64 floor: take a little more, the exact rescoring cuts
_STAMP_WINDOW_SECONDS = 5.0  # a write stamped up to this long before its commit is still found by the next check
_SQL_BATCH = 500
_F32_WIDTH = 4

# Rows whose vector may have changed since a time: (id, embedding); {vector_rows} = _vector_rows_sql().
_CHANGED_SQL = {
    "items": "SELECT id,embedding FROM items WHERE updated_at>? AND {vector_rows}",
}

_registry: dict[tuple[str, str], _TableIndex] = {}
_registry_lock = threading.Lock()


class _TableIndex:
    """One table's HNSW index of one database file (vectors + row ids only) and its sync state."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.index = None
        self.meta: tuple[str, str] | None = None
        self.synced_at = 0.0  # wall clock of the last build/sync start


def _database_file(connection: sqlite3.Connection) -> str:
    for row in connection.execute("PRAGMA database_list"):
        if row[1] == "main":
            return os.path.normcase(os.path.realpath(row[2])) if row[2] else ""
    return ""


def _vector_rows_sql() -> str:
    """SQL condition, one `?` = the byte width: a stored vector (the present-vector partial index takes it) of
    exactly this width (a row of another width never reaches the reshape in _add or the indexed-copy compare)."""
    from .store import vector_present_sql  # store imports this module
    return f"{vector_present_sql()} AND length(embedding)=?"


def _blobs(connection: sqlite3.Connection, table: str, ids: Sequence[int]) -> dict[int, bytes]:
    found: dict[int, bytes] = {}
    for start in range(0, len(ids), _SQL_BATCH):
        batch = ids[start:start + _SQL_BATCH]
        for row in connection.execute(
                f"SELECT id,embedding FROM {table} WHERE id IN ({','.join('?' * len(batch))})", batch):
            found[int(row[0])] = row[1]
    return found


def _embedding_meta(connection: sqlite3.Connection) -> tuple[str, str]:
    values = dict(connection.execute(
        "SELECT key,value FROM meta_entries WHERE key IN ('embedding_model','embedding_dimensions')").fetchall())
    return str(values.get("embedding_model", "")), str(values.get("embedding_dimensions", ""))


def _add(state: _TableIndex, pairs: Sequence[tuple[int, bytes]], dims: int) -> None:
    import numpy as np
    if not pairs:
        return
    vectors = np.frombuffer(b"".join(blob for _, blob in pairs), dtype=np.float32).reshape(len(pairs), dims)
    state.index.add_with_ids(vectors, np.fromiter((rowid for rowid, _ in pairs), dtype=np.int64, count=len(pairs)))


def _build(state: _TableIndex, connection: sqlite3.Connection, table: str, dims: int, meta: tuple[str, str]) -> None:
    import faiss
    hnsw = faiss.IndexHNSWFlat(dims, _HNSW_M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = _EF_CONSTRUCTION
    state.index = faiss.IndexIDMap2(hnsw)
    state.meta = meta
    state.synced_at = time.time()
    rows = connection.execute(f"SELECT id,embedding FROM {table} WHERE {_vector_rows_sql()}",
                              (dims * _F32_WIDTH,)).fetchall()
    _add(state, [(int(row[0]), row[1]) for row in rows], dims)


def _sync(state: _TableIndex, connection: sqlite3.Connection, table: str, ids: Sequence[int], dims: int) -> None:
    """Bring the index to the database: build (first use, model change, too many stale copies), re-add rows
    changed in place since the last sync, add allowed rows the index lacks."""
    import faiss
    import numpy as np
    width = dims * _F32_WIDTH
    meta = _embedding_meta(connection)
    if state.index is None or state.meta != meta or state.index.d != dims:
        _build(state, connection, table, dims, meta)
        return
    vector_rows = _vector_rows_sql()
    live = int(connection.execute(f"SELECT count(*) FROM {table} WHERE {vector_rows}", (width,)).fetchone()[0])
    if state.index.ntotal - live > DEAD_RATIO * state.index.ntotal:
        _build(state, connection, table, dims, meta)
        return
    present = set(faiss.vector_to_array(state.index.id_map).tolist())
    since, state.synced_at = state.synced_at - _STAMP_WINDOW_SECONDS, time.time()
    recent = [(int(row[0]), row[1]) for row in connection.execute(_CHANGED_SQL[table].format(vector_rows=vector_rows),
                                                                               (since, width))]
    pending = [pair for pair in recent if pair[0] not in present]
    known = [pair for pair in recent if pair[0] in present]
    if known:  # same id: added again only when the stored vector differs from the indexed copy
        indexed = state.index.reconstruct_batch(np.fromiter((rowid for rowid, _ in known), dtype=np.int64))
        stored = np.frombuffer(b"".join(blob for _, blob in known), dtype=np.float32).reshape(len(known), dims)
        pending.extend(pair for pair, differs in zip(known, (indexed != stored).any(axis=1)) if differs)
    present.update(rowid for rowid, _ in pending)
    missing = [rowid for rowid in ids if rowid not in present]
    pending.extend((rowid, blob) for rowid, blob in _blobs(connection, table, missing).items()
                   if blob is not None and len(blob) == width)
    _add(state, pending, dims)


def hnsw_candidates(index, query: Sequence[float], ids: Sequence[int], threshold: float | None,
                    limit: int | None) -> list[int]:
    """Candidate extraction (one place to change), only among ids, no repeats: with a cosine threshold every row
    at or above it (range_search), else the top `limit` (search)."""
    import faiss
    import numpy as np
    selector = faiss.IDSelectorBatch(np.asarray(ids, dtype=np.int64))
    vector = np.asarray([query], dtype=np.float32)
    if threshold is not None:
        params = faiss.SearchParametersHNSW(sel=selector, efSearch=_EF_SEARCH)
        _, _, labels = index.range_search(vector, float(threshold) - _RADIUS_MARGIN, params=params)
    else:
        # An old copy (same id added again) takes at most one slot each: ask for that many more, so after the
        # repeats are merged the true top `limit` are all still there.
        stale = index.ntotal - len(np.unique(faiss.vector_to_array(index.id_map)))
        k = min(int(limit) + stale, index.ntotal)  # not len(ids): old copies of allowed ids also sit among them
        params = faiss.SearchParametersHNSW(sel=selector, efSearch=max(_EF_SEARCH, k))
        labels = index.search(vector, k, params=params)[1][0]
    return list(dict.fromkeys(int(rowid) for rowid in labels if rowid >= 0))


def similarities(connection: sqlite3.Connection, table: str, ids: Sequence[int], query: Sequence[float], *,
                 threshold: float | None = None, limit: int | None = None) -> list[float]:
    """Cosine per id in order, as dot_similarities over those rows' blobs. Pass the caller's own cosine floor
    (threshold) or, for a plain top-N caller, its N (limit). For a database file just the HNSW candidates
    (hnsw_candidates) have a value, the rest are NaN (every caller skips non-finite); an in-memory database
    compares every id."""
    from .store import dot_similarities
    ids = [int(rowid) for rowid in ids]
    path = _database_file(connection) if len(ids) > EXACT_LIMIT else ""
    if not path:
        blobs = _blobs(connection, table, ids)
        sims = dot_similarities(query, [blobs.get(rowid) for rowid in ids])
        return [sim if rowid in blobs else math.nan for rowid, sim in zip(ids, sims)]
    with _registry_lock:
        state = _registry.setdefault((path, table), _TableIndex())
    with state.lock:
        _sync(state, connection, table, ids, len(query))
        picked = hnsw_candidates(state.index, query, ids, threshold, limit)
    blobs = _blobs(connection, table, picked)
    scored = dict(zip(picked, dot_similarities(query, [blobs.get(rowid) for rowid in picked])))
    return [scored[rowid] if rowid in blobs and rowid in scored else math.nan for rowid in ids]


def drop_all() -> None:
    """Forget every cached index (tests)."""
    with _registry_lock:
        _registry.clear()
