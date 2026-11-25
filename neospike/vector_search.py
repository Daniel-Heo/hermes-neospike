"""faiss HNSW vector search inside the embedding service (op 'search'; its own worker thread).

SQLite stays the source. Per (database file, table, vector column) one IndexIDMap2(IndexHNSWFlat) holds only the
vectors and their row ids; it is filled from the database on first use and never written to disk. A request names
the database file, the table and column (SEARCH_SOURCES), the allowed row ids (the caller picks them with its own
SQL), one or more query vectors and either the caller's cosine floor (threshold) or its N (limit). Per query only the
HNSW candidates among the allowed ids (every row at or above the floor, or the top `limit`) get a cosine, exact from
their stored blob; the reply lists those ids and scores, and the caller gives every other id NaN.

The database is opened read-only for each request and closed after it, so the service sees committed rows only and
never keeps a profile's file open. Changes by any process: a row id the index lacks is added on the next search; a
vector changed in place is found by SQL (updated_at) and added again under the same id (HNSW cannot remove: the old
copy stays, its hits are rescored with the current blob). The index is built again when the copies no longer in the
database exceed DEAD_RATIO of it or meta embedding_model/dimensions change.
"""
from __future__ import annotations

import math
import os
import pathlib
import sqlite3
import time
from collections.abc import Sequence

DEAD_RATIO = 0.3  # stale copies / index size above this: build again
MAX_IDS = 100_000  # allowed ids per request (the frame cap also bounds it)
MAX_QUERIES = 64  # query vectors per request
MAX_DIMENSIONS = 4096  # query vector length
# (table, vector column) a search may read; every one has an updated_at that a vector write moves.
SEARCH_SOURCES = frozenset((("items", "embedding"), ("preferences", "embedding"),
                            ("preferences", "situation_embedding")))
DEFAULT_COLUMN = "embedding"
_HNSW_M = 32
_EF_CONSTRUCTION = 200
_EF_SEARCH = 512
_RADIUS_MARGIN = 1e-4  # float32 scores vs the callers' float64 floor: take a little more, the exact rescoring cuts
_STAMP_WINDOW_SECONDS = 5.0  # a write stamped up to this long before its commit is still found by the next check
_SQL_BATCH = 500
_F32_WIDTH = 4
_BUSY_TIMEOUT_SECONDS = 5.0

# Rows whose vector may have changed since a time: (id, vector); {vector_rows} = vector_rows_sql(column).
CHANGED_SQL = "SELECT id,{column} FROM {table} WHERE updated_at>? AND {vector_rows}"

_ERR = {
    "bad_path": "Vector search needs an existing database file",
    "bad_table": "Unknown vector search table or column",
    "bad_ids": "Invalid vector search ids",
    "bad_vector": "Invalid vector search query",
    "bad_bound": "Vector search needs one positive limit or one finite threshold",
}

_INT_TYPES = frozenset((int,))
_NUMBER_TYPES = frozenset((int, float))


class TableIndex:
    """One vector column's HNSW index of one database file (vectors + row ids only) and its sync state."""

    def __init__(self) -> None:
        self.index = None
        self.meta: tuple[str, str] | None = None
        self.synced_at = 0.0  # wall clock of the last build/sync start


class SearchRequest:
    """A checked 'search' request."""

    def __init__(self, path: str, table: str, column: str, ids: list[int], queries: list[list[float]],
                 threshold: float | None, limit: int | None) -> None:
        self.path, self.table, self.column, self.ids, self.queries = path, table, column, ids, queries
        self.threshold, self.limit = threshold, limit


def _is_int(value) -> bool:
    return type(value) is int


def _is_number(value) -> bool:
    return type(value) in _NUMBER_TYPES and math.isfinite(value)


def _all_ints(values: list) -> bool:
    """Every element is a plain int (not bool), checked at C speed."""
    return set(map(type, values)) <= _INT_TYPES


def _all_numbers(values: list) -> bool:
    import numpy as np
    return set(map(type, values)) <= _NUMBER_TYPES and bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def _is_query(vector, dims: int) -> bool:
    return isinstance(vector, list) and len(vector) == dims and _all_numbers(vector)


def _checked_queries(queries) -> list[list[float]]:
    if not isinstance(queries, list) or not 0 < len(queries) <= MAX_QUERIES or not isinstance(queries[0], list):
        raise ValueError(_ERR["bad_vector"])
    dims = len(queries[0])
    if not 0 < dims <= MAX_DIMENSIONS or not all(_is_query(vector, dims) for vector in queries):
        raise ValueError(_ERR["bad_vector"])
    return queries


def parse_request(request: dict) -> SearchRequest:
    """The request's fields checked; ValueError for a malformed one."""
    path = request.get("path")
    if not isinstance(path, str) or not os.path.isabs(path) or not os.path.isfile(path):
        raise ValueError(_ERR["bad_path"])
    table, column = request.get("table"), request.get("column", DEFAULT_COLUMN)
    if (table, column) not in SEARCH_SOURCES:
        raise ValueError(_ERR["bad_table"])
    ids = request.get("ids")
    if not isinstance(ids, list) or len(ids) > MAX_IDS or not _all_ints(ids):
        raise ValueError(_ERR["bad_ids"])
    queries = _checked_queries(request.get("vectors"))
    threshold, limit = request.get("threshold"), request.get("limit")
    if threshold is not None:
        if not _is_number(threshold):
            raise ValueError(_ERR["bad_bound"])
        threshold, limit = float(threshold), None
    elif not _is_int(limit) or limit <= 0:
        raise ValueError(_ERR["bad_bound"])
    return SearchRequest(path, table, column, ids, queries, threshold, limit)


def vector_rows_sql(column: str = DEFAULT_COLUMN) -> str:
    """SQL condition, one `?` = the byte width: a stored vector (the present-vector partial index takes it) of
    exactly this width (a row of another width never reaches the reshape in _add or the indexed-copy compare)."""
    from .store import vector_present_sql
    return f"{vector_present_sql(column)} AND length({column})=?"


def _open_readonly(path: str) -> sqlite3.Connection:
    return sqlite3.connect(pathlib.Path(path).as_uri() + "?mode=ro", uri=True, timeout=_BUSY_TIMEOUT_SECONDS,
                           isolation_level=None)


def _blobs(connection: sqlite3.Connection, table: str, column: str, ids: Sequence[int]) -> dict[int, bytes]:
    found: dict[int, bytes] = {}
    for start in range(0, len(ids), _SQL_BATCH):
        batch = ids[start:start + _SQL_BATCH]
        for row in connection.execute(
                f"SELECT id,{column} FROM {table} WHERE id IN ({','.join('?' * len(batch))})", batch):
            found[int(row[0])] = row[1]
    return found


def _embedding_meta(connection: sqlite3.Connection) -> tuple[str, str]:
    values = dict(connection.execute(
        "SELECT key,value FROM meta_entries WHERE key IN ('embedding_model','embedding_dimensions')").fetchall())
    return str(values.get("embedding_model", "")), str(values.get("embedding_dimensions", ""))


def _add(state: TableIndex, pairs: Sequence[tuple[int, bytes]], dims: int) -> None:
    import numpy as np
    if not pairs:
        return
    vectors = np.frombuffer(b"".join(blob for _, blob in pairs), dtype=np.float32).reshape(len(pairs), dims)
    state.index.add_with_ids(vectors, np.fromiter((rowid for rowid, _ in pairs), dtype=np.int64, count=len(pairs)))


def _build(state: TableIndex, connection: sqlite3.Connection, table: str, column: str, dims: int,
           meta: tuple[str, str]) -> None:
    import faiss
    hnsw = faiss.IndexHNSWFlat(dims, _HNSW_M, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = _EF_CONSTRUCTION
    state.index = faiss.IndexIDMap2(hnsw)
    state.meta = meta
    state.synced_at = time.time()
    rows = connection.execute(f"SELECT id,{column} FROM {table} WHERE {vector_rows_sql(column)}",
                              (dims * _F32_WIDTH,)).fetchall()
    _add(state, [(int(row[0]), row[1]) for row in rows], dims)


def _sync(state: TableIndex, connection: sqlite3.Connection, table: str, column: str, ids: Sequence[int],
          dims: int) -> None:
    """Bring the index to the database: build (first use, model change, too many stale copies), re-add rows
    changed in place since the last sync, add allowed rows the index lacks."""
    import faiss
    import numpy as np
    width = dims * _F32_WIDTH
    meta = _embedding_meta(connection)
    if state.index is None or state.meta != meta or state.index.d != dims:
        _build(state, connection, table, column, dims, meta)
        return
    vector_rows = vector_rows_sql(column)
    live = int(connection.execute(f"SELECT count(*) FROM {table} WHERE {vector_rows}", (width,)).fetchone()[0])
    if state.index.ntotal - live > DEAD_RATIO * state.index.ntotal:
        _build(state, connection, table, column, dims, meta)
        return
    present = set(faiss.vector_to_array(state.index.id_map).tolist())
    since, state.synced_at = state.synced_at - _STAMP_WINDOW_SECONDS, time.time()
    recent = [(int(row[0]), row[1]) for row in connection.execute(
        CHANGED_SQL.format(column=column, table=table, vector_rows=vector_rows), (since, width))]
    pending = [pair for pair in recent if pair[0] not in present]
    known = [pair for pair in recent if pair[0] in present]
    if known:  # same id: added again only when the stored vector differs from the indexed copy
        indexed = state.index.reconstruct_batch(np.fromiter((rowid for rowid, _ in known), dtype=np.int64))
        stored = np.frombuffer(b"".join(blob for _, blob in known), dtype=np.float32).reshape(len(known), dims)
        pending.extend(pair for pair, differs in zip(known, (indexed != stored).any(axis=1)) if differs)
    present.update(rowid for rowid, _ in pending)
    missing = [rowid for rowid in ids if rowid not in present]
    pending.extend((rowid, blob) for rowid, blob in _blobs(connection, table, column, missing).items()
                   if blob is not None and len(blob) == width)
    _add(state, pending, dims)


def hnsw_candidates(index, query: Sequence[float], ids: Sequence[int], threshold: float | None,
                    limit: int | None) -> list[int]:
    """Candidate extraction (one place to change), only among ids, no repeats: with a cosine threshold every row
    at or above it (range_search), else the top `limit` (search)."""
    import faiss
    import numpy as np
    if index.ntotal == 0:
        return []
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


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


class VectorSearcher:
    """Every index of the service, keyed by (database file, table, column). Used by the single search thread."""

    def __init__(self) -> None:
        self.indexes: dict[tuple[str, str, str], TableIndex] = {}

    def state(self, path: str, table: str = "items", column: str = DEFAULT_COLUMN) -> TableIndex | None:
        """The index of one database file and vector column, None before its first search."""
        return self.indexes.get((path, table, column))

    def search(self, request: dict) -> dict:
        """{'results': per query {'ids': candidate ids that have a stored row, 'scores': their cosines}} (a
        non-finite cosine is None: the wire carries no NaN)."""
        from .store import dot_similarities  # looked up per call: tests compare against a patched scorer
        checked = parse_request(request)
        if not checked.ids:
            return {"results": [{"ids": [], "scores": []} for _ in checked.queries]}
        state = self.indexes.setdefault((checked.path, checked.table, checked.column), TableIndex())
        connection = _open_readonly(checked.path)
        try:
            connection.execute("BEGIN")  # one snapshot for the sync and the rescoring
            _sync(state, connection, checked.table, checked.column, checked.ids, len(checked.queries[0]))
            picks = [hnsw_candidates(state.index, query, checked.ids, checked.threshold, checked.limit)
                     for query in checked.queries]
            wanted = sorted({rowid for picked in picks for rowid in picked})
            blobs = _blobs(connection, checked.table, checked.column, wanted)
        finally:
            connection.close()
        results = []
        for query, picked in zip(checked.queries, picks):
            found = [rowid for rowid in picked if rowid in blobs]
            scores = dot_similarities(query, [blobs[rowid] for rowid in found])
            results.append({"ids": found, "scores": [_finite_or_none(score) for score in scores]})
        return {"results": results}
