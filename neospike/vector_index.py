"""Vector search of the items vectors (memories and document parts; v22 one table) through the embedding service.

The faiss HNSW index lives in the embedding service (op 'search', vector_search.py), one per database file and
table, shared by every process that searches that file. similarities() is a drop-in for dot_similarities(): the
caller still picks the allowed rows with its own SQL; only the HNSW candidates among them (every row at or above the
caller's cosine floor, or its top `limit`) get a cosine (exact, from their stored blob) and every other row gets
NaN, which every caller already skips (math.isfinite). The service reads the file in its own read-only connection,
so it sees committed rows only. There is no other path: a database without a file (in memory) or a service that
cannot answer raises EmbeddingUnavailable, as a failed embedding does.
"""
from __future__ import annotations

import math
import os
import sqlite3
from collections.abc import Sequence

SEARCH_OP = "search"
DEFAULT_COLUMN = "embedding"
_QUERY_BATCH = 16  # query vectors per request: 16 x 1,024 floats stay well under the frame cap
_NO_FILE = "Vector search needs a database file (an in-memory database has none)"


def _database_file(connection: sqlite3.Connection) -> str:
    for row in connection.execute("PRAGMA database_list"):
        if row[1] == "main":
            return os.path.normcase(os.path.realpath(row[2])) if row[2] else ""
    return ""


def _search_client():
    """Client of the shared embedding service of this process's Hermes home."""
    from ._wincompat import default_hermes_home
    from .inference import EmbeddingClient
    return EmbeddingClient(default_hermes_home())


def _scored(ids: list[int], found: dict) -> list[float]:
    scored = dict(zip(found["ids"], found["scores"]))
    return [math.nan if scored.get(rowid) is None else float(scored[rowid]) for rowid in ids]


def similarities_many(connection: sqlite3.Connection, table: str, ids: Sequence[int],
                      queries: Sequence[Sequence[float]], *, threshold: float | None = None,
                      limit: int | None = None, column: str = DEFAULT_COLUMN) -> list[list[float]]:
    """similarities() for several query vectors at once against the same allowed rows: one list per query."""
    from .inference import EmbeddingUnavailable
    ids = [int(rowid) for rowid in ids]
    vectors = [[float(value) for value in query] for query in queries]
    if not ids or not vectors:
        return [[] if not ids else [math.nan] * len(ids) for _ in vectors]
    path = _database_file(connection)
    if not path:
        raise EmbeddingUnavailable(_NO_FILE)
    bound = {"threshold": None if threshold is None else float(threshold),
             "limit": None if limit is None else int(limit)}
    client = _search_client()
    out: list[list[float]] = []
    try:
        for start in range(0, len(vectors), _QUERY_BATCH):
            found = client.request(SEARCH_OP, path=path, table=table, column=column, ids=ids,
                                   vectors=vectors[start:start + _QUERY_BATCH], **bound)
            out.extend(_scored(ids, result) for result in found["results"])
    except ValueError as error:  # a request over the frame cap (too many ids)
        raise EmbeddingUnavailable(str(error)) from None
    return out


def similarities(connection: sqlite3.Connection, table: str, ids: Sequence[int], query: Sequence[float], *,
                 threshold: float | None = None, limit: int | None = None,
                 column: str = DEFAULT_COLUMN) -> list[float]:
    """Cosine per id in order, as dot_similarities over those rows' blobs (column: the vector column). Pass the
    caller's own cosine floor (threshold) or, for a plain top-N caller, its N (limit). Just the HNSW candidates
    have a value, the rest are NaN (every caller skips non-finite)."""
    if not ids:
        return []
    return similarities_many(connection, table, ids, [query], threshold=threshold, limit=limit, column=column)[0]
