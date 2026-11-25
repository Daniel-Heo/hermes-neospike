"""Faiss HNSW candidates for the items vectors in the embedding service (memories and document parts; synthetic
vectors only)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import faiss
import numpy as np
import pytest

from neospike import mcp_server, notes, vector_index, vector_search
from neospike.inference import EmbeddingUnavailable
from neospike.tests.support_vector import use_full_comparison
from neospike.store import FORMULA_AUTHOR, MemoryStore, dot_similarities

DIMENSIONS = 32
CLUSTERS = 60
NOISE = 0.35
SEED = 449
NAMESPACE = "owner"
RECALL_ROWS = 6000
SYNC_ROWS = 1500
QUERY_COUNT = 30
TOP = 10
THRESHOLD = 0.75
MIN_RECALL = 0.99
SCORE_TOLERANCE = 1e-6  # freshness moves with each call's own now
FULL_DIMENSIONS = 1024
RECALL_ROWS_1024 = 5000
THRESHOLD_1024 = 0.5
STALE_CHANGES = 30
OTHER_OWNER_ROWS = 3000  # keeps the old copies under the rebuild share (DEAD_RATIO)
NEAR_REWRITES = 2  # same-id rewrites that stay next to the anchor before the last one moves away
OLD_COPY_NEIGHBOURS = 1001  # allowed rows of the old-copies test: enough that half of them changed outnumber the rest
SMALL_ALLOWED = (1, 5, 50)  # allowed sets far smaller than the index (one owner or project of a big table)
SMALL_QUERY_NOISE = 0.05
RESCORE_TOLERANCE = 1e-12  # the same cosine from a batch of another size: last-bit rounding only


def _clustered(count: int, seed: int = SEED, dims: int = DIMENSIONS) -> np.ndarray:
    generator = np.random.default_rng(seed)
    centers = generator.normal(size=(CLUSTERS, dims))
    rows = centers[generator.integers(0, CLUSTERS, count)] + NOISE * generator.normal(size=(count, dims))
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    return rows.astype(np.float32)


def _unit(vector: np.ndarray) -> np.ndarray:
    return (vector / np.linalg.norm(vector)).astype(np.float32)


def _bulk_memories(store: MemoryStore, vectors: np.ndarray, *, importance: float = 0.5, prefix: str = "m",
                   owner: str = NAMESPACE) -> list[int]:
    now = time.time()
    rows = [(owner, hashlib.sha256(f"{prefix}{index}".encode()).hexdigest(), f"{prefix} {index}", "agent:s",
             vector.tobytes(), now, "memory_item", 1.0, importance, "", json.dumps({}))
            for index, vector in enumerate(vectors)]
    with store._write_scope():
        store._conn.executemany(
            "INSERT INTO items(type,owner,text_hash,text,origin,embedding,created_at,kind,confidence,importance,"
            "project,metadata_json) VALUES ('memory',?,?,?,?,?,?,?,?,?,?,?)", rows)
    return [int(row[0]) for row in store._conn.execute("SELECT id FROM items WHERE type='memory' AND owner=? ORDER BY id",
                                                        (owner,))]


@pytest.fixture
def store(tmp_path: Path):
    opened = MemoryStore(tmp_path / "memory.db", dimensions=DIMENSIONS, semantic_threshold=THRESHOLD)
    yield opened
    opened.close()


@pytest.fixture
def store_1024(tmp_path: Path):
    opened = MemoryStore(tmp_path / "memory1024.db", dimensions=FULL_DIMENSIONS)
    yield opened
    opened.close()


def _exact(store: MemoryStore, ids: list[int], query: np.ndarray) -> list[float]:
    blobs = dict(store._conn.execute("SELECT id,embedding FROM items WHERE type='memory'").fetchall())
    return dot_similarities(query, [blobs[rowid] for rowid in ids])


def _state(service, store: MemoryStore, table: str = "items"):
    return service.state(vector_index._database_file(store._conn), table)


def _full_comparison(patch: pytest.MonkeyPatch) -> None:
    use_full_comparison(patch)


def test_recall_top_and_threshold(store: MemoryStore) -> None:
    pool = _clustered(RECALL_ROWS + QUERY_COUNT)  # queries from the same clusters, not stored
    ids = _bulk_memories(store, pool[:RECALL_ROWS])
    queries = pool[RECALL_ROWS:]
    top_hits = top_total = range_hits = range_total = 0
    for query in queries:
        exact = np.asarray(_exact(store, ids, query))
        found = vector_index.similarities(store._conn, "items", ids, query, limit=TOP)
        want = set(np.argsort(-exact)[:TOP].tolist())
        got = {index for index, sim in enumerate(found) if np.isfinite(sim)}
        top_hits, top_total = top_hits + len(want & got), top_total + TOP
        above = set(np.flatnonzero(exact >= THRESHOLD).tolist())
        ranged = vector_index.similarities(store._conn, "items", ids, query, threshold=THRESHOLD)
        got_range = {index for index, sim in enumerate(ranged) if np.isfinite(sim) and sim >= THRESHOLD}
        assert got_range <= above
        range_hits, range_total = range_hits + len(above & got_range), range_total + len(above)
    print(f"recall top{TOP}={top_hits / top_total:.4f} range={range_hits / range_total:.4f} ({range_total} rows)")
    assert top_hits / top_total >= MIN_RECALL
    assert range_total > QUERY_COUNT and range_hits / range_total >= MIN_RECALL


def test_recall_1024_dimensions(store_1024: MemoryStore) -> None:
    pool = _clustered(RECALL_ROWS_1024 + QUERY_COUNT, dims=FULL_DIMENSIONS)
    ids = _bulk_memories(store_1024, pool[:RECALL_ROWS_1024])
    top_hits = top_total = range_hits = range_total = 0
    for query in pool[RECALL_ROWS_1024:]:
        exact = np.asarray(_exact(store_1024, ids, query))
        found = vector_index.similarities(store_1024._conn, "items", ids, query, limit=TOP)
        want = set(np.argsort(-exact)[:TOP].tolist())
        top_hits += len(want & {index for index, sim in enumerate(found) if np.isfinite(sim)})
        top_total += TOP
        above = set(np.flatnonzero(exact >= THRESHOLD_1024).tolist())
        ranged = vector_index.similarities(store_1024._conn, "items", ids, query, threshold=THRESHOLD_1024)
        range_hits += len(above & {index for index, sim in enumerate(ranged) if np.isfinite(sim) and sim >= THRESHOLD_1024})
        range_total += len(above)
    print(f"recall 1024d top{TOP}={top_hits / top_total:.4f} range={range_hits / range_total:.4f} ({range_total} rows)")
    assert top_hits / top_total >= MIN_RECALL
    assert range_total > QUERY_COUNT and range_hits / range_total >= MIN_RECALL


def test_top_n_after_same_id_changes_by_other_connection(store: MemoryStore, search_service) -> None:
    """Old copies of rows changed in place (another connection) never push the true top N out."""
    pool = _clustered(SYNC_ROWS + 1)
    ids = _bulk_memories(store, pool[:SYNC_ROWS])
    anchor = ids[0]
    assert len(store.closest_memories(NAMESPACE, anchor, limit=TOP)) == TOP
    near = [hit[0] for hit in store.closest_memories(NAMESPACE, anchor, limit=TOP)]
    far = [int(index) for index in np.argsort(_exact(store, ids, pool[0]))[:STALE_CHANGES]]
    other = sqlite3.connect(store.path)
    with other:  # the nearest rows move away and far rows move next to the anchor, ids unchanged
        for rowid in near:
            other.execute("UPDATE items SET embedding=? WHERE id=?", (_unit(-pool[0]).tobytes(), rowid))
        for index in far:
            other.execute("UPDATE items SET embedding=? WHERE id=?",
                          (_unit(pool[0] + 0.05 * pool[index]).tobytes(), ids[index]))
    other.close()
    hits = store.closest_memories(NAMESPACE, anchor, limit=TOP)
    assert _state(search_service, store).index.ntotal > SYNC_ROWS  # old copies are in the index
    assert len({hit[0] for hit in hits}) == TOP
    assert not set(near) & {hit[0] for hit in hits}
    with pytest.MonkeyPatch.context() as patch:
        _full_comparison(patch)
        assert store.closest_memories(NAMESPACE, anchor, limit=TOP) == hits


def _near(anchor: np.ndarray, spread: float, count: int, seed: int) -> np.ndarray:
    noise = np.random.default_rng(seed).normal(size=(count, DIMENSIONS))
    noise *= spread / np.linalg.norm(noise, axis=1, keepdims=True)
    rows = anchor + noise
    return (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)


def test_top_n_when_old_copies_outnumber_allowed_minus_limit(store: MemoryStore, search_service) -> None:
    """One owner's 1,002 rows (1,001 neighbours) of a bigger table; its rows changed twice leave more old copies near the anchor than
    allowed ids - limit (k = allowed count gave 0 of them). k must not stop at the allowed count,
    or the true neighbours never reach the scoring."""
    anchor = _clustered(1, seed=SEED + 9)[0]
    _bulk_memories(store, _clustered(OTHER_OWNER_ROWS), owner="other")
    allowed = OLD_COPY_NEIGHBOURS
    moved_count = allowed // 2
    kept_count = allowed - moved_count
    ids = _bulk_memories(store, np.concatenate([anchor[None, :], _near(anchor, 0.2, moved_count, 1),
                                                _near(anchor, 1.5, kept_count, 2)]))
    moved = ids[1:1 + moved_count]
    assert len(ids) - 1 == allowed
    store.closest_memories(NAMESPACE, ids[0], limit=TOP)  # build
    other = sqlite3.connect(store.path)
    rewrites = [_near(anchor, 0.2, moved_count, 10 + step) for step in range(NEAR_REWRITES)]
    for step, vectors in enumerate([*rewrites, _near(-anchor, 0.2, moved_count, 4)]):
        with other:  # same ids, another connection: first still next to the anchor, then on the far side
            other.executemany("UPDATE items SET embedding=? WHERE id=?",
                              [(vector.tobytes(), rowid) for vector, rowid in zip(vectors, moved)])
        hits = store.closest_memories(NAMESPACE, ids[0], limit=TOP)
        assert len({hit[0] for hit in hits}) == TOP, step
    other.close()
    index = _state(search_service, store).index
    stale = index.ntotal - len(np.unique(faiss.vector_to_array(index.id_map)))
    assert stale == (NEAR_REWRITES + 1) * moved_count and stale > allowed - TOP
    assert not set(moved) & {hit[0] for hit in hits}
    with pytest.MonkeyPatch.context() as patch:
        _full_comparison(patch)
        assert store.closest_memories(NAMESPACE, ids[0], limit=TOP) == hits


def test_memory_database_is_refused() -> None:
    """No file path, no search: the same error as an embedding that cannot be made."""
    vectors = _clustered(SYNC_ROWS)
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, embedding BLOB)")
    connection.executemany("INSERT INTO items(id,embedding) VALUES (?,?)",
                           [(index + 1, vector.tobytes()) for index, vector in enumerate(vectors)])
    ids = list(range(1, SYNC_ROWS + 1))
    query = _clustered(1, seed=SEED + 2)[0]
    with pytest.raises(EmbeddingUnavailable, match="database file"):
        vector_index.similarities(connection, "items", ids, query, limit=TOP)
    assert vector_index.similarities(connection, "items", [], query, limit=TOP) == []
    connection.close()


@pytest.mark.parametrize("allowed", SMALL_ALLOWED)
def test_small_allowed_set_takes_hnsw_and_matches_full_comparison(store: MemoryStore, allowed: int,
                                                                   search_service) -> None:
    """A database file uses the index even for 1 or 5 allowed rows; top N and threshold hits equal the full
    comparison, scores included."""
    vectors = _clustered(SYNC_ROWS)
    ids = _bulk_memories(store, vectors)
    picked = ids[::SYNC_ROWS // allowed][:allowed]
    query = _unit(vectors[ids.index(picked[0])] + SMALL_QUERY_NOISE * _clustered(1, seed=SEED + 3)[0])
    exact = np.asarray(_exact(store, picked, query))
    found = vector_index.similarities(store._conn, "items", picked, query, limit=TOP)
    assert _state(search_service, store).index.ntotal == SYNC_ROWS
    got = {index for index, sim in enumerate(found) if np.isfinite(sim)}
    assert got == set(np.argsort(-exact)[:TOP].tolist())
    assert all(abs(found[index] - exact[index]) < RESCORE_TOLERANCE for index in got)
    ranged = vector_index.similarities(store._conn, "items", picked, query, threshold=THRESHOLD)
    above = set(np.flatnonzero(exact >= THRESHOLD).tolist())
    assert above and {index for index, sim in enumerate(ranged) if np.isfinite(sim) and sim >= THRESHOLD} == above
    assert all(abs(ranged[index] - exact[index]) < RESCORE_TOLERANCE for index in above)


def _search_ids(store: MemoryStore, query: np.ndarray, limit: int = TOP) -> list[int]:
    return [hit.id for hit in store.search(NAMESPACE, "q", query.tolist(), limit=limit)]


def test_other_connection_add_change_delete(store: MemoryStore, search_service) -> None:
    vectors = _clustered(SYNC_ROWS)
    ids = _bulk_memories(store, vectors)
    query = vectors[0]
    first = _search_ids(store, query)
    assert first and _state(search_service, store).index.ntotal == SYNC_ROWS
    other = sqlite3.connect(store.path)
    with other:
        other.execute(
            "INSERT INTO items(type,owner,text_hash,text,origin,embedding,created_at,kind) VALUES ('memory',?,?,?,?,?,?,?)",
            (NAMESPACE, "new-hash", "new row", "agent:s", query.tobytes(), time.time(), "memory_item"))
        changed = ids[-1]
        other.execute("UPDATE items SET text=?,text_hash=?,embedding=? WHERE id=?",
                      ("changed row", "changed-hash", _unit(query + 0.01).tobytes(), changed))
        other.execute("DELETE FROM items WHERE type='memory' AND id=?", (first[1],))
    other.close()
    new_id = int(store._conn.execute("SELECT id FROM items WHERE type='memory' AND text='new row'").fetchone()[0])
    after = _search_ids(store, query)
    assert new_id in after and changed in after and first[1] not in after
    with pytest.MonkeyPatch.context() as patch:
        _full_comparison(patch)
        assert _search_ids(store, query) == after


def test_note_chunk_id_reused_by_other_connection(store: MemoryStore) -> None:
    vectors = _clustered(SYNC_ROWS)
    now = time.time()
    with store._write_scope():
        note_id = notes.create_formula_note(store._conn, "auto", [(f"c{index}", vector.tobytes())
                                                                  for index, vector in enumerate(vectors)], now)["id"]
    target = _unit(-vectors[0])
    assert notes.nearest_formula_chunks(store._conn, target.tobytes(), limit=1)
    top_id = int(store._conn.execute("SELECT MAX(id) FROM items WHERE author=? AND title IS NULL",
                                     (FORMULA_AUTHOR,)).fetchone()[0])
    other = sqlite3.connect(store.path)
    with other:  # a part rewrite: the highest id deleted and inserted again (the same id) with another vector
        before = other.execute("SELECT src_id FROM item_links WHERE relation='next' AND dst_id=?", (top_id,)).fetchone()[0]
        other.execute("DELETE FROM items WHERE id=?", (top_id,))
        other.execute("INSERT INTO items(id,type,title,text,text_hash,author,kind,embedding,created_at,updated_at) "
                      "VALUES (?,'note',NULL,'rewritten','hr',?,'note',?,?,?)",
                      (top_id, FORMULA_AUTHOR, target.tobytes(), now, time.time() + 1))
        other.execute("INSERT INTO item_links(src_id,dst_id,relation) VALUES (?,?,'next')", (before, top_id))
    other.close()
    assert note_id
    assert notes.nearest_formula_chunks(store._conn, target.tobytes(), limit=1)[0][0] == top_id


def test_rebuild_on_stale_share_and_model_change(store: MemoryStore, search_service) -> None:
    vectors = _clustered(SYNC_ROWS * 2)
    ids = _bulk_memories(store, vectors)
    query = vectors[0]
    vector_index.similarities(store._conn, "items", ids, query, limit=TOP)
    built = _state(search_service, store).index
    gone = ids[: int(len(ids) * (vector_search.DEAD_RATIO + 0.05))]
    with store._write_scope():
        store._conn.executemany("DELETE FROM items WHERE type='memory' AND id=?", [(rowid,) for rowid in gone])
    left = ids[len(gone):]
    vector_index.similarities(store._conn, "items", left, query, limit=TOP)
    assert _state(search_service, store).index is not built and _state(search_service, store).index.ntotal == len(left)
    rebuilt = _state(search_service, store).index
    vector_index.similarities(store._conn, "items", left, query, limit=TOP)
    assert _state(search_service, store).index is rebuilt
    with store._write_scope():
        store._conn.execute("UPDATE meta_entries SET value='other-model' WHERE key='embedding_model'")
    vector_index.similarities(store._conn, "items", left, query, limit=TOP)
    assert _state(search_service, store).index is not rebuilt and _state(search_service, store).meta[0] == "other-model"


def test_threshold_callers_match_full_comparison(store: MemoryStore) -> None:
    """Every row over the floor reaches the old scoring: a lower-cosine durable/important row still wins."""
    pool = _clustered(SYNC_ROWS + 1)
    base, query = pool[:SYNC_ROWS], pool[SYNC_ROWS]
    near = np.stack([_unit(query + 0.15 * np.random.default_rng(index).normal(size=DIMENSIONS))
                     for index in range(40)])
    _bulk_memories(store, base)
    _bulk_memories(store, near[:-1], importance=0.0, prefix="near")
    _bulk_memories(store, near[-1:], importance=1.0, prefix="star")
    with store._write_scope():
        store._conn.execute("UPDATE items SET kind='durable' WHERE text LIKE 'star%'")
    star = int(store._conn.execute("SELECT id FROM items WHERE type='memory' AND text LIKE 'star%'").fetchone()[0])
    hnsw = [(hit.id, hit.score) for hit in store.search(NAMESPACE, "q", query.tolist(), limit=5)]
    memory = mcp_server.search_memory_and_formula_readonly(store._conn, NAMESPACE, query, dimensions=DIMENSIONS,
                                                           limit=5, project="", session_id="", threshold=THRESHOLD)
    with pytest.MonkeyPatch.context() as patch:
        _full_comparison(patch)
        exact = [(hit.id, hit.score) for hit in store.search(NAMESPACE, "q", query.tolist(), limit=5)]
        exact_memory = mcp_server.search_memory_and_formula_readonly(
            store._conn, NAMESPACE, query, dimensions=DIMENSIONS, limit=5, project="", session_id="",
            threshold=THRESHOLD)
    assert hnsw[0][0] == star
    assert [hit[0] for hit in hnsw] == [hit[0] for hit in exact]
    assert all(abs(a[1] - b[1]) < SCORE_TOLERANCE for a, b in zip(hnsw, exact))
    assert [hit["id"] for hit in memory] == [hit["id"] for hit in exact_memory]


def test_measure_search_and_build(store: MemoryStore) -> None:
    """Report only: one search (HNSW vs full comparison) and the first build on 10,000 rows."""
    pool = _clustered(10_001)
    ids = _bulk_memories(store, pool[:10_000])
    query = pool[10_000]
    with store._write_scope():  # rows written a minute ago (an existing database), not inside the recheck window
        store._conn.execute("UPDATE items SET updated_at=updated_at-60")
    started = time.perf_counter()
    vector_index.similarities(store._conn, "items", ids, query, threshold=THRESHOLD)
    build = time.perf_counter() - started
    rounds = 20
    started = time.perf_counter()
    for _ in range(rounds):
        vector_index.similarities(store._conn, "items", ids, query, threshold=THRESHOLD)
    hnsw_ms = (time.perf_counter() - started) / rounds * 1000
    with pytest.MonkeyPatch.context() as patch:
        _full_comparison(patch)
        started = time.perf_counter()
        for _ in range(rounds):
            vector_index.similarities(store._conn, "items", ids, query, threshold=THRESHOLD)
        exact_ms = (time.perf_counter() - started) / rounds * 1000
    print(f"measure 10k: first build+search {build:.2f}s, search hnsw {hnsw_ms:.2f}ms, full {exact_ms:.2f}ms")
