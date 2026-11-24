"""Vectorized similarity search (dot_similarities, one numpy product) returns what the per-row
MemoryStore._cosine reference loop returns: same row ids in the same order, scores within SCORE_TOLERANCE."""
from __future__ import annotations

import hashlib
import math
import random
from array import array
from pathlib import Path

import pytest

from neospike import mcp_server, store as store_module
from neospike.notes import NoteService, search_note_parts
from neospike.store import MemoryStore, dot_similarities
from neospike.tests.support_items import WordCounter

DIMENSIONS = 16
SCORE_TOLERANCE = 1e-5
LOW_THRESHOLD = 0.2      # low enough that many random 16-d vectors pass, so ordering is exercised
MEMORY_ROWS = 300
NOTE_COUNT = 60
QUERY_COUNT = 5
LIMIT = 50
SEED = 431
NAMESPACE = "owner"


def _unit(seed_text: str) -> list[float]:
    generator = random.Random(hashlib.sha256(seed_text.encode()).digest())
    vector = [generator.gauss(0.0, 1.0) for _ in range(DIMENSIONS)]
    norm = math.sqrt(math.fsum(value * value for value in vector))
    return [value / norm for value in vector]


class RandomUnitEmbedder:
    dimensions = DIMENSIONS

    def embed_document(self, text: str, **_kwargs) -> list[float]:
        return _unit(text)

    def embed_query(self, text: str, **_kwargs) -> list[float]:
        return _unit(text)


def _reference_scores(query, blobs) -> list[float]:
    """Reference scoring: MemoryStore._cosine per row."""
    return [MemoryStore._cosine(query, array("f", blob if blob is not None else b"")) for blob in blobs]


def _assert_same(new: list[dict], old: list[dict], key: str) -> None:
    assert [item.get(key) for item in new] == [item.get(key) for item in old]
    assert all(abs(a["score"] - b["score"]) < SCORE_TOLERANCE for a, b in zip(new, old))
    stripped = [{k: v for k, v in item.items() if k != "score"} for item in new]
    assert stripped == [{k: v for k, v in item.items() if k != "score"} for item in old]


@pytest.fixture
def filled(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db", dimensions=DIMENSIONS)
    embedder = RandomUnitEmbedder()
    for index in range(MEMORY_ROWS):
        text = f"memory {index}"
        store.insert(NAMESPACE, text, embedder.embed_document(text), source="agent:s", kind="memory_item")
    service = NoteService(store, embedder, token_counter=WordCounter())
    for index in range(NOTE_COUNT):
        service.save(f"Note {index}", f"# Note {index}\n\nParagraph one of {index}.\n\n## Part\n\nParagraph two of {index}.",
                     author="human")
    yield store
    store.close()


def _queries() -> list[list[float]]:
    return [_unit(f"query {SEED} {index}") for index in range(QUERY_COUNT)]


def test_dot_scores_equal_cosine_including_odd_widths() -> None:
    generator = random.Random(SEED)
    blobs = [array("f", _unit(f"row {index}")).tobytes() for index in range(MEMORY_ROWS)]
    blobs[3] = b""                                              # embedding pending
    blobs[7] = array("f", _unit("short")[:DIMENSIONS - 1]).tobytes()  # stale width
    blobs[11] = None
    generator.shuffle(blobs)
    query = _unit("the query")
    new, old = dot_similarities(query, blobs), _reference_scores(query, blobs)
    assert len(new) == len(old) and all(abs(a - b) < SCORE_TOLERANCE for a, b in zip(new, old))
    assert dot_similarities(query, []) == []


def test_memory_note_and_mcp_search_match_former_loop(filled: MemoryStore, monkeypatch) -> None:
    filled.semantic_threshold = LOW_THRESHOLD
    calls = {
        "store.search": lambda query: [{"id": hit.id, "score": hit.score} for hit in filled.search(
            NAMESPACE, "q", query, limit=LIMIT)],
        "notes": lambda query: search_note_parts(filled._conn, "q", query, dimensions=DIMENSIONS,
                                                  semantic_threshold=LOW_THRESHOLD, limit=LIMIT),
        "mcp memory_search": lambda query: mcp_server.search_memory_and_formula_readonly(
            filled._conn, NAMESPACE, query, dimensions=DIMENSIONS, limit=LIMIT, project="", session_id="",
            threshold=LOW_THRESHOLD),
    }
    for name, call in calls.items():
        for query in _queries():
            new = call(query)
            with monkeypatch.context() as patch:
                # vector_index.similarities takes store.dot_similarities at call time.
                patch.setattr(store_module, "dot_similarities", _reference_scores)
                old = call(query)
            assert new, name
            _assert_same(new, old, "chunk_id" if name == "notes" else "id")
