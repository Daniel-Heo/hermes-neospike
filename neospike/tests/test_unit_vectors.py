"""Every vector write path stores a unit vector, so similarity is a plain dot product."""
from __future__ import annotations

import math
import random
from array import array
from pathlib import Path

import pytest

from neospike.conversations import ConversationIndexer
from neospike.notes import NoteService
from neospike.reembed import _pack as reembed_pack
from neospike.store import MemoryStore

DIMENSIONS = 8
NORM_TOLERANCE = 1e-3
COSINE_TOLERANCE = 1e-6
SCALE = 3.0            # a non-unit input: every write path must store it scaled back to norm 1
REGRESSION_PAIRS = 50
REGRESSION_SEED = 429


class _WordCounter:
    @staticmethod
    def count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def cut(text: str, tokens: int) -> str:
        return " ".join(text.split()[:tokens])


def _scaled(dimensions: int) -> list[float]:
    return [SCALE * (index + 1) for index in range(dimensions)]


def _norm(blob: bytes) -> float:
    vector = array("f", blob)
    return math.sqrt(math.fsum(value * value for value in vector))


def _old_cosine(left: list[float], right: list[float]) -> float:
    """Reference cosine (both norms recomputed on every call)."""
    dot = sum(x * y for x, y in zip(left, right))
    norm = math.sqrt(sum(x * x for x in left) * sum(y * y for y in right))
    return dot / norm if norm else 0.0


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(math.fsum(value * value for value in vector))
    return [value / norm for value in vector]


@pytest.fixture
def store(tmp_path: Path):
    value = MemoryStore(tmp_path / "memory.db", dimensions=DIMENSIONS)
    yield value
    value.close()


def test_every_write_path_stores_a_unit_vector(store: MemoryStore, tmp_path: Path) -> None:
    blobs: dict[str, bytes] = {}
    memory_id = store.insert("owner", "a stored memory", _scaled(DIMENSIONS), source="agent:s")
    pending_id = store.insert("owner", "a pending memory", None, source="agent:s")
    assert store.write_missing_embedding(pending_id, _scaled(DIMENSIONS))
    for name, row_id in (("memories.insert", memory_id), ("memories.fill_embedding", pending_id)):
        blobs[name] = store._conn.execute("SELECT embedding FROM items WHERE type='memory' AND id=?", (row_id,)).fetchone()[0]
    blobs["memories.replacement"] = store._pack_embedding(_scaled(DIMENSIONS))  # update/supersede/restore/split
    blobs["note_parts"] = NoteService(store, None, token_counter=_WordCounter())._pack(_scaled(DIMENSIONS))
    blobs["conversation_chunks"] = ConversationIndexer(store, None, token_counter=_WordCounter())._pack(
        _scaled(DIMENSIONS))
    blobs["reembed"] = reembed_pack(_scaled(DIMENSIONS), DIMENSIONS, allow_zero=False)
    for name, blob in blobs.items():
        assert abs(_norm(blob) - 1.0) <= NORM_TOLERANCE, name


def test_dot_product_equals_previous_cosine_on_unit_vectors() -> None:
    generator = random.Random(REGRESSION_SEED)
    for _ in range(REGRESSION_PAIRS):
        left = _unit([generator.uniform(-1.0, 1.0) for _ in range(DIMENSIONS)])
        right = array("f", _unit([generator.uniform(-1.0, 1.0) for _ in range(DIMENSIONS)]))
        assert abs(MemoryStore._cosine(left, right) - _old_cosine(left, list(right))) <= COSINE_TOLERANCE
    assert MemoryStore._cosine([1.0, 0.0], array("f", b"")) == 0.0  # embedding-pending empty BLOB
