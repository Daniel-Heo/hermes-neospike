"""Vectorized similarity search (dot_similarities, one numpy product) returns what the per-row
MemoryStore._cosine reference loop returns: same row ids in the same order, scores within SCORE_TOLERANCE."""
from __future__ import annotations

import hashlib
import math
import random
from array import array
from operator import itemgetter
from pathlib import Path

import pytest

from neospike import conversations as conv
from neospike import mcp_server, store as store_module
from neospike.conversations import merge_adjacent, split_chains
from neospike.notes import NOTE_IMPORTANCE, NoteService, search_note_parts
from neospike.store import MemoryStore, dot_similarities, freshness, blend_score
from neospike.tests.test_conversations import WordCounter

DIMENSIONS = 16
SCORE_TOLERANCE = 1e-5
LOW_THRESHOLD = 0.2      # low enough that many random 16-d vectors pass, so ordering is exercised
MEMORY_ROWS = 300
NOTE_COUNT = 60
SESSIONS = 40
TURNS_PER_SESSION = 6
TURN_WORDS = 60
CHUNK_TOKENS, CONTEXT_TOKENS = 120, 20   # several chunks per conversation (test_conversations values)
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


def _old_conversation_search(connection, query_embedding, *, semantic_threshold: float, limit: int) -> list[dict]:
    """Reference search_conversation_chunks (one JOIN, per-row _cosine)."""
    qvec = list(query_embedding)
    now = conv.time.time()
    rows = connection.execute(
        "SELECT ch.id,ch.conversation_id,ch.seq_start,ch.seq_end,ch.part,ch.text,ch.prefix_chars,ch.embedding,"
        "c.last_message_at,c.session_id,c.title FROM conversation_chunks ch "
        "JOIN conversations c ON c.id=ch.conversation_id WHERE ch.embedding IS NOT NULL").fetchall()
    hits = []
    for row in rows:
        semantic = MemoryStore._cosine(qvec, array("f", row[7]))
        if not math.isfinite(semantic) or semantic < semantic_threshold:
            continue
        score = blend_score(semantic, NOTE_IMPORTANCE, freshness(now, float(row[8])))
        hits.append({"chunk_id": int(row[0]), "conversation_id": int(row[1]), "seq_start": int(row[2]),
                     "seq_end": int(row[3]), "part": int(row[4]), "text": str(row[5]), "prefix_chars": int(row[6]),
                     "score": score, "session_id": row[9], "title": row[10]})
    merged = merge_adjacent(hits, split_chains(connection, hits))
    merged.sort(key=itemgetter("score"), reverse=True)
    return merged[:limit]


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
    conversations = conv.ConversationStore(store)
    for session in range(SESSIONS):
        conversations.append(f"s{session}", [("user" if turn % 2 == 0 else "assistant",
                                              " ".join(f"w{session}x{turn}x{word}" for word in range(TURN_WORDS)))
                                             for turn in range(TURNS_PER_SESSION)])
    conv.ConversationIndexer(store, embedder, token_counter=WordCounter(), max_tokens=CHUNK_TOKENS,
                             context_tokens=CONTEXT_TOKENS).index_pending()
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


def test_conversation_search_matches_former_loop(filled: MemoryStore) -> None:
    rows = filled._conn.execute("SELECT COUNT(*) FROM conversation_chunks WHERE embedding IS NOT NULL").fetchone()[0]
    assert rows >= 100
    for query in _queries():
        new = conv.search_conversation_chunks(filled._conn, "q", query, dimensions=DIMENSIONS,
                                              semantic_threshold=LOW_THRESHOLD, limit=LIMIT)
        old = _old_conversation_search(filled._conn, query, semantic_threshold=LOW_THRESHOLD, limit=LIMIT)
        assert new
        _assert_same(new, old, "chunk_ids")


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
