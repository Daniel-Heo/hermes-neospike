"""MCP mem_search = memories + note chunks (agent, automatic) + doc chunks (notes the user wrote),
cosine >= MEMORY_SEARCH_COSINE_THRESHOLD, highest first, cut to limit, kind memory / note / doc. The evaluation
lane search_memory_and_formula_readonly keeps memories + automatic-note chunks only.
Vectors are built so each item's cosine to the query is known exactly."""
from __future__ import annotations

import json
import math
import time
from array import array
from functools import partial
from pathlib import Path

import pytest

from neospike.config import EMBEDDING_DIMENSIONS, NeospikeConfig
from neospike.database import canonical_database_path
from neospike.mcp_server import (KIND_DOC, KIND_MEMORY, KIND_NOTE, KIND_NOTE_CHUNK, MEMORY_SEARCH_COSINE_THRESHOLD,
                                 NeospikeTools, search_memory_and_formula_readonly)
from neospike.store import DEFAULT_SEMANTIC_CUTOFF, FORMULA_AUTHOR, MemoryStore
from neospike.store import document_type, write_chain

JUST_BELOW = 0.001  # cosine margin under the floor for the "just below" memory
# float32 storage rounds a cosine of exactly 0.7 down to 0.69999999; this lifts "on the floor" past that rounding.
ON_FLOOR_FLOAT32 = 1e-6

DIMS = EMBEDDING_DIMENSIONS
NS = "task411"
OTHER_PROJECT = "D:/somewhere/else"


def _vector(cosine: float) -> list[float]:
    """Unit vector whose cosine to the query axis e0 is *cosine*."""
    vector = [0.0] * DIMS
    vector[0], vector[1] = cosine, math.sqrt(1.0 - cosine * cosine)
    return vector


QUERY = _vector(1.0)


_ITEM_COLUMNS = "(type,owner,title,text,text_hash,project,tags_json,author,kind,embedding,created_at,updated_at,deleted_at)"


def _chunk(store: MemoryStore, kind: str, author: str, text: str, cosine: float) -> int:
    """One part row of a document (chained to its head by _note)."""
    now = time.time()
    cursor = store._conn.execute(
        f"INSERT INTO items{_ITEM_COLUMNS} VALUES (?,'',NULL,?,?,'','[]',?,?,?,?,?,NULL)",
        (kind, text, f"h:{text}", author, kind, array("f", _vector(cosine)).tobytes(), now, now))
    return int(cursor.lastrowid)


def _note(store: MemoryStore, title: str, author: str, chunks: list[tuple[str, float]], *, trashed: bool = False,
          tags_json: str = "[]") -> int:
    """A v22 document: head row (title only) + its part rows along next links."""
    now = time.time()
    kind = document_type(author)
    cursor = store._conn.execute(
        f"INSERT INTO items{_ITEM_COLUMNS} VALUES (?,'',?,'','',?,?,?,?,X'',?,?,?)",
        (kind, title, "", tags_json, author, kind, now, now, now if trashed else None))
    note_id = int(cursor.lastrowid)
    parts = [_chunk(store, kind, author, text, cosine) for text, cosine in chunks]
    write_chain(store._conn, [note_id, *parts], now=now)
    return note_id


@pytest.fixture()
def tools(tmp_path: Path) -> NeospikeTools:
    home = tmp_path / "home"
    NeospikeConfig().save(home)
    store = MemoryStore(canonical_database_path(home / "neospike", "memory"), dimensions=DIMS)
    try:
        def remember(text: str, cosine: float, **metadata) -> int:
            return store.insert(NS, text, _vector(cosine), source="agent:s", kind="durable",
                                project=metadata.pop("project", ""), metadata={"scope": "global", **metadata})
        remember("memory high", 0.80)
        remember("memory just below", MEMORY_SEARCH_COSINE_THRESHOLD - JUST_BELOW)
        old = remember("memory superseded", 0.95)
        new = remember("memory replacement", MEMORY_SEARCH_COSINE_THRESHOLD + ON_FLOOR_FLOAT32)  # on the floor: kept
        remember("memory of another project", 0.97, scope="project", project=OTHER_PROJECT)
        with store._conn:
            store._conn.execute("UPDATE items SET invalid_at=?, superseded_by=? WHERE id=?", (time.time(), new, old))
            # Floor 0.61, so the chunks sit at 0.75 / 0.72 above it and 0.55 below it.
            _note(store, "auto A", FORMULA_AUTHOR, [("auto chunk 0.90", 0.90), ("auto chunk 0.72", 0.72)])
            _note(store, "auto B", FORMULA_AUTHOR, [("auto chunk 0.75", 0.75), ("auto chunk 0.55", 0.55)])
            _note(store, "auto trashed", FORMULA_AUTHOR, [("trashed auto chunk", 0.98)], trashed=True)
            _note(store, "human plan", "human", [("human chunk 0.99", 0.99)])
            _note(store, "TASK_900", "agent", [("agent chunk 0.85", 0.85)], tags_json='["task"]')
    finally:
        store.close()
    result = NeospikeTools(home, NS)
    result._query_embedding = _FixedQuery()
    return result


class _FixedQuery:
    """Query embedding stand-in: always the e0 axis."""

    def __call__(self, query: str) -> list[float]:
        return list(QUERY)


def _search(tools: NeospikeTools, **arguments) -> list[dict]:
    return json.loads(tools.mem_search("anything", **arguments))["results"]


def test_threshold_is_the_store_cutoff() -> None:
    assert MEMORY_SEARCH_COSINE_THRESHOLD == DEFAULT_SEMANTIC_CUTOFF  # one constant


def test_memories_notes_and_docs_by_cosine_cut_to_limit(tools: NeospikeTools) -> None:
    every = _search(tools, limit=20)
    assert [item["text"] for item in every] == ["human chunk 0.99", "auto chunk 0.90", "agent chunk 0.85", "memory high",
                                                "auto chunk 0.75", "auto chunk 0.72", "memory replacement"]
    # just below, 0.55, trashed, other project: out
    assert [round(item["score"], 3) for item in every] == [0.99, 0.9, 0.85, 0.8, 0.75, 0.72,
                                                           round(MEMORY_SEARCH_COSINE_THRESHOLD, 3)]
    assert [item["kind"] for item in every] == [KIND_DOC, KIND_NOTE, KIND_NOTE, KIND_MEMORY, KIND_NOTE, KIND_NOTE,
                                                KIND_MEMORY]
    assert [item["text"] for item in _search(tools, limit=3)] == ["human chunk 0.99", "auto chunk 0.90",
                                                                  "agent chunk 0.85"]
    assert len(_search(tools)) == 5  # default limit
    doc, memory = every[0], every[3]
    assert set(doc) == {"kind", "note_id", "title", "chunk_id", "heading_path", "text", "score"}
    assert doc["title"] == "human plan" and every[2]["title"] == "TASK_900"
    assert memory["item_kind"] == "durable" and {"id", "text", "origin", "score", "confidence"} <= set(memory)


def test_one_query_returns_all_three_kinds(tools: NeospikeTools) -> None:
    assert {item["kind"] for item in _search(tools)} == {KIND_MEMORY, KIND_NOTE, KIND_DOC}


def test_tag_keeps_only_items_with_the_tag(tools: NeospikeTools) -> None:
    assert [item["text"] for item in _search(tools, limit=20, tag="task")] == ["agent chunk 0.85"]


def test_evaluation_lane_keeps_memories_and_automatic_chunks_only(tools: NeospikeTools) -> None:
    lane = partial(search_memory_and_formula_readonly, dimensions=DIMS, limit=20,
                   project=str(Path.cwd().resolve()), session_id="")
    every = json.loads(tools._guarded(lane, NS, QUERY))
    assert [item["text"] for item in every] == ["auto chunk 0.90", "memory high", "auto chunk 0.75", "auto chunk 0.72",
                                                "memory replacement"]  # human and agent chunks never come here
    assert every[0]["kind"] == KIND_NOTE_CHUNK and every[1]["kind"] == "durable"


def test_superseded_memory_only_with_history(tools: NeospikeTools) -> None:
    assert "memory superseded" not in [item["text"] for item in _search(tools, limit=20, include_history=False)]
    full = _search(tools, limit=20, include_history=True)
    [old] = [item for item in full if item["text"] == "memory superseded"]
    assert old["score"] == pytest.approx(0.95 * 0.5, abs=1e-6) and "superseded_by" in old
    assert full[-1] is old  # half score ranks it last here
