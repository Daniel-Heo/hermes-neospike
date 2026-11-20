"""neospike_memory_reconcile splits an over-long body like neospike_memory_update; the pieces of every split
(reconcile, update correct/supersede, dashboard edit) form a one-way chain of item_links relation 'next' (no
refers/supersedes row); deleting a piece re-links the chain, link-prune clears a dangling pointer; the Connected
list and the map's lines show the chain."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neospike.store import MemoryStore, next_of
from neospike.tests.test_dashboard_notes import NAMESPACE, _app
from neospike.tests.test_notes import DIMENSIONS as BOW_DIMENSIONS
from neospike.tests.test_remember_dedupe import _remember, provider  # noqa: F401 (fixture)
from neospike.tests.test_save_conflicts import (FRIDAYS, MONDAYS, _chunk_ids, _formula_note, _note, _reconcile,
                                                _seed, _snapshot)
from neospike.tests.test_sphere_map import _client, _db, _seed as _seed_map, home  # noqa: F401 (fixture)

PIECE_CHARS = 900
PIECE_COUNT = 3


def _long_text(word: str = "Mondays") -> str:
    """PIECE_COUNT paragraphs of PIECE_CHARS characters: 2 paragraphs fit one 2000-char piece, so 2 pieces."""
    paragraphs = []
    for index in range(1, PIECE_COUNT + 1):
        sentence = f"Paragraph {index} says the release train left on {word} until October. "
        paragraphs.append((sentence * (PIECE_CHARS // len(sentence) + 1))[:PIECE_CHARS].rsplit(".", 1)[0] + ".")
    return "\n\n".join(paragraphs)


def _chain(connection: sqlite3.Connection, first: int) -> list[int]:
    order, current = [], first
    while current is not None and current not in order:
        order.append(current)
        current = next_of(connection, current)
    return order


def _row(connection: sqlite3.Connection, memory_id: int) -> dict:
    row = connection.execute("SELECT * FROM items WHERE type='memory' AND id=?", (memory_id,)).fetchone()
    return {**dict(zip([column[0] for column in connection.execute("SELECT * FROM items WHERE type='memory' LIMIT 0").description], row))}


def test_reconcile_splits_long_body_keeps_first_id_and_chains(provider):
    first = _seed(provider, MONDAYS, 0.85)
    _remember(provider)
    text = _long_text()
    assert len(text) > 2000
    done = _reconcile(provider, [{"id": first, "text": text}])
    connection = provider._memory_db._conn
    chain = _chain(connection, first)
    assert done == {"changed_ids": [first], "piece_ids": chain[1:]} and len(chain) == 2
    pieces = [_row(connection, memory_id) for memory_id in chain]
    assert "\n\n".join(piece["text"] for piece in pieces) == text and all(len(p["text"]) <= 2000 for p in pieces)
    metadata = [json.loads(piece["metadata_json"]) for piece in pieces]
    assert [meta["subject"] for meta in metadata] == ["release (1/2)", "release (2/2)"]
    assert {(piece["owner"], piece["project"]) for piece in pieces} == {(provider._ns_label, "")}
    assert {(piece["scope"], meta["durability"]) for piece, meta in zip(pieces, metadata)} == {("global", "ongoing")}
    assert connection.execute("SELECT text FROM item_revisions WHERE item_id=?", (first,)).fetchone()[0] == MONDAYS
    assert connection.execute(
        "SELECT COUNT(*) FROM item_links WHERE relation IN ('refers','supersedes')").fetchone()[0] == 0  # no such link


def test_reconcile_split_rolls_back_when_one_piece_fails(provider):
    first = _seed(provider, MONDAYS, 0.85)
    _remember(provider)
    text = _long_text()
    tail = text.split("\n\n")[-1]
    _seed(provider, tail, 0.1)  # a visible item already holds the second piece's text
    before = _snapshot(provider)
    assert _reconcile(provider, [{"id": first, "text": text}]) == {}
    assert _snapshot(provider) == before


def test_reconcile_splits_chunk_into_consecutive_chunks(provider):
    note = _formula_note(provider, [("Trains now leave on Mondays only.", 0.86), ("Lunch is at noon.", 0.1)])
    reply = _remember(provider)
    target = reply["older"][0]["chunk_id"]
    text = _long_text()
    done = _reconcile(provider, [{"chunk_id": target, "text": text}])
    chunks = _chunk_ids(provider, note["id"])
    assert len(chunks) == 3 and done == {"changed_chunk_ids": [chunks[0]], "piece_chunk_ids": [chunks[1]]}
    one, two, three = text.split("\n\n")  # inside one chunk a blank line becomes a line break (formula_chunk_text)
    assert _note(provider, note["id"]) == (f"{one}\n{two}\n\n{three}\n\nLunch is at noon.", None)


def _update(provider, memory_id: int, expected: str, text: str, mode: str) -> dict:
    return json.loads(provider.handle_tool_call("neospike_memory_update", {
        "id": memory_id, "expected_text": expected, "new_text": text, "subject": "release", "durability": "ongoing",
        "mode": mode}))


@pytest.mark.parametrize("mode", ["correct", "supersede"])
def test_update_split_is_chained_and_delete_relinks(provider, mode):
    first = _seed(provider, FRIDAYS, 0.1)
    words = " ".join(f"Rule {n} keeps the release train on Fridays." for n in range(1, 140))
    done = _update(provider, first, FRIDAYS, words, mode)
    ids = done["ids"]
    connection = provider._memory_db._conn
    assert len(ids) >= 3 and _chain(connection, ids[0]) == ids
    assert ids[0] == first if mode == "correct" else ids[0] != first
    middle = ids[1]
    assert json.loads(provider.handle_tool_call("neospike_memory_delete", {"id": middle}))
    assert _chain(connection, ids[0]) == [ids[0], *ids[2:]]


def test_link_prune_clears_dangling_piece_pointer(provider):
    store = provider._memory_db
    first = _seed(provider, FRIDAYS, 0.1)
    second = _seed(provider, MONDAYS, 0.2)
    with store._lock, store._conn:
        store._conn.execute("INSERT INTO item_links(src_id,dst_id,relation) VALUES (?,?,'next')", (first, second))
        store._conn.execute("DROP TRIGGER items_next_ad")  # e.g. rows written without the trigger
        store._conn.execute("DELETE FROM items WHERE type='memory' AND id=?", (second,))
    assert store.prune_orphan_links()["piece_chain"] == 1
    assert _chain(store._conn, first) == [first]
    assert store.prune_orphan_links()["piece_chain"] == 0


def test_dashboard_edit_split_is_chained(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from neospike.tests.test_notes import BowEmbedder
    original = "Project Falcon deploys from the release branch only."
    seed = MemoryStore(_db(home), dimensions=BOW_DIMENSIONS)
    try:
        item_id = seed.insert(
            NAMESPACE, original, BowEmbedder().embed_document(original), source="agent:s1", kind="memory_item",
            project="D:/work/falcon", metadata={"scope": "project", "session_id": "s1", "subject": "Falcon deploys",
                                                "durability": "stable"})
    finally:
        seed.close()
    client = TestClient(_app(monkeypatch))
    edited = " ".join(f"Falcon rule {n} deploys from the release branch after review." for n in range(1, 80))
    result = client.put(f"/memories/{item_id}", json={"expected_text": original, "new_text": edited}).json()
    assert result["updated"] is True and result["ids"][0] == item_id and len(result["ids"]) >= 2
    connection = sqlite3.connect(f"file:{_db(home).as_posix()}?mode=ro", uri=True)
    try:
        assert _chain(connection, item_id) == result["ids"]
    finally:
        connection.close()


def test_connected_list_and_map_lines_show_the_chain(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    memories, _saved = _seed_map(home, 4)
    store = MemoryStore(_db(home), dimensions=BOW_DIMENSIONS)
    try:
        with store._conn:
            store._conn.executemany("INSERT INTO item_links(src_id,dst_id,relation) VALUES (?,?,'next')",
                                    [(memories[0], memories[1]), (memories[1], memories[2])])
    finally:
        store.close()
    client = _client(monkeypatch)
    middle = client.get(f"/connections/memory/{memories[1]}").json()["items"]
    assert [(item["key"], item["relations"]) for item in middle] == [
        (f"mem:{memories[0]}", ["previous piece"]), (f"mem:{memories[2]}", ["next piece"])]
    points = {point["key"]: point for point in client.get("/map").json()["points"]}
    assert f"mem:{memories[0]}" in points[f"mem:{memories[1]}"]["links"]
    assert f"mem:{memories[2]}" in points[f"mem:{memories[1]}"]["links"]
    assert f"mem:{memories[3]}" not in points[f"mem:{memories[1]}"]["links"]
