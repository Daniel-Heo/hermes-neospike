"""v22 items layout: (a) a long document is a head row plus part rows along item_links 'next' and
reads back byte for byte; (b) an automatic note's body is the memories it absorbed (parts with their source);
(c) a topic has no items row, only topic links."""
from __future__ import annotations

import json
import time

from neospike import topics
from neospike.notes import NoteService, note_body, part_ids
from neospike.store import next_of
from neospike.tests.support_items import chunk_rows, members_of, note_rows
from neospike.tests.support_items import WordCounter
from neospike.tests.test_formula import NS as FORMULA_NS, _axis as _formula_axis, _remember as _formula_remember
from neospike.tests.test_formula import _store as _formula_store, _texts
from neospike.store import MemoryStore
from neospike.tests.test_notes import DIMENSIONS as BOW_DIMENSIONS, BowEmbedder
from neospike.tests.test_topics import NS, _mix, _remember, _store

PART_TOKENS = 12  # a small part ceiling (WordCounter: one token per word), so the body splits into several parts


def test_long_document_is_a_head_and_chained_parts_that_rebuild_the_body(tmp_path) -> None:
    """(a)"""
    store = MemoryStore(tmp_path / "neospike" / "memory.db", dimensions=BOW_DIMENSIONS)
    service = NoteService(store, BowEmbedder(), token_counter=WordCounter(), max_tokens=PART_TOKENS)
    paragraphs = [" ".join(f"word{section}x{index}" for index in range(9)) for section in range(5)]
    body = "# Long doc\n\n" + "\n\n".join(paragraphs) + "\n\n## Tail\nlast line\n"
    saved = service.save("Long doc", body, author="human")
    connection = store._conn
    head = connection.execute("SELECT type,title,text,length(embedding) FROM items WHERE id=?", (saved["id"],)).fetchone()
    assert tuple(head) == ("doc", "Long doc", "", 0)  # the head row holds the title only
    parts = part_ids(connection, saved["id"])
    assert saved["chunks"] == len(parts) > 2
    rows = connection.execute(f"SELECT id,title,text,length(embedding) FROM items WHERE id IN ({','.join('?' * len(parts))})",
                              parts).fetchall()
    assert all(row[1] is None and row[3] > 0 for row in rows)  # every part has its own vector
    assert next_of(connection, saved["id"]) == parts[0] and next_of(connection, parts[-1]) is None
    assert note_body(connection, saved["id"], "human") == body  # byte for byte, no overlap
    assert all(WordCounter.count(row["text"]) <= PART_TOKENS for row in chunk_rows(connection, saved["id"]))
    # A rewrite replaces the part rows (SQLite may hand the freed ids out again), the head id stays.
    again = service.save("Long doc", body.replace("last line", "the new last line"), author="human", expected_rev=1)
    assert again["id"] == saved["id"] and again["rev"] == 2
    assert note_body(connection, saved["id"], "human").endswith("the new last line\n")
    store.close()


def test_automatic_note_body_is_its_absorbed_memories(tmp_path) -> None:
    """(b) topic-to-note: each member memory becomes one part (text, vector, source) and is deleted."""
    store = _formula_store(tmp_path)
    texts = _texts(11)
    ids = [_formula_remember(store, text, _formula_axis(0)) for text in texts]
    with store._lock, store._conn:
        store._conn.execute("DELETE FROM item_links WHERE relation='topic'")
        topics._create_topic(store._conn, topics._member_rows(store._conn, ids), time.time())
    blobs = dict(store._conn.execute("SELECT id,embedding FROM items WHERE type='memory'").fetchall())
    assert topics.topic_to_note(store, FORMULA_NS)["converted"] == 1
    [note] = note_rows(store._conn)
    parts = chunk_rows(store._conn, note["id"])
    assert note["author"] == "formula" and note["body"] == "\n\n".join(texts)
    assert [row["text"] for row in parts] == texts
    assert [json.loads(row["source_json"])["memory_id"] for row in parts] == ids
    assert [row["embedding"] for row in parts] == [blobs[memory_id] for memory_id in ids]
    assert store._conn.execute("SELECT COUNT(*) FROM items WHERE type='memory'").fetchone()[0] == 0
    store.close()


def test_topic_lives_only_in_item_links(tmp_path) -> None:
    """(c)"""
    store = _store(tmp_path)
    ids = [_remember(store, f"otter {axis}", _mix(0.95, 0, axis), subject="otter") for axis in range(1, 7)]
    before = store._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    [name] = topics.discover(store, NS)["topic_ids"]
    assert store._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == before  # no row for the topic
    assert store._conn.execute("SELECT COUNT(*) FROM items WHERE title=? OR text=?", (name, name)).fetchone()[0] == 0
    assert members_of(store._conn, name) == ids
    assert {row[0] for row in store._conn.execute(
        "SELECT dst_id FROM item_links WHERE relation='topic'")} == {None}  # identified by label, no target row
    tables = {row[0] for row in store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not {name for name in tables if name.startswith("topic")}
    store.close()
