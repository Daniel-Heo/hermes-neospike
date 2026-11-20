"""Main formula: automatic notes (chunks are the source), topic-to-note, note-absorb, formula-migrate.

Acceptance items of the work order are named in each test (A1 .. A8). A1 (no page for a new topic, the name
never changes) is in test_topics.py (test_new_topic_has_no_page, test_assign_threshold_and_dirty);
A9 (daily order) is in test_cron_scripts.py.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest

from neospike import notes as notes_module
from neospike import topics
from neospike.config import EMBEDDING_DIMENSIONS
from neospike.notes import ORIGIN_HUMAN, NoteService
from neospike.store import SCHEMA_VERSION, FORMULA_AUTHOR, MemoryStore
from neospike.tests.supersede_links import link_supersede
from neospike.tests.support_items import chunk_rows, members_of, note_row, note_rows, topic_names
from neospike.tests.test_remember_dedupe import no_remember_tidy  # noqa: F401 (fixture)

pytestmark = pytest.mark.usefixtures("no_remember_tidy")

NS = "test:formula"
DIMS = 8
CLI_DIMS = EMBEDDING_DIMENSIONS  # the CLI opens the store at the fixed model dimension
HUMAN_AXIS = 5
WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo",
         "lima", "mike", "november", "oscar")


def _axis(index: int, dims: int = DIMS) -> list[float]:
    vector = [0.0] * dims
    vector[index] = 1.0
    return vector


def _mix(cosine: float, main: int = 0, other: int = 1, dims: int = DIMS) -> list[float]:
    vector = [0.0] * dims
    vector[main], vector[other] = cosine, math.sqrt(1 - cosine * cosine)
    return vector


class CountingEmbedder:
    """Note chunk vectors on HUMAN_AXIS; counts embedding calls."""

    def __init__(self, dims: int = DIMS) -> None:
        self.calls = 0
        self.dims = dims

    def embed_document(self, text: str, **_kwargs) -> list[float]:
        self.calls += 1
        return _axis(HUMAN_AXIS, self.dims)


def _store(tmp_path: Path, dims: int = DIMS) -> MemoryStore:
    directory = tmp_path / "neospike"
    directory.mkdir(parents=True, exist_ok=True)
    return MemoryStore(directory / "memory.db", dimensions=dims)


def _remember(store: MemoryStore, text: str, vector: list[float]) -> int:
    memory_id = store.insert(NS, text, vector, source="agent:s", kind="memory_item",
                             metadata={"scope": "global", "subject": "formula", "durability": "ongoing", "tags": [],
                                       "session_id": "s"})
    assert memory_id is not None
    return memory_id


def _texts(count: int, prefix: str = "rule") -> list[str]:
    """Distinct memory bodies without digits (A2: no id or date string may reach a note body)."""
    return [f"{prefix} {WORDS[index]} keeps the build green" for index in range(count)]


def _topic(store: MemoryStore, texts: list[str], vector: list[float]) -> tuple[str, list[int]]:
    ids = [_remember(store, text, vector) for text in texts]
    with store._lock, store._conn:
        store._conn.execute("DELETE FROM item_links WHERE relation='topic'")  # the stored items may have joined a topic
        topic_id = topics._create_topic(store._conn, topics._member_rows(store._conn, ids), time.time())
    return topic_id, ids


def _rows(store: MemoryStore, sql: str, args: tuple = ()) -> list[tuple]:
    return [tuple(row) for row in store._conn.execute(sql, args).fetchall()]


def _formula_note(store: MemoryStore) -> dict:
    [note] = [row for row in note_rows(store._conn) if row["author"] == FORMULA_AUTHOR]
    return note


def _chunks(store: MemoryStore, note_id: int) -> list[tuple]:
    return [(row["id"], row["ord"], row["text"], row["embedding"]) for row in chunk_rows(store._conn, note_id)]


def _converted_note(store: MemoryStore, count: int = 11) -> tuple[sqlite3.Row, list[int]]:
    _, ids = _topic(store, _texts(count), _axis(0))
    assert topics.topic_to_note(store, NS)["converted"] == 1
    return _formula_note(store), ids


# A2 -------------------------------------------------------------------------------------------
def test_ten_members_stay_a_topic(tmp_path):
    store = _store(tmp_path)
    topic_id, ids = _topic(store, _texts(10), _axis(0))
    result = topics.topic_to_note(store, NS)
    assert result["converted"] == 0 and result["errors"] == []
    assert _rows(store, "SELECT COUNT(*) FROM items WHERE type='memory'")[0][0] == 10
    assert topic_names(store._conn) == [topic_id] and note_rows(store._conn) == []
    store.close()


def test_eleven_members_become_one_automatic_note(tmp_path):
    store = _store(tmp_path)
    texts = _texts(11)
    topic_id, ids = _topic(store, texts, _axis(0))
    name = topic_id
    blobs = {row[0]: row[1] for row in _rows(store, "SELECT id,embedding FROM items WHERE type='memory'")}
    result = topics.topic_to_note(store, NS)
    assert result["converted"] == 1 and result["errors"] == []
    assert _rows(store, "SELECT COUNT(*) FROM items WHERE type='memory'")[0][0] == 0
    assert topic_names(store._conn) == [] and members_of(store._conn, name) == []
    note = _formula_note(store)
    assert note["title"] == name and note["project"] == "" and note["rev"] == 1 and json.loads(note["tags_json"]) == []
    chunks = _chunks(store, note["id"])
    assert len(chunks) == 11
    assert [chunk[2] for chunk in chunks] == texts  # memory id order
    assert [chunk[3] for chunk in chunks] == [blobs[memory_id] for memory_id in ids]  # embedding bytes copied
    assert note["body"] == "\n\n".join(texts)
    assert not re.search(r"\d", note["body"]) and "mem:" not in note["body"]
    # Memory tombstones were written by the delete trigger (incremental backup).
    assert _rows(store, "SELECT COUNT(*) FROM tombstones WHERE table_name='items'")[0][0] == 11
    store.close()


def test_note_title_follows_the_duplicate_rule(tmp_path):
    store = _store(tmp_path)
    name, _ = _topic(store, _texts(11), _axis(0))
    NoteService(store, CountingEmbedder()).save(name, "A human note that already has this title.", author="human",
                                                origin=ORIGIN_HUMAN)
    topics.topic_to_note(store, NS)
    assert _formula_note(store)["title"] == f"{name} (2)"
    store.close()


# A3 -------------------------------------------------------------------------------------------
def test_invalid_members_are_deleted_and_archived_members_stay(tmp_path):
    store = _store(tmp_path)
    texts = _texts(13)
    topic_id, ids = _topic(store, texts, _axis(0))
    old, new, archived = ids[0], ids[1], ids[2]
    assert link_supersede(store, new, old)["linked"]
    assert store.archive(NS, archived)
    result = topics.topic_to_note(store, NS)
    assert result["converted"] == 1
    [converted] = result["topics"]
    assert (converted["valid"], converted["invalid"], converted["archived"]) == (11, 1, 1)
    chunk_texts = [chunk[2] for chunk in _chunks(store, _formula_note(store)["id"])]
    assert texts[0] not in chunk_texts and texts[2] not in chunk_texts and len(chunk_texts) == 11
    assert _rows(store, "SELECT id FROM items WHERE type='memory'") == [(archived,)]
    assert topic_names(store._conn) == []
    store.close()


# A4 -------------------------------------------------------------------------------------------
def test_absorb_threshold_human_note_and_second_run(tmp_path):
    store = _store(tmp_path)
    note, _ = _converted_note(store)
    human = NoteService(store, CountingEmbedder()).save("Human page", "A human note about the release process.",
                                                         author="human", origin=ORIGIN_HUMAN)
    near = _remember(store, "release gate stays manual", _mix(0.85))  # cos 0.85 to the note chunks
    far = _remember(store, "lunch is at noon", _mix(0.75, 0, 2))  # cos 0.75: stays
    human_twin = _remember(store, "release process twin", _mix(0.99, HUMAN_AXIS, 6))  # cos 0.99 to the human note
    near_blob = _rows(store, "SELECT embedding FROM items WHERE type='memory' AND id=?", (near,))[0][0]
    first = topics.absorb(store, NS)
    assert first["absorbed"] == 1 and [item["memory_ids"] for item in first["notes"]] == [[near]]
    assert first["compared"] == 3 * 11 + 2  # 3 memories x 11 chunks, then the 2 left x the absorbed one
    chunks = _chunks(store, note["id"])
    assert len(chunks) == 12 and chunks[-1][2] == "release gate stays manual" and chunks[-1][3] == near_blob
    body = note_row(store._conn, note["id"])["body"]
    assert body == "\n\n".join(chunk[2] for chunk in chunks)
    assert {row[0] for row in _rows(store, "SELECT id FROM items WHERE type='memory'")} == {far, human_twin}
    assert len(NoteService(store, None).chunks(human["id"])) == 1  # the human note is never a target
    second = topics.absorb(store, NS)
    assert second["compared"] == 0 and second["absorbed"] == 0
    store.close()


def test_changed_memory_is_compared_again(tmp_path):
    store = _store(tmp_path)
    _converted_note(store)
    far = _remember(store, "lunch is at noon", _mix(0.75, 0, 2))
    assert topics.absorb(store, NS)["compared"] == 11
    row = store.fetch_item_for_update(NS, far, project="", session_id="s")
    assert store.update_memory_item(NS, far, expected_text=row["text"], expected_row=row,
                                    item={"text": "lunch moved to one"}, embedding=_mix(0.75, 0, 2),
                                    project="", session_id="s") == "updated"
    assert topics.absorb(store, NS)["compared"] == 11  # another content hash: every chunk again
    store.close()


# A5 -------------------------------------------------------------------------------------------


def test_whole_body_write_splits_at_blank_lines_and_reuses_embeddings(tmp_path):
    store = _store(tmp_path)
    note, _ = _converted_note(store)
    embedder = CountingEmbedder()
    service = NoteService(store, embedder)
    before = _chunks(store, note["id"])
    paragraphs = [chunk[2] for chunk in before]
    paragraphs[2] = "rule edited by hand"
    saved = service.save(note["title"], "\n\n".join(paragraphs) + "\n\n\n", author="human", origin=ORIGIN_HUMAN,
                         expected_rev=1, note_id=note["id"])
    assert saved["chunks"] == 11 and embedder.calls == 1  # only the edited paragraph is embedded
    after = _chunks(store, note["id"])
    assert [chunk[2] for chunk in after] == paragraphs
    assert [chunk[3] for index, chunk in enumerate(after) if index != 2] == [
        chunk[3] for index, chunk in enumerate(before) if index != 2]
    row = note_row(store._conn, note["id"])
    assert row["author"] == FORMULA_AUTHOR and row["body"] == "\n\n".join(paragraphs)
    store.close()


def test_long_paragraph_is_split_by_the_note_splitter(tmp_path):
    store = _store(tmp_path)
    note, _ = _converted_note(store)
    service = NoteService(store, CountingEmbedder(), max_tokens=120)
    long_paragraph = " ".join(f"Sentence {WORDS[index % len(WORDS)]} number {index}." for index in range(40))
    service.save(note["title"], long_paragraph, author="human", origin=ORIGIN_HUMAN, expected_rev=1,
                 note_id=note["id"])
    chunks = _chunks(store, note["id"])
    assert len(chunks) > 1 and all(service.token_counter.count(chunk[2]) <= 120 for chunk in chunks)
    store.close()


# A6 -------------------------------------------------------------------------------------------
def test_three_absorbs_make_one_revision(tmp_path):
    store = _store(tmp_path)
    note, _ = _converted_note(store)
    for other in (2, 3, 4):
        _remember(store, f"close {WORDS[other]} memory", _mix(0.9, 0, other))
    result = topics.absorb(store, NS)
    assert result["absorbed"] == 3 and result["notes"][0]["rev"] == 2
    assert _rows(store, "SELECT rev,edited_by FROM item_revisions WHERE item_id=?", (note["id"],)) == [
        (1, topics.FORMULA_EDITOR)]
    assert len(_chunks(store, note["id"])) == 14
    store.close()


# A7 -------------------------------------------------------------------------------------------
class _Boom:
    def __call__(self, *args, **kwargs):
        raise RuntimeError("note write failed")


def _snapshot(store: MemoryStore) -> dict:
    return {table: _rows(store, f"SELECT * FROM {table} ORDER BY 1")
            for table in ("items", "item_links", "item_revisions")}


def test_failed_conversion_changes_nothing(tmp_path, monkeypatch):
    store = _store(tmp_path)
    _topic(store, _texts(11), _axis(0))
    before = _snapshot(store)
    monkeypatch.setattr(notes_module, "create_formula_note", _Boom())
    result = topics.topic_to_note(store, NS)
    assert result["converted"] == 0 and len(result["errors"]) == 1 and "note write failed" in result["errors"][0]["error"]
    assert _snapshot(store) == before
    store.close()


# A8 -------------------------------------------------------------------------------------------
def _cli(home: Path, monkeypatch, capsys, *arguments: str, code: int = 0) -> dict:
    from neospike import cli
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(home), "--owner", NS, *arguments])
    assert cli.main() == code
    return json.loads(capsys.readouterr().out)


def _counts(db: Path) -> tuple:
    with closing(sqlite3.connect(db)) as connection:
        return (connection.execute("SELECT COUNT(*) FROM items WHERE type='memory'").fetchone()[0],
                connection.execute("SELECT author, COUNT(*), SUM(deleted_at IS NULL) FROM items WHERE title IS NOT NULL "
                                   "AND type IN ('note','doc') GROUP BY author ORDER BY author").fetchall(),
                connection.execute("SELECT COUNT(DISTINCT label) FROM item_links WHERE relation='topic'").fetchone()[0])


# schema v15/v16 -----------------------------------------------------------------------------------
def test_v14_notes_table_accepts_formula_after_migration(tmp_path):
    database = tmp_path / "neospike" / "memory.db"
    database.parent.mkdir(parents=True)
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "CREATE TABLE notes (id INTEGER PRIMARY KEY, title TEXT NOT NULL UNIQUE, body TEXT NOT NULL, "
            "tags_json TEXT NOT NULL DEFAULT '[]', project TEXT NOT NULL DEFAULT '', "
            "author TEXT NOT NULL CHECK (author IN ('human','agent')), created_at REAL NOT NULL, "
            "updated_at REAL NOT NULL, rev INTEGER NOT NULL DEFAULT 1, deleted_at REAL)")
        connection.execute("INSERT INTO notes(id,title,body,author,created_at,updated_at) VALUES (7,'Old','b','human',1,1)")
    store = MemoryStore(database, dimensions=DIMS)
    # v22: the note becomes the doc head 1 (new ids start above the highest memory id; there is none).
    assert [(row["id"], row["title"], row["author"], row["type"], row["body"]) for row in note_rows(store._conn)] == [
        (1, "Old", "human", "doc", "b")]
    insert = ("INSERT INTO items(type,title,text,text_hash,author,embedding,created_at) "
              "VALUES ('note',?,'','',?,X'',2)")
    with store._lock, store._conn:
        store._conn.execute(insert, ("New", FORMULA_AUTHOR))
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute(insert, ("Bad", "bot"))
    assert _rows(store, "SELECT value FROM meta_entries WHERE key='schema_version'") == [(str(SCHEMA_VERSION),)]
    store._conn.execute("DELETE FROM items WHERE id=1")
    assert ("items", "1") in _rows(store, "SELECT table_name,row_key FROM tombstones WHERE table_name='items'")
    store.close()
