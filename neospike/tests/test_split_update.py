"""A correction over 2,000 characters is split into several memory items in one transaction."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pytest

from neospike.policy import ITEM_TEXT_CEILING
from neospike.provider import NeospikeProvider
from neospike.service import cut_memory_text, numbered_subject


OLD = "The pilot requires the GLASS-OTTER approval mark."
PARAGRAPH_CHARS = 745
PARAGRAPH_COUNT = 6


class Embedder:
    dimensions = 3

    def embed_document(self, text):
        return [0., 1., 0.] if "COPPER" in text else [1., 0., 0.]

    embed_query = embed_document


@pytest.fixture
def provider(tmp_path):
    with closing(sqlite3.connect(tmp_path / "state.db")) as conn, conn:
        conn.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, cwd TEXT, git_repo_root TEXT)")
        conn.execute("INSERT INTO sessions VALUES (?,?,?)", ("a", str(tmp_path / "project-a"), None))
    p = NeospikeProvider(embedder=Embedder())
    p.initialize("a", hermes_home=str(tmp_path), cwd=str(tmp_path / "wrong-cwd"))
    yield p
    p.shutdown()


def rows(p):
    return [dict(r) for r in p._memory_db._conn.execute("SELECT * FROM items WHERE type='memory' ORDER BY id")]


def table(p, name):
    return [tuple(r) for r in p._memory_db._conn.execute(f"SELECT * FROM {name}")]


def remember(p, text=OLD):
    result = json.loads(p.handle_tool_call("neospike_memory_save", {
        "text": text, "scope": "project", "subject": "pilot approval", "durability": "ongoing",
        "importance": .8, "confidence": .9, "tags": ["pilot"], "kind": "fact", "basis": "user_said",
    }))
    assert result == {"stored": True}
    return rows(p)[-1]["id"]


def update(p, memory_id, new_text, **overrides):
    return json.loads(p.handle_tool_call("neospike_memory_update", {
        "id": memory_id, "expected_text": OLD, "new_text": new_text,
        "subject": "pilot approval", "durability": "ongoing", **overrides,
    }))


def paragraph(index):
    words = []
    sentence = 0
    while len(" ".join(words)) < PARAGRAPH_CHARS:
        sentence += 1
        words.append(f"Paragraph {index} sentence {sentence} records COPPER approval detail {sentence}.")
    return " ".join(words)[:PARAGRAPH_CHARS].strip()


def paragraphs_text():
    return "\n\n".join(paragraph(index) for index in range(1, PARAGRAPH_COUNT + 1))


def by_id(p):
    return {row["id"]: row for row in rows(p)}


def test_split_rule_paragraph_sentence_and_hard_cut():
    text = paragraphs_text()
    assert 4400 <= len(text) <= 4600
    pieces = cut_memory_text(text)
    assert len(pieces) == 3
    expected = [paragraph(1) + "\n\n" + paragraph(2), paragraph(3) + "\n\n" + paragraph(4),
                paragraph(5) + "\n\n" + paragraph(6)]
    assert pieces == expected
    sentences = " ".join(f"Sentence {n} is about the COPPER pilot approval." for n in range(1, 120))
    pieces = cut_memory_text(sentences)
    assert len(pieces) > 2 and all(len(piece) <= ITEM_TEXT_CEILING for piece in pieces)
    assert all(piece.endswith(".") for piece in pieces)
    assert " ".join(pieces) == sentences
    hard = "a" * 4500
    assert [len(piece) for piece in cut_memory_text(hard)] == [2000, 2000, 500]
    assert cut_memory_text("short text stays whole") == ["short text stays whole"]
    assert numbered_subject("s" * 200, 2, 3) == "s" * 194 + " (2/3)"


def test_at_most_limit_is_not_split(provider):
    memory_id = remember(provider)
    text = ("COPPER pilot approval wording. " * 70)[:ITEM_TEXT_CEILING]
    result = update(provider, memory_id, text)
    assert result == {"updated": True, "id": memory_id}
    assert len(rows(provider)) == 1
    assert json.loads(rows(provider)[0]["metadata_json"])["subject"] == "pilot approval"


def test_correct_splits_into_three_items_at_paragraphs(provider):
    memory_id = remember(provider)
    result = update(provider, memory_id, paragraphs_text())
    assert result["updated"] is True and result["id"] == memory_id
    ids = result["ids"]
    assert len(ids) == 3 and ids[0] == memory_id
    current = by_id(provider)
    pieces = cut_memory_text(paragraphs_text())
    original_project = current[memory_id]["project"]
    for index, (item_id, piece) in enumerate(zip(ids, pieces), 1):
        row = current[item_id]
        meta = json.loads(row["metadata_json"])
        assert row["text"] == piece and len(piece) <= ITEM_TEXT_CEILING
        assert meta["subject"] == f"pilot approval ({index}/3)"
        assert (row["scope"], meta["durability"], meta["tags"]) == ("project", "ongoing", ["pilot"])
        assert (row["project"], row["kind"]) == (original_project, "memory_item")
    assert len({current[item_id]["owner"] for item_id in ids}) == 1
    revisions = provider._memory_db._conn.execute("SELECT item_id,rev,text FROM item_revisions").fetchall()
    assert [tuple(r) for r in revisions] == [(memory_id, 1, OLD)]


def test_sentence_only_text_splits_at_sentence_ends(provider):
    memory_id = remember(provider)
    text = " ".join(f"Sentence {n} is about the COPPER pilot approval." for n in range(1, 120))
    result = update(provider, memory_id, text)
    ids = result["ids"]
    assert len(ids) == len(cut_memory_text(text)) > 2
    current = by_id(provider)
    assert all(current[item_id]["text"].endswith(".") for item_id in ids)
    assert all(len(current[item_id]["text"]) <= ITEM_TEXT_CEILING for item_id in ids)


def test_supersede_first_piece_replaces_the_old_item(provider):
    memory_id = remember(provider)
    result = update(provider, memory_id, paragraphs_text(), mode="supersede")
    assert result["superseded"] is True and result["old_id"] == memory_id
    ids = result["ids"]
    assert len(ids) == 3 and ids[0] == result["id"] != memory_id
    links = provider._memory_db._conn.execute(
        "SELECT src_id,dst_id FROM item_links WHERE relation='supersedes' AND reverted_at IS NULL").fetchall()
    assert [tuple(link) for link in links] == [(ids[0], memory_id)]
    current = by_id(provider)
    assert current[memory_id]["text"] == OLD
    subjects = [json.loads(current[item_id]["metadata_json"])["subject"] for item_id in ids]
    assert subjects == [f"pilot approval ({index}/3)" for index in (1, 2, 3)]


def test_rejected_piece_saves_nothing(provider):
    memory_id = remember(provider)
    before, revisions = rows(provider), table(provider, "item_revisions")
    blocked = paragraphs_text().replace(paragraph(5), "token=do-not-store-this-value " + paragraph(5)[:600])
    result = update(provider, memory_id, blocked)
    assert "error" in result
    assert rows(provider) == before and table(provider, "item_revisions") == revisions


def test_duplicate_later_piece_rolls_back_the_written_first_piece(provider):
    memory_id = remember(provider)
    text = paragraphs_text()
    remember(provider, text=cut_memory_text(text)[2])  # the third piece already exists as a visible item
    before, revisions = rows(provider), table(provider, "item_revisions")
    result = update(provider, memory_id, text)
    assert result == {"updated": False, "reason": "conflict"}
    assert rows(provider) == before and table(provider, "item_revisions") == revisions
