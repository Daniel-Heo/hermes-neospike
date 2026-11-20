"""Right after neospike_memory_save stores a memory item, its 3 closest valid memory items
are compared: cos >= 0.90 deletes the old one, 0.80 <= cos < 0.90 lists it in the reply's
`older`, below 0.80 nothing happens. Archived memories are never
compared or deleted, and a failure in this step never fails the save. Automatic-note chunks:
test_save_conflicts.py."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from neospike.config import NeospikeConfig
from neospike.provider import NeospikeProvider
from neospike.store import MemoryStore

SESSION = "session-dedupe"
DIMENSIONS = 128
NEW_TEXT = "The release train leaves on Thursdays."


def _at_cosine(similarity: float) -> list[float]:
    """A unit vector whose cosine with the NEW_TEXT vector e0 is `similarity`."""
    return [similarity, math.sqrt(1.0 - similarity * similarity)] + [0.0] * (DIMENSIONS - 2)


class Embedder:
    """NEW_TEXT -> e0; any other text -> the last axis (orthogonal). Seeds carry their own vectors."""
    dimensions = DIMENSIONS

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def embed_document(self, text: str, **_kwargs) -> list[float]:
        if text == NEW_TEXT:
            return [1.0] + [0.0] * (DIMENSIONS - 1)
        return [0.0] * (DIMENSIONS - 1) + [1.0]

    embed_query = embed_document

    def take_warnings(self) -> list[str]:
        return []


@pytest.fixture
def provider(tmp_path: Path):
    home_dir = tmp_path / "home"
    NeospikeConfig().save(home_dir)
    instance = NeospikeProvider(embedder=Embedder())
    instance.initialize(SESSION, hermes_home=str(home_dir), cwd=str(home_dir))
    yield instance
    instance.shutdown()


def _seed(provider: NeospikeProvider, text: str, similarity: float) -> int:
    """An existing global memory item at `similarity` to NEW_TEXT, stored without the tidy step."""
    memory_id = provider._memory_db.insert(provider._ns_label, text, _at_cosine(similarity), source="agent:seed",
                                       kind="memory_item", metadata={"scope": "global", "subject": "release"})
    assert memory_id is not None
    return memory_id


def _remember(provider: NeospikeProvider) -> dict:
    return json.loads(provider.handle_tool_call("neospike_memory_save", {
        "text": NEW_TEXT, "scope": "global", "subject": "release", "durability": "stable", "kind": "fact", "basis": "user_said"}))


def _ids(store: MemoryStore) -> set[int]:
    return {int(row[0]) for row in store._conn.execute("SELECT id FROM items WHERE type='memory'")}


def _new_id(store: MemoryStore) -> int:
    return int(store._conn.execute("SELECT id FROM items WHERE type='memory' AND text=?", (NEW_TEXT,)).fetchone()[0])


def test_duplicate_neighbor_is_deleted_and_new_kept(provider: NeospikeProvider) -> None:
    old = _seed(provider, "Release train departs every Thursday.", 0.95)
    result = _remember(provider)
    store = provider._memory_db
    assert result == {"stored": True, "deleted": [old]}
    assert _ids(store) == {_new_id(store)}


def test_contradiction_neighbor_is_listed_not_filed(provider: NeospikeProvider) -> None:
    old = _seed(provider, "The release train leaves on Mondays.", 0.85)
    reply = _remember(provider)
    store = provider._memory_db
    new = _new_id(store)
    assert reply == {"stored": True, "id": new, "older": [{"id": old, "text": "The release train leaves on Mondays."}]}
    assert _ids(store) == {old, new}


def test_far_neighbor_changes_nothing(provider: NeospikeProvider) -> None:
    old = _seed(provider, "Lunch is served at noon.", 0.70)
    assert _remember(provider) == {"stored": True}
    store = provider._memory_db
    assert _ids(store) == {old, _new_id(store)}


def test_only_three_closest_are_processed(provider: NeospikeProvider) -> None:
    olds = [_seed(provider, f"Release train note {index}.", similarity)
            for index, similarity in enumerate((0.99, 0.97, 0.95, 0.93))]
    result = _remember(provider)
    store = provider._memory_db
    assert sorted(result["deleted"]) == sorted(olds[:3])
    assert _ids(store) == {olds[3], _new_id(store)}


def test_archived_memory_is_not_deleted(provider: NeospikeProvider) -> None:
    store = provider._memory_db
    old = _seed(provider, "Release train departs every Thursday.", 0.99)
    with store._lock, store._conn:
        store._conn.execute("UPDATE items SET archived_at=1.0 WHERE id=?", (old,))
    assert _remember(provider) == {"stored": True}
    assert old in _ids(store)


def test_tidy_failure_does_not_fail_the_save(provider: NeospikeProvider, monkeypatch) -> None:
    _seed(provider, "Release train departs every Thursday.", 0.99)

    def broken(*_args, **_kwargs):
        raise RuntimeError("neighbor lookup failed")

    monkeypatch.setattr(provider._memory_db, "closest_memories", broken)
    assert _remember(provider) == {"stored": True}
    assert _new_id(provider._memory_db) > 0


def _skip_tidy(*_args, **_kwargs) -> dict:
    return {}


@pytest.fixture
def no_remember_tidy(monkeypatch) -> None:
    """For tests about other features whose toy embedders give distinct memories one vector."""
    monkeypatch.setattr(NeospikeProvider, "_dedupe_after_save", _skip_tidy)


def test_duplicate_hands_pinned_topic_to_new_memory(provider: NeospikeProvider) -> None:
    """The deleted duplicate's hand-pinned topic membership passes to the new memory."""
    from neospike import topics
    store = provider._memory_db
    old = _seed(provider, "Release train departs every Thursday.", 0.95)
    topic_id = "release"
    with store._lock, store._conn:
        store._conn.execute("INSERT INTO item_links(src_id,relation,label,created_at) VALUES (?,'topic',?,1.0)",
                            (old, topic_id))
        marks = topics.manual_marks(store._conn)
        marks.pin(old, topic_id)
        topics.save_manual_marks(store._conn, marks)
    assert _remember(provider) == {"stored": True, "deleted": [old]}
    new = _new_id(store)
    members = {int(row[0]): str(row[1]) for row in store._conn.execute(
        "SELECT src_id,label FROM item_links WHERE relation='topic'")}
    assert members == {new: topic_id}
    assert topics.manual_marks(store._conn).pinned_to(new) == topic_id
    assert store._conn.execute(f"SELECT {topics.topic_changed_at_sql('?')}", (topic_id,)).fetchone()[0] > 1.0


def test_duplicate_within_two_words_is_deleted(provider: NeospikeProvider) -> None:
    """cos >= 0.90 and a word gap of 2 (6 vs 4 words) is a duplicate."""
    old = _seed(provider, "Release train every Thursday.", 0.95)
    assert _remember(provider) == {"stored": True, "deleted": [old]}
    assert _ids(provider._memory_db) == {_new_id(provider._memory_db)}


def test_duplicate_cosine_with_three_word_gap_is_a_contradiction(provider: NeospikeProvider) -> None:
    """cos >= 0.90 but a word gap of 3 (6 vs 3 words): both stay, listed in older."""
    old = _seed(provider, "Thursday release train.", 0.95)
    reply = _remember(provider)
    store = provider._memory_db
    new = _new_id(store)
    assert reply == {"stored": True, "id": new, "older": [{"id": old, "text": "Thursday release train."}]}
    assert _ids(store) == {old, new}


def test_dedupe_delete_releases_links(provider: NeospikeProvider) -> None:
    """The dedupe deletion goes through store.delete, so the old memory's links go too."""
    store = provider._memory_db
    old = _seed(provider, "Release train departs every Thursday.", 0.95)
    keep = _seed(provider, "Lunch is served at noon.", 0.10)
    with store._lock, store._conn:
        store._conn.execute("INSERT INTO item_links(src_id,dst_id,dst_title,relation) VALUES (?,?,'',?)",
                            (keep, old, "supersedes"))
    assert _remember(provider) == {"stored": True, "deleted": [old]}
    assert store._conn.execute("SELECT COUNT(*) FROM item_links WHERE src_id=? OR dst_id=?", (old, old)).fetchone()[0] == 0
