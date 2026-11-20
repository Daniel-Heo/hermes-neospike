"""Topic members edited by hand stay put through topic-assign / topic-discover (and the
formula-migrate rebuild: test_topic_links.py).

A topic is a tag without a page; a topic may still carry a page note (a topic link from the note, the
dashboard's handle for a hand move), but no page is written or rewritten."""
from __future__ import annotations

import pytest

from neospike import topics
from neospike.notes import NoteService
from neospike.tests.support_items import body, members_of
from neospike.tests.test_topics import NS, PageEmbedder, _axis, _clean, _dirty, _remember, _store, _topic_from

LEGACY_PAGE = "## 요약\nold summary\n\n## 근거 기억\n- [[mem:999]] old evidence line\n"


def _page(store, note_id: int) -> str:
    return body(store._conn, note_id)


def _members(store, topic_id: str) -> set[int]:
    return {item_id for item_id in members_of(store._conn, topic_id)
            if store._conn.execute("SELECT type FROM items WHERE id=?", (item_id,)).fetchone()[0] == "memory"}


def _topic_with_page(store, texts, vector):
    """A topic that carries a page note (written here; no topic page is written by the code)."""
    topic_id, ids = _topic_from(store, texts, vector)
    saved = NoteService(store, PageEmbedder()).save(f"TOPIC: legacy {topic_id}", LEGACY_PAGE, tags=["topic"],
                                                    author="agent")
    with store._lock, store._conn:
        store._conn.execute("INSERT INTO item_links(src_id,relation,label) VALUES (?,'topic',?)", (saved["id"], topic_id))
    _clean(store, topic_id)
    return topic_id, ids, saved["id"]


def test_removed_members_leave_page_and_stay_out_after_assign(tmp_path):
    store = _store(tmp_path)
    topic_id, ids, note_id = _topic_with_page(store, ["A one", "A two", "A three", "A four"], _axis(0))
    assert not _dirty(store, topic_id)
    result = topics.move_members(store, NS, [ids[0], ids[1], 999], PageEmbedder(), remove_from_note=note_id)
    assert result["moved_ids"] == ids[:2] and result["skipped_ids"] == [999] and result["dirty_topics"] == [topic_id]
    assert result["evidence"]["rewritten"] == 0  # The page is not rewritten
    assert _members(store, topic_id) == set(ids[2:]) and _page(store, note_id) == LEGACY_PAGE
    # v22: the topic time is computed from its links and members, so a removal does not move it (dirty_topics above
    # still names the topic).
    assert not _dirty(store, topic_id)
    # Linked to the members at cos 1.0, yet topic-assign leaves them out of this topic.
    assert topics.assign_all(store, NS)["assigned"] == 0
    assert _members(store, topic_id) == set(ids[2:])
    # A brand-new memory with the same vector joins through its links (the mark is per memory, not on store).
    fresh = _remember(store, "A five", _axis(0))
    assert _members(store, topic_id) == set(ids[2:])
    assert topics.assign_all(store, NS)["assigned"] == 1
    assert _members(store, topic_id) == {*ids[2:], fresh}
    store.close()


def test_added_member_is_pinned(tmp_path):
    store = _store(tmp_path)
    topic_id, ids, note_id = _topic_with_page(store, ["A one", "A two", "A three"], _axis(0))
    stray = _remember(store, "B far away", _axis(3))
    assert topics.assign_all(store, NS)["assigned"] == 0  # no link to a member
    result = topics.move_members(store, NS, [stray], PageEmbedder(), target_note=note_id)
    assert result["moved_ids"] == [stray] and _dirty(store, topic_id)
    assert stray in _members(store, topic_id) and _page(store, note_id) == LEGACY_PAGE
    marks = topics.manual_marks(store._conn)
    assert marks.pinned_to(stray) == topic_id
    store.close()


def test_move_to_other_topic_and_back_clears_exclusion(tmp_path):
    store = _store(tmp_path)
    first, first_ids, first_note = _topic_with_page(store, ["A one", "A two", "A three"], _axis(0))
    second, second_ids, second_note = _topic_with_page(store, ["C one", "C two", "C three"], _axis(4))
    topics.move_members(store, NS, [first_ids[0]], PageEmbedder(), remove_from_note=first_note)
    assert topics.manual_marks(store._conn).excluded_from(first_ids[0]) == [first]
    moved = topics.move_members(store, NS, [first_ids[0]], PageEmbedder(), target_note=second_note)
    assert moved["dirty_topics"] == [second] and first_ids[0] in _members(store, second)
    back = topics.move_members(store, NS, [first_ids[0]], PageEmbedder(), target_note=first_note)
    assert back["dirty_topics"] == [first, second]
    marks = topics.manual_marks(store._conn)
    assert marks.excluded_from(first_ids[0]) == [] and marks.pinned_to(first_ids[0]) == first
    assert _page(store, first_note) == LEGACY_PAGE and _page(store, second_note) == LEGACY_PAGE
    store.close()


class DownEmbedder:
    def embed_document(self, text: str, **_kwargs):
        raise PermissionError("Insecure inference directory")


def test_member_move_embeds_and_writes_no_page(tmp_path):
    """No page rewrite, so an embedder that is down does not matter."""
    store = _store(tmp_path)
    topic_id, ids, note_id = _topic_with_page(store, ["A one", "A two", "A three"], _axis(0))
    result = topics.move_members(store, NS, [ids[0]], DownEmbedder(), remove_from_note=note_id)
    assert result["moved_ids"] == [ids[0]] and result["evidence"] == {"pages": [], "rewritten": 0}
    assert _members(store, topic_id) == set(ids[1:]) and result["dirty_topics"] == [topic_id]
    store.close()


def test_move_refuses_non_topic_note_and_both_targets(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(topics.TopicError):
        topics.move_members(store, NS, [1], PageEmbedder(), target_note=123)
    with pytest.raises(topics.TopicError):
        topics.move_members(store, NS, [1], PageEmbedder(), target_note=1, remove_from_note=1)
    store.close()


def test_marks_pin_and_exclude():
    marks = topics.ManualMarks({"pinned": {"5": "beta", "6": "gamma"}, "excluded": {"7": ["beta"]}})
    assert marks.pinned_to(5) == "beta" and marks.pinned_to(6) == "gamma" and marks.excluded_from(7) == ["beta"]
    marks.pin(7, "beta")
    assert marks.excluded_from(7) == [] and marks.pinned_to(7) == "beta"
    marks.exclude(7, "beta")
    assert marks.pinned_to(7) is None and marks.excluded_from(7) == ["beta"]
    assert topics.ManualMarks("garbage").as_json() == '{"excluded": {}, "pinned": {}}'
