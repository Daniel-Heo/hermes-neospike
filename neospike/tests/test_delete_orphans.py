"""Deleting a memory also removes its topic link and the links it starts or ends (a `refers` link from a
note keeps its row with dst_id NULL, so the broken-link lint finds the [[mem:N]] text). Every memory delete
path shares this. v22: a topic is its links, so its time follows its remaining members (no dirty flag)."""
from __future__ import annotations

from neospike import topics
from neospike.tests.test_topics import NS, _axis, _clean, _dirty, _remember, _store, _topic_from

_HEAD_SQL = ("INSERT INTO items(type,owner,title,text,text_hash,author,kind,embedding,created_at,updated_at,deleted_at) "
             "VALUES ('doc','',?,'','',?,'doc',X'',1.0,1.0,?)")


def _note(store, title: str, *, deleted: float | None = None) -> int:
    """A document head row (the note id)."""
    with store._lock, store._conn:
        return int(store._conn.execute(_HEAD_SQL, (title, "human", deleted)).lastrowid)


def _link(store, src_id: int, dst_id: int | None, *, to_memory: bool, relation: str = "refers") -> None:
    with store._lock, store._conn:
        store._conn.execute(
            "INSERT INTO item_links(src_id,dst_id,dst_title,relation) VALUES (?,?,?,?)",
            (src_id, dst_id, f"mem:{dst_id}" if to_memory else "Page", relation))


def _member_rows(store, memory_id: int) -> int:
    return int(store._conn.execute("SELECT COUNT(*) FROM item_links WHERE relation='topic' AND src_id=?",
                                   (memory_id,)).fetchone()[0])


def _link_rows(store, memory_id: int) -> int:
    return int(store._conn.execute(
        "SELECT COUNT(*) FROM item_links WHERE relation IN ('refers','supersedes') AND (src_id=? OR dst_id=?)",
        (memory_id, memory_id)).fetchone()[0])


def _seed_orphans(store):
    topic_id, ids = _topic_from(store, ["A one", "A two", "A three"], _axis(0))
    victim, other = ids[0], ids[1]
    page = _note(store, "Page")
    _link(store, other, victim, to_memory=True, relation="supersedes")
    _link(store, victim, other, to_memory=True, relation="supersedes")
    _link(store, victim, page, to_memory=False)
    _link(store, page, victim, to_memory=True)
    _clean(store, topic_id)
    return topic_id, victim, other, page


def _assert_cleaned(store, topic_id: str, victim: int, other: int, page: int) -> None:
    assert _member_rows(store, victim) == 0
    assert _link_rows(store, victim) == 0
    # The note's [[mem:N]] text is untouched; its link row now reads as broken (dst_id NULL).
    broken = store._conn.execute(
        "SELECT dst_title FROM item_links WHERE src_id=? AND relation='refers' AND dst_id IS NULL", (page,)).fetchall()
    assert [row[0] for row in broken] == [f"mem:{victim}"]
    assert _member_rows(store, other) == 1
    assert not _dirty(store, topic_id)  # the remaining members did not change


def test_delete_removes_topic_member_and_links(tmp_path):
    store = _store(tmp_path)
    topic_id, victim, other, page = _seed_orphans(store)
    assert store.delete(NS, victim)
    _assert_cleaned(store, topic_id, victim, other, page)
    store.close()


def test_delete_source_containing_shares_the_cleanup(tmp_path):
    store = _store(tmp_path)
    topic_id, victim, other, page = _seed_orphans(store)
    assert store.delete_sources_matching(NS, "agent:s", "A one") == 1
    _assert_cleaned(store, topic_id, victim, other, page)
    store.close()


def test_delete_with_heir_moves_pinned_membership(tmp_path):
    store = _store(tmp_path)
    topic_id, ids = _topic_from(store, ["A one", "A two", "A three"], _axis(0))
    old = ids[0]
    with store._lock, store._conn:
        marks = topics.manual_marks(store._conn)
        marks.pin(old, topic_id)
        topics.save_manual_marks(store._conn, marks)
    heir = _remember(store, "B far away", _axis(3))  # too far to join the topic by itself
    assert _member_rows(store, heir) == 0
    _clean(store, topic_id)
    assert store.delete(NS, old, heir=heir)
    member = store._conn.execute("SELECT label FROM item_links WHERE relation='topic' AND src_id=?", (heir,)).fetchone()
    assert member is not None and member[0] == topic_id
    assert topics.manual_marks(store._conn).pinned_to(heir) == topic_id
    assert _member_rows(store, old) == 0 and _dirty(store, topic_id)
    store.close()


def test_delete_with_heir_without_pin_changes_nothing_for_heir(tmp_path):
    store = _store(tmp_path)
    topic_id, ids = _topic_from(store, ["A one", "A two", "A three"], _axis(0))
    heir = _remember(store, "B far away", _axis(3))
    assert store.delete(NS, ids[0], heir=heir)
    assert _member_rows(store, heir) == 0
    assert topics.manual_marks(store._conn).pinned_to(heir) is None
    store.close()


def _rows(store, sql: str, *params) -> list[tuple]:
    return [tuple(row) for row in store._conn.execute(sql, params)]


def test_prune_dangling_links_drops_only_rows_with_a_gone_endpoint(tmp_path):
    """Rows left by a raw-SQL delete go; healthy rows and title-only links stay."""
    store = _store(tmp_path)
    topic_id, ids = _topic_from(store, ["A one", "A two", "A three"], _axis(0))
    gone, live = ids[0], ids[1]
    free_a, free_b = _remember(store, "B far", _axis(3)), _remember(store, "C far", _axis(4))
    note = _note(store, "Live")
    trashed = _note(store, "Trashed", deleted=2.0)
    with store._lock, store._conn:
        folder = int(store._conn.execute("INSERT INTO folders(name,color,created_at) VALUES ('f','#000000',1.0)").lastrowid)
    _link(store, live, gone, to_memory=True, relation="supersedes")   # dst gone -> deleted
    _link(store, gone, note, to_memory=False)                         # src gone -> deleted
    _link(store, trashed, live, to_memory=True)                       # src trashed -> deleted
    _link(store, note, gone, to_memory=True)                          # refers into gone -> unresolved
    _link(store, note, live, to_memory=True)                          # healthy
    _link(store, live, trashed, to_memory=False)                      # into a trashed note: kept (restorable)
    _link(store, note, None, to_memory=False)                         # title-only unresolved: kept
    with store._lock, store._conn:
        store._conn.execute("INSERT INTO folder_items(folder_id,item_type,item_id) VALUES (?,?,?)", (folder, "memory", gone))
        store._conn.execute("INSERT INTO folder_items(folder_id,item_type,item_id) VALUES (?,?,?)", (folder, "note", 999))
        store._conn.execute("INSERT INTO folder_items(folder_id,item_type,item_id) VALUES (?,?,?)", (folder, "memory", live))
        # discovery rows: one link and one check mark of the gone memory, one healthy pair
        store._conn.execute("INSERT INTO memory_pairs(low_id,high_id,cosine,created_at) VALUES (?,?,0.9,1.0)",
                            (gone, free_a))
        store._conn.execute("INSERT INTO memory_pairs(low_id,high_id,cosine,created_at) VALUES (?,?,0.9,1.0)",
                            (free_a, free_b))
        store._conn.executemany("INSERT INTO memory_checks(memory_id,stage,checked_at) VALUES (?,'topic',1.0)",
                                [(gone,), (free_a,)])
        # absorb marks: one of the gone memory, one healthy
        store._conn.executemany("INSERT INTO memory_checks(memory_id,stage,text_hash,chunk_id) VALUES (?,'absorb','h',0)",
                                [(gone,), (free_a,)])
        store._conn.execute("DELETE FROM items WHERE type='memory' AND id=?", (gone,))  # a delete that skips store.delete leaves the dependent rows
    assert store.prune_orphan_links() == {"topic_links": 1, "links": 3, "links_unresolved": 1, "folder_items": 2,
                                            "memory_pairs": 1, "memory_checks": 2, "revisions": 0, "piece_chain": 0}
    assert _rows(store, "SELECT memory_id FROM memory_checks WHERE stage='absorb'") == [(free_a,)]
    assert _rows(store, "SELECT low_id,high_id FROM memory_pairs") == [(free_a, free_b)]
    assert _rows(store, "SELECT memory_id FROM memory_checks WHERE stage='topic'") == [(free_a,)]
    assert _member_rows(store, gone) == 0 and _member_rows(store, live) == 1
    assert _link_rows(store, gone) == 0
    assert sorted(_rows(store, "SELECT src_id,dst_id,dst_title FROM item_links WHERE relation IN ('refers','supersedes')"),
                  key=repr) == sorted([(note, None, f"mem:{gone}"), (note, live, f"mem:{live}"), (live, trashed, "Page"),
                                       (note, None, "Page")], key=repr)
    assert _rows(store, "SELECT item_type,item_id FROM folder_items") == [("memory", live)]
    assert store.prune_orphan_links() == {"topic_links": 0, "links": 0, "links_unresolved": 0, "folder_items": 0,
                                            "memory_pairs": 0, "memory_checks": 0, "revisions": 0, "piece_chain": 0}
    assert _rows(store, "SELECT COUNT(*) FROM items WHERE type='memory'")[0][0] == 4
    assert _rows(store, "SELECT COUNT(*) FROM items WHERE title IS NOT NULL")[0][0] == 2
    store.close()
