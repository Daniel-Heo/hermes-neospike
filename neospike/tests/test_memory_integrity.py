"""Memory integrity, one section per behaviour:
recall leaves updated_at alone; the save reply's older list is in content-time order
a duplicate visible more widely lifts the new memory to its scope
recall lane errors and embedding failures are logged with their cause
the same content failing twice is stored embedding-pending; embed-pending fills it and dedupes
deleting a memory removes its revisions/folder rows, empty topics, and the memories it superseded"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest

from neospike import mcp_server, sphere_map, topics
from neospike.config import NeospikeConfig
from neospike.notes import embed_pending
from neospike.provider import NeospikeProvider
from neospike.store import MemoryStore, install_change_triggers
from neospike.tests.supersede_links import link_supersede
from neospike.tests.test_remember_dedupe import DIMENSIONS, NEW_TEXT, SESSION, _at_cosine, _new_id

PROJECT = "D:/work/project-a"


class FlakyEmbedder:
    """NEW_TEXT -> e0, other texts -> the last axis; `failing` makes every call raise (the daemon down)."""
    dimensions = DIMENSIONS

    def __init__(self) -> None:
        self.failing = False

    def _vector(self, text: str) -> list[float]:
        if self.failing:
            raise RuntimeError("Inference startup recently failed; retry on later demand")
        if text == NEW_TEXT:
            return [1.0] + [0.0] * (DIMENSIONS - 1)
        return [0.0] * (DIMENSIONS - 1) + [1.0]

    def embed_document(self, text: str, **_kwargs) -> list[float]:
        return self._vector(text)

    def embed_query(self, text: str, **_kwargs) -> list[float]:
        return self._vector(text)

    def take_warnings(self) -> list[str]:
        return []


@pytest.fixture
def flaky():
    return FlakyEmbedder()


@pytest.fixture
def provider(tmp_path: Path, flaky: FlakyEmbedder):
    home_dir = tmp_path / "home"
    NeospikeConfig().save(home_dir)
    instance = NeospikeProvider(embedder=flaky)
    instance.initialize(SESSION, hermes_home=str(home_dir), cwd=str(home_dir))
    yield instance
    instance.shutdown()


def _seed(provider: NeospikeProvider, text: str, similarity: float, *, scope: str = "global",
          project: str = "", session_id: str = "earlier") -> int:
    memory_id = provider._memory_db.insert(provider._ns_label, text, _at_cosine(similarity), source=f"agent:{session_id}",
                                       kind="memory_item", project=project,
                                       metadata={"scope": scope, "subject": "release", "durability": "ongoing",
                                                 "session_id": session_id})
    assert memory_id is not None
    return memory_id


def _remember(provider: NeospikeProvider, scope: str = "global") -> dict:
    return json.loads(provider.handle_tool_call("neospike_memory_save", {
        "text": NEW_TEXT, "scope": scope, "subject": "release", "durability": "stable", "kind": "fact", "basis": "user_said"}))


def _one(store: MemoryStore, sql: str, *args):
    return store._conn.execute(sql, args).fetchone()


def _logs(provider: NeospikeProvider, prefix: str) -> list[dict]:
    directory = provider._home_root / "neospike" / "logs"
    return [json.loads(line) for path in sorted(directory.glob(f"{prefix}-*.jsonl"))
            for line in path.read_text(encoding="utf-8").splitlines()]


# Recall and updated_at ---------------------------------------------------------------------------------------------
def test_recall_leaves_updated_at_and_a_content_change_still_bumps_it(provider: NeospikeProvider) -> None:
    store = provider._memory_db
    memory_id = _seed(provider, NEW_TEXT + " (seed)", 1.0)
    with store._lock, store._conn:
        store._conn.execute("UPDATE items SET updated_at=1000.0 WHERE id=?", (memory_id,))
    hits = store.search(provider._ns_label, "release", [1.0] + [0.0] * (DIMENSIONS - 1), limit=3)
    assert [hit.id for hit in hits] == [memory_id]
    row = _one(store, "SELECT updated_at,accessed_at,access_count FROM items WHERE type='memory' AND id=?", memory_id)
    assert row[0] == 1000.0 and row[1] is not None and row[2] == 1
    assert store.archive(provider._ns_label, memory_id)
    assert _one(store, "SELECT updated_at FROM items WHERE type='memory' AND id=?", memory_id)[0] > 1000.0


def test_v16_trigger_is_replaced_on_open_and_stored_values_stay(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = MemoryStore(path, dimensions=4)
    memory_id = store.insert("ns", "kept text", [1.0, 0.0, 0.0, 0.0], source="s")
    with store._lock, store._conn:  # the v16 database: the update trigger fires on any UPDATE
        store._conn.execute("DROP TRIGGER items_track_au")
        store._conn.execute("CREATE TRIGGER items_track_au AFTER UPDATE ON items WHEN new.updated_at IS "
                            "old.updated_at BEGIN UPDATE items SET updated_at=12345.0 WHERE id=new.id; END")
        store._conn.execute("UPDATE meta_entries SET value='16' WHERE key='schema_version'")
        store._conn.execute("UPDATE items SET updated_at=777.0 WHERE id=?", (memory_id,))
    store.close()
    store = MemoryStore(path, dimensions=4)
    assert "UPDATE OF" in _one(store, "SELECT sql FROM sqlite_master WHERE name='items_track_au'")[0]
    assert _one(store, "SELECT updated_at FROM items WHERE type='memory' AND id=?", memory_id)[0] == 777.0  # not rewritten
    store.search("ns", "kept", [1.0, 0.0, 0.0, 0.0])
    assert _one(store, "SELECT updated_at FROM items WHERE type='memory' AND id=?", memory_id)[0] == 777.0
    before = _one(store, "SELECT sql FROM sqlite_master WHERE name='items_track_au'")[0]
    with store._lock, store._conn:
        install_change_triggers(store._conn)  # same SQL: kept, not rebuilt
    assert _one(store, "SELECT sql FROM sqlite_master WHERE name='items_track_au'")[0] == before
    store.close()


def test_older_list_is_in_content_time_order_not_recall_order(provider: NeospikeProvider) -> None:
    store = provider._memory_db
    first = _seed(provider, "The release train leaves on Mondays.", 0.85)
    second = _seed(provider, "The release train leaves on Fridays now.", 0.82)
    with store._lock, store._conn:
        store._conn.execute("UPDATE items SET created_at=100.0, updated_at=100.0 WHERE id=?", (first,))
        store._conn.execute("UPDATE items SET created_at=200.0, updated_at=200.0 WHERE id=?", (second,))
    store.search(provider._ns_label, "release", _at_cosine(0.85), limit=1)  # recall touches the first one
    assert [item["id"] for item in _remember(provider)["older"]] == [first, second]


def test_a_later_revision_makes_a_memory_newer(provider: NeospikeProvider) -> None:
    store = provider._memory_db
    first = _seed(provider, "The release train leaves on Mondays.", 0.85)
    second = _seed(provider, "The release train leaves on Fridays now.", 0.82)
    with store._lock, store._conn:
        store._conn.execute("UPDATE items SET created_at=100.0 WHERE id=?", (first,))
        store._conn.execute("UPDATE items SET created_at=200.0 WHERE id=?", (second,))
        store._conn.execute("INSERT INTO item_revisions(item_id,rev,text,metadata_json,edited_at,edited_by) "
                            "VALUES (?,1,'old body','{}',300.0,'agent')", (first,))
    assert [item["id"] for item in _remember(provider)["older"]] == [second, first]


# Scope of a duplicate ---------------------------------------------------------------------------------------------
def test_session_memory_duplicating_a_global_one_becomes_global(provider: NeospikeProvider) -> None:
    store = provider._memory_db
    old = _seed(provider, "Release train departs every Thursday.", 0.95)
    reply = _remember(provider, scope="session")
    new = _new_id(store)
    assert reply == {"stored": True, "deleted": [old], "scope": "global"}
    meta, project, scope = _one(store, "SELECT metadata_json,project,scope FROM items WHERE type='memory' AND id=?", new)
    assert scope == "global" and project == "" and json.loads(meta)["session_id"] == SESSION
    assert _one(store, "SELECT COUNT(*) FROM items WHERE type='memory' AND id=?", old)[0] == 0
    assert [hit.id for hit in store.search(provider._ns_label, "x", [1.0] + [0.0] * (DIMENSIONS - 1),
                                           project="", session_id="another-session")] == [new]


def test_project_memory_duplicating_a_global_one_becomes_global(provider: NeospikeProvider) -> None:
    provider._project_dir = PROJECT
    old = _seed(provider, "Release train departs every Thursday.", 0.95)
    reply = _remember(provider, scope="project")
    assert reply["scope"] == "global" and reply["deleted"] == [old]
    assert _one(provider._memory_db, "SELECT project FROM items WHERE type='memory' AND id=?", _new_id(provider._memory_db))[0] == ""


def test_session_memory_duplicating_a_project_one_takes_that_project(provider: NeospikeProvider) -> None:
    provider._project_dir = PROJECT
    old = _seed(provider, "Release train departs every Thursday.", 0.95, scope="project", project=PROJECT)
    reply = _remember(provider, scope="session")
    scope, project = _one(provider._memory_db, "SELECT scope,project FROM items WHERE type='memory' AND id=?",
                          _new_id(provider._memory_db))
    assert reply["scope"] == "project" and scope == "project" and project == PROJECT
    assert reply["deleted"] == [old]


def test_wider_new_memory_keeps_its_scope(provider: NeospikeProvider) -> None:
    old = _seed(provider, "Release train departs every Thursday.", 0.95, scope="session", session_id=SESSION)
    reply = _remember(provider, scope="global")
    assert reply == {"stored": True, "deleted": [old]}
    assert _one(provider._memory_db, "SELECT scope FROM items WHERE type='memory' AND id=?",
                _new_id(provider._memory_db))[0] == "global"


# Failure causes ---------------------------------------------------------------------------------------------
def test_recall_log_keeps_each_lane_cause(provider: NeospikeProvider, flaky: FlakyEmbedder) -> None:
    flaky.failing = True
    provider.prefetch("anything at all", session_id=SESSION)
    entry = _logs(provider, "recall")[-1]
    assert "memory" in entry["unavailable_lanes"] and sorted(entry["lane_errors"]) == sorted(entry["unavailable_lanes"])
    assert entry["lane_errors"]["memory"] == "RuntimeError: Inference startup recently failed; retry on later demand"


def test_embedding_failure_of_a_save_is_logged(provider: NeospikeProvider, flaky: FlakyEmbedder) -> None:
    flaky.failing = True
    _remember(provider)
    entry = _logs(provider, "embedding-failures")[-1]
    assert entry["operation"] == "remember"
    assert entry["cause"] == "RuntimeError: Inference startup recently failed; retry on later demand"
    assert entry["ts"]


# Embedding-pending ---------------------------------------------------------------------------------------------
def test_two_failures_store_pending_then_embed_pending_fills_and_dedupes(provider: NeospikeProvider,
                                                                          flaky: FlakyEmbedder) -> None:
    store = provider._memory_db
    old = _seed(provider, "Release train departs every Thursday.", 0.95)
    flaky.failing = True
    first = _remember(provider, scope="session")
    assert first["stored"] is False and first["error"] == "embedding_failed"
    assert "RuntimeError" in first["message"] and "once more" in first["message"]
    assert _one(store, "SELECT COUNT(*) FROM items WHERE type='memory' AND text=?", NEW_TEXT)[0] == 0
    second = _remember(provider, scope="session")
    assert second == {"stored": True, "embedding": "pending"}
    new = _new_id(store)
    assert _one(store, "SELECT length(embedding) FROM items WHERE type='memory' AND id=?", new)[0] == 0
    assert _one(store, "SELECT COUNT(*) FROM items WHERE type='memory' AND id=?", old)[0] == 1  # no comparison without a vector
    assert [row[0] for row in store._conn.execute(
        "SELECT rowid FROM items_fts WHERE items_fts MATCH 'Thursdays'")] == [new]  # text search lists it
    assert [entry["operation"] for entry in _logs(provider, "embedding-failures")] == ["remember", "remember_pending"]
    flaky.failing = False
    done = embed_pending(store, flaky, chunk_editor="formula")
    assert done["filled_ids"] == [new] and done["deleted_ids"] == [old] and done["scope_raised_ids"] == [new]
    assert done["left"] == 0
    assert _one(store, "SELECT length(embedding) FROM items WHERE type='memory' AND id=?", new)[0] == DIMENSIONS * 4
    assert _one(store, "SELECT scope FROM items WHERE type='memory' AND id=?", new)[0] == "global"
    assert embed_pending(store, flaky, chunk_editor="formula")["pending"] == 0


def test_failure_window_is_per_content(provider: NeospikeProvider, flaky: FlakyEmbedder) -> None:
    flaky.failing = True
    assert _remember(provider)["error"] == "embedding_failed"
    other = json.loads(provider.handle_tool_call("neospike_memory_save", {
        "text": "Some other fact entirely.", "scope": "global", "subject": "x", "durability": "stable", "kind": "fact", "basis": "user_said"}))
    assert other["error"] == "embedding_failed"  # another body is a first failure
    assert _remember(provider, scope="session")["error"] == "embedding_failed"  # another scope too
    assert _remember(provider)["embedding"] == "pending"


def test_failed_fill_stays_pending_and_is_logged(provider: NeospikeProvider, flaky: FlakyEmbedder) -> None:
    flaky.failing = True
    _remember(provider)
    _remember(provider)
    logged: list[tuple[str, str]] = []
    done = embed_pending(provider._memory_db, flaky, chunk_editor="formula", failure_log=_Collect(logged))
    assert done["failed"] == 1 and done["left"] == 1 and logged[0][0] == "embed-pending"


class _Collect:
    def __init__(self, into: list) -> None:
        self.into = into

    def __call__(self, operation: str, cause: str) -> None:
        self.into.append((operation, cause))


def test_every_vector_reader_skips_a_pending_memory(provider: NeospikeProvider, flaky: FlakyEmbedder) -> None:
    store, ns = provider._memory_db, provider._ns_label
    for index in range(7):  # enough live memories for links, a topic and a conversion pass
        _seed(provider, f"Release note number {index} about trains.", 0.6 + index * 0.01)
    flaky.failing = True
    _remember(provider)
    assert _remember(provider)["embedding"] == "pending"
    pending = _new_id(store)
    # The first row of each vector batch is the pending one (lowest id), as in a home where it came first.
    with store._lock, store._conn:
        store._conn.execute("UPDATE items SET id=0 WHERE id=?", (pending,))
    flaky.failing = False
    query = [1.0] + [0.0] * (DIMENSIONS - 1)
    assert store.search(ns, "x", query, limit=10) is not None
    assert store.closest_memories(ns, 0, limit=3) == []
    assert all(memory_id != 0 for memory_id, _ in store.closest_memories(ns, 1, limit=10))
    provider.prefetch("release trains", session_id=SESSION)
    assert provider._failed_lanes == []
    assert topics.assign_all(store, ns) is not None
    assert topics.discover(store, ns) is not None
    assert topics.absorb(store, ns) is not None
    assert topics.topic_to_note(store, ns)["errors"] == []
    assert all(hit["id"] != 0 for hit in mcp_server.search_memory_and_formula_readonly(
        store._conn, ns, query, dimensions=DIMENSIONS, limit=10, project="", session_id=SESSION, threshold=-1.0))
    assert all(item.get("id") != "mem:0" for item in sphere_map.build_map(store._conn).get("items", []))
    assert _one(store, "SELECT COUNT(*) FROM memory_checks WHERE memory_id=0")[0] == 0
    assert _one(store, "SELECT COUNT(*) FROM item_links WHERE relation='topic' AND src_id=0")[0] == 0
    assert _one(store, "SELECT length(embedding) FROM items WHERE type='memory' AND id=0")[0] == 0  # still pending, not lost


def test_mirror_replace_keeps_the_new_text_when_embedding_fails(provider: NeospikeProvider,
                                                                  flaky: FlakyEmbedder) -> None:
    provider.on_memory_write("add", "memory", "User prefers short replies.")
    flaky.failing = True
    provider.on_memory_write("replace", "memory", "User prefers very short replies.",
                             {"old_text": "short replies"})
    rows = provider._memory_db._conn.execute(
        "SELECT text,length(embedding) FROM items WHERE type='memory' AND origin='builtin:memory'").fetchall()
    assert [tuple(row) for row in rows] == [("User prefers very short replies.", 0)]


# helpers of delete cleanup -------------------------------------------------------------------------------------
def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "neospike" / "memory.db", dimensions=DIMENSIONS)


# Delete cleanup ---------------------------------------------------------------------------------------------
def _item(store: MemoryStore, text: str, similarity: float = 0.5) -> int:
    memory_id = store.insert("ns", text, _at_cosine(similarity), source="agent:s", kind="memory_item",
                             metadata={"scope": "global", "subject": "s", "session_id": "s"})
    assert memory_id is not None
    return memory_id


def test_delete_removes_revisions_and_folder_rows(tmp_path: Path) -> None:
    store = _store(tmp_path)
    memory_id = _item(store, "a memory with history")
    with store._lock, store._conn:
        store._push_revision(memory_id, "older body", "{}", edited_by="agent", session_id="s", now=1.0)
        folder = store._conn.execute("INSERT INTO folders(name,color,created_at) VALUES ('f','red',1.0)").lastrowid
        store._conn.execute("INSERT INTO folder_items(folder_id,item_type,item_id) VALUES (?,'memory',?)", (folder, memory_id))
    assert store.delete("ns", memory_id)
    assert _one(store, "SELECT COUNT(*) FROM item_revisions")[0] == 0
    assert _one(store, "SELECT COUNT(*) FROM folder_items")[0] == 0
    store.close()


def test_link_prune_removes_revisions_left_by_older_deletes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    memory_id = _item(store, "a memory with history")
    with store._lock, store._conn:
        store._push_revision(memory_id, "older body", "{}", edited_by="agent", session_id="s", now=1.0)
        store._conn.execute("DELETE FROM items WHERE type='memory' AND id=?", (memory_id,))  # a delete that skips store.delete leaves the revisions
    assert store.prune_orphan_links()["revisions"] == 1
    assert _one(store, "SELECT COUNT(*) FROM item_revisions")[0] == 0
    store.close()


def test_topic_left_without_members_is_deleted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    one, two = _item(store, "first topic member"), _item(store, "second topic member")
    with store._lock, store._conn:
        name = topics._create_topic(store._conn, topics._member_rows(store._conn, [one, two]), time.time())
    assert store.delete("ns", one)
    assert topics.topic_exists(store._conn, name)
    assert store.delete("ns", two)
    assert not topics.topic_exists(store._conn, name)  # v22: a topic is its links; the last one went with the member
    with store._lock, store._conn:  # a topic link left by a delete that skipped store.delete
        store._conn.execute("INSERT INTO item_links(src_id,relation,label) VALUES (999,'topic','left over')")
    assert store.prune_orphan_links()["topic_links"] == 1
    assert not topics.topic_exists(store._conn, "left over")
    store.close()


def test_deleting_a_replacer_deletes_what_it_superseded_except_archived(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old, kept, chain, replacer = (_item(store, text) for text in ("old plan", "archived plan", "older plan", "plan v3"))
    for target in (old, kept):
        assert link_supersede(store, replacer, target)["linked"]
    assert link_supersede(store, old, chain)["linked"]  # chain -> old -> replacer
    assert store.archive("ns", kept)
    assert store.delete("ns", replacer)
    alive = {int(row[0]) for row in store._conn.execute("SELECT id FROM items WHERE type='memory'")}
    assert alive == {kept}
    assert tuple(_one(store, "SELECT superseded_by,archived_at IS NOT NULL FROM items WHERE type='memory' AND id=?", kept)) == (None, 1)
    store.close()


def test_another_live_replacer_keeps_the_superseded_memory(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old, first, second = (_item(store, text) for text in ("old plan", "plan a", "plan b"))
    assert link_supersede(store, first, old)["linked"] and link_supersede(store, second, old)["linked"]
    assert store.delete("ns", second)
    assert _one(store, "SELECT superseded_by FROM items WHERE type='memory' AND id=?", old)[0] == first
    store.close()


def test_duplicate_deletion_moves_the_supersedes_link_to_the_new_memory(provider: NeospikeProvider) -> None:
    store = provider._memory_db
    superseded = _seed(provider, "The release train leaves on Tuesdays.", 0.1)
    duplicate = _seed(provider, "Release train departs every Thursday.", 0.95)
    assert link_supersede(store, duplicate, superseded)["linked"]
    assert _remember(provider)["deleted"] == [duplicate]
    new = _new_id(store)
    assert _one(store, "SELECT superseded_by FROM items WHERE type='memory' AND id=?", superseded)[0] == new
    assert [tuple(row) for row in store._conn.execute(
        "SELECT src_id,dst_id FROM item_links WHERE relation='supersedes' AND reverted_at IS NULL")] == [(new, superseded)]


def test_topic_to_note_deletes_the_memories_its_members_superseded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    members = [_item(store, f"member fact number {index}", 0.9) for index in range(topics._Tuning.NOTE_MEMBER_LIMIT + 1)]
    outside = _item(store, "an older fact the first member replaced", 0.2)
    assert link_supersede(store, members[0], outside)["linked"]
    with store._lock, store._conn:
        topics._create_topic(store._conn, topics._member_rows(store._conn, members), time.time())
    done = topics.topic_to_note(store, "ns")
    assert done["converted"] == 1 and done["errors"] == []
    assert _one(store, "SELECT COUNT(*) FROM items WHERE type='memory'")[0] == 0
    store.close()
