"""The map panel's Connected list and the map's lines read one edge set (plugin_api._graph_edges)."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from neospike.store import MemoryStore, topic_handle
from neospike.tests.support_items import body as note_body_of
from neospike.tests.test_notes import DIMENSIONS
from neospike.tests.test_sphere_map import _client, _db, _seed, home  # noqa: F401  (home is a fixture)

SEED_MEMORIES = 6


def _connect(store: MemoryStore, *, topic_refers: list[int], topic_memories: list[int], supersedes: list[tuple[int, int]],
             reverted: tuple[int, int]) -> tuple[int, int]:
    """The Kitchen note citing topic_refers, a page-less topic with topic_memories as members, and supersedes
    links (one of them reverted). Returns (note id, topic id)."""
    connection = store._conn
    now = time.time()
    note_id = connection.execute("SELECT id FROM items WHERE title='Kitchen'").fetchone()[0]
    topic_id = topic_handle("kitchen")
    connection.executemany("INSERT INTO item_links(src_id,dst_id,dst_title,relation) VALUES (?,?,?,'refers')",
                           [(note_id, memory_id, f"mem:{memory_id}") for memory_id in topic_refers])
    connection.executemany("INSERT INTO item_links(src_id,relation,label,created_at) VALUES (?,'topic','kitchen',?)",
                           [(memory_id, now) for memory_id in topic_memories])
    connection.executemany("INSERT INTO item_links(src_id,dst_id,relation) VALUES (?,?,'supersedes')",
                           supersedes + [reverted])
    connection.execute("UPDATE item_links SET reverted_at=? WHERE relation='supersedes' AND src_id=? AND dst_id=?", (now, *reverted))
    connection.commit()
    return note_id, topic_id


def _seed_connections(home: Path) -> tuple[list[int], tuple[int, int]]:
    memories, _saved = _seed(home, SEED_MEMORIES)
    store = MemoryStore(_db(home), dimensions=DIMENSIONS)
    try:
        topic = _connect(store, topic_refers=[memories[0], memories[1]], topic_memories=[memories[1], memories[2]],
                         supersedes=[(memories[3], memories[1])], reverted=(memories[4], memories[1]))
    finally:
        store.close()
    return memories, topic


def test_topic_lists_members_note_lists_refers_and_memory_lists_both_directions(home: Path, monkeypatch) -> None:
    memories, (note, topic) = _seed_connections(home)
    client = _client(monkeypatch)
    listed = client.get(f"/connections/note/{note}").json()["items"]
    assert [(item["key"], item["relations"]) for item in listed] == [
        (f"mem:{memories[0]}", ["refers"]), (f"mem:{memories[1]}", ["refers"])]
    members = client.get(f"/connections/topic/{topic}").json()
    assert members["is_topic"] is True and members["title"] == "kitchen" and members["key"] == f"topic:{topic}"
    assert [(item["key"], item["relations"]) for item in members["items"]] == [
        (f"mem:{memories[1]}", ["topic member"]), (f"mem:{memories[2]}", ["topic member"])]
    middle = client.get(f"/connections/memory/{memories[1]}").json()["items"]
    # The reverted supersedes link (memories[4]) is gone; the topic shows as kind "topic", first.
    assert [(item["key"], item["kind"], item["relations"]) for item in middle] == [
        (f"topic:{topic}", "topic", ["topic member"]), (f"note:{note}", "note", ["referenced by"]),
        (f"mem:{memories[3]}", "memory", ["superseded by"])]
    newer = client.get(f"/connections/memory/{memories[3]}").json()["items"]
    assert [(item["key"], item["relations"]) for item in newer] == [(f"mem:{memories[1]}", ["supersedes"])]
    assert client.get(f"/connections/memory/{memories[5]}").json()["items"] == []
    assert client.get("/connections/memory/999999").status_code == 404
    assert client.get("/connections/folder/1").status_code == 404
    assert client.get("/connections/topic/999999").status_code == 404


def test_map_lines_equal_the_connected_lists(home: Path, monkeypatch) -> None:
    _seed_connections(home)
    client = _client(monkeypatch)
    points = client.get("/map").json()["points"]
    on_map = {other["key"] for other in points}
    for point in points:
        listed = client.get(f"/connections/{point['lane']}/{point['id']}").json()["items"]
        assert point["links"] == sorted(item["key"] for item in listed if item["key"] in on_map), point["key"]
    assert any(point["links"] for point in points)


def test_connections_carry_db_text_and_topic_flag(home: Path, monkeypatch) -> None:
    """Each Connected item carries its stored text unchanged; a topic says it is a topic; /topics lists it."""
    memories, (note, topic) = _seed_connections(home)
    client = _client(monkeypatch)
    reply = client.get(f"/connections/topic/{topic}").json()
    assert reply["is_topic"] is True and client.get(f"/connections/note/{note}").json()["is_topic"] is False
    connection = sqlite3.connect(f"file:{_db(home).as_posix()}?mode=ro", uri=True)  # the dashboard guard allows ro only
    try:
        texts = dict(connection.execute(f"SELECT id,text FROM items WHERE type='memory' AND id IN ({memories[1]},{memories[2]})"))
        body = note_body_of(connection, note)
    finally:
        connection.close()
    assert [item["text"] for item in reply["items"]] == [texts[memories[1]], texts[memories[2]]]
    back = client.get(f"/connections/memory/{memories[1]}").json()
    assert back["is_topic"] is False and back["items"][1]["text"] == body
    assert client.get("/topics").json()["topics"] == [{"id": topic, "title": "kitchen", "members": 2}]
    assert client.post("/topics/members/move", json={"items": ["mem:1"]}).status_code == 422
