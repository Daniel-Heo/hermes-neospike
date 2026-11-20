"""Topic points on the map (mean of the members, projected like a folder marker, never cached), the topic
panel's hand edits by topic id (kept as manual marks), and no API that renames a topic."""
from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path

import numpy as np

from neospike import topics as topic_pages
from neospike.dashboard import plugin_api as api
from neospike.store import META_LIVE_NAMESPACE, topic_handle
from neospike.tests.support_items import chunk_rows
from neospike.tests.test_sphere_map import (NAMESPACE, UNIT_TOLERANCE, _client, _db, _full_vectors, _reference_pca, _seed,
                                            home)  # noqa: F401  (home is a fixture)

SEED_MEMORIES = 40
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
MOVE_ROUTE = "/topics/members/move"


class _Writer:
    """Writes the scratch database with the real sqlite3.connect (the client's guard allows mode=ro only)."""

    def __init__(self, home: Path) -> None:
        self.path = _db(home)
        self.connect = sqlite3.connect
        # _seed stores durable memories; only memory_item rows can be topic members (topics._ASSIGNABLE_SQL).
        self.run("UPDATE items SET kind='memory_item'", [()])
        # The dashboard's CLI child works on the live namespace recorded in meta (store.pick_namespace).
        self.run("INSERT OR REPLACE INTO meta_entries(key,value) VALUES (?,?)", [(META_LIVE_NAMESPACE, NAMESPACE)])

    def run(self, sql: str, rows: list[tuple]) -> None:
        with closing(self.connect(self.path)) as connection:
            connection.executemany(sql, rows)
            connection.commit()

    def topic(self, name: str, members: list[int], *, deleted: bool = False) -> int:
        """A topic of these members (v22: its topic links); a deleted topic has none. Returns its handle."""
        now = time.time()
        with closing(self.connect(self.path)) as connection:
            if not deleted:
                connection.executemany("INSERT INTO item_links(src_id,relation,label,created_at) VALUES (?,'topic',?,?)",
                                       [(memory_id, name, now) for memory_id in members])
            connection.commit()
        return topic_handle(name)

    def read(self, sql: str, args: tuple = ()) -> list[tuple]:
        with closing(self.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)) as connection:
            return connection.execute(sql, args).fetchall()

    def marks(self) -> topic_pages.ManualMarks:
        with closing(self.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)) as connection:
            return topic_pages.manual_marks(connection)


def _items(payload: dict) -> list[dict]:
    return [point for point in payload["points"] if point["lane"] != "topic"]


def _topic(payload: dict, topic_id: int) -> dict | None:
    return next((point for point in payload["points"] if point["key"] == f"topic:{topic_id}"), None)


def _expected_position(home: Path, payload: dict, member_ids: list[int]) -> np.ndarray:
    """Independent numpy reference: full-dimension member mean, projected with the map's PCA, normalized."""
    items = _items(payload)
    matrix = _full_vectors(home, items)
    mean, axes = _reference_pca(matrix)
    wanted = {f"mem:{memory_id}" for memory_id in member_ids}
    rows = [index for index, point in enumerate(items) if point["key"] in wanted]
    position = (matrix[rows].mean(axis=0) - mean) @ axes.T
    return position / np.linalg.norm(position)


def _xyz(point: dict) -> np.ndarray:
    return np.array([point["x"], point["y"], point["z"]])


# acceptance 3 -----------------------------------------------------------------------------
def test_topic_point_is_the_projected_member_mean_and_follows_member_changes(home: Path, monkeypatch) -> None:
    memories, _saved = _seed(home, SEED_MEMORIES)
    writer = _Writer(home)
    first_members = memories[2:5]
    topic = writer.topic("Synthetic group", first_members)
    empty = writer.topic("No members", [])
    gone = writer.topic("Deleted", [memories[5]], deleted=True)
    client = _client(monkeypatch)
    payload = client.get("/map").json()
    point = _topic(payload, topic)
    assert point is not None and _topic(payload, empty) is None and _topic(payload, gone) is None
    assert (point["lane"], point["kind"], point["title"], point["count"], point["topic"]) == \
        ("topic", "topic", "Synthetic group", 3, True)
    assert point["folder_id"] is None and point["neighbors"] == []
    assert sorted(point["links"]) == sorted(f"mem:{memory_id}" for memory_id in first_members)
    assert np.abs(_xyz(point) - _expected_position(home, payload, first_members)).max() <= UNIT_TOLERANCE

    # Hand edit through the panel API (topic id), then the next /map: same cached layout, new topic position.
    moved = client.post(MOVE_ROUTE, json={"items": [f"mem:{memories[6]}", f"mem:{memories[7]}"], "to_topic": topic})
    assert moved.status_code == 200 and moved.json()["moved"] == 2, moved.text
    again = client.get("/map").json()
    assert again["cached"] is True and again["hash"] == payload["hash"]
    after = _topic(again, topic)
    members = [*first_members, memories[6], memories[7]]
    assert after["count"] == 5 and np.abs(_xyz(after) - _xyz(point)).max() > UNIT_TOLERANCE
    assert np.abs(_xyz(after) - _expected_position(home, again, members)).max() <= UNIT_TOLERANCE

    # Every member taken out: no point.
    removed = client.post(MOVE_ROUTE, json={"items": [f"mem:{memory_id}" for memory_id in members], "remove_from_topic": topic})
    assert removed.status_code == 200 and removed.json()["moved"] == 5, removed.text
    assert _topic(client.get("/map").json(), topic) is None


def test_note_position_follows_its_chunks(home: Path, monkeypatch) -> None:
    _memories, saved = _seed(home, SEED_MEMORIES)
    writer = _Writer(home)
    client = _client(monkeypatch)
    note_id = saved[2]["id"]
    before = client.get("/map").json()
    with closing(sqlite3.connect(f"file:{writer.path.as_posix()}?mode=ro", uri=True)) as reader:
        reader.row_factory = sqlite3.Row
        first_part = chunk_rows(reader, note_id)[0]
    blob, dimensions, part_id = first_part["embedding"], len(first_part["embedding"]) // 4, first_part["id"]
    vector = np.random.default_rng(410).normal(size=int(dimensions)).astype(np.float32)
    assert vector.tobytes() != blob
    writer.run("UPDATE items SET embedding=? WHERE id=?", [(vector.tobytes(), part_id)])
    after = client.get("/map").json()
    assert after["cached"] is False and after["hash"] != before["hash"]
    old = next(point for point in before["points"] if point["key"] == f"note:{note_id}")
    new = next(point for point in after["points"] if point["key"] == f"note:{note_id}")
    assert np.abs(_xyz(new) - _xyz(old)).max() > UNIT_TOLERANCE
    # Still the normalized chunk-sum rule (independent reference over the new vectors).
    items = _items(after)
    matrix = _full_vectors(home, items)
    mean, axes = _reference_pca(matrix)
    row = next(index for index, point in enumerate(items) if point["key"] == f"note:{note_id}")
    expected = (matrix[row] - mean) @ axes.T
    assert np.abs(_xyz(new) - expected / np.linalg.norm(expected)).max() <= UNIT_TOLERANCE


# acceptance 4 -----------------------------------------------------------------------------
def test_panel_edits_are_pinned_or_excluded_and_the_title_cannot_change(home: Path, monkeypatch) -> None:
    memories, _saved = _seed(home, SEED_MEMORIES)
    writer = _Writer(home)
    topic = writer.topic("Fixed title", memories[2:4])
    other = writer.topic("Other", [memories[8]])
    client = _client(monkeypatch)
    added = client.post(MOVE_ROUTE, json={"items": [f"mem:{memories[5]}"], "to_topic": topic, "name": "Renamed", "title": "Renamed"})
    assert added.status_code == 200 and added.json()["moved_ids"] == [memories[5]], added.text
    assert writer.marks().pinned_to(memories[5]) == "Fixed title"
    out = client.post(MOVE_ROUTE, json={"items": [f"mem:{memories[2]}"], "remove_from_topic": topic})
    assert out.status_code == 200 and out.json()["moved_ids"] == [memories[2]], out.text
    assert writer.marks().excluded_from(memories[2]) == ["Fixed title"]
    assert sorted(row[0] for row in writer.read("SELECT src_id FROM item_links WHERE relation='topic' AND label=?",
                                                ("Fixed title",))) == \
        sorted([memories[3], memories[5]])
    # Into another topic from the panel's Move to...: pinned there.
    across = client.post(MOVE_ROUTE, json={"items": [f"mem:{memories[3]}"], "to_topic": other})
    assert across.status_code == 200 and writer.marks().pinned_to(memories[3]) == "Other"
    # The title is read-only everywhere: no write route but the member move, and nothing above renamed it.
    writes = sorted((route.path, method) for route in api.router.routes if "topic" in route.path
                    for method in getattr(route, "methods", set()) & WRITE_METHODS)
    assert writes == [(MOVE_ROUTE, "POST")]
    assert client.put(f"/topics/{topic}", json={"name": "Renamed"}).status_code in (404, 405)
    assert client.patch(f"/topics/{topic}", json={"name": "Renamed"}).status_code in (404, 405)
    assert writer.read("SELECT DISTINCT label FROM item_links WHERE relation='topic' AND label=?", ("Fixed title",)) == [
        ("Fixed title",)]
    panel = client.get(f"/connections/topic/{topic}").json()
    assert panel["title"] == "Fixed title" and [item["id"] for item in panel["items"]] == [memories[5]]
    assert client.post(MOVE_ROUTE, json={"items": ["mem:1"], "to_topic": topic, "remove_from_topic": topic}).status_code == 422
    assert client.post(MOVE_ROUTE, json={"items": ["mem:1"], "to_topic": 999999}).status_code >= 400
