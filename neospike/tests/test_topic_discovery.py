"""New topics from incremental neighbour links (between all live memories; the links stay
when a memory gets a topic)."""
from __future__ import annotations

import time
from pathlib import Path

from neospike import backup as db_backup
from neospike import topics
from neospike.store import MemoryStore
from neospike.tests.test_topics import NS, _axis, _mix, _remember, _store


def _links(store: MemoryStore) -> list[tuple]:
    return [tuple(row) for row in store._conn.execute(
        "SELECT low_id,high_id FROM memory_pairs ORDER BY low_id,high_id")]


def _checked(store: MemoryStore) -> list[int]:
    return [int(row[0]) for row in store._conn.execute(
        "SELECT memory_id FROM memory_checks WHERE stage='topic' ORDER BY memory_id")]


def _members(store: MemoryStore, topic_id: str) -> list[int]:
    return [int(row[0]) for row in store._conn.execute(
        "SELECT src_id FROM item_links WHERE relation='topic' AND label=? ORDER BY src_id", (topic_id,))]


class _SourceCounter:
    """Wraps topics.nearest_links and records how many memories each call scores."""

    def __init__(self) -> None:
        self.original = topics.nearest_links
        self.sources: list[int] = []

    def __call__(self, rows, sources):
        self.sources.append(len(sources))
        return self.original(rows, sources)


def test_constants_are_the_user_decision():
    assert (topics._Tuning.LINK_COSINE, topics._Tuning.LINK_NEIGHBORS,
            topics._Tuning.GROUP_MAX_WITHOUT_TOPIC) == (0.6, 3, 5)


def test_four_neighbours_give_three_links_and_below_threshold_none(tmp_path, monkeypatch):
    store = _store(tmp_path)
    # Each neighbour is >= 0.6 from the hub only; between neighbours cos <= 0.72 * 0.71 < 0.6.
    neighbours = [_remember(store, f"n{axis}", _mix(cosine, 0, axis))
                  for axis, cosine in ((1, 0.72), (2, 0.71), (3, 0.70), (4, 0.69))]
    _remember(store, "far", _mix(0.59, 0, 5))  # below the 0.6 threshold to the hub
    first = topics.discover(store, NS)
    assert (first["link_checked"], first["links_added"]) == (5, 0) and _links(store) == []
    counter = _SourceCounter()
    monkeypatch.setattr(topics, "nearest_links", counter)
    hub = _remember(store, "hub", _axis(0))
    second = topics.discover(store, NS)
    assert (second["link_checked"], second["links_added"]) == (1, 3)
    assert counter.sources == [1]  # only the new memory was scored
    assert _links(store) == [(neighbours[0], hub), (neighbours[1], hub), (neighbours[2], hub)]
    assert hub in _checked(store) and len(_checked(store)) == 6
    third = topics.discover(store, NS)
    assert (third["link_checked"], third["links_added"]) == (0, 0) and counter.sources == [1]
    store.close()


def test_five_make_no_topic_six_make_one_and_keep_their_rows(tmp_path):
    store = _store(tmp_path)
    ids = [_remember(store, f"otter {axis}", _mix(0.95, 0, axis), subject="otter") for axis in range(1, 6)]
    result = topics.discover(store, NS)
    assert result["topic_new"] == 0 and len(_checked(store)) == 5 and _links(store)
    assert store._conn.execute("SELECT COUNT(*) FROM item_links WHERE relation='topic'").fetchone()[0] == 0
    ids.append(_remember(store, "otter six", _mix(0.95, 0, 6), subject="otter"))
    result = topics.discover(store, NS)
    assert result["link_checked"] == 1 and result["topic_new"] == 1
    [topic_id] = result["topic_ids"]
    assert _members(store, topic_id) == ids
    assert store._conn.execute(f"SELECT {topics.topic_changed_at_sql('?')}", (topic_id,)).fetchone()[0] > 0
    assert _links(store) and _checked(store) == ids  # links stay for later joins
    store.close()


def test_memory_with_a_topic_is_a_neighbour(tmp_path):
    """Neighbours are chosen whatever their topic."""
    store = _store(tmp_path)
    members = [_remember(store, f"member {index}", _axis(0)) for index in range(3)]
    with store._lock, store._conn:
        topics._create_topic(store._conn, topics._member_rows(store._conn, members), time.time())
    free = _remember(store, "free near the topic", _mix(0.95))
    result = topics.discover(store, NS)
    assert result["link_checked"] == 4
    assert {(free, member) for member in members} <= {(high, low) for low, high in _links(store)}
    store.close()


def test_joining_a_topic_keeps_the_rows(tmp_path):
    store = _store(tmp_path)
    first, second = _remember(store, "a", _mix(0.9, 0, 1)), _remember(store, "b", _mix(0.9, 0, 2))
    topics.discover(store, NS)
    assert _links(store) == [(first, second)] and _checked(store) == [first, second]
    members = [_remember(store, f"member {index}", _axis(0)) for index in range(3)]
    with store._lock, store._conn:
        topic_id = topics._create_topic(store._conn, topics._member_rows(store._conn, members), time.time())
    assigned = topics.assign_all(store, NS)
    assert assigned["assigned"] == 2 and _members(store, topic_id) == [first, second, *members]
    assert (first, second) in _links(store) and _checked(store) == [first, second, *members]
    store.close()


def test_delete_drops_the_rows_of_the_memory(tmp_path):
    store = _store(tmp_path)
    first, second = _remember(store, "a", _mix(0.9, 0, 1)), _remember(store, "b", _mix(0.9, 0, 2))
    topics.discover(store, NS)
    assert _links(store) == [(first, second)]
    assert store.delete(NS, first)
    assert _links(store) == [] and _checked(store) == [second]
    store.close()


def test_backup_carries_the_discovery_tables(tmp_path: Path):
    store = _store(tmp_path)
    directory = tmp_path / "backups"
    db_backup.run_backup(store.path, directory)
    first, second = _remember(store, "a", _mix(0.9, 0, 1)), _remember(store, "b", _mix(0.9, 0, 2))
    topics.discover(store, NS)
    increment = db_backup.run_backup(store.path, directory)
    assert {"memory_pairs", "memory_checks"} <= set(increment["replace_tables"])
    target = tmp_path / "restored.db"
    db_backup.restore(directory, target)
    restored = MemoryStore(target, dimensions=store.dimensions)
    try:
        assert _links(restored) == [(first, second)] and _checked(restored) == [first, second]
    finally:
        restored.close()
        store.close()
