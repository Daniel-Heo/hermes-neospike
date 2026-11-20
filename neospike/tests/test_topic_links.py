"""Topics only through links. No topic on save; a memory without a topic joins the topic of its closest
linked topic member (chains too); topics never merge; checks are per body; formula-migrate rebuilds membership.

Acceptance items of the work order are named in each test (A1 .. A6)."""
from __future__ import annotations

import json
import math
import time

import pytest

from neospike import topics
from neospike.config import NeospikeConfig
from neospike.provider import NeospikeProvider
from neospike.tests.test_remember_dedupe import no_remember_tidy  # noqa: F401 (fixture)
from neospike.tests.test_topics import NS, AxisEmbedder, _axis, _remember, _store, _topic_from

# Toy vectors give distinct memories one vector; post-save dedupe is not under test here.
pytestmark = pytest.mark.usefixtures("no_remember_tidy")


def _vector(**parts: float) -> list[float]:
    """A unit vector from axis weights (a0=..., a1=...); the rest of the length goes to axis 6."""
    vector = [0.0] * 8
    for name, value in parts.items():
        vector[int(name[1:])] = value
    rest = 1.0 - sum(value * value for value in vector)
    vector[6] = math.sqrt(rest) if rest > 1e-12 else 0.0
    return vector


def _topic_of(store, memory_id: int) -> str | None:
    row = store._conn.execute("SELECT label FROM item_links WHERE relation='topic' AND src_id=?", (memory_id,)).fetchone()
    return None if row is None else str(row[0])


def _members(store, topic_id: str) -> list[int]:
    return [int(row[0]) for row in store._conn.execute(
        "SELECT src_id FROM item_links WHERE relation='topic' AND label=? ORDER BY src_id", (topic_id,))]


def _daily(store) -> None:
    topics.assign_all(store, NS)
    topics.discover(store, NS)


# A1 -------------------------------------------------------------------------------------------
def test_a1_remember_stores_without_topic(tmp_path):
    home = tmp_path / "home"
    NeospikeConfig().save(home)
    provider = NeospikeProvider(embedder=AxisEmbedder())
    provider.initialize("session", hermes_home=str(home), cwd=str(home))
    try:
        def remember(text: str) -> int:
            reply = json.loads(provider.handle_tool_call("neospike_memory_save", {
                "text": text, "scope": "global", "subject": "item", "durability": "ongoing", "kind": "fact", "basis": "user_said"}))
            assert reply == {"stored": True}
            return int(provider._memory_db._conn.execute("SELECT MAX(id) FROM items WHERE type='memory'").fetchone()[0])

        store = provider._memory_db
        members = [remember(f"alpha rule number {word}") for word in ("one", "two", "three")]
        with store._lock, store._conn:
            topic_id = topics._create_topic(store._conn, topics._member_rows(store._conn, members), time.time())
        fresh = remember("alpha rule number four")  # same vector as the members (cos 1.0)
        assert _topic_of(store, fresh) is None
        assert topics.assign_all(store, provider._ns_label)["members"] == [{"memory_id": fresh, "topic_id": topic_id}]
    finally:
        provider.shutdown()


# A2 -------------------------------------------------------------------------------------------
def test_a2_close_to_the_mean_but_no_neighbour_stays_without_topic(tmp_path):
    store = _store(tmp_path)
    member_ids = [_remember(store, f"member {axis}", _axis(axis)) for axis in (1, 2, 3, 4)]
    with store._lock, store._conn:
        topic_id = topics._create_topic(store._conn, topics._member_rows(store._conn, member_ids), time.time())
    weight = 0.45
    lone = _remember(store, "close to the mean", _vector(a0=math.sqrt(1 - 4 * weight * weight), a1=weight, a2=weight,
                                                         a3=weight, a4=weight))
    mean = [0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]  # unit mean of the four members
    stored = store._conn.execute("SELECT embedding FROM items WHERE type='memory' AND id=?", (lone,)).fetchone()[0]
    vectors, _ = topics._unit_rows([stored])
    assert abs(float(vectors[0] @ mean) - 0.9) < 1e-6 and max(abs(float(vectors[0][axis])) for axis in (1, 2, 3, 4)) < 0.6
    _daily(store)
    assert _topic_of(store, lone) is None and _members(store, topic_id) == member_ids
    store.close()


# A3 -------------------------------------------------------------------------------------------
def test_a3_join_closest_topic_and_along_chains(tmp_path):
    store = _store(tmp_path)
    first, first_ids = _topic_from(store, ["otter a", "otter b", "otter c"], _axis(0))
    second, second_ids = _topic_from(store, ["heron a", "heron b", "heron c"], _axis(1))
    both = _remember(store, "between", _vector(a0=0.7, a1=0.65))  # 0.70 to the first, 0.65 to the second
    chain = [_remember(store, "chain one", _vector(a0=0.8, a3=0.6)),  # 0.8 to the first topic's members
             _remember(store, "chain two", _vector(a0=0.2, a3=0.98)),  # 0.75 to chain one, 0.2 to the members
             _remember(store, "chain three", _vector(a3=0.7, a4=0.714))]  # 0.69 to chain two only
    joined = topics.assign_all(store, NS)
    assert {item["memory_id"]: item["topic_id"] for item in joined["members"]} == {
        both: first, chain[0]: first, chain[1]: first, chain[2]: first}
    assert _members(store, second) == second_ids
    again = topics.assign_all(store, NS)
    assert (again["link_checked"], again["assigned"]) == (0, 0)
    store.close()


def test_a3_tie_and_hand_exclusion(tmp_path):
    store = _store(tmp_path)
    first, _ = _topic_from(store, ["otter a", "otter b", "otter c"], _axis(0))
    second, _ = _topic_from(store, ["heron a", "heron b", "heron c"], _axis(1))
    taken_out = _remember(store, "taken out", _vector(a0=0.75, a1=0.65))
    with store._lock, store._conn:
        marks = topics.manual_marks(store._conn)
        marks.exclude(taken_out, first)
        topics.save_manual_marks(store._conn, marks)
    topics.assign_all(store, NS)
    assert _topic_of(store, taken_out) == second  # the closer topic was refused by hand
    assert topics.join_plan([(1, 3, 0.7), (2, 3, 0.7)], {1: 9, 2: 4}, topics.ManualMarks({})) == {3: 4}  # tie: lower id
    store.close()


# A4 -------------------------------------------------------------------------------------------
def test_a4_links_between_topics_never_merge_them(tmp_path):
    store = _store(tmp_path)
    first, first_ids = _topic_from(store, ["otter a", "otter b", "otter c"], _axis(0))
    second, second_ids = _topic_from(store, ["heron a", "heron b", "heron c"], _vector(a0=0.8, a1=0.6))
    _daily(store)
    links = store._conn.execute("SELECT low_id,high_id FROM memory_pairs").fetchall()
    assert any(int(low) in first_ids and int(high) in second_ids for low, high in links)  # the topics are linked
    assert _members(store, first) == first_ids and _members(store, second) == second_ids
    assert store._conn.execute("SELECT COUNT(DISTINCT label) FROM item_links WHERE relation='topic'").fetchone()[0] == 2
    store.close()


# A5 -------------------------------------------------------------------------------------------
def test_a5_second_run_checks_nothing_and_a_changed_body_is_checked_again(tmp_path):
    store = _store(tmp_path)
    ids = [_remember(store, f"otter {index}", _vector(a0=0.9, a1=0.1 * index)) for index in range(4)]
    assert topics.assign_all(store, NS)["link_checked"] == 4
    assert topics.assign_all(store, NS)["link_checked"] == 0
    assert topics.discover(store, NS)["link_checked"] == 0
    row = store.fetch_item_for_update(NS, ids[0], project="", session_id="s")
    assert store.update_memory_item(NS, ids[0], expected_text=row["text"], expected_row=row,
                                    item={"text": "heron now"}, embedding=_axis(5), project="", session_id="s") == "updated"
    assert store._conn.execute("SELECT COUNT(*) FROM memory_pairs WHERE low_id=? OR high_id=?",
                               (ids[0], ids[0])).fetchone()[0] > 0  # the old links until it is checked again
    result = topics.assign_all(store, NS)
    assert result["link_checked"] == 1
    assert store._conn.execute("SELECT COUNT(*) FROM memory_pairs WHERE low_id=? OR high_id=?",
                               (ids[0], ids[0])).fetchone()[0] == 0  # far from all now
    assert topics.assign_all(store, NS)["link_checked"] == 0
    store.close()


# A6 -------------------------------------------------------------------------------------------
