"""The save follow-up (note-absorb -> topic-assign) runs after the saving turn's sync_turn,
never inside the turn; preference/constraint memories stay memories; an absorbed chunk carries source_json; a
chunk absorbed from this conversation's save after the cut stays out of recall and search; the daily
embed-pending stage runs the same follow-up. Fake state.db and a keyword embedder; no LLM, no CLI process."""
from __future__ import annotations

import json
import sys
import time
from array import array
from pathlib import Path

import pytest

from neospike import topics
from neospike.config import EMBEDDING_DIMENSIONS, NeospikeConfig
from neospike.notes import create_formula_note
from neospike.tests.support_items import chunk_rows
from neospike.provider import NeospikeProvider
from neospike.store import FORMULA_AUTHOR, MemoryStore
from neospike.tests.test_live_context import (CUT_AGO_S, NAMESPACE, SID, KeywordEmbedder, compact, make_state,
                                              remember, search)

WAIT_S = 10.0
CHUNK = "The rust compiler is pinned."
LONG_RUST = "The build now uses the rust nightly toolchain for every single release candidate."
COFFEE = "The team drinks coffee at ten every morning in the office kitchen."
SAVE = {"scope": "global", "subject": "build", "durability": "stable", "kind": "fact", "basis": "user_said"}


@pytest.fixture
def home(tmp_path: Path) -> Path:
    NeospikeConfig().save(tmp_path)
    return tmp_path


@pytest.fixture
def provider(home: Path):
    instance = NeospikeProvider(embedder=KeywordEmbedder())
    instance.initialize(SID, hermes_home=str(home), agent_identity="default", user_id="owner",
                        agent_context="primary", cwd=str(home))
    yield instance
    instance.shutdown()


def formula_note(provider: NeospikeProvider, texts: tuple[str, ...]) -> int:
    store = provider._memory_db
    with store._lock, store._conn:
        note = create_formula_note(store._conn, "Rust toolchain",
                                   [(text, _blob(text)) for text in texts], time.time())
    return int(note["id"])


def _blob(text: str) -> bytes:
    return array("f", KeywordEmbedder.vector(text)).tobytes()


def save(provider: NeospikeProvider, text: str, **fields) -> dict:
    return json.loads(provider.handle_tool_call("neospike_memory_save", {"text": text, **SAVE, **fields}))


def end_turn(provider: NeospikeProvider) -> None:
    provider.sync_turn("the question", "the answer", session_id=SID)
    assert provider._follow_up.wait_idle(WAIT_S)
    assert provider._follow_up.last_error is None


def memory_ids(provider: NeospikeProvider) -> set[int]:
    return {int(row[0]) for row in provider._memory_db._conn.execute("SELECT id FROM items WHERE type='memory'")}


def chunk_sources(provider: NeospikeProvider, note_id: int) -> list[dict | None]:
    return [None if row["source_json"] is None else json.loads(row["source_json"])
            for row in chunk_rows(provider._memory_db._conn, note_id)]


def topic_checked(provider: NeospikeProvider) -> set[int]:
    return {int(row[0]) for row in provider._memory_db._conn.execute(
        "SELECT memory_id FROM memory_checks WHERE stage='topic'")}


# 11. the saving turn's sync_turn runs absorb and assign; never inside the turn ------------------------------------
def test_save_then_update_in_the_same_turn_and_absorb_after_sync_turn(provider):
    note_id = formula_note(provider, (CHUNK,))
    saved = save(provider, LONG_RUST, importance=0.8)
    coffee = save(provider, COFFEE)
    assert saved["stored"] is True and coffee["stored"] is True
    [rust_id] = [int(row[0]) for row in provider._memory_db._conn.execute(
        "SELECT id FROM items WHERE type='memory' AND text=?", (LONG_RUST,))]
    edited = LONG_RUST.replace("every single", "each")
    update = json.loads(provider.handle_tool_call("neospike_memory_update", {
        "id": rust_id, "expected_text": LONG_RUST, "new_text": edited, "subject": "build", "durability": "stable"}))
    assert update.get("updated") is True  # the same turn still finds the id: nothing absorbed yet
    assert provider._follow_up.runs == 0 and rust_id in memory_ids(provider)
    end_turn(provider)
    assert provider._follow_up.runs == 1
    assert rust_id not in memory_ids(provider)  # absorbed (cos 1.0 >= 0.8 to the rust chunk)
    [first, absorbed] = chunk_sources(provider, note_id)
    assert first is None  # a chunk written without a source keeps NULL
    assert absorbed["memory_id"] == rust_id and absorbed["origin"] == f"agent:{SID}" and absorbed["session_id"] == SID
    assert (absorbed["memory_kind"], absorbed["basis"], absorbed["subject"]) == ("fact", "user_said", "build")
    assert absorbed["created_at"] > 0 and absorbed["importance"] == pytest.approx(0.8)
    assert set(absorbed) == {"memory_id", "origin", "session_id", "created_at", "memory_kind", "basis", "subject",
                             "importance"}
    assert topic_checked(provider) >= memory_ids(provider) - {rust_id}  # topic-assign checked the coffee memory


def test_turn_without_a_save_wakes_nothing(provider):
    formula_note(provider, (CHUNK,))
    remember(provider, LONG_RUST)  # written outside the tools (e.g. the dashboard): left for embed-pending
    provider.sync_turn("the question", "the answer", session_id=SID)
    assert provider._follow_up.wait_idle(WAIT_S) and provider._follow_up.runs == 0
    assert len(memory_ids(provider)) == 1


def test_profile_memories_are_not_absorbed_merged_or_assigned(provider):
    """A preference/constraint is a preferences row, so absorb, chunk merge and topics never see it."""
    note_id = formula_note(provider, (CHUNK,))
    preference = "The rust compiler stays pinned."  # cos 1.0 and a word gap of 0: a chunk duplicate for a fact
    reply = save(provider, preference, kind="preference")
    assert reply["stored"] is True and "replaced_chunk" not in json.dumps(reply)
    constraint = save(provider, LONG_RUST, kind="constraint", situation="building release candidates")
    assert constraint["stored"] is True
    assert memory_ids(provider) == set()
    assert [row[0] for row in provider._memory_db._conn.execute("SELECT type FROM preferences ORDER BY id")] == [
        "preference", "constraint"]
    end_turn(provider)
    assert memory_ids(provider) == set() and chunk_sources(provider, note_id) == [None]
    assert topics._live(provider._memory_db._conn, NAMESPACE) == []
    assert "The rust compiler stays pinned." in provider.prefetch("anything else", session_id="20261119_next")


def test_topic_to_note_parts_carry_their_source(tmp_path):
    store = MemoryStore(tmp_path / "memory.db", dimensions=KeywordEmbedder.dimensions)
    texts = [f"rust rule {word} holds" for word in ("one", "two", "three", "four", "five", "six", "seven", "eight",
                                                    "nine", "ten", "eleven")]
    ids = [store.insert(NAMESPACE, text, KeywordEmbedder.vector(text), source="agent:20261001_old",
                        kind="memory_item", importance=0.7,
                        metadata={"scope": "global", "subject": "rust", "durability": "stable", "tags": [],
                                  "session_id": "20261001_old", "memory_kind": "decision", "basis": "agent_inferred"})
           for text in texts]
    with store._lock, store._conn:
        store._conn.execute("DELETE FROM item_links WHERE relation='topic'")
        topics._create_topic(store._conn, topics._member_rows(store._conn, ids), time.time())
    assert topics.topic_to_note(store, NAMESPACE)["converted"] == 1
    sources = [json.loads(row["source_json"]) for row in chunk_rows(store._conn)
               if store._conn.execute("SELECT author FROM items WHERE id=?", (row["note_id"],)).fetchone()[0]
               == FORMULA_AUTHOR]
    assert [item["memory_id"] for item in sources] == ids
    assert {(item["memory_kind"], item["basis"], item["session_id"], round(item["importance"], 3))
            for item in sources} == {("decision", "agent_inferred", "20261001_old", 0.7)}
    store.close()


def test_save_time_chunk_replacement_carries_the_source(provider):
    note_id = formula_note(provider, (CHUNK,))
    reply = save(provider, "The rust compiler is pinned now.")  # cos 1.0, word gap 1: the chunk takes the body
    assert reply["stored"] is True
    [source] = chunk_sources(provider, note_id)
    assert source["session_id"] == SID and source["memory_kind"] == "fact" and source["subject"] == "build"
    assert source["memory_id"] not in memory_ids(provider)


# a chunk absorbed from this conversation's save after the cut is left out -------------------------
def test_chunk_absorbed_from_a_save_here_after_the_cut_is_left_out(home, provider):
    make_state(home)
    compact(home, time.time() - CUT_AGO_S)
    note_id = formula_note(provider, (CHUNK,))
    save(provider, LONG_RUST)
    end_turn(provider)
    assert len(chunk_sources(provider, note_id)) == 2
    recall = provider.prefetch("rust?", session_id=SID)
    assert CHUNK in recall and "rust nightly" not in recall
    found = search(provider, "rust?")
    texts = [hit["text"] for hit in found["results"]]
    assert CHUNK in texts and not any("rust nightly" in text for text in texts)
    other = provider.prefetch("rust?", session_id="20261119_next")  # another conversation sees the chunk
    assert "rust nightly" in other


# 12. the daily embed-pending stage runs the follow-up ----------------------------------------------------------
class _NoModel:
    def __init__(self, *_args, **_kwargs) -> None:
        raise AssertionError("no embedding model in this test")


def test_embed_pending_stage_runs_the_follow_up(tmp_path, monkeypatch, capsys):
    from neospike import cli
    home = tmp_path / "home"
    NeospikeConfig().save(home)
    store = MemoryStore(home / "neospike" / "memory.db", dimensions=EMBEDDING_DIMENSIONS)
    axis = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)
    with store._lock, store._conn:
        note_id = create_formula_note(store._conn, "Rust toolchain", [(CHUNK, array("f", axis).tobytes())],
                                      time.time())["id"]
    memory_id = store.insert(NAMESPACE, LONG_RUST, axis, source="dashboard", kind="memory_item",
                             metadata={"scope": "global", "subject": "build", "durability": "stable", "tags": [],
                                       "memory_kind": "fact"})
    store.close()
    monkeypatch.setattr(cli, "EmbeddingClient", _NoModel)
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(home), "--owner", NAMESPACE, "embed-pending"])
    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["pending"] == 0 and output["follow_up"]["absorb"]["absorbed"] == 1
    assert "assign" in output["follow_up"]
    store = MemoryStore(home / "neospike" / "memory.db", dimensions=EMBEDDING_DIMENSIONS)
    assert store._conn.execute("SELECT 1 FROM items WHERE type='memory' AND id=?", (memory_id,)).fetchone() is None
    [source] = [json.loads(row["source_json"]) for row in chunk_rows(store._conn, note_id)
                if row["source_json"] is not None]
    assert source["memory_id"] == memory_id and source["origin"] == "dashboard" and source["session_id"] == ""
    store.close()
