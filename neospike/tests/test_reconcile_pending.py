"""Save-time older pairs wait in a list until the same conversation's neospike_memory_reconcile (change, delete
or keep) or the daily LLM check settles them; one reconcile call writes everything or nothing."""
from __future__ import annotations

import json
import time

import pytest

from neospike import memory_reconcile as mr
from neospike import provider as provider_module
from neospike.notes import create_formula_note
from neospike.provider import NeospikeProvider
from neospike.tests.test_remember_dedupe import Embedder, _at_cosine, _ids, _remember, provider  # noqa: F401
from neospike.tests.test_save_conflicts import (FRIDAYS, MONDAYS, _chunk_ids, _formula_note, _packed, _reconcile,
                                                _seed, _snapshot, _text)

CORRECTED = "The release train left on Mondays until October."


def _pending(provider: NeospikeProvider) -> list[tuple[int, str, int]]:
    with provider._memory_db._lock:
        return [(entry["memory_id"], entry["lane"], entry["other_id"])
                for entry in mr.read_pending(provider._memory_db._conn)]


def _memory(provider: NeospikeProvider, text: str, similarity: float, basis: str = "") -> int:
    metadata = {"scope": "global", "subject": "release", "durability": "ongoing", "session_id": "earlier"}
    if basis:
        metadata["basis"] = basis
    memory_id = provider._memory_db.insert(provider._ns_label, text, _at_cosine(similarity), source="agent:earlier",
                                           kind="memory_item", metadata=metadata)
    assert memory_id is not None
    return memory_id


def _wait(provider: NeospikeProvider, newer: int, older: int, lane: str = mr.LANE_MEMORY) -> None:
    store = provider._memory_db
    texts = {int(row[0]): str(row[1]) for row in store._conn.execute("SELECT id,text FROM items")}
    assert mr.add_pending(store, newer, texts[newer], [{lane: older, "text": texts[older]}]) == 1


class FakeLlm:
    """Returns the prepared answers in order and records every message list."""

    def __init__(self, *answers: list[dict]) -> None:
        self.answers = [json.dumps(answer) for answer in answers]
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.answers.pop(0) if self.answers else "[]"


def _judge(provider: NeospikeProvider, llm: FakeLlm) -> dict:
    return mr.judge(provider._memory_db, llm, Embedder(), language="en")


def _alive(provider: NeospikeProvider, memory_id: int) -> bool:
    return memory_id in _ids(provider._memory_db)


# waiting list at save time and the same conversation's reconcile ------------------------------------------
def test_older_pairs_wait_and_a_correction_or_deletion_takes_them_off(provider):
    first = _seed(provider, MONDAYS, 0.85)
    second = _seed(provider, FRIDAYS, 0.82)
    new = _remember(provider)["id"]
    assert sorted(_pending(provider)) == [(new, "id", first), (new, "id", second)]
    assert _reconcile(provider, [{"id": first, "text": CORRECTED}]) == {"changed_ids": [first]}
    assert _pending(provider) == [(new, "id", second)]
    assert _reconcile(provider, [{"id": second}]) == {"deleted_ids": [second]}
    assert _pending(provider) == []


def test_keep_writes_nothing_and_takes_the_pair_off(provider):
    first = _seed(provider, MONDAYS, 0.85)
    second = _seed(provider, FRIDAYS, 0.82)
    new = _remember(provider)["id"]
    before = _snapshot(provider)
    assert _reconcile(provider, [{"id": first, "keep": True}]) == {"kept_ids": [first]}
    assert _snapshot(provider) == before and _pending(provider) == [(new, "id", second)]


def test_deleting_the_new_memory_takes_all_its_pairs_off(provider):
    _seed(provider, MONDAYS, 0.85)
    _seed(provider, FRIDAYS, 0.82)
    new = _remember(provider)["id"]
    assert _reconcile(provider, [{"id": new}]) == {"deleted_ids": [new]}
    assert _pending(provider) == []


def test_chunk_pair_waits_and_keep_takes_it_off(provider):
    note = _formula_note(provider, [("Trains now leave on Mondays only.", 0.86)])
    new = _remember(provider)["id"]
    chunk = note["chunk_ids"][0]
    assert _pending(provider) == [(new, "chunk_id", chunk)]
    assert _reconcile(provider, [{"chunk_id": chunk, "keep": True}]) == {"kept_chunk_ids": [chunk]}
    assert _pending(provider) == []


@pytest.mark.parametrize("change", [{"text": CORRECTED, "keep": True}, {"keep": False}])
def test_keep_with_text_or_not_true_is_refused(provider, change):
    first = _seed(provider, MONDAYS, 0.85)
    new = _remember(provider)["id"]
    before = _snapshot(provider)
    assert "error" in _reconcile(provider, [{"id": first, **change}])
    assert _snapshot(provider) == before and _pending(provider) == [(new, "id", first)]


def test_pairs_outlive_the_conversation(provider):
    first = _seed(provider, MONDAYS, 0.85)
    new = _remember(provider)["id"]
    provider.on_session_end([])
    assert _reconcile(provider, [{"id": first, "keep": True}])["error"] == "not_allowed"
    assert _pending(provider) == [(new, "id", first)]


# one call: all or nothing --------------------------------------------------------------------------------
def _broken_chunks(*_args, **_kwargs):
    raise RuntimeError("chunk step failed")


class SeenThenBroken:
    """The chunk step: records the memory bodies the open transaction already holds, then fails."""

    def __init__(self, memory_ids: list[int]) -> None:
        self.memory_ids = memory_ids
        self.seen: dict[int, str | None] = {}

    def __call__(self, connection, *_args, **_kwargs):
        for memory_id in self.memory_ids:
            row = connection.execute("SELECT text FROM items WHERE id=?", (memory_id,)).fetchone()
            self.seen[memory_id] = None if row is None else row[0]
        raise RuntimeError("chunk step failed")


def test_failed_chunk_step_leaves_no_memory_change(provider, monkeypatch):
    note = _formula_note(provider, [("Trains now leave on Mondays only.", 0.86)])
    first = _seed(provider, MONDAYS, 0.85)
    second = _seed(provider, FRIDAYS, 0.82)
    new = _remember(provider)["id"]
    before, waiting = _snapshot(provider), _pending(provider)
    step = SeenThenBroken([first, second])
    monkeypatch.setattr(provider_module, "reconcile_formula_chunks", step)
    done = _reconcile(provider, [{"id": first, "text": CORRECTED}, {"id": second},
                                 {"chunk_id": note["chunk_ids"][0]}])
    assert done == {"error": "Reconcile failed"}
    assert step.seen == {first: CORRECTED, second: None}  # both memory writes ran before the failure
    assert _snapshot(provider) == before and _pending(provider) == waiting and _alive(provider, new)
    assert provider._memory_db._conn.execute("SELECT COUNT(*) FROM item_revisions WHERE item_id=?",
                                             (first,)).fetchone()[0] == 0
    assert not provider._memory_db._conn.in_transaction


def test_failed_chunk_step_leaves_no_save_update(provider, monkeypatch):
    note = _formula_note(provider, [("Trains now leave on Mondays only.", 0.86)])
    first = _seed(provider, MONDAYS, 0.85)
    before = _snapshot(provider)
    monkeypatch.setattr(provider_module, "reconcile_formula_chunks", _broken_chunks)
    reply = json.loads(provider.handle_tool_call("neospike_memory_save", {"updates": [
        {"ref": f"id:{first}", "text": CORRECTED},
        {"ref": f"note:{note['id']}.{note['chunk_ids'][0]}", "text": "Trains leave on Tuesdays."}]}))
    assert reply == {"error": "Update failed"} and _snapshot(provider) == before


# daily LLM judgement ------------------------------------------------------------------------------------
def test_three_decisions_apply_and_a_kept_pair_is_not_asked_again(provider):
    keep_new, keep_old = _memory(provider, "Builds run at night.", 0.1), _memory(provider, "Tests run at night.", 0.2)
    upd_new, upd_old = _memory(provider, "Deploys happen on Fridays.", 0.3), _memory(provider, "Deploys happen on Mondays; reviews on Tuesdays.", 0.4)
    del_new, del_old = _memory(provider, "The office is in Seoul.", 0.5), _memory(provider, "The office is in Busan.", 0.6)
    for newer, older in ((keep_new, keep_old), (upd_new, upd_old), (del_new, del_old)):
        _wait(provider, newer, older)
    llm = FakeLlm([{"pair": [f"id:{keep_new}", f"id:{keep_old}"], "action": "keep"},
                   {"pair": [f"id:{upd_new}", f"id:{upd_old}"], "action": "update", "keep_id": f"id:{upd_old}",
                    "text": "Reviews happen on Tuesdays."},
                   {"pair": [f"id:{del_new}", f"id:{del_old}"], "action": "delete", "keep_id": f"id:{del_new}"}])
    result = _judge(provider, llm)
    assert (result["judged"], result["applied"], result["keep"], result["update"], result["delete"],
            result["skipped"], result["calls"]) == (3, 2, 1, 1, 1, 0, 1)
    assert _text(provider, upd_old) == "Reviews happen on Tuesdays." and not _alive(provider, upd_new)
    assert [tuple(row) for row in provider._memory_db._conn.execute(
        "SELECT text,edited_by FROM item_revisions WHERE item_id=?", (upd_old,))] == [
        ("Deploys happen on Mondays; reviews on Tuesdays.", "formula")]
    assert not _alive(provider, del_old) and _alive(provider, del_new)
    assert _alive(provider, keep_new) and _alive(provider, keep_old) and _pending(provider) == []
    store = provider._memory_db
    assert mr.add_pending(store, keep_new, "Builds run at night.", [{"id": keep_old, "text": "Tests run at night."}]) == 0
    again = FakeLlm()
    assert _judge(provider, again)["judged"] == 0 and again.calls == []
    assert mr.summary_part(result).startswith("memory pairs: judged 3 of 3 in 1 calls, applied 2, keep 1")


def test_user_said_item_is_not_removed_or_rewritten_for_an_inferred_one(provider):
    newer = _memory(provider, "Reports go out weekly.", 0.1, basis="agent_inferred")
    stated = _memory(provider, "Reports go out monthly.", 0.2, basis="user_said")
    _wait(provider, newer, stated)
    deleted = _judge(provider, FakeLlm([{"pair": [f"id:{newer}", f"id:{stated}"], "action": "delete",
                                        "keep_id": f"id:{newer}"}]))
    assert deleted["skip_reasons"] == {mr.SKIP_PROTECTED: 1} and _alive(provider, stated)
    _wait(provider, newer, stated)
    rewritten = _judge(provider, FakeLlm([{"pair": [f"id:{newer}", f"id:{stated}"], "action": "update",
                                          "keep_id": f"id:{stated}", "text": "Reports go out weekly."}]))
    assert rewritten["skip_reasons"] == {mr.SKIP_PROTECTED: 1}
    assert _text(provider, stated) == "Reports go out monthly." and _alive(provider, newer)


def test_changed_or_deleted_pairs_leave_without_a_question(provider):
    changed_new, changed_old = _memory(provider, "Lunch is at noon.", 0.1), _memory(provider, "Lunch is at one.", 0.2)
    gone_new, gone_old = _memory(provider, "Coffee is free.", 0.3), _memory(provider, "Coffee costs a dollar.", 0.4)
    _wait(provider, changed_new, changed_old)
    _wait(provider, gone_new, gone_old)
    store = provider._memory_db
    with store._lock, store._conn:
        store._conn.execute("UPDATE items SET text=? WHERE id=?", ("Lunch is at two.", changed_old))
    assert store.delete(provider._ns_label, gone_old)
    llm = FakeLlm()
    result = _judge(provider, llm)
    assert (result["pending"], result["judged"], result["dropped"]) == (2, 0, 2)
    assert llm.calls == [] and _pending(provider) == []


def test_memory_absorbed_into_a_chunk_is_judged_as_that_chunk(provider):
    newer = _memory(provider, "The train leaves at nine.", 0.1)
    older = _memory(provider, "The train leaves at eight.", 0.2)
    _wait(provider, newer, older)
    store = provider._memory_db
    source = json.dumps({"memory_id": newer, "basis": "", "created_at": time.time()})
    with store._lock, store._conn:
        note = create_formula_note(store._conn, "train", [("The train leaves at nine.", _packed(0.1, 3), source)],
                                   time.time())
    assert store.delete(provider._ns_label, newer)
    chunk = _chunk_ids(provider, note["id"])[0]
    llm = FakeLlm([{"pair": [f"chunk:{chunk}", f"id:{older}"], "action": "delete", "keep_id": f"chunk:{chunk}"}])
    result = _judge(provider, llm)
    assert f"chunk:{chunk}" in llm.calls[0][1]["content"]
    assert (result["judged"], result["delete"]) == (1, 1) and not _alive(provider, older)
    assert _pending(provider) == []
