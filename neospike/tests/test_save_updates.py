"""neospike_memory_save updates: a recalled line that proved wrong (id:N memory, note:N.C automatic-note chunk)
is fixed in place by its number; text in the same call is saved as before; a refused update writes nothing."""
from __future__ import annotations

import json

import pytest

from neospike.notes import NoteService
from neospike.provider import NeospikeProvider
from neospike.tests.support_items import chunk_rows, note_row
from neospike.tests.test_remember_dedupe import NEW_TEXT, _ids, provider  # noqa: F401 (fixture)
from neospike.tests.test_save_conflicts import (FRIDAYS, MONDAYS, _formula_note, _human_note, _seed, _snapshot,
                                                _text)

CORRECTED = "The release train leaves on Tuesdays."
SAVE_FIELDS = {"text": NEW_TEXT, "scope": "global", "subject": "release", "durability": "stable", "kind": "fact",
               "basis": "user_said"}
LONG_TEXT = " ".join(f"Rule {n} keeps the release train on Fridays." for n in range(1, 140))


def _save(provider: NeospikeProvider, **args) -> dict:
    return json.loads(provider.handle_tool_call("neospike_memory_save", args))


def _revisions(provider: NeospikeProvider, item_id: int) -> list[str]:
    return [row[0] for row in provider._memory_db._conn.execute(
        "SELECT text FROM item_revisions WHERE item_id=? ORDER BY rev", (item_id,))]


def _chunk_texts(provider: NeospikeProvider, note_id: int) -> list[str]:
    return [row["text"] for row in chunk_rows(provider._memory_db._conn, note_id)]


def _automatic(provider: NeospikeProvider) -> dict:
    return _formula_note(provider, [(MONDAYS, 0.10), ("Lunch is at noon.", 0.10), ("Deploys need two reviews.", 0.10)])


def test_memory_ref_is_corrected_in_place(provider):
    memory_id = _seed(provider, MONDAYS, 0.10)
    before = provider._memory_db._conn.execute("SELECT embedding FROM items WHERE id=?", (memory_id,)).fetchone()[0]
    reply = _save(provider, updates=[{"ref": f"[id:{memory_id}]", "text": CORRECTED}])
    assert reply == {"updated": [{"ref": f"id:{memory_id}"}]}
    assert _text(provider, memory_id) == CORRECTED and _revisions(provider, memory_id) == [MONDAYS]
    assert _ids(provider._memory_db) == {memory_id}
    assert provider._memory_db._conn.execute("SELECT embedding FROM items WHERE id=?", (memory_id,)).fetchone()[0] != before
    assert memory_id in provider._reconcile_allowed(provider._active_sid)["id"]


def test_chunk_ref_changes_only_that_chunk_and_the_note_body(provider):
    note = _automatic(provider)
    first, second, third = note["chunk_ids"]
    reply = _save(provider, updates=[{"ref": f" note:{note['id']}.{first} ", "text": CORRECTED}])
    (entry,) = reply["updated"]
    assert entry["ref"] == f"note:{note['id']}.{first}" and entry["now"].startswith(f"note:{note['id']}.")
    texts = _chunk_texts(provider, note["id"])
    assert texts == [CORRECTED, "Lunch is at noon.", "Deploys need two reviews."]
    assert CORRECTED in note_row(provider._memory_db._conn, note["id"])["body"]
    assert MONDAYS not in note_row(provider._memory_db._conn, note["id"])["body"]
    now = int(entry["now"].rsplit(".", 1)[1])
    assert [int(row["id"]) for row in chunk_rows(provider._memory_db._conn, note["id"])] == [now, second, third]
    assert now in provider._reconcile_allowed(provider._active_sid)["chunk_id"]


def test_update_and_save_in_one_call(provider):
    memory_id = _seed(provider, MONDAYS, 0.10)
    reply = _save(provider, updates=[{"ref": f"id:{memory_id}", "text": CORRECTED}], **SAVE_FIELDS)
    assert reply["updated"] == [{"ref": f"id:{memory_id}"}] and reply["stored"] is True
    assert _text(provider, memory_id) == CORRECTED and len(_ids(provider._memory_db)) == 2


def test_update_only_adds_no_memory_and_has_no_stored_key(provider):
    memory_id = _seed(provider, MONDAYS, 0.10)
    reply = _save(provider, updates=[{"ref": f"id:{memory_id}", "text": CORRECTED}])
    assert "stored" not in reply and _ids(provider._memory_db) == {memory_id}


def test_save_without_updates_replies_as_before(provider):
    assert _save(provider, **SAVE_FIELDS) == {"stored": True}
    missing = {name: value for name, value in SAVE_FIELDS.items() if name != "scope"}
    assert _save(provider, **missing) == {"error": "Unsupported memory scope"}


def test_call_without_text_or_updates_is_refused(provider):
    assert "updates" in _save(provider, scope="global")["error"]


def _refusal_cases(provider: NeospikeProvider) -> list[tuple[str, str]]:
    automatic = _automatic(provider)
    agent = NoteService(provider._memory_db, provider._vectorizer).save(
        "Agent plan", "# One\nfirst agent part\n\n# Two\nsecond agent part\n", author="agent")
    agent_chunk = int(chunk_rows(provider._memory_db._conn, agent["id"])[0]["id"])
    doc_id, doc_chunks = _human_note(provider, 0.10)
    mirror = provider._memory_db.insert(provider._ns_label, FRIDAYS, [1.0] + [0.0] * 127, source="agent:earlier",
                                        kind="durable", metadata={"scope": "global", "session_id": "earlier"})
    return [
        ("agent note chunk", f"note:{agent['id']}.{agent_chunk}"),
        ("doc chunk", f"note:{doc_id}.{doc_chunks[0]}"),
        ("unknown id", "id:999999"),
        ("chunk of another note", f"note:{agent['id']}.{automatic['chunk_ids'][0]}"),
        ("preference", "pref:1"),
        ("builtin copy", f"id:{mirror}"),
        ("conversation", "conv:3"),
        ("bad format", "note:12"),
        ("bad format", "memory 4"),
    ]


def test_refused_refs_change_nothing(provider):
    for label, ref in _refusal_cases(provider):
        before = _snapshot(provider)
        reply = _save(provider, updates=[{"ref": ref, "text": CORRECTED}])
        assert "error" in reply and "updated" not in reply, label
        assert _snapshot(provider) == before, label
    assert "target=preference" in _save(provider, updates=[{"ref": "pref:1", "text": CORRECTED}])["error"]


@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_text_is_refused(provider, text):
    memory_id = _seed(provider, MONDAYS, 0.10)
    before = _snapshot(provider)
    assert "error" in _save(provider, updates=[{"ref": f"id:{memory_id}", "text": text}])
    assert _snapshot(provider) == before


def test_refused_update_skips_the_save(provider):
    memory_id = _seed(provider, MONDAYS, 0.10)
    good, bad = {"ref": f"id:{memory_id}", "text": CORRECTED}, {"ref": "id:999999", "text": CORRECTED}
    before = _snapshot(provider)
    reply = _save(provider, updates=[good, bad], **SAVE_FIELDS)
    assert reply["error"] == "not_found" and "stored" not in reply
    assert _snapshot(provider) == before


def test_long_correction_is_split_not_refused(provider):
    memory_id = _seed(provider, MONDAYS, 0.10)
    (entry,) = _save(provider, updates=[{"ref": f"id:{memory_id}", "text": LONG_TEXT}])["updated"]
    assert entry["ref"] == f"id:{memory_id}" and entry["pieces"]
    pieces = [memory_id, *(int(piece.removeprefix("id:")) for piece in entry["pieces"])]
    assert all(len(_text(provider, piece)) <= 2000 for piece in pieces)
    assert _ids(provider._memory_db) == set(pieces)


def test_schema_prompt_and_description_name_the_updates_path(provider):
    (save,) = [tool for tool in provider.get_tool_schemas() if tool["name"] == "neospike_memory_save"]
    assert save["parameters"]["properties"]["updates"]["items"]["required"] == ["ref", "text"]
    assert "updates" in save["description"] and "[note:N.C" in save["description"]
    prompt = provider.system_prompt_block()
    assert "neospike_memory_save updates" in prompt and "neospike_memory_update" in prompt
