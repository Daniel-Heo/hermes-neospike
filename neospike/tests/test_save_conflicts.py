"""Save-time older items are handled on the spot: the save reply lists them oldest first with
the new memory's id, and one neospike_memory_reconcile call corrects (text) or deletes (no text) old items, never merging
them. Automatic-note chunks take part (duplicate -> the chunk takes the new body); chunks are their own search
results.

Acceptance items of the work order are named in each test (A1 .. A6)."""
from __future__ import annotations

import json
import math
import time
from array import array

from neospike.notes import ORIGIN_HUMAN, NoteService, create_formula_note
from neospike.tests.support_items import chunk_rows, note_row, note_rows
from neospike.provider import NeospikeProvider
from neospike.tests.test_remember_dedupe import (DIMENSIONS, NEW_TEXT, _at_cosine, _ids, _new_id, _remember,
                                                 provider)  # noqa: F401 (fixture)

PROMPT_CHARS_BEFORE = 2688  # system_prompt_block length ceiling
TOOLS_BEFORE = 15
TOOLS_REMOVED_INBOX = 3  # neospike_review, neospike_approve, neospike_reject
TOOLS_REMOVED_IMAGE = 3  # neospike_index_image, neospike_search_images, neospike_forget_image
TOOLS_REMOVED_ADMIN = 3  # neospike_status, neospike_index_file, neospike_prune
TOOLS_ADDED_DOC = 2  # neospike_doc_read, neospike_doc_write (docs split from the note tools)
# forget gained the optional target (memory | preference); the rest is unchanged.
FORGET_SCHEMA_BEFORE = {"type": "object", "properties": {"id": {"type": "integer", "minimum": 1, "maximum": 2**63 - 1},
                                                         "target": {"type": "string", "enum": ["memory", "preference"],
                                                                    "default": "memory",
                                                                    "description": "preference: id is pref:N (a preference/constraint)."}},
                        "additionalProperties": False, "required": ["id"]}
MONDAYS, FRIDAYS = "The release train leaves on Mondays.", "The release train leaves on Fridays now."


def _seed(provider: NeospikeProvider, text: str, similarity: float) -> int:
    """An earlier global memory item as neospike_memory_save stores it (correctable), at `similarity` to NEW_TEXT."""
    memory_id = provider._memory_db.insert(provider._ns_label, text, _at_cosine(similarity), source="agent:earlier",
                                       kind="memory_item", metadata={"scope": "global", "subject": "release",
                                                                     "durability": "ongoing", "session_id": "earlier"})
    assert memory_id is not None
    return memory_id


def _packed(similarity: float, axis: int = 1) -> bytes:
    """A vector at `similarity` to NEW_TEXT (e0), the rest on `axis` (distinct axes keep items apart)."""
    vector = [0.0] * DIMENSIONS
    vector[0], vector[axis] = similarity, math.sqrt(1.0 - similarity * similarity)
    return array("f", vector).tobytes()


def _formula_note(provider: NeospikeProvider, items: list[tuple[str, float]], title: str = "release rules") -> dict:
    store = provider._memory_db
    with store._lock, store._conn:
        note = create_formula_note(store._conn, title, [(text, _packed(similarity, 2 + index))
                                                        for index, (text, similarity) in enumerate(items)], time.time())
    return {**note, "chunk_ids": _chunk_ids(provider, note["id"])}


def _chunk_ids(provider: NeospikeProvider, note_id: int) -> list[int]:
    return [int(row["id"]) for row in chunk_rows(provider._memory_db._conn, note_id)]


def _note(provider: NeospikeProvider, note_id: int) -> tuple:
    row = note_row(provider._memory_db._conn, note_id)
    return row["body"], row["deleted_at"]


def _text(provider: NeospikeProvider, memory_id: int) -> str:
    return provider._memory_db._conn.execute("SELECT text FROM items WHERE type='memory' AND id=?", (memory_id,)).fetchone()[0]


def _reconcile(provider: NeospikeProvider, changes: list[dict]) -> dict:
    return json.loads(provider.handle_tool_call("neospike_memory_reconcile", {"changes": changes}))


def _human_note(provider: NeospikeProvider, similarity: float) -> tuple[int, list[int]]:
    """A human note of three sections whose chunks sit at `similarity` to NEW_TEXT (set directly)."""
    saved = NoteService(provider._memory_db, provider._vectorizer).save(
        "Human release page", "# One\nfirst human part\n\n# Two\nsecond human part\n\n# Three\nthird human part\n",
        author="human", origin=ORIGIN_HUMAN)
    chunk_ids = _chunk_ids(provider, saved["id"])
    with provider._memory_db._lock, provider._memory_db._conn:
        provider._memory_db._conn.executemany("UPDATE items SET embedding=? WHERE id=?",
                                          [(_packed(similarity, 10 + index), chunk_id)
                                           for index, chunk_id in enumerate(chunk_ids)])
    return saved["id"], chunk_ids


def _snapshot(provider: NeospikeProvider) -> tuple:
    connection = provider._memory_db._conn
    return (tuple(map(tuple, connection.execute("SELECT id,text,embedding FROM items WHERE type='memory' ORDER BY id"))),
            tuple(sorted((row["id"], row["note_id"], row["text"]) for row in chunk_rows(connection))),
            tuple((row["id"], row["body"], row["rev"], row["deleted_at"]) for row in note_rows(connection)))


# A1 -------------------------------------------------------------------------------------------
def test_a1_two_older_items_with_bodies_and_no_candidate(provider):
    first = _seed(provider, MONDAYS, 0.85)
    second = _seed(provider, FRIDAYS, 0.82)
    _seed(provider, "Lunch is at noon.", 0.70)  # below 0.80: not listed
    reply = _remember(provider)
    assert reply == {"stored": True, "id": _new_id(provider._memory_db),
                     "older": [{"id": first, "text": MONDAYS}, {"id": second, "text": FRIDAYS}]}


# A2 -------------------------------------------------------------------------------------------
def test_a2_older_list_is_oldest_first_by_last_write(provider):
    first = _seed(provider, MONDAYS, 0.85)
    second = _seed(provider, FRIDAYS, 0.82)
    with provider._memory_db._lock, provider._memory_db._conn:  # the first body was rewritten later (revision time)
        provider._memory_db._conn.execute(
            "INSERT INTO item_revisions(item_id,rev,text,metadata_json,edited_at,edited_by) VALUES (?,1,?,'{}',?,?)",
            (first, "an earlier body", time.time() + 60, "agent"))
    assert [item["id"] for item in _remember(provider)["older"]] == [second, first]


def test_a2_one_reconcile_call_corrects_one_and_deletes_one(provider):
    first = _seed(provider, MONDAYS, 0.85)
    second = _seed(provider, FRIDAYS, 0.82)
    with provider._memory_db._lock, provider._memory_db._conn:
        provider._memory_db._conn.execute("INSERT INTO item_links(src_id,dst_id,dst_title,relation) "
                                      "VALUES (?,?,'',?)", (second, first, "supersedes"))
    reply = _remember(provider)
    new = reply["id"]
    before = provider._memory_db._conn.execute("SELECT embedding FROM items WHERE type='memory' AND id=?", (first,)).fetchone()[0]
    done = _reconcile(provider, [{"id": first, "text": "The release train left on Mondays until October."},
                                 {"id": second}])
    assert done == {"changed_ids": [first], "deleted_ids": [second]}
    assert _text(provider, first) == "The release train left on Mondays until October."
    assert provider._memory_db._conn.execute("SELECT embedding FROM items WHERE type='memory' AND id=?", (first,)).fetchone()[0] != before
    assert [row[0] for row in provider._memory_db._conn.execute(
        "SELECT text FROM item_revisions WHERE item_id=?", (first,))] == [MONDAYS]
    assert _ids(provider._memory_db) == {first, new} and _text(provider, new) == NEW_TEXT
    assert provider._memory_db._conn.execute(
        "SELECT COUNT(*) FROM item_links WHERE relation='supersedes'").fetchone()[0] == 0  # the deletion's clean-up


def test_a2_unlisted_id_refuses_the_whole_call(provider):
    first = _seed(provider, MONDAYS, 0.85)
    outsider = _seed(provider, "Lunch is at noon.", 0.70)
    _remember(provider)
    before = _snapshot(provider)
    refused = _reconcile(provider, [{"id": first}, {"id": outsider, "text": "Lunch is at one o'clock now."}])
    assert refused["error"] == "not_allowed" and refused["items"] == [{"id": outsider}]
    assert _snapshot(provider) == before


def test_a2_no_call_keeps_all_and_nothing_waits(provider):
    old = _seed(provider, MONDAYS, 0.85)
    _remember(provider)
    assert _ids(provider._memory_db) == {old, _new_id(provider._memory_db)}


def test_a2_archived_memory_is_never_listed(provider):
    old = _seed(provider, MONDAYS, 0.85)
    with provider._memory_db._lock, provider._memory_db._conn:
        provider._memory_db._conn.execute("UPDATE items SET archived_at=1.0 WHERE id=?", (old,))
    assert _remember(provider) == {"stored": True}


def test_a2_targets_end_with_the_session(provider):
    first = _seed(provider, MONDAYS, 0.85)
    _remember(provider)
    provider.on_session_switch("another-session")
    assert _reconcile(provider, [{"id": first}])["error"] == "not_allowed"


# A3 -------------------------------------------------------------------------------------------
def test_a3_duplicate_chunk_takes_the_new_body(provider):
    note = _formula_note(provider, [("Release train departs every Thursday.", 0.95), ("Lunch is at noon.", 0.1)])
    reply = _remember(provider)
    new_chunks = _chunk_ids(provider, note["id"])
    assert reply == {"stored": True, "replaced_chunk": {"note_id": note["id"], "chunk_id": new_chunks[0]}}
    assert _ids(provider._memory_db) == set()  # the new memory is gone: the note holds the content
    assert _note(provider, note["id"]) == (f"{NEW_TEXT}\n\nLunch is at noon.", None)
    stored = provider._memory_db._conn.execute("SELECT embedding FROM items WHERE id=?", (new_chunks[0],)).fetchone()[0]
    assert list(array("f", stored))[:2] == [1.0, 0.0]  # the new memory's embedding (e0)


def test_a3_chunks_first_then_one_call_for_chunks_and_memories(provider):
    note = _formula_note(provider, [("Trains now leave on Mondays only.", 0.86), ("Trains wait on Tuesdays.", 0.84),
                                    ("Lunch is at noon.", 0.1)])
    memory = _seed(provider, FRIDAYS, 0.85)
    _human_note(provider, 0.99)  # never compared
    reply = _remember(provider)
    first, second = note["chunk_ids"][:2]
    assert reply["older"] == [{"chunk_id": first, "text": "Trains now leave on Mondays only."},
                              {"chunk_id": second, "text": "Trains wait on Tuesdays."},
                              {"id": memory, "text": FRIDAYS}]
    done = _reconcile(provider, [{"chunk_id": first, "text": "Trains left on Mondays until October."},
                                 {"chunk_id": second}, {"id": memory}])
    chunks = _chunk_ids(provider, note["id"])
    assert done == {"deleted_ids": [memory], "changed_chunk_ids": [chunks[0]], "deleted_chunk_ids": [second]}
    assert _note(provider, note["id"]) == ("Trains left on Mondays until October.\n\nLunch is at noon.", None)
    assert _ids(provider._memory_db) == {reply["id"]}


def test_a3_emptied_automatic_note_is_trashed(provider):
    note = _formula_note(provider, [("Trains now leave on Mondays only.", 0.86)])
    reply = _remember(provider)
    assert _reconcile(provider, [{"chunk_id": reply["older"][0]["chunk_id"]}]) == {"deleted_chunk_ids": note["chunk_ids"]}
    assert _note(provider, note["id"])[1] is not None


def test_a3_human_chunk_is_never_listed_and_refuses_the_call(provider):
    note = _formula_note(provider, [("Trains now leave on Mondays only.", 0.86)])
    _, human_chunks = _human_note(provider, 0.99)
    reply = _remember(provider)
    assert reply["older"] == [{"chunk_id": note["chunk_ids"][0], "text": "Trains now leave on Mondays only."}]
    before = _snapshot(provider)
    refused = _reconcile(provider, [{"chunk_id": note["chunk_ids"][0]}, {"chunk_id": human_chunks[0]}])
    assert refused["error"] == "not_allowed" and _snapshot(provider) == before


# A4 -------------------------------------------------------------------------------------------
def test_a4_automatic_chunks_are_separate_results_human_note_collapses(provider):
    note = _formula_note(provider, [("Release train rule one.", 0.95), ("Release train rule two.", 0.93),
                                    ("Release train rule three.", 0.91)])
    human_id, _ = _human_note(provider, 0.94)
    found = json.loads(provider.handle_tool_call("neospike_memory_search", {"query": NEW_TEXT, "limit": 20}))["results"]
    automatic = [row["chunk_id"] for row in found if row.get("id") == note["id"]]
    assert automatic == note["chunk_ids"]  # three results, in score order, each with its chunk id
    assert [row["id"] for row in found if row.get("kind") == "note"].count(human_id) == 1
    text = provider.prefetch(NEW_TEXT)
    assert all(f"note:{note['id']}.{chunk_id} #" in text for chunk_id in note["chunk_ids"])


# A6 -------------------------------------------------------------------------------------------
def test_a6_one_new_tool_and_forget_unchanged(provider):
    schemas = {item["name"]: item for item in provider.get_tool_schemas()}
    assert len(schemas) == (TOOLS_BEFORE + 1 - TOOLS_REMOVED_INBOX - TOOLS_REMOVED_IMAGE - TOOLS_REMOVED_ADMIN
                            + TOOLS_ADDED_DOC) and "neospike_memory_reconcile" in schemas
    assert schemas["neospike_memory_delete"]["parameters"] == FORGET_SCHEMA_BEFORE
    prompt = provider.system_prompt_block()
    assert len(prompt) <= PROMPT_CHARS_BEFORE and "TOPIC" not in prompt and "topic page" not in prompt


def test_a6_empty_or_secret_text_refuses_the_whole_call(provider):
    """A body over 2000 characters is split (test_piece_chain.py), not refused."""
    first = _seed(provider, MONDAYS, 0.85)
    second = _seed(provider, FRIDAYS, 0.82)
    _remember(provider)
    before = _snapshot(provider)
    assert _reconcile(provider, [{"id": second}, {"id": first, "text": "   "}])["error"] == "invalid"
    secret = "The deploy key is OPENAI_API_KEY=sk-secret-value now."
    assert _reconcile(provider, [{"id": second}, {"id": first, "text": secret}])["error"] == "policy"
    assert _snapshot(provider) == before


def test_a6_changed_item_is_not_compared_again(provider):
    first = _seed(provider, MONDAYS, 0.85)
    second = _seed(provider, FRIDAYS, 0.82)
    _remember(provider)
    done = _reconcile(provider, [{"id": first, "text": "The release train left on Mondays until October."}])
    assert done == {"changed_ids": [first]} and "older" not in done
    assert _text(provider, second) == FRIDAYS
