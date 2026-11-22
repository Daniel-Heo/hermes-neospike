"""Agents may edit human notes + origin rules."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from neospike.notes import (
    ORIGIN_AUTOMATION, NoteConflict, NotePermissionError, NotePolicyError, NoteService,
)
from neospike.store import MemoryStore
from neospike.tests.test_dashboard_notes import NAMESPACE
from neospike.tests.test_notes import DIMENSIONS, BowEmbedder, _call, provider  # noqa: F401 (fixture)

SECRET_BODY = "Deploy with api_key = sk-live-0123456789abcdefghijklmnop please."
INDEX_JS = Path(__file__).resolve().parents[1] / "dashboard" / "dist" / "index.js"


@pytest.fixture
def notes(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db", dimensions=DIMENSIONS)
    yield NoteService(store, BowEmbedder())
    store.close()


def _revisions(notes: NoteService, note_id: int) -> list[tuple]:
    return [tuple(row) for row in notes.store._conn.execute(
        "SELECT rev,edited_by FROM item_revisions WHERE item_id=? ORDER BY rev", (note_id,))]


# 1 + 3 + 5 ----------------------------------------------------------------------------------
def test_agent_edits_human_note_and_human_rolls_back(notes: NoteService) -> None:
    human = notes.save("Plan", "# Plan\nWritten by a person.", author="human")
    edited = notes.save("Plan", "# Plan\nRewritten by the agent.", author="agent", expected_rev=human["rev"])
    assert edited["rev"] == human["rev"] + 1 and edited["author"] == "human"
    assert notes.read(note_id=human["id"])["author"] == "human"
    assert _revisions(notes, human["id"]) == [(1, "agent")]
    with pytest.raises(NoteConflict):  # 3: stale rev
        notes.save("Plan", "# Plan\nStale agent write.", author="agent", expected_rev=human["rev"])
    restored = notes.restore(human["id"], 1, author="human", expected_rev=edited["rev"])  # 5
    current = notes.read(note_id=human["id"])
    assert restored["rev"] == 3 and current["body"] == "# Plan\nWritten by a person." and current["author"] == "human"
    assert _revisions(notes, human["id"]) == [(1, "agent"), (2, "human")]


# 2 ------------------------------------------------------------------------------------------
def test_human_edit_keeps_agent_author(notes: NoteService) -> None:
    agent = notes.save("Report", "# Report\nAgent text.", author="agent")
    edited = notes.save("Report", "# Report\nPerson text.", author="human", expected_rev=agent["rev"])
    assert edited["author"] == "agent" and notes.read(note_id=agent["id"])["author"] == "agent"
    assert _revisions(notes, agent["id"]) == [(1, "human")]


# 4 ------------------------------------------------------------------------------------------
def test_secret_body_refused_for_agent_edit_of_human_note(notes: NoteService) -> None:
    human = notes.save("Keys", "# Keys\nNothing secret here.", author="human")
    before = dict(notes.read(note_id=human["id"]))
    with pytest.raises(NotePolicyError):
        notes.save("Keys", SECRET_BODY, author="agent", expected_rev=human["rev"])
    assert dict(notes.read(note_id=human["id"])) == before and _revisions(notes, human["id"]) == []


def test_agent_tool_edits_human_note(provider) -> None:  # noqa: F811
    provider._core.save_note("Person note", "A human wrote this note body.", author="human")
    written = _call(provider, "neospike_doc_write",
                    {"title": "Person note", "body": "Agent edit of the person note.", "expected_rev": 1})
    assert written["doc"]["rev"] == 2 and written["doc"]["author"] == "human"


# 7 ------------------------------------------------------------------------------------------
def test_automation_is_refused_for_human_notes_and_writes_agent_notes(notes: NoteService) -> None:
    """An automation edit of a human note is refused (no proposal); the note stays as it was."""
    human = notes.save("Human page", "# Human\nPerson text.", author="human")
    agent = notes.save("Agent page", "# Agent\nAgent text.", author="agent")
    touched = notes.save("Touched page", "# Touched\nAgent text.", author="agent")
    touched = notes.save("Touched page", "# Touched\nPerson edit.", author="human", expected_rev=touched["rev"])
    for note in (human, touched):
        before = notes.read(note_id=note["id"])
        with pytest.raises(NotePermissionError):
            notes.save(before["title"], "# New\nAutomation rewrite.", author="agent", origin=ORIGIN_AUTOMATION,
                       namespace=NAMESPACE, expected_rev=before["rev"], note_id=note["id"])
        assert notes.read(note_id=note["id"]) == before
    direct = notes.save("Agent page", "# Agent\nAutomation rewrite.", author="agent", origin=ORIGIN_AUTOMATION,
                        namespace=NAMESPACE, expected_rev=agent["rev"])
    assert direct["rev"] == agent["rev"] + 1
    with pytest.raises(NotePolicyError):  # the policy check comes first
        notes.save("Human page", SECRET_BODY, author="agent", origin=ORIGIN_AUTOMATION, namespace=NAMESPACE,
                   expected_rev=human["rev"])


# 9 ------------------------------------------------------------------------------------------
def test_dashboard_save_failure_keeps_editor_text() -> None:
    source = INDEX_JS.read_text(encoding="utf-8-sig")
    catch = re.search(r"function save\(expectedRev\) \{.*?\}\)\.catch\(function \(e\) \{(.*?)\n      \}\);", source, re.S)
    assert catch and "failure.status === 422" in catch.group(1)
    assert not re.search(r"set(Title|Body|Tags)\(|showNote\(", catch.group(1))

