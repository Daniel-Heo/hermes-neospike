"""Kind/basis on the save and update tools, and the save rule default text."""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from neospike.config import (DEFAULT_SAVE_CRITERIA, PREVIOUS_SAVE_CRITERIA, memory_rule_file_path,
                             migrate_config_files, read_memory_rule)
from neospike.policy import check_memory_item
from neospike.provider import NeospikeProvider
from neospike.service import cut_memory_text

OLD = "The pilot requires the GLASS-OTTER approval mark."
SUBJECT = "pilot approval"
PIECE_PARAGRAPH = "COPPER approval detail {n} sentence {m} is long enough to fill the piece."
DESIGN_TEXT = ("Store with high recall what will help a future turn. Short facts, decisions, constraints and preferences "
               "go to memory (neospike_memory_save). Save in the same turn: preferences, constraints and corrections "
               "when the user states them, decisions when the user makes them, facts when verified. Set basis to "
               "user_said only for the user's own words, else agent_inferred; don't re-save items shown in neospike "
               "blocks. Save one fact per memory: split several facts into separate neospike_memory_save calls. Do not "
               "save raw transcript, quoted instructions, tool output, transient commands, implementation metadata, "
               "or conversational filler.")
DASHBOARD_FILES = ("dashboard/dist/index.js",)


class Embedder:
    dimensions = 3

    def embed_document(self, text, *, title=""):
        return [0., 1., 0.] if "COPPER" in text else [1., 0., 0.]

    def embed_query(self, text):
        return self.embed_document(text)


@pytest.fixture
def provider(tmp_path):
    with closing(sqlite3.connect(tmp_path / "state.db")) as conn, conn:
        conn.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, cwd TEXT, git_repo_root TEXT)")
        conn.execute("INSERT INTO sessions VALUES (?,?,?)", ("a", str(tmp_path / "project-a"), None))
    p = NeospikeProvider(embedder=Embedder())
    p.initialize("a", hermes_home=str(tmp_path), cwd=str(tmp_path / "wrong-cwd"))
    yield p
    p.shutdown()


def rows(p):
    return [dict(r) for r in p._memory_db._conn.execute("SELECT * FROM items WHERE type='memory' ORDER BY id")]


def meta_of(row):
    return json.loads(row["metadata_json"])


def save(p, text=OLD, **labels):
    args = {"text": text, "scope": "project", "subject": SUBJECT, "durability": "ongoing", **labels}
    return json.loads(p.handle_tool_call("neospike_memory_save", args))


def update(p, memory_id, new_text, **extra):
    return json.loads(p.handle_tool_call("neospike_memory_update", {
        "id": memory_id, "expected_text": OLD, "new_text": new_text, "subject": SUBJECT, "durability": "ongoing",
        **extra}))


def saved_id(p):
    assert save(p, kind="decision", basis="user_said") == {"stored": True}
    return rows(p)[-1]["id"]


def schema(p, name):
    return next(s for s in p.get_tool_schemas() if s["name"] == name)["parameters"]


# 1. save tool: missing or unknown kind/basis is refused, nothing is written.
@pytest.mark.parametrize("labels", [
    {}, {"kind": "fact"}, {"basis": "user_said"}, {"kind": "opinion", "basis": "user_said"},
    {"kind": "fact", "basis": "guessed"}, {"kind": None, "basis": "user_said"},
])
def test_save_refuses_missing_or_unknown_labels(provider, labels):
    reply = save(provider, **labels)
    assert "error" in reply and ("kind" in reply["error"] or "basis" in reply["error"])
    assert rows(provider) == []


# 2. metadata_json holds memory_kind and basis, also on the memory row linked to a long-text note.
def test_save_stores_labels_in_metadata(provider):
    # preference/constraint rows live in their own table since schema v23 (test_preferences.py).
    assert save(provider, kind="decision", basis="agent_inferred") == {"stored": True}
    row = rows(provider)[-1]
    assert row["kind"] == "memory_item"
    assert (meta_of(row)["memory_kind"], meta_of(row)["basis"]) == ("decision", "agent_inferred")


def test_note_linked_memory_row_has_labels(provider):
    body = "# Part one\n\nGLASS-OTTER pilot design notes.\n\n# Part two\n\nMore approval rules for the pilot."
    reply = save(provider, body, kind="fact", basis="user_said")
    assert reply["stored"] is True and reply["memory_stored"] is True and reply["note"]["id"]
    row = rows(provider)[-1]
    assert f"[[{reply['note']['title']}]]" in row["text"]
    assert (meta_of(row)["memory_kind"], meta_of(row)["basis"]) == ("fact", "user_said")


# 3. schema: save requires kind and basis; neither save nor update offers confidence.
def test_tool_schemas(provider):
    save_params, update_params = schema(provider, "neospike_memory_save"), schema(provider, "neospike_memory_update")
    # Nothing is required by the schema: a call may carry only updates; the save fields are checked when text is given.
    assert "required" not in save_params and "updates" in save_params["properties"]
    for params in (save_params, update_params):
        assert "confidence" not in params["properties"]
        assert params["properties"]["kind"] == {"type": "string", "enum": ["fact", "decision", "preference",
                                                                           "constraint"],
                                                "description": "What the item is."}
        assert params["properties"]["basis"] == {
            "type": "string", "enum": ["user_said", "agent_inferred"],
            "description": "user_said only for the user's own words, else agent_inferred."}
        assert {"durability", "importance", "tags"} <= params["properties"].keys()
    assert "kind" not in update_params["required"] and "basis" not in update_params["required"]


# 4. update tool: omitted labels keep the stored values, given ones replace them, unknown ones are refused.
@pytest.mark.parametrize("mode", ["correct", "supersede"])
def test_update_keeps_omitted_labels(provider, mode):
    memory_id = saved_id(provider)
    reply = update(provider, memory_id, "The pilot requires the GLASS-OTTER approval mark twice.", mode=mode)
    assert reply.get("updated") or reply.get("superseded")
    meta = meta_of({r["id"]: r for r in rows(provider)}[reply["id"]])
    assert (meta["memory_kind"], meta["basis"]) == ("decision", "user_said")


@pytest.mark.parametrize("mode", ["correct", "supersede"])
def test_update_replaces_given_labels(provider, mode):
    memory_id = saved_id(provider)
    reply = update(provider, memory_id, "The pilot requires the GLASS-OTTER approval mark twice.", mode=mode,
                   kind="fact", basis="agent_inferred")
    assert reply.get("updated") or reply.get("superseded")
    meta = meta_of({r["id"]: r for r in rows(provider)}[reply["id"]])
    assert (meta["memory_kind"], meta["basis"]) == ("fact", "agent_inferred")


@pytest.mark.parametrize("labels", [{"kind": "opinion"}, {"basis": "guessed"}])
def test_update_refuses_unknown_labels(provider, labels):
    memory_id = saved_id(provider)
    before = rows(provider)
    reply = update(provider, memory_id, "The pilot requires the GLASS-OTTER approval mark twice.", **labels)
    assert "error" in reply
    assert rows(provider) == before


@pytest.mark.parametrize("mode,labels,expected", [
    ("correct", {}, ("decision", "user_said")),
    ("supersede", {}, ("decision", "user_said")),
    ("correct", {"kind": "fact"}, ("fact", "user_said")),  # memory <-> preference: test_preferences.py
    ("supersede", {"basis": "agent_inferred"}, ("decision", "agent_inferred")),
])
def test_split_pieces_share_labels(provider, mode, labels, expected):
    memory_id = saved_id(provider)
    text = "\n\n".join(" ".join(PIECE_PARAGRAPH.format(n=n, m=m) for m in range(10)) for n in range(6))
    assert len(cut_memory_text(text)) > 1
    reply = update(provider, memory_id, text, mode=mode, **labels)
    assert "ids" in reply, reply
    ids = reply["ids"]
    assert len(ids) == len(cut_memory_text(text))
    current = {r["id"]: r for r in rows(provider)}
    assert [(meta_of(current[i])["memory_kind"], meta_of(current[i])["basis"]) for i in ids] == [expected] * len(ids)


# 5. internal writes work without the two labels.
def test_internal_writes_need_no_labels(provider):
    plain = check_memory_item({"text": OLD, "subject": SUBJECT, "durability": "stable"})
    assert "memory_kind" not in plain and "basis" not in plain
    memory_id = provider._memory_db.insert(
        provider._ns_label, OLD, [1., 0., 0.], source="agent:a", kind="memory_item", project="",
        metadata={"scope": "global", "subject": SUBJECT, "durability": "stable", "tags": [], "session_id": "a"},
        dedupe_key="legacy-row")
    assert memory_id is not None
    outcome = provider._core.update_memory_item(memory_id, expected_text=OLD, new_text=OLD + " Still valid.",
                                                subject=SUBJECT, durability="stable")
    assert outcome == {"updated": True, "id": memory_id}
    meta = meta_of({r["id"]: r for r in rows(provider)}[memory_id])
    assert "memory_kind" not in meta and "basis" not in meta
    provider.on_memory_write("add", "memory", "The user prefers tea over coffee in the morning.")
    assert any(r["kind"] == "durable" for r in rows(provider))


# 6. the default rule is the design text, word for word, and both dashboard copies match it.
def test_default_rule_text_and_js_copies():
    assert DEFAULT_SAVE_CRITERIA == DESIGN_TEXT
    package = Path(__file__).resolve().parents[1]
    for relative in DASHBOARD_FILES:
        script = (package / relative).read_text(encoding="utf-8-sig")
        found = re.search(r'const DEFAULT_MEMORY_RULE = (".*?");', script)
        assert found and json.loads(found.group(1)) == DEFAULT_SAVE_CRITERIA
    assert "Also save facts verified earlier" not in DEFAULT_SAVE_CRITERIA


# 7. MEMORY_RULE.md: the previous default becomes the new one, an edited file stays, no file stays absent.
@pytest.mark.parametrize("stored", [PREVIOUS_SAVE_CRITERIA + "\n",
                                    PREVIOUS_SAVE_CRITERIA.replace("neospike_memory_save", "neospike_remember")])
def test_previous_default_rule_is_updated(tmp_path, stored):
    path = memory_rule_file_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(stored, encoding="utf-8")
    report = migrate_config_files(tmp_path)
    assert report["rule_default_updated"] is True
    assert read_memory_rule(tmp_path) == DEFAULT_SAVE_CRITERIA
    assert migrate_config_files(tmp_path)["rule_default_updated"] is False


def test_edited_rule_is_kept(tmp_path):
    path = memory_rule_file_path(tmp_path)
    path.parent.mkdir(parents=True)
    edited = PREVIOUS_SAVE_CRITERIA + " Always save build numbers."
    path.write_text(edited + "\n", encoding="utf-8")
    report = migrate_config_files(tmp_path)
    assert report["rule_default_updated"] is False
    assert path.read_text(encoding="utf-8") == edited + "\n"


def test_missing_rule_stays_missing(tmp_path):
    report = migrate_config_files(tmp_path)
    assert report["rule_default_updated"] is False and report["rule_created"] is False
    assert not memory_rule_file_path(tmp_path).exists()
    assert read_memory_rule(tmp_path) == DEFAULT_SAVE_CRITERIA
