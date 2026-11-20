"""Preferences/constraints in their own table. Common ones (no situation) go out once per conversation in
<neospike_profile>; situational ones go out in <neospike_situation> when the message is close to the situation
(cos >= store.DEFAULT_SEMANTIC_CUTOFF), once per conversation. Unit-vector keyword embedder; no LLM, no CLI except
the dashboard child process."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import time
from contextlib import closing
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from neospike import backup as db_backup
from neospike import cli
from neospike import preferences as prefs
from neospike.provider import PROFILE_OPEN, SITUATION_CLOSE, SITUATION_HEADER, SITUATION_OPEN, SituationTuning
from neospike.store import DEFAULT_SEMANTIC_CUTOFF, SCHEMA_VERSION, MemoryStore
from neospike.config import NeospikeConfig
from neospike.database import canonical_database_path
from neospike.dashboard import plugin_api as api
from neospike.tests.test_dashboard_notes import _child_prefix, _CliSpy, _ReadOnlyConnectGuard
from neospike.tests.test_live_context import (CUT_AGO_S, NAMESPACE, SID, KeywordEmbedder, compact, deliver,  # noqa: F401
                                              home, log_entries, make_state, provider)
from neospike.tests.test_sphere_map import _NUMPY_BLOCKER, _run_script

SAVE = {"scope": "global", "subject": "style", "durability": "stable"}
PYTHON_SITUATION = "writing python code"
PYTHON_RULE = "Use four-space indents and type hints in python."
GUITAR_RULE = "Answer guitar questions with short sentences."


def save(provider, text: str, **fields) -> dict:
    return json.loads(provider.handle_tool_call("neospike_memory_save", {"text": text, **SAVE, **fields}))


def update(provider, **fields) -> dict:
    return json.loads(provider.handle_tool_call("neospike_memory_update",
                                                {"subject": "style", "durability": "stable", **fields}))


def pref_rows(store: MemoryStore) -> list[tuple]:
    return [tuple(row) for row in store._conn.execute(
        "SELECT id,type,situation,basis,text FROM preferences ORDER BY id")]


def item_count(store: MemoryStore) -> int:
    return int(store._conn.execute("SELECT COUNT(*) FROM items").fetchone()[0])


# 1. migration v22 -> v23 ------------------------------------------------------------------------------------------
def _labelled(store: MemoryStore, text: str, scope: str, kind: str, *, subject: str = "", basis: str | None = None,
              project: str = "") -> int:
    metadata = {"scope": scope, "subject": subject, "durability": "stable", "session_id": "s1", "memory_kind": kind}
    if basis is not None:
        metadata["basis"] = basis
    memory_id = store.insert(NAMESPACE, text, KeywordEmbedder.vector(text), source="agent:s1", kind="memory_item",
                             project=project, metadata=metadata)
    assert memory_id is not None
    return memory_id


def test_v22_preferences_move_to_their_table(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = MemoryStore(path, dimensions=KeywordEmbedder.dimensions)
    try:
        common = _labelled(store, "Prefers guitar examples.", "global", "preference", basis="user_said")
        scoped = _labelled(store, "Keep python answers short.", "project", "constraint", subject="python work",
                           project="D:/proj")
        bare = _labelled(store, "Likes tea breaks.", "session", "preference")
        gone = _labelled(store, "Liked rust once.", "global", "preference", basis="user_said")
        fact = _labelled(store, "The coffee machine is broken.", "global", "fact")
        with store._lock, store._conn:
            store._conn.execute("UPDATE items SET invalid_at=1 WHERE id=?", (gone,))
            store._conn.execute("INSERT INTO item_links(src_id,dst_id,relation) VALUES (?,?,'supersedes')",
                                (common, gone))
            store._conn.execute("INSERT INTO item_revisions(item_id,rev,text,edited_by,edited_at) VALUES (?,1,'x','agent',1)",
                                (scoped,))
            store._conn.execute("DROP TABLE preferences")  # rewind to v22
            store._conn.execute("UPDATE meta_entries SET value='22' WHERE key='schema_version'")
            store._conn.execute("PRAGMA user_version=22")
    finally:
        store.close()
    store = MemoryStore(path, dimensions=KeywordEmbedder.dimensions)
    try:
        assert pref_rows(store) == [
            (common, "preference", "", "user_said", "Prefers guitar examples."),
            (scoped, "constraint", "python work", "agent_inferred", "Keep python answers short."),
            (bare, "preference", "Likes tea breaks.", "agent_inferred", "Likes tea breaks.")]
        assert [row[0] for row in store._conn.execute("SELECT id FROM items")] == [fact]
        assert store._conn.execute("SELECT COUNT(*) FROM items WHERE json_extract(metadata_json,'$.memory_kind') "
                                   "IN ('preference','constraint')").fetchone()[0] == 0
        assert store._conn.execute("SELECT COUNT(*) FROM item_links").fetchone()[0] == 0
        assert store._conn.execute("SELECT COUNT(*) FROM item_revisions").fetchone()[0] == 0
        moved = store._conn.execute("SELECT embedding,situation_embedding FROM preferences WHERE id=?",
                                    (common,)).fetchone()
        assert bytes(moved[0]) and not bytes(moved[1])  # text vector kept, situation vector left to embed-pending
        assert store._conn.execute("SELECT value FROM meta_entries WHERE key='schema_version'").fetchone()[0] \
            == str(SCHEMA_VERSION) == "23"
        assert store._conn.execute("PRAGMA user_version").fetchone()[0] == 23
        store._conn.execute("INSERT INTO items_fts(items_fts) VALUES ('integrity-check')")
    finally:
        store.close()
    assert (tmp_path / "memory.pre-v23.db").exists()


def test_newer_database_is_not_opened_for_writing(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    MemoryStore(path, dimensions=KeywordEmbedder.dimensions).close()
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA user_version=24")
    with pytest.raises(Exception, match="newer"):
        MemoryStore(path, dimensions=KeywordEmbedder.dimensions)


# 2. save ----------------------------------------------------------------------------------------------------------
def test_inferred_without_situation_is_refused_and_stores_nothing(provider) -> None:
    reply = save(provider, PYTHON_RULE, kind="preference", basis="agent_inferred")
    assert reply["stored"] is False and reply["error"] == "situation_required"
    assert "situation" in reply["message"] and "save again" in reply["message"]
    assert pref_rows(provider._memory_db) == [] and item_count(provider._memory_db) == 0


def test_user_said_without_situation_is_common_and_not_an_item(provider) -> None:
    reply = save(provider, GUITAR_RULE, kind="preference", basis="user_said")
    assert reply["stored"] is True and reply["id"].startswith("pref:") and reply["common"] is True
    [row] = pref_rows(provider._memory_db)
    assert row[1:] == ("preference", "", "user_said", GUITAR_RULE)
    assert item_count(provider._memory_db) == 0
    inferred = save(provider, PYTHON_RULE, kind="constraint", basis="agent_inferred", situation=PYTHON_SITUATION)
    assert inferred["stored"] is True and inferred["common"] is False
    assert pref_rows(provider._memory_db)[-1][1:3] == ("constraint", PYTHON_SITUATION)


def test_near_duplicate_replaces_the_old_row(provider) -> None:
    first = save(provider, "Use python type hints everywhere.", kind="preference", basis="user_said")
    second = save(provider, "Use python type hints everywhere please.", kind="preference", basis="agent_inferred",
                  situation=PYTHON_SITUATION)
    assert second["deleted"] == [first["id"]]
    # The deleted duplicate was common and user_said: the new row takes both (the old scope-widening rule).
    [row] = pref_rows(provider._memory_db)
    assert (prefs.pref_ref(row[0]), row[2], row[3]) == (second["id"], "", "user_said")



def test_near_duplicate_constraint_keeps_its_situation_and_the_delete(provider) -> None:
    first = save(provider, "Use python type hints everywhere.", kind="preference", basis="user_said")
    second = save(provider, "Use python type hints everywhere please.", kind="constraint", basis="user_said",
                  situation=PYTHON_SITUATION)
    assert second["stored"] is True and second["deleted"] == [first["id"]] and second["common"] is False
    [row] = pref_rows(provider._memory_db)  # the duplicate's delete is not rolled back
    assert (prefs.pref_ref(row[0]), row[1], row[2]) == (second["id"], "constraint", PYTHON_SITUATION)


# 3. common preferences: once per conversation in <neospike_profile>, situational ones never there --------------
def test_common_goes_out_once_and_situational_stays_out_of_the_profile(home, provider) -> None:
    make_state(home)
    # Saved by an earlier conversation.
    store = provider._memory_db
    for pref_type, text, situation in (("preference", GUITAR_RULE, ""), ("constraint", PYTHON_RULE, PYTHON_SITUATION)):
        fields = prefs.clean_fields({"type": pref_type, "text": text, "situation": situation})
        store.add_preference(NAMESPACE, fields, embedding=KeywordEmbedder.vector(text),
                             situation_embedding=KeywordEmbedder.vector(situation) if situation else None)
    first = provider.prefetch("tea time?", session_id=SID)
    profile = first[first.index(PROFILE_OPEN):]
    assert GUITAR_RULE in profile and PYTHON_RULE not in first and "[pref:" in profile
    deliver(home, first)
    assert PROFILE_OPEN not in provider.prefetch("tea time?", session_id=SID)


# 4. situational preferences ---------------------------------------------------------------------------------------
def test_situational_goes_out_above_the_cutoff_on_every_matching_turn(home, provider) -> None:
    make_state(home)
    pref_id = save(provider, PYTHON_RULE, kind="constraint", basis="agent_inferred", situation=PYTHON_SITUATION)["id"]
    assert SituationTuning.CUTOFF == DEFAULT_SEMANTIC_CUTOFF
    below = provider.prefetch("guitar chords?", session_id=SID)
    assert SITUATION_OPEN not in below
    first = provider.prefetch("python question", session_id=SID)
    expected = f"- [{pref_id} constraint] When {PYTHON_SITUATION}: {PYTHON_RULE}"
    assert f"{SITUATION_OPEN}\n{SITUATION_HEADER}\n{expected}\n{SITUATION_CLOSE}" in first
    entry = log_entries(home)[-1]
    assert entry["situation"] == [{"pref": int(pref_id.split(":")[1]), "score": 1.0}]
    deliver(home, first, content="python question")
    again = provider.prefetch("another python question", session_id=SID)  # the same line is already in the context
    assert f"{SITUATION_OPEN}\n{SITUATION_HEADER}\n{expected}\n{SITUATION_CLOSE}" in again


def test_situational_without_live_context_goes_out_on_every_matching_turn(provider) -> None:
    save(provider, PYTHON_RULE, kind="constraint", basis="agent_inferred", situation=PYTHON_SITUATION)
    assert SITUATION_OPEN in provider.prefetch("python question", session_id=SID)
    assert SITUATION_OPEN in provider.prefetch("python again", session_id=SID)


def test_row_without_situation_vector_waits(provider) -> None:
    save(provider, PYTHON_RULE, kind="constraint", basis="agent_inferred", situation=PYTHON_SITUATION)
    store = provider._memory_db
    with store._lock, store._conn:
        store._conn.execute("UPDATE preferences SET situation_embedding=x''")
    assert SITUATION_OPEN not in provider.prefetch("python question", session_id=SID)


def test_situation_block_makes_no_embedding_call(provider, monkeypatch) -> None:
    save(provider, PYTHON_RULE, kind="constraint", basis="agent_inferred", situation=PYTHON_SITUATION)
    calls: list[str] = []
    original = provider._vectorizer.embed_query

    def counted(text: str) -> list[float]:
        calls.append(text)
        return original(text)

    monkeypatch.setattr(provider._vectorizer, "embed_query", counted)
    assert SITUATION_OPEN in provider.prefetch("python question", session_id=SID)
    assert calls == ["python question"]  # the recall search's one query embedding


# 5. search never returns preferences ------------------------------------------------------------------------------
def test_search_leaves_preferences_out(provider) -> None:
    save(provider, "Prefers python examples.", kind="preference", basis="user_said")
    save(provider, "The python build uses version 3.12.", kind="fact", basis="user_said")
    found = json.loads(provider.handle_tool_call("neospike_memory_search", {"query": "python"}))
    texts = [hit["text"] for hit in found["results"]]
    assert "The python build uses version 3.12." in texts and "Prefers python examples." not in texts


# 6. update / delete with target preference ------------------------------------------------------------------------
def test_update_and_delete_a_preference(provider) -> None:
    pref = save(provider, PYTHON_RULE, kind="constraint", basis="agent_inferred", situation=PYTHON_SITUATION)["id"]
    number = int(pref.split(":")[1])
    store = provider._memory_db
    emptied = update(provider, id=number, target="preference", expected_text=PYTHON_RULE, new_text=PYTHON_RULE,
                     situation="")
    assert emptied["updated"] is False and emptied["error"] == "situation_required"
    kind_change = update(provider, id=number, target="preference", expected_text=PYTHON_RULE, new_text=PYTHON_RULE,
                         kind="fact")
    assert kind_change["error"] == "kind_change"
    changed = update(provider, id=number, target="preference", expected_text=PYTHON_RULE, mode="supersede",
                     new_text="Use tabs in python code.", situation="editing python files")
    assert changed == {"updated": True, "id": pref}
    assert pref_rows(store) == [(number, "constraint", "editing python files", "agent_inferred",
                                 "Use tabs in python code.")]
    assert store._conn.execute("SELECT COUNT(*) FROM item_revisions").fetchone()[0] == 0  # no history
    removed = json.loads(provider.handle_tool_call("neospike_memory_delete", {"id": number, "target": "preference"}))
    assert removed == {"removed": True, "id": pref} and pref_rows(store) == []


def test_memory_cannot_become_a_preference(provider) -> None:
    save(provider, "The python build uses version 3.12.", kind="fact", basis="user_said")
    memory_id = int(provider._memory_db._conn.execute("SELECT id FROM items").fetchone()[0])
    reply = update(provider, id=memory_id, expected_text="The python build uses version 3.12.",
                   new_text="The python build uses version 3.12.", kind="preference")
    assert reply["error"] == "kind_change"


# 7. dashboard API --------------------------------------------------------------------------------------------------
@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """The dashboard router with writes through the real CLI child (test_dashboard_notes.py setup)."""
    home_dir = tmp_path / "hermes"
    NeospikeConfig().save(home_dir)
    monkeypatch.setenv("HERMES_HOME", str(home_dir))
    monkeypatch.setattr(api, "_cli_argv_base", _child_prefix)
    monkeypatch.setattr(api.subprocess, "run", _CliSpy(subprocess.run))
    monkeypatch.setattr(sqlite3, "connect", _ReadOnlyConnectGuard(sqlite3.connect))
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


def test_dashboard_preference_routes(client) -> None:
    assert client.get("/preferences").json() == {"preferences": []}
    added = client.post("/preferences", json={"type": "preference", "text": "Answer in Korean please.",
                                              "situation": ""})
    assert added.status_code == 200, added.text
    [row] = client.get("/preferences").json()["preferences"]
    assert (row["type"], row["situation"], row["basis"], row["common"]) == ("preference", "", "user_said", True)
    edited = client.put(f"/preferences/{row['id']}", json={"expected_text": row["text"], "type": "constraint",
                                                           "situation": "writing reports", "text": "Answer in Korean."})
    assert edited.status_code == 200, edited.text
    [row] = client.get("/preferences").json()["preferences"]
    assert (row["type"], row["situation"], row["text"], row["common"]) == (
        "constraint", "writing reports", "Answer in Korean.", False)
    stale = client.put(f"/preferences/{row['id']}", json={"expected_text": "old", "text": "x y z"})
    assert stale.status_code == 409
    assert client.delete(f"/preferences/{row['id']}").json() == {"removed": True, "id": row["id"]}
    assert client.get("/preferences").json() == {"preferences": []}


_PREF_ROUTE_SCRIPT = _NUMPY_BLOCKER + (
    "import json\n"
    "from fastapi import FastAPI\n"
    "from fastapi.testclient import TestClient\n"
    "from neospike.dashboard import plugin_api as api\n"
    "app = FastAPI()\n"
    "app.include_router(api.router)\n"
    "response = TestClient(app).get('/preferences')\n"
    "sys.stdout.write(json.dumps({'status': response.status_code, 'body': response.json(),\n"
    "                             'numpy_loaded': 'numpy' in sys.modules}))\n"
)


def test_dashboard_preferences_import_and_list_without_numpy(tmp_path: Path) -> None:
    home_dir = tmp_path / "hermes"
    store = MemoryStore(home_dir / "neospike" / "memory.db", dimensions=KeywordEmbedder.dimensions)
    try:
        fields = prefs.clean_fields({"type": "preference", "text": GUITAR_RULE, "situation": ""})
        store.add_preference(NAMESPACE, fields, embedding=None, situation_embedding=None)
    finally:
        store.close()
    routed = _run_script(_PREF_ROUTE_SCRIPT, home_dir)
    assert routed["status"] == 200 and routed["numpy_loaded"] is False
    assert [row["text"] for row in routed["body"]["preferences"]] == [GUITAR_RULE]


# 8. incremental backup -> restore ---------------------------------------------------------------------------------
def test_backup_restore_keeps_preferences_and_their_deletes(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "live" / "memory.db", dimensions=KeywordEmbedder.dimensions)
    directory = tmp_path / "backups"
    query = "SELECT * FROM preferences ORDER BY id"
    try:
        def add(text: str, situation: str) -> int:
            fields = prefs.clean_fields({"type": "preference", "text": text, "situation": situation})
            return store.add_preference(NAMESPACE, fields, embedding=KeywordEmbedder.vector(text),
                                        situation_embedding=KeywordEmbedder.vector(situation) if situation else None)
        kept = add(GUITAR_RULE, "")
        dropped = add("Prefers tea answers.", "tea talk")
        assert db_backup.run_backup(store.path, directory)["kind"] == "full"
        assert store.change_preference(NAMESPACE, kept, {"situation": "guitar talk"},
                                       situation_embedding=KeywordEmbedder.vector("guitar talk")) == "updated"
        assert store.remove_preference(NAMESPACE, dropped)
        added = add(PYTHON_RULE, PYTHON_SITUATION)
        increment = db_backup.run_backup(store.path, directory)
        assert increment["kind"] == "incremental" and increment["rows"]["preferences"] == 2
        target = tmp_path / "restored.db"
        db_backup.restore(directory, target)
        with closing(sqlite3.connect(target)) as restored:
            assert restored.execute(query).fetchall() == [tuple(row) for row in store._conn.execute(query)]
            assert [row[0] for row in restored.execute("SELECT id FROM preferences ORDER BY id")] == [kept, added]
    finally:
        store.close()


# 9. the rescope commands are gone ---------------------------------------------------------------------------------
@pytest.mark.parametrize("command", ["memory-rescope", "rescope-session"])
def test_rescope_commands_do_not_exist(command: str) -> None:
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(cli.forward_old_command([command]))
    assert command not in cli.OLD_COMMANDS


# embed-pending fills the preference vectors ----------------------------------------------------------------------
def test_embed_pending_fills_preference_vectors(tmp_path: Path) -> None:
    from neospike.notes import embed_pending
    store = MemoryStore(tmp_path / "memory.db", dimensions=KeywordEmbedder.dimensions)
    try:
        fields = prefs.clean_fields({"type": "constraint", "text": PYTHON_RULE, "situation": PYTHON_SITUATION,
                                     "basis": "agent_inferred"})
        pref_id = store.add_preference(NAMESPACE, fields, embedding=None, situation_embedding=None)
        result = embed_pending(store, KeywordEmbedder(), chunk_editor="formula")
        assert result["preferences"]["filled"] == 2 and result["preferences"]["left"] == 0
        row = store.preference(NAMESPACE, pref_id)
        assert row["has_vector"] and row["has_situation_vector"]
    finally:
        store.close()


# 8. a constraint always needs a situation, whatever its basis --------------------------------------------------------
def test_user_said_constraint_without_situation_is_refused(provider) -> None:
    reply = save(provider, PYTHON_RULE, kind="constraint", basis="user_said")
    assert reply["stored"] is False and reply["error"] == "situation_required"
    assert reply["message"] == prefs.MSG_CONSTRAINT_SITUATION_REQUIRED
    assert pref_rows(provider._memory_db) == []


def _saved_earlier(store: MemoryStore, pref_type: str, text: str, situation: str = "") -> None:
    """A row an earlier conversation saved."""
    fields = prefs.clean_fields({"type": pref_type, "text": text, "situation": situation})
    store.add_preference(NAMESPACE, fields, embedding=KeywordEmbedder.vector(text),
                         situation_embedding=KeywordEmbedder.vector(situation) if situation else None)


def test_user_said_constraint_with_situation_is_situational(home, provider) -> None:
    make_state(home)
    _saved_earlier(provider._memory_db, "preference", GUITAR_RULE)
    reply = save(provider, PYTHON_RULE, kind="constraint", basis="user_said", situation=PYTHON_SITUATION)
    assert reply["stored"] is True and reply["common"] is False
    first = provider.prefetch("python question", session_id=SID)
    profile = first[first.index(PROFILE_OPEN):]
    situation = first[first.index(SITUATION_OPEN):first.index(SITUATION_CLOSE)]
    assert GUITAR_RULE in profile and f"When {PYTHON_SITUATION}: {PYTHON_RULE}" in situation
    assert PYTHON_RULE not in first.replace(situation, "")


def test_update_cannot_make_a_constraint_common(provider) -> None:
    pref = save(provider, PYTHON_RULE, kind="constraint", basis="user_said", situation=PYTHON_SITUATION)["id"]
    common = save(provider, GUITAR_RULE, kind="preference", basis="user_said")["id"]
    emptied = update(provider, id=int(pref.split(":")[1]), target="preference", expected_text=PYTHON_RULE,
                     new_text=PYTHON_RULE, situation="")
    retyped = update(provider, id=int(common.split(":")[1]), target="preference", expected_text=GUITAR_RULE,
                     new_text=GUITAR_RULE, kind="constraint")
    for reply in (emptied, retyped):
        assert reply["updated"] is False and reply["error"] == "situation_required"
    assert [row[1:3] for row in pref_rows(provider._memory_db)] == [("constraint", PYTHON_SITUATION),
                                                                    ("preference", "")]


def test_store_layer_refuses_a_common_constraint(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db", dimensions=KeywordEmbedder.dimensions)
    try:
        common = prefs.clean_fields({"type": "constraint", "text": PYTHON_RULE, "situation": ""})
        with pytest.raises(prefs.PreferenceError):
            store.add_preference(NAMESPACE, common, embedding=None, situation_embedding=None)
        fields = prefs.clean_fields({"type": "constraint", "text": PYTHON_RULE, "situation": PYTHON_SITUATION})
        pref_id = store.add_preference(NAMESPACE, fields, embedding=None, situation_embedding=None)
        with pytest.raises(prefs.PreferenceError):
            store.change_preference(NAMESPACE, pref_id, {"situation": ""})
        assert store.preference(NAMESPACE, pref_id)["situation"] == PYTHON_SITUATION
    finally:
        store.close()


def test_dashboard_refuses_a_common_constraint(client) -> None:
    created = client.post("/preferences", json={"type": "constraint", "text": PYTHON_RULE, "situation": ""})
    assert created.status_code == 422, created.text
    added = client.post("/preferences", json={"type": "constraint", "text": PYTHON_RULE,
                                              "situation": PYTHON_SITUATION})
    assert added.status_code == 200, added.text
    [row] = client.get("/preferences").json()["preferences"]
    emptied = client.put(f"/preferences/{row['id']}", json={"expected_text": PYTHON_RULE, "type": "constraint",
                                                            "situation": "", "text": PYTHON_RULE})
    assert emptied.status_code == 422, emptied.text
    [row] = client.get("/preferences").json()["preferences"]
    assert (row["type"], row["situation"], row["common"]) == ("constraint", PYTHON_SITUATION, False)


def test_old_common_constraint_stays_out_of_the_profile_but_is_listed(home, provider) -> None:
    make_state(home)
    store = provider._memory_db
    with store._conn:  # a row written before the rule, straight into the table
        store._conn.execute(
            "INSERT INTO preferences(owner,type,text,text_hash,situation,basis,created_at) VALUES (?,?,?,?,?,?,?)",
            (NAMESPACE, "constraint", PYTHON_RULE, prefs.text_hash("constraint", PYTHON_RULE), "", "user_said", 1.0))
    _saved_earlier(store, "preference", GUITAR_RULE)
    first = provider.prefetch("tea time?", session_id=SID)
    assert GUITAR_RULE in first[first.index(PROFILE_OPEN):] and PYTHON_RULE not in first
    listed = prefs.list_preferences(store._conn, NAMESPACE)
    assert [(row["type"], row["text"]) for row in listed if row["type"] == "constraint"] == [
        ("constraint", PYTHON_RULE)]


# common preferences: a line already in the live context is left out, whoever saved the row ---------------------
def test_common_line_already_in_the_context_is_left_out(home, provider) -> None:
    make_state(home)
    store = provider._memory_db
    _saved_earlier(store, "preference", GUITAR_RULE)
    _saved_earlier(store, "preference", "Likes tea examples in every answer.")
    [guitar_id] = [row[0] for row in pref_rows(store) if row[4] == GUITAR_RULE]
    guitar_line = f"- [{prefs.pref_ref(guitar_id)} preference] {GUITAR_RULE}"
    deliver(home, f"pasted note\n{guitar_line}")  # the line is in the context without a profile block
    first = provider.prefetch("tea time?", session_id=SID)
    profile = first[first.index(PROFILE_OPEN):]
    assert "Likes tea examples" in profile and guitar_line not in profile


def test_common_saved_here_after_the_cut_goes_out_until_its_line_is_in_the_context(home, provider) -> None:
    make_state(home)
    now = time.time()
    compact(home, now - CUT_AGO_S)
    reply = save(provider, GUITAR_RULE, kind="preference", basis="user_said")
    assert reply["stored"] is True and reply["common"] is True
    first = provider.prefetch("tea time?", session_id=SID)
    expected = f"- [{reply['id']} preference] {GUITAR_RULE}"
    assert expected in first[first.index(PROFILE_OPEN):]
    deliver(home, first)
    assert PROFILE_OPEN not in provider.prefetch("tea time?", session_id=SID)


def preference_columns(connection: sqlite3.Connection) -> list[str]:
    return [str(rec[1]) for rec in connection.execute("PRAGMA table_info(preferences)")]


def test_new_preference_table_has_no_origin(provider) -> None:
    reply = save(provider, GUITAR_RULE, kind="preference", basis="user_said")
    assert reply["stored"] is True
    store = provider._memory_db
    assert "origin" not in preference_columns(store._conn)
    assert "origin" not in store.preference(NAMESPACE, int(reply["id"].split(":")[1]))


def test_dashboard_preference_table_has_no_origin(client, tmp_path: Path) -> None:
    added = client.post("/preferences", json={"type": "preference", "text": "Answer in Korean please.",
                                              "situation": ""})
    assert added.status_code == 200, added.text
    database = canonical_database_path(tmp_path / "hermes" / "neospike", "memory")
    with closing(sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)) as connection:
        assert "origin" not in preference_columns(connection)
        assert connection.execute("SELECT COUNT(*) FROM preferences").fetchone()[0] == 1


# 4. the retired origin column of an earlier v23 layout ------------------------------------------------------------
_IMPORTANCE_LINE = "    importance REAL NOT NULL DEFAULT 0.5,\n"


def _old_layout_database(tmp_path: Path) -> tuple[Path, list[tuple]]:
    """A v23 database whose preferences table still has the origin column (filled), and its rows without it."""
    path = tmp_path / "memory.db"
    store = MemoryStore(path, dimensions=KeywordEmbedder.dimensions)
    try:
        for pref_type, text, situation in (("preference", GUITAR_RULE, ""),
                                           ("constraint", PYTHON_RULE, PYTHON_SITUATION)):
            fields = prefs.clean_fields({"type": pref_type, "text": text, "situation": situation})
            assert store.add_preference(NAMESPACE, fields, embedding=KeywordEmbedder.vector(text),
                                        situation_embedding=KeywordEmbedder.vector(situation) if situation else None)
        with store._lock, store._conn:
            store._conn.execute("DELETE FROM preferences WHERE type='preference'")  # ids keep a gap
        fields = prefs.clean_fields({"type": "preference", "text": "Short answers.", "situation": ""})
        assert store.add_preference(NAMESPACE, fields, embedding=None, situation_embedding=None)
    finally:
        store.close()
    old_ddl = prefs.PREFERENCES_DDL.replace(_IMPORTANCE_LINE, _IMPORTANCE_LINE + "    origin TEXT NOT NULL DEFAULT '',\n")
    assert old_ddl != prefs.PREFERENCES_DDL
    with closing(sqlite3.connect(path)) as connection, connection:
        columns = preference_columns(connection)
        rows = connection.execute(f"SELECT {','.join(columns)} FROM preferences ORDER BY id").fetchall()
        connection.execute("DROP TABLE preferences")
        connection.executescript(old_ddl)
        connection.executemany(
            f"INSERT INTO preferences({','.join(columns)},origin) VALUES ({','.join('?' * (len(columns) + 1))})",
            [(*row, f"agent:old{row[0]}") for row in rows])
        assert "origin" in preference_columns(connection)
    return path, rows


def _rows_without_origin(path: Path) -> tuple[list[str], list[tuple]]:
    with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as connection:
        columns = preference_columns(connection)
        return columns, connection.execute(
            f"SELECT {','.join(columns)} FROM preferences ORDER BY id").fetchall()


def test_write_open_drops_the_origin_column_and_keeps_rows(tmp_path: Path) -> None:
    path, rows = _old_layout_database(tmp_path)
    for _ in range(2):  # a second open finds nothing to drop and changes nothing
        MemoryStore(path, dimensions=KeywordEmbedder.dimensions).close()
        columns, after = _rows_without_origin(path)
        assert "origin" not in columns and after == rows and len(after) == 2
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 23
        assert connection.execute("SELECT value FROM meta_entries WHERE key='schema_version'").fetchone()[0] == "23"
        assert prefs.drop_origin_column(connection) is False
    assert not (tmp_path / "memory.pre-v24.db").exists()
    store = MemoryStore(path, dimensions=KeywordEmbedder.dimensions)
    try:
        fields = prefs.clean_fields({"type": "preference", "text": "Use metric units.", "situation": ""})
        assert store.add_preference(NAMESPACE, fields, embedding=None, situation_embedding=None) is not None
    finally:
        store.close()


def test_read_only_open_leaves_the_origin_column(tmp_path: Path) -> None:
    path, rows = _old_layout_database(tmp_path)
    before = path.read_bytes()
    with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as connection:
        listed = prefs.list_preferences(connection)
        assert "origin" in preference_columns(connection)
    assert sorted(item["id"] for item in listed) == [row[0] for row in rows]
    assert path.read_bytes() == before


def test_v22_migration_makes_no_origin_column(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    store = MemoryStore(path, dimensions=KeywordEmbedder.dimensions)
    try:
        kept = _labelled(store, "Prefers guitar examples.", "global", "preference", basis="user_said")
        with store._lock, store._conn:
            store._conn.execute("DROP TABLE preferences")  # rewind to v22
            store._conn.execute("UPDATE meta_entries SET value='22' WHERE key='schema_version'")
            store._conn.execute("PRAGMA user_version=22")
    finally:
        store.close()
    store = MemoryStore(path, dimensions=KeywordEmbedder.dimensions)
    try:
        assert "origin" not in preference_columns(store._conn)
        assert [row[0] for row in pref_rows(store)] == [kept]
    finally:
        store.close()
