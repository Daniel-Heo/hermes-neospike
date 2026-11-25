"""Correct vs supersede, invalid_at/superseded_by, revert, revisions."""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

from neospike import backup as db_backup
from neospike import cli
from neospike.config import EMBEDDING_DIMENSIONS, NeospikeConfig
from neospike.tests.supersede_links import link_supersede
from neospike.tests.support_vector import use_full_comparison
from neospike.provider import NeospikeProvider
from neospike.recall_log import REASON_SUPERSEDED
from neospike.store import RELATION_SUPERSEDES, SCHEMA_VERSION, MemoryStore

SESSION = "session-live"
OLD = "The pilot uses the OTTER approval mark for release."
MID = "The pilot now uses the HERON approval mark for release."
NEW = "The pilot finally uses the FALCON approval mark for release."
QUERY_OLD = "OTTER approval mark"
FAKE_KEY = "sk-FAKEKEY1234567890abcdef"
USER_MESSAGE = f"Decision changed: use HERON from now on. my key is {FAKE_KEY}"
KEYWORDS = ("OTTER", "HERON", "FALCON", "SPARROW")


class Embedder:
    dimensions = EMBEDDING_DIMENSIONS  # the CLI opens the store with it

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def embed_document(self, text: str, **_kwargs) -> list[float]:
        axes = [float(word in text) for word in KEYWORDS]
        if not any(axes):
            axes = [0.0] * len(KEYWORDS) + [1.0]
        return axes + [0.0] * (self.dimensions - len(axes))

    embed_query = embed_document

    def take_warnings(self) -> list[str]:
        return []


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    NeospikeConfig().save(home)
    return home


@pytest.fixture
def provider(tmp_path: Path):
    home = _home(tmp_path)
    instance = NeospikeProvider(embedder=Embedder())
    instance.initialize(SESSION, hermes_home=str(home), cwd=str(home))
    yield instance
    instance.shutdown()


def _call(provider: NeospikeProvider, name: str, args: dict) -> dict:
    return json.loads(provider.handle_tool_call(name, args))


def _remember(provider: NeospikeProvider, text: str = OLD) -> int:
    assert _call(provider, "neospike_memory_save", {"text": text, "scope": "global", "subject": "pilot approval",
                                                  "durability": "ongoing", "kind": "fact", "basis": "user_said"}) == {"stored": True}
    return int(provider._memory_db._conn.execute("SELECT MAX(id) FROM items WHERE type='memory'").fetchone()[0])


def _update(provider: NeospikeProvider, memory_id: int, expected: str, new: str, mode: str | None = None) -> dict:
    args = {"id": memory_id, "expected_text": expected, "new_text": new, "subject": "pilot approval",
            "durability": "ongoing"}
    if mode is not None:
        args["mode"] = mode
    return _call(provider, "neospike_memory_update", args)


def _memory(provider: NeospikeProvider, memory_id: int) -> sqlite3.Row:
    return provider._memory_db._conn.execute("SELECT * FROM items WHERE type='memory' AND id=?", (memory_id,)).fetchone()


def _run_cli(provider: NeospikeProvider, monkeypatch, capsys, *arguments: str, code: int = 0) -> dict:
    monkeypatch.setattr(cli, "EmbeddingClient", Embedder)
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(provider._home_root), *arguments])
    assert cli.main() == code
    return json.loads(capsys.readouterr().out)


def _search_ids(provider: NeospikeProvider, **extra) -> dict[int, dict]:
    results = _call(provider, "neospike_memory_search", {"query": QUERY_OLD, **extra})["results"]
    return {item["id"]: item for item in results if item.get("kind") == "memory_item"}


# 1 + 6. supersede: prefetch swap, recall log reason, link row with session and masked evidence -----
def test_supersede_swaps_prefetch_and_records_masked_evidence(provider: NeospikeProvider) -> None:
    old_id = _remember(provider)
    provider.on_turn_start(1, USER_MESSAGE)
    result = _update(provider, old_id, OLD, MID, mode="supersede")
    assert result["superseded"] is True and result["old_id"] == old_id
    new_id = result["id"]
    old = _memory(provider, old_id)
    assert old["invalid_at"] is not None and old["superseded_by"] == new_id and old["text"] == OLD

    injected = provider.prefetch(QUERY_OLD, session_id=SESSION)
    # Only the old text scores for this query; its slot goes to the replacing memory.
    assert MID in injected and OLD not in injected

    log = sorted((provider._home_root / "neospike" / "logs").glob("recall-*.jsonl"))[-1]
    entry = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert [item["id"] for item in entry["injected"] if item["lane"] == "memory"] == [new_id]
    dropped = next(item for item in entry["dropped"] if item["id"] == old_id)
    assert dropped["reason"] == REASON_SUPERSEDED and REASON_SUPERSEDED not in entry["untracked_reasons"]

    link = provider._memory_db._conn.execute("SELECT * FROM item_links WHERE relation=?", (RELATION_SUPERSEDES,)).fetchone()
    assert (link["src_id"], link["dst_id"], link["created_by"], link["origin"], link["session_id"]) == (
        new_id, old_id, "human", "conversation", SESSION)
    assert "Decision changed: use HERON" in link["evidence"] and FAKE_KEY not in link["evidence"]
    assert len(link["evidence"]) <= 500 and link["reverted_at"] is None


# 2. direct search: excluded by default, half score + superseded_by with include_history ---------
def test_direct_search_hides_history_unless_requested(provider: NeospikeProvider, monkeypatch, capsys) -> None:
    old_id = _remember(provider)
    before = _search_ids(provider)[old_id]["score"]
    new_id = _update(provider, old_id, OLD, MID, mode="supersede")["id"]

    assert old_id not in _search_ids(provider)
    history = _search_ids(provider, include_history=True)[old_id]
    assert history["superseded_by"] == new_id and math.isclose(history["score"], before * 0.5, rel_tol=1e-3)

    cli_default = _run_cli(provider, monkeypatch, capsys, "search", QUERY_OLD)["results"]
    cli_history = _run_cli(provider, monkeypatch, capsys, "search", QUERY_OLD, "--include-history")["results"]
    assert old_id not in {item["id"] for item in cli_default}
    assert {item["id"]: item.get("superseded_by") for item in cli_history}[old_id] == new_id


# 3. chains and cycles -------------------------------------------------------------------------
def test_chain_end_is_injected_and_cycles_are_refused(provider: NeospikeProvider) -> None:
    a = _remember(provider)
    b = _update(provider, a, OLD, MID, mode="supersede")["id"]
    c = _update(provider, b, MID, NEW, mode="supersede")["id"]
    injected = provider.prefetch(QUERY_OLD, session_id=SESSION)
    assert NEW in injected and OLD not in injected and MID not in injected

    store = provider._memory_db
    assert link_supersede(store, a, b) == {"linked": False, "reason": "cycle"}  # A->B->C->A
    assert link_supersede(store, b, c)["reason"] == "cycle"
    assert store._conn.execute("SELECT invalid_at FROM items WHERE type='memory' AND id=?", (c,)).fetchone()[0] is None


def test_two_memories_replace_one_and_both_stay_active(provider: NeospikeProvider) -> None:
    # Y and Z each replace a different part of X; superseding X again must not touch Y.
    x = _remember(provider)
    first = _update(provider, x, OLD, MID, mode="supersede")
    assert link_supersede(provider._memory_db, x, first["id"])["reason"] == "cycle"  # X->Y then Y->X
    second = _update(provider, x, OLD, NEW, mode="supersede")
    y, z = first["id"], second["id"]
    assert second["superseded"] is True and second["old_id"] == x
    assert _memory(provider, y)["invalid_at"] is None and _memory(provider, z)["invalid_at"] is None
    assert _memory(provider, x)["invalid_at"] is not None and _memory(provider, x)["superseded_by"] == z
    live = provider._memory_db._conn.execute(
        "SELECT src_id FROM item_links WHERE relation=? AND dst_id=? AND reverted_at IS NULL ORDER BY id",
        (RELATION_SUPERSEDES, x)).fetchall()
    assert [row[0] for row in live] == [y, z]
    injected = provider.prefetch(QUERY_OLD, session_id=SESSION)
    assert MID in injected and NEW in injected and OLD not in injected
    hits = provider._memory_db.search(provider._ns_label, QUERY_OLD, Embedder().embed_query(QUERY_OLD), limit=8)
    assert {item.id: item.replaced_id for item in hits} == {z: x, y: x}


def test_replace_mode_respects_the_limit(provider: NeospikeProvider) -> None:
    x = _remember(provider)
    _update(provider, x, OLD, MID, mode="supersede")
    z = _update(provider, x, OLD, NEW, mode="supersede")["id"]
    hits = provider._memory_db.search(provider._ns_label, QUERY_OLD, Embedder().embed_query(QUERY_OLD), limit=1)
    assert [item.id for item in hits] == [z]  # newest link first, the cap holds


def test_revert_keeps_old_invalid_until_no_replacer_is_left(provider: NeospikeProvider) -> None:
    x = _remember(provider)
    first = _update(provider, x, OLD, MID, mode="supersede")
    second = _update(provider, x, OLD, NEW, mode="supersede")
    store = provider._memory_db
    assert store.undo_supersede(first["link_id"])["reactivated"] is False
    assert _memory(provider, x)["invalid_at"] is not None and _memory(provider, x)["superseded_by"] == second["id"]
    assert store.undo_supersede(second["link_id"])["reactivated"] is True
    assert _memory(provider, x)["invalid_at"] is None and _memory(provider, x)["superseded_by"] is None


# 4. revert ------------------------------------------------------------------------------------
def test_revert_reactivates_old_memory_and_keeps_the_link(provider: NeospikeProvider, monkeypatch, capsys) -> None:
    old_id = _remember(provider)
    result = _update(provider, old_id, OLD, MID, mode="supersede")
    reverted = _run_cli(provider, monkeypatch, capsys, "supersede-revert", "--link", str(result["link_id"]))
    assert reverted["reverted"] is True and reverted["reactivated"] is True
    old = _memory(provider, old_id)
    assert old["invalid_at"] is None and old["superseded_by"] is None
    link = provider._memory_db._conn.execute("SELECT reverted_at FROM item_links WHERE id=?", (result["link_id"],)).fetchone()
    assert link is not None and link["reverted_at"] is not None
    assert _memory(provider, result["id"]) is not None  # the new memory stays
    assert OLD in provider.prefetch(QUERY_OLD, session_id=SESSION)
    again = _run_cli(provider, monkeypatch, capsys, "supersede-revert", "--link", str(result["link_id"]), code=1)
    assert again["reason"] == "not_found"


# 5. correct keeps rev 1; restore keeps growing the history -------------------------------------
def test_correct_records_revision_and_restore_grows_history(provider: NeospikeProvider, monkeypatch, capsys) -> None:
    memory_id = _remember(provider)
    assert _update(provider, memory_id, OLD, MID) == {"updated": True, "id": memory_id}  # default mode=correct
    listed = _run_cli(provider, monkeypatch, capsys, "memory-revisions", "--id", str(memory_id))
    assert listed["text"] == MID and [(item["rev"], item["text"], item["edited_by"]) for item in listed["revisions"]] == [
        (1, OLD, "agent")]

    restored = _run_cli(provider, monkeypatch, capsys, "memory-restore", "--id", str(memory_id), "--rev", "1")
    assert restored == {"restored": True, "id": memory_id, "rev": 1}
    after = _run_cli(provider, monkeypatch, capsys, "memory-revisions", "--id", str(memory_id))
    assert after["text"] == OLD and [item["rev"] for item in after["revisions"]] == [2, 1]
    assert after["revisions"][0]["text"] == MID and after["revisions"][0]["edited_by"] == "human"
    _run_cli(provider, monkeypatch, capsys, "memory-restore", "--id", str(memory_id), "--rev", "2")
    final = _run_cli(provider, monkeypatch, capsys, "memory-revisions", "--id", str(memory_id))
    assert final["text"] == MID and len(final["revisions"]) == 3
    assert MID in provider.prefetch("HERON approval mark", session_id=SESSION)  # FTS and vector follow the text
    assert _memory(provider, memory_id)["invalid_at"] is None


def test_unknown_mode_is_rejected(provider: NeospikeProvider) -> None:
    memory_id = _remember(provider)
    assert "error" in _update(provider, memory_id, OLD, MID, mode="rewrite")
    assert _memory(provider, memory_id)["text"] == OLD


# 8. link-import ---------------------------------------------------------------------------------
def _state(db: Path) -> dict:
    with closing(sqlite3.connect(db)) as connection:
        return {table: db_backup.table_rows(connection, table) for table in db_backup.data_tables(connection)}


# 9. backup restore keeps the new columns row for row ---------------------------------------------
def test_backup_restore_keeps_links_and_invalid_columns(provider: NeospikeProvider, tmp_path: Path) -> None:
    store = provider._memory_db
    directory = tmp_path / "backups"
    a = _remember(provider)
    first = _update(provider, a, OLD, MID, mode="supersede")
    assert db_backup.run_backup(store.path, directory)["kind"] == "full"
    store.undo_supersede(first["link_id"])
    second = _update(provider, a, OLD, NEW, mode="supersede")
    _update(provider, second["id"], NEW, NEW.replace("finally", "really"))
    assert db_backup.run_backup(store.path, directory)["kind"] == "incremental"
    target = tmp_path / "restored.db"
    db_backup.restore(directory, target)
    live, restored = _state(store.path), _state(target)
    for table in ("item_links", "items", "item_revisions"):
        assert restored[table] == live[table], table
    with closing(sqlite3.connect(target)) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(item_links)")}
        assert {"id", "relation", "created_by", "origin", "session_id", "evidence", "created_at",
                "reverted_at"} <= columns
        assert connection.execute("SELECT COUNT(*) FROM item_links WHERE reverted_at IS NOT NULL").fetchone()[0] == 1


# schema v9 migration ----------------------------------------------------------------------------
def test_v8_links_table_becomes_refers_code_links(tmp_path: Path, monkeypatch) -> None:
    """A v8 file (memories without invalid_at/superseded_by, links without id/relation) goes through the v22
    migration: a link keeps its rowid as id and becomes refers/code; a link whose source note is gone is dropped."""
    use_full_comparison(monkeypatch)  # the migration's top-5 check reads the old memories table
    db = tmp_path / "memory.db"
    with closing(sqlite3.connect(db)) as connection:
        connection.executescript(
            "CREATE TABLE memories (id INTEGER PRIMARY KEY, namespace TEXT NOT NULL, content_hash TEXT NOT NULL, "
            "text TEXT NOT NULL, source TEXT NOT NULL, embedding BLOB NOT NULL, created_at REAL NOT NULL);"
            "CREATE TABLE links (src_type TEXT NOT NULL, src_id INTEGER NOT NULL, dst_type TEXT NOT NULL, "
            "dst_id INTEGER, dst_title TEXT);"
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
        vector = Embedder().embed_document(OLD)
        import array
        connection.execute("INSERT INTO memories VALUES (1,'owner','h',?,'s',?,1.0)",
                           (OLD, array.array("f", vector).tobytes()))
        connection.executemany("INSERT INTO links VALUES (?,?,?,?,?)",
                               [("note", 5, "note", None, "Missing"), ("memory", 1, "note", 7, "Seven")])
        connection.executemany("INSERT INTO meta VALUES (?,?)", [("schema_version", "8")])
        connection.commit()
    store = MemoryStore(db, dimensions=Embedder.dimensions)
    try:
        assert store._conn.execute("SELECT value FROM meta_entries WHERE key=?", ("schema_version",)).fetchone()[0] == str(SCHEMA_VERSION) and (tmp_path / "memory.pre-v9.db").exists()
        rows = [tuple(row) for row in store._conn.execute(
            "SELECT id,src_id,dst_id,dst_title,relation,created_by,reverted_at FROM item_links ORDER BY id")]
        assert rows == [(2, 1, None, "Seven", "refers", "code", None)]
        assert _memory_columns(store) >= {"invalid_at", "superseded_by"}
        store._conn.execute("DELETE FROM item_links WHERE id=2")  # tracking trigger is there, keyed by id
        assert store._conn.execute("SELECT row_key FROM tombstones WHERE table_name='item_links'").fetchone()[0] == "2"
    finally:
        store.close()


def _memory_columns(store: MemoryStore) -> set[str]:
    return {row[1] for row in store._conn.execute("PRAGMA table_info(items)")}
