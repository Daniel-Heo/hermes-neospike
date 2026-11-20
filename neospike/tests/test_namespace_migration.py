"""Live namespace in meta, namespace-migrate, the removed rescope commands, item_revisions backup."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path


import pytest

from neospike import backup as db_backup
from neospike import cli, eval_recall
from neospike.config import EMBEDDING_DIMENSIONS, NeospikeConfig
from neospike.mcp_server import NeospikeTools
from neospike.provider import NeospikeProvider
from neospike.store import DEFAULT_NAMESPACE, META_LIVE_NAMESPACE, MemoryStore

LIVE = "default:sid:S-1-5-21-1000"
LOCAL_TEXTS = ("python alpha decision", "python beta decision", "python shared decision")
LIVE_TEXTS = ("python delta decision", "python shared decision")  # second one: same hash as a local row
SHARED_SOURCE = "agent:s1"


class FakeEmbedder:
    dimensions = EMBEDDING_DIMENSIONS  # the CLI opens the store with it

    def embed_query(self, text: str, **_kwargs) -> list[float]:
        return self.embed_document(text)

    def embed_document(self, text: str, **_kwargs) -> list[float]:
        value = text.lower()
        axes = [float("python" in value), float("rust" in value), 1.0 if "python" not in value else 0.0]
        return axes + [0.0] * (self.dimensions - len(axes))

    def take_warnings(self) -> list[str]:
        return []


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    NeospikeConfig().save(home)
    return home


def _db(home: Path) -> Path:
    return home / "neospike" / "memory.db"


def _provider(home: Path, *, context: str = "primary") -> NeospikeProvider:
    identity, _, user_id = LIVE.partition(":")
    provider = NeospikeProvider(embedder=FakeEmbedder())
    provider.initialize("session-live", hermes_home=str(home), agent_identity=identity, user_id=user_id,
                        agent_context=context, cwd=str(home))
    return provider


def _run_cli(home: Path, monkeypatch, capsys, *arguments: str) -> dict:
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(home), *arguments])
    assert cli.main() == 0
    return json.loads(capsys.readouterr().out)


def _state_hash(db: Path) -> str:
    with closing(sqlite3.connect(db)) as connection:
        state = {table: db_backup.table_rows(connection, table) for table in db_backup.data_tables(connection)}
    return hashlib.sha256(repr(state).encode()).hexdigest()


# 1. live namespace defaults -------------------------------------------------------------------
def test_tools_default_to_live_namespace(tmp_path: Path, monkeypatch, capsys) -> None:
    home = _home(tmp_path)
    assert NeospikeTools(home).namespace == DEFAULT_NAMESPACE  # nothing recorded yet
    _provider(home).shutdown()
    store = MemoryStore(_db(home), dimensions=FakeEmbedder.dimensions)
    try:
        assert store._conn.execute("SELECT value FROM meta_entries WHERE key=?", (META_LIVE_NAMESPACE,)).fetchone()[0] == LIVE
        assert store.remember_live_namespace(LIVE) is False  # unchanged value is not rewritten
        memory = store.insert(LIVE, "The user prefers clay pots.", FakeEmbedder().embed_document("clay pots"),
                              source="t")
    finally:
        store.close()
    # A non-primary provider (eval, subagent) never overwrites the live value.
    eval_provider = NeospikeProvider(embedder=FakeEmbedder())
    eval_provider.initialize("eval", hermes_home=str(home), agent_identity="other", user_id="x",
                             agent_context="eval", cwd=str(home))
    eval_provider.shutdown()

    # The CLI default namespace is the live one (archive matches the memory's namespace).
    assert _run_cli(home, monkeypatch, capsys, "archive", "--id", str(memory))["archived"] is True
    assert NeospikeTools(home).namespace == LIVE
    assert NeospikeTools(home, "x:y").namespace == "x:y"
    with eval_recall.Evaluator(home, embedder=FakeEmbedder()) as runner:
        assert runner.provider._ns_label == LIVE


# 2-3. namespace-migrate -----------------------------------------------------------------------
# 4. rescope commands (removed) ---------------------------------------------------------------------------
def _seed_scopes(store: MemoryStore) -> dict[str, int]:
    embedder = FakeEmbedder()
    ids = {}
    for name, scope, durability in (("stable", "session", "stable"), ("ongoing", "session", "ongoing"),
                                    ("transient", "session", "transient"), ("global", "global", "stable")):
        text = f"python {name} memory"
        ids[name] = store.insert(LIVE, text, embedder.embed_document(text), source="historical-session:s9",
                                 kind="memory_item", metadata={"scope": scope, "durability": durability,
                                                               "session_id": "s9", "subject": name})
    return ids


def _scope(store: MemoryStore, memory_id: int) -> str:
    row = store._conn.execute("SELECT scope FROM items WHERE type='memory' AND id=?", (memory_id,))
    return row.fetchone()[0]


@pytest.mark.parametrize("command", ["memory-rescope", "rescope-session"])
def test_rescope_commands_are_gone(tmp_path: Path, monkeypatch, capsys, command: str) -> None:
    """Memory-rescope (old name rescope-session) widened every session memory to global; removed."""
    home = _home(tmp_path)
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(home), command])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2 and "invalid choice" in capsys.readouterr().err
    assert command not in cli.OLD_COMMANDS and not hasattr(MemoryStore, "reassign_session_scope")


# 5. backup / restore carries item_revisions -----------------------------------------------------
def test_backup_restore_item_revisions(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "live" / "memory.db", dimensions=FakeEmbedder.dimensions)
    directory = tmp_path / "backups"
    try:
        ids = _seed_scopes(store)
        assert db_backup.run_backup(store.path, directory)["kind"] == "full"
        with store._lock, store._conn:  # two in-place edits of the memories (revisions)
            for name in ("stable", "ongoing"):
                store._push_revision(ids[name], f"python {name} memory", "{}", edited_by="human", session_id="s9",
                                     now=time.time())
        increment = db_backup.run_backup(store.path, directory)
        assert increment["rows"]["item_revisions"] == 2
        with store._conn:  # a deleted revision travels as a tombstone
            store._conn.execute("DELETE FROM item_revisions WHERE item_id=?", (ids["ongoing"],))
        assert db_backup.run_backup(store.path, directory)["kind"] == "incremental"
        target = tmp_path / "restored.db"
        db_backup.restore(directory, target)
        query = "SELECT * FROM item_revisions ORDER BY item_id, rev"
        with closing(sqlite3.connect(target)) as restored:
            assert restored.execute(query).fetchall() == [tuple(row) for row in store._conn.execute(query)]
            assert len(restored.execute(query).fetchall()) == 1
    finally:
        store.close()


# schema v7 -> v8 ------------------------------------------------------------------------------
def test_v7_database_migrates_to_v8_with_item_revisions(tmp_path: Path) -> None:
    from neospike.store import SCHEMA_VERSION
    db = tmp_path / "memory.db"
    MemoryStore(db, dimensions=FakeEmbedder.dimensions).close()
    with closing(sqlite3.connect(db)) as connection:  # rewind to the v7 layout
        connection.execute("DROP TRIGGER item_revisions_track_ad")
        connection.execute("DROP TABLE item_revisions")
        connection.execute("UPDATE meta_entries SET value='7' WHERE key='schema_version'")
        connection.commit()
    store = MemoryStore(db, dimensions=FakeEmbedder.dimensions)
    try:
        assert SCHEMA_VERSION >= 8 and (tmp_path / "memory.pre-v8.db").exists()
        assert store._conn.execute("SELECT value FROM meta_entries WHERE key=?", ("schema_version",)).fetchone()[0] == str(SCHEMA_VERSION)
        assert store._conn.execute("SELECT COUNT(*) FROM item_revisions").fetchone()[0] == 0
    finally:
        store.close()
