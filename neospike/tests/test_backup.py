"""Monthly incremental backup (full snapshot + inc zips), restore, schema v5, reject fix."""
from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

from neospike import backup as db_backup
from neospike import cli
from neospike.config import NeospikeConfig
from neospike.database import canonical_database_path
from neospike.notes import NoteService
from neospike.store import CHANGE_TRIGGER_SPECS, SCHEMA_VERSION, MemoryStore
from neospike.tests.test_notes import DIMENSIONS, BowEmbedder
from neospike.tests.support_items import chunk_rows

NAMESPACE = "owner"


def _vector(seed: int) -> list[float]:
    vector = [0.0] * DIMENSIONS
    vector[seed % DIMENSIONS] = 1.0
    vector[(seed * 7 + 3) % DIMENSIONS] = 0.5
    norm = sum(value * value for value in vector) ** 0.5  # stored and query vectors are unit vectors
    return [value / norm for value in vector]


def _state(db: Path) -> dict[str, list[tuple]]:
    """Every data table (FTS excluded), meta without backup bookkeeping, BLOBs included."""
    with closing(sqlite3.connect(db)) as connection:
        return {table: db_backup.table_rows(connection, table) for table in db_backup.data_tables(connection)}


@pytest.fixture
def env(tmp_path: Path):
    store = MemoryStore(tmp_path / "live" / "memory.db", dimensions=DIMENSIONS)
    notes = NoteService(store, BowEmbedder())
    yield store, notes, tmp_path / "backups"
    store.close()


def _backup(store: MemoryStore, directory: Path, **kwargs) -> dict:
    return db_backup.run_backup(store.path, directory, **kwargs)


def _purge(store: MemoryStore, note_id: int) -> None:
    """Permanent delete of a trashed note (the delete triggers record tombstones); no product path purges."""
    with store._lock, store._conn:
        parts = [int(row["id"]) for row in chunk_rows(store._conn, note_id)]
        for sql in ("DELETE FROM item_revisions WHERE item_id=?", "DELETE FROM item_links WHERE src_id=?",
                    "DELETE FROM items WHERE id=?"):
            store._conn.execute(sql, (note_id,))
        store._conn.executemany("DELETE FROM items WHERE id=?", [(part,) for part in parts])


def _two_rounds(store: MemoryStore, notes: NoteService, directory: Path) -> list[dict]:
    """Full snapshot, then two rounds of changes, each followed by an incremental backup."""
    garden = notes.save("Garden plan", "# Garden\nTomatoes and basil near [[Compost]].", author="human")
    compost = notes.save("Compost", "# Compost\nTurn the pile weekly.", author="human")
    memory_a = store.insert(NAMESPACE, "The user waters plants at dawn.", _vector(1), source="agent:s1")
    store.insert(NAMESPACE, "The user dislikes slugs.", _vector(2), source="agent:s1")
    notes.link_memory(memory_a, "See [[Garden plan]]")
    results = [_backup(store, directory)]

    # round 1: note edit + new note, memory add
    notes.save("Garden plan", "# Garden\nTomatoes, basil and chili near [[Compost]].", author="human",
               expected_rev=garden["rev"])
    trash = notes.save("Scratch", "# Scratch\nTemporary thoughts about [[Garden plan]].", author="human")
    store.insert(NAMESPACE, "The user grows chili.", _vector(3), source="agent:s2")
    results.append(_backup(store, directory))

    # round 2: deletes (forget, permanent note delete), rename (links of other notes change)
    assert store.delete(NAMESPACE, memory_a) is True
    deleted = notes.delete(trash["id"], author="human", expected_rev=trash["rev"])
    _purge(store, deleted["id"])
    notes.save("Compost heap", "# Compost\nTurn the pile weekly.", author="human",
               expected_rev=compost["rev"], note_id=compost["id"])
    results.append(_backup(store, directory))
    return results


# 1 + 7: snapshot + 2 increments restore row-for-row; FTS rebuilt ---------------------------
def test_restore_matches_original_row_for_row(env, tmp_path: Path) -> None:
    store, notes, directory = env
    full, first, second = _two_rounds(store, notes, directory)
    assert full["kind"] == "full" and full["name"].startswith("full-") and full["name"].endswith(".db")
    assert first["kind"] == second["kind"] == "incremental"
    assert first["previous"] == full["name"] and second["previous"] == first["name"]
    manifest = db_backup.read_manifest(Path(second["path"]))
    assert manifest["schema_version"] == SCHEMA_VERSION and manifest["previous"] == first["name"]
    assert manifest["rows"]["tombstones"] >= 4  # memory, note, revisions, chunks' note links
    assert "item_links" in manifest["replace_tables"]
    target = tmp_path / "restored" / "memory.db"
    result = db_backup.restore(directory, target)
    assert result["increments"] == [first["name"], second["name"]] and result["integrity"] == "ok"
    original, restored = _state(store.path), _state(target)
    assert set(original) == set(restored)
    for table in original:
        assert restored[table] == original[table], table
    assert any(row[0] == "items" for row in original["tombstones"])
    assert any(row[0] == "item_links" for row in original["tombstones"])
    with closing(sqlite3.connect(target)) as connection:
        triggers = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        assert set(CHANGE_TRIGGER_SPECS) <= triggers
        hits = [row[0] for row in connection.execute(
            "SELECT i.type FROM items_fts JOIN items i ON i.id=items_fts.rowid WHERE items_fts MATCH 'chili'")]
        assert "doc" in hits and hits.count("memory") == 1
    restored_store = MemoryStore(target, dimensions=DIMENSIONS)
    try:
        restored_notes = NoteService(restored_store, BowEmbedder())
        query = "tomatoes basil chili compost"  # BoW query long enough to clear the 0.61 cosine floor
        found = restored_notes.search(query, BowEmbedder().embed_query(query))
        assert found and "chili" in json.dumps(found, ensure_ascii=False)
    finally:
        restored_store.close()


# 2: no change -> no file --------------------------------------------------------------------
def test_unchanged_backup_creates_no_file(env) -> None:
    store, notes, directory = env
    _two_rounds(store, notes, directory)
    before = sorted(path.name for path in directory.iterdir())
    assert _backup(store, directory) == {"created": False}
    store.search(NAMESPACE, "nothing matches this", _vector(99))  # read path without hits
    assert _backup(store, directory) == {"created": False}
    assert sorted(path.name for path in directory.iterdir()) == before


# reembed rewrites every vector without moving updated_at; a later increment would pair the old
# snapshot's vectors with the new meta. A changed embedding model or dimension starts a new chain instead.
@pytest.mark.parametrize("key, value", [("embedding_dimensions", "512"), ("embedding_model", "other/model")])
def test_embedding_change_starts_a_new_chain(env, key: str, value: str) -> None:
    store, notes, directory = env
    store.insert(NAMESPACE, "The user waters plants at dawn.", _vector(1), source="agent:s1")
    assert _backup(store, directory)["kind"] == "full"
    store.insert(NAMESPACE, "The user grows chili.", _vector(3), source="agent:s2")
    assert _backup(store, directory)["kind"] == "incremental"  # unchanged meta: the chain continues
    with store._lock, store._conn:
        assert store._conn.execute("SELECT value FROM meta_entries WHERE key=?", (key,)).fetchone()[0] != value
        store._conn.execute("INSERT OR REPLACE INTO meta_entries(key, value) VALUES (?, ?)", (key, value))
    result = _backup(store, directory)
    assert result["kind"] == "full" and result["reason"] == "embedding_changed"
    assert db_backup.chain(directory)["full"].name == result["name"]
    assert _backup(store, directory) == {"created": False}


# A search hit changes only accessed_at/access_count (updated_at stays); the next
# increment still carries the row, so a restore keeps the read bookkeeping. The note's chunks are not resent.
def test_access_only_change_goes_into_next_increment(env, tmp_path: Path) -> None:
    store, notes, directory = env
    memory_id = store.insert(NAMESPACE, "The user waters plants at dawn.", _vector(1), source="agent:s1")
    note = notes.save("Garden plan", "# Garden\nTomatoes and basil.", author="human")
    assert _backup(store, directory)["kind"] == "full"
    before = {row_id: store._conn.execute("SELECT updated_at FROM items WHERE id=?", (row_id,)).fetchone()[0]
              for row_id in (memory_id, note["id"])}
    store.search(NAMESPACE, "The user waters plants at dawn.", _vector(1))
    assert store.log_note_access([note["id"]]) == 1
    for row_id in (memory_id, note["id"]):
        updated, count = store._conn.execute("SELECT updated_at,access_count FROM items WHERE id=?",
                                             (row_id,)).fetchone()
        assert (updated, count) == (before[row_id], 1), row_id
    increment = _backup(store, directory)
    assert increment.get("kind") == "incremental"
    manifest = db_backup.read_manifest(Path(increment["path"]))
    assert manifest["rows"]["items"] == 2  # the memory and the note's head row; its parts are not resent
    target = tmp_path / "restored" / "memory.db"
    db_backup.restore(directory, target)
    original, restored = _state(store.path), _state(target)
    for table in ("items", "item_links"):
        assert restored[table] == original[table], table


# 3: broken chain -> refused -----------------------------------------------------------------
def test_broken_chain_is_refused(env, tmp_path: Path) -> None:
    store, notes, directory = env
    _, first, _ = _two_rounds(store, notes, directory)
    Path(first["path"]).unlink()
    target = tmp_path / "restored.db"
    with pytest.raises(db_backup.BackupError, match="broken"):
        db_backup.restore(directory, target)
    assert not target.exists()
    assert db_backup.list_backups(directory)["chain"] == "broken"


# 4: existing target -> refused, untouched ---------------------------------------------------
def test_restore_refuses_existing_file(env, tmp_path: Path) -> None:
    store, notes, directory = env
    _two_rounds(store, notes, directory)
    target = tmp_path / "keep.db"
    target.write_bytes(b"precious")
    with pytest.raises(db_backup.BackupError, match="existing"):
        db_backup.restore(directory, target)
    assert target.read_bytes() == b"precious"
    live = store.path.read_bytes()
    with pytest.raises(db_backup.BackupError):
        db_backup.restore(directory, store.path)
    assert store.path.read_bytes() == live


# 5: v4 -> v5 fills updated_at ---------------------------------------------------------------
def test_v4_database_migrates_with_updated_at(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    store = MemoryStore(db, dimensions=DIMENSIONS)
    store.insert(NAMESPACE, "Legacy memory row.", _vector(5), source="agent:s1")
    store.close()
    with closing(sqlite3.connect(db)) as connection:  # rewind to the v4 layout
        for name in CHANGE_TRIGGER_SPECS:
            connection.execute(f"DROP TRIGGER {name}")
        connection.execute("DROP TABLE tombstones")
        connection.execute("ALTER TABLE items DROP COLUMN updated_at")
        connection.execute("UPDATE meta_entries SET value='4' WHERE key='schema_version'")
        connection.commit()
    store = MemoryStore(db, dimensions=DIMENSIONS)
    try:
        rows = store._conn.execute("SELECT created_at,updated_at FROM items WHERE type='memory'").fetchall()
        assert rows and all(row["updated_at"] == row["created_at"] for row in rows)
        assert store._conn.execute("SELECT value FROM meta_entries WHERE key='schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
        assert (tmp_path / "memory.pre-v5.db").exists()
        store.archive(NAMESPACE, 1)
        row = store._conn.execute("SELECT created_at,updated_at FROM items WHERE type='memory' AND id=1").fetchone()
        assert row["updated_at"] > row["created_at"]  # write paths keep it current
    finally:
        store.close()


# CLI: backup / --list / restore ------------------------------------------------------------
def _run_cli(home: Path, monkeypatch, capsys, *arguments: str) -> dict:
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(home), "--owner", NAMESPACE, *arguments])
    assert cli.main() == 0
    return json.loads(capsys.readouterr().out)


def test_cli_backup_list_restore(tmp_path: Path, monkeypatch, capsys) -> None:
    home = tmp_path / "home"
    NeospikeConfig().save(home)
    db = canonical_database_path(home / "neospike", "memory")
    store = MemoryStore(db, dimensions=DIMENSIONS)
    store.insert(NAMESPACE, "The user waters plants at dawn.", _vector(1), source="agent:s1")
    store.close()
    full = _run_cli(home, monkeypatch, capsys, "backup")
    assert full["kind"] == "full" and Path(full["path"]).parent == home / "neospike" / "backups"
    assert _run_cli(home, monkeypatch, capsys, "backup") == {"created": False}
    store = MemoryStore(db, dimensions=DIMENSIONS)
    store.insert(NAMESPACE, "The user grows chili.", _vector(3), source="agent:s2")
    store.close()
    inc = _run_cli(home, monkeypatch, capsys, "backup")
    assert inc["kind"] == "incremental" and inc["rows"] == {"items": 1}
    listed = _run_cli(home, monkeypatch, capsys, "backup", "--list")
    assert listed["chain"] == "ok" and listed["chain_tail"] == inc["name"] == listed["last_recorded"]
    assert [entry["name"] for entry in listed["backups"]] == [full["name"], inc["name"]]
    target = tmp_path / "restored.db"
    restored = _run_cli(home, monkeypatch, capsys, "restore", "--to", str(target))
    assert restored["increments"] == [inc["name"]] and _state(target) == _state(db)
    assert _run_cli(home, monkeypatch, capsys, "backup", "--full")["kind"] == "full"
