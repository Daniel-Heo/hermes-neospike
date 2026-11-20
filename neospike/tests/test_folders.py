"""Folders — single membership, trash, colours, origin, backup."""
from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

from neospike import backup as db_backup
from neospike import cli
from neospike import folders as folder_family
from neospike.config import NeospikeConfig
from neospike.database import canonical_database_path
from neospike.notes import NoteService
from neospike.store import MemoryStore
from neospike.tests.test_notes import DIMENSIONS, BowEmbedder

NAMESPACE = "owner"
EXTRA_COLORS = 14


def _vector(seed: int) -> list[float]:
    vector = [0.0] * DIMENSIONS
    vector[seed % DIMENSIONS] = 1.0
    vector[(seed * 7 + 3) % DIMENSIONS] = 0.5
    return vector


@pytest.fixture
def env(tmp_path: Path):
    store = MemoryStore(tmp_path / "live" / "memory.db", dimensions=DIMENSIONS)
    notes = NoteService(store, BowEmbedder())
    yield store, notes
    store.close()


def _items(store: MemoryStore) -> list[tuple]:
    return store._conn.execute("SELECT folder_id,item_type,item_id FROM folder_items ORDER BY item_type,item_id").fetchall()


# 4 -----------------------------------------------------------------------------------------
def test_item_moved_to_second_folder_leaves_the_first(env) -> None:
    store, notes = env
    memory = store.insert(NAMESPACE, "The user waters plants at dawn.", _vector(1), source="agent:s1")
    note = notes.save("Garden plan", "# Garden\nTomatoes and basil.", author="human")
    first = folder_family.create_folder(store._conn, "Garden")
    second = folder_family.create_folder(store._conn, "Kitchen")
    folder_family.move_items(store._conn, first["id"], [("memory", memory), ("note", note["id"])])
    moved = folder_family.move_items(store._conn, second["id"], [("memory", memory)])
    assert moved["moved"] == 1 and moved["left"] == [first["id"]]
    assert [tuple(row) for row in _items(store)] == [(second["id"], "memory", memory), (first["id"], "note", note["id"])]
    with pytest.raises(sqlite3.IntegrityError):  # the table itself refuses a second membership
        store._conn.execute("INSERT INTO folder_items(folder_id,item_type,item_id) VALUES (?,?,?)",
                            (first["id"], "memory", memory))
    counts = {folder["id"]: folder["count"] for folder in folder_family.list_folders(store._conn)}
    assert counts == {first["id"]: 1, second["id"]: 1}
    unfiled = folder_family.move_items(store._conn, None, [("memory", memory)])
    assert unfiled["moved"] == 1 and [tuple(row)[1:] for row in _items(store)] == [("note", note["id"])]
    with pytest.raises(folder_family.FolderNotFound):
        folder_family.move_items(store._conn, first["id"], [("note", 999)])


# 5 -----------------------------------------------------------------------------------------
def test_deleted_folder_releases_members_and_keeps_items(env) -> None:
    store, notes = env
    memory = store.insert(NAMESPACE, "The user dislikes slugs.", _vector(2), source="agent:s1")
    note = notes.save("Compost", "# Compost\nTurn the pile weekly.", author="human")
    folder = folder_family.create_folder(store._conn, "Garden")
    folder_family.move_items(store._conn, folder["id"], [("memory", memory), ("note", note["id"])])
    result = folder_family.delete_folder(store._conn, folder["id"])
    assert result == {"id": folder["id"], "deleted": True, "released": 2}
    assert _items(store) == [] and folder_family.list_folders(store._conn) == []
    trashed = folder_family.list_folders(store._conn, include_deleted=True)
    assert trashed[0]["deleted_at"] is not None
    assert store._conn.execute("SELECT COUNT(*) FROM items WHERE type='memory' AND id=?", (memory,)).fetchone()[0] == 1
    assert notes.read(note_id=note["id"])["deleted_at"] is None
    with pytest.raises(folder_family.FolderNotFound):
        folder_family.move_items(store._conn, folder["id"], [("memory", memory)])


# 6 -----------------------------------------------------------------------------------------
def test_member_colours_follow_palette_then_golden_angle_outside_reserved_hues(env) -> None:
    store, _ = env
    created = [folder_family.create_folder(store._conn, f"F{index}")["color"]
               for index in range(len(folder_family.MEMBER_PALETTE) + EXTRA_COLORS)]
    assert tuple(created[:len(folder_family.MEMBER_PALETTE)]) == folder_family.MEMBER_PALETTE
    extra = created[len(folder_family.MEMBER_PALETTE):]
    assert len(set(created)) == len(created)
    for color in extra:
        hue = folder_family.hue_degrees(color)
        assert not (40 <= hue <= 65) and not (90 <= hue <= 160), (color, hue)
    # A trashed folder's colour is free again; the first unused palette colour comes back.
    second = folder_family.list_folders(store._conn)[1]
    folder_family.delete_folder(store._conn, second["id"])
    assert folder_family.create_folder(store._conn, "Again")["color"] == folder_family.MEMBER_PALETTE[1]
    assert folder_family.create_folder(store._conn, "Custom", color="#123abc")["color"] == "#123ABC"
    with pytest.raises(folder_family.FolderError):
        folder_family.create_folder(store._conn, "Bad", color="red")


def test_human_edit_of_auto_folder_records_human_origin(env) -> None:
    store, _ = env
    memory = store.insert(NAMESPACE, "The user grows chili.", _vector(3), source="agent:s2")
    auto = folder_family.create_folder(store._conn, "Topic: chili", actor="auto")
    assert auto["origin"] == "auto"
    folder_family.move_items(store._conn, auto["id"], [("memory", memory)], actor="auto")
    assert folder_family.update_folder(store._conn, auto["id"], name="Chili (auto)", actor="auto")["origin"] == "auto"
    renamed = folder_family.update_folder(store._conn, auto["id"], name="Chili")
    assert renamed["origin"] == "human" and renamed["name"] == "Chili"
    other = folder_family.create_folder(store._conn, "Topic: basil", actor="auto")
    folder_family.move_items(store._conn, other["id"], [("memory", memory)])  # human moves out of an auto folder
    origins = {folder["id"]: folder["origin"] for folder in folder_family.list_folders(store._conn)}
    assert origins == {auto["id"]: "human", other["id"]: "human"}


# 7 -----------------------------------------------------------------------------------------
def _folder_rows(db: Path) -> dict[str, list[tuple]]:
    with closing(sqlite3.connect(db)) as connection:
        return {table: db_backup.table_rows(connection, table) for table in folder_family.FOLDER_TABLES}


def test_backup_restore_keeps_folder_tables_row_for_row(env, tmp_path: Path) -> None:
    store, notes = env
    directory = tmp_path / "backups"
    memory_a = store.insert(NAMESPACE, "The user waters plants at dawn.", _vector(1), source="agent:s1")
    memory_b = store.insert(NAMESPACE, "The user dislikes slugs.", _vector(2), source="agent:s1")
    note = notes.save("Garden plan", "# Garden\nTomatoes and basil.", author="human")
    garden = folder_family.create_folder(store._conn, "Garden")
    folder_family.move_items(store._conn, garden["id"], [("memory", memory_a)])
    assert db_backup.run_backup(store.path, directory)["kind"] == "full"
    kitchen = folder_family.create_folder(store._conn, "Kitchen", actor="auto")
    folder_family.move_items(store._conn, kitchen["id"], [("memory", memory_a), ("note", note["id"])])
    folder_family.update_folder(store._conn, garden["id"], color="#112233")
    first = db_backup.run_backup(store.path, directory)
    assert first["kind"] == "incremental" and first["rows"]["folders"] == 2 and "folder_items" in first["replace_tables"]
    folder_family.delete_folder(store._conn, kitchen["id"])
    folder_family.move_items(store._conn, garden["id"], [("memory", memory_b)])
    assert db_backup.run_backup(store.path, directory)["kind"] == "incremental"
    target = tmp_path / "restored" / "memory.db"
    db_backup.restore(directory, target)
    original, restored = _folder_rows(store.path), _folder_rows(target)
    assert restored == original and len(original["folders"]) == 2 and len(original["folder_items"]) == 1


# CLI ---------------------------------------------------------------------------------------
def _run_cli(home: Path, monkeypatch, capsys, *arguments: str, expect: int = 0) -> dict:
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(home), "--owner", NAMESPACE, *arguments])
    assert cli.main() == expect
    return json.loads(capsys.readouterr().out)


def test_cli_folder_commands(tmp_path: Path, monkeypatch, capsys) -> None:
    home = tmp_path / "home"
    NeospikeConfig().save(home)
    db = canonical_database_path(home / "neospike", "memory")
    store = MemoryStore(db, dimensions=DIMENSIONS)
    memory = store.insert(NAMESPACE, "The user waters plants at dawn.", _vector(1), source="agent:s1")
    note = NoteService(store, BowEmbedder()).save("Garden plan", "# Garden\nTomatoes.", author="human")
    store.close()
    folder = _run_cli(home, monkeypatch, capsys, "folder-create", "--name", "정원")["folder"]
    assert folder["color"] == folder_family.MEMBER_PALETTE[0] and folder["origin"] == "human"
    moved = _run_cli(home, monkeypatch, capsys, "folder-move", "--folder", str(folder["id"]),
                     "--items", f"mem:{memory},note:{note['id']}")
    assert moved["moved"] == 2
    renamed = _run_cli(home, monkeypatch, capsys, "folder-update", "--id", str(folder["id"]), "--name", "Garden",
                       "--color", "#0072b2")["folder"]
    assert (renamed["name"], renamed["color"], renamed["count"]) == ("Garden", "#0072B2", 2)
    assert _run_cli(home, monkeypatch, capsys, "folder-move", "--folder", "none", "--items", f"mem:{memory}")["moved"] == 1
    assert _run_cli(home, monkeypatch, capsys, "folder-list")["folders"][0]["count"] == 1
    bad = _run_cli(home, monkeypatch, capsys, "folder-move", "--folder", "x", "--items", "mem:1", expect=1)
    assert bad["error"] == "invalid"
    assert _run_cli(home, monkeypatch, capsys, "folder-delete", "--id", str(folder["id"]))["released"] == 1
    assert _run_cli(home, monkeypatch, capsys, "folder-list")["folders"] == []


# migration v6 -> v7 --------------------------------------------------------------------------
def test_v6_database_migrates_to_v7_with_folders_and_note_access(tmp_path: Path) -> None:
    from neospike.store import CHANGE_TRIGGER_SPECS, SCHEMA_VERSION
    db = tmp_path / "memory.db"
    store = MemoryStore(db, dimensions=DIMENSIONS)
    NoteService(store, BowEmbedder()).save("Legacy note", "# Legacy\nBody text.", author="human")
    store.close()
    with closing(sqlite3.connect(db)) as connection:  # rewind to the v6 layout
        for name in CHANGE_TRIGGER_SPECS:
            if name.startswith(("folders_track_", "folder_items_track_")):
                connection.execute(f"DROP TRIGGER {name}")
        connection.execute("DROP TABLE folder_items")
        connection.execute("DROP TABLE folders")
        connection.execute("UPDATE meta_entries SET value='6' WHERE key='schema_version'")
        connection.commit()
    store = MemoryStore(db, dimensions=DIMENSIONS)
    try:
        assert SCHEMA_VERSION >= 7 and (tmp_path / "memory.pre-v7.db").exists()  # backup is named after the first step from v6
        assert store._conn.execute("SELECT value FROM meta_entries WHERE key='schema_version'").fetchone()[0] == str(SCHEMA_VERSION)
        assert tuple(store._conn.execute("SELECT accessed_at,access_count FROM items WHERE title='Legacy note'").fetchone()) == (
            None, 0)
        folder = folder_family.create_folder(store._conn, "After migration")
        assert folder["updated_at"] is not None and folder["color"] == folder_family.MEMBER_PALETTE[0]
    finally:
        store.close()
