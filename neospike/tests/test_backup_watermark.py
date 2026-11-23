"""Overlapping increments: rows stamped >= since up to the backup moment N; the next since is B = (N - 1 s)
rounded down to the minute; the no-file rule ignores the overlap [B, N)."""
from __future__ import annotations

import json
import sqlite3
import time
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pytest

from neospike import backup as db_backup
from neospike import conversations as conv
from neospike import folders as folder_family
from neospike.notes import NoteService
from neospike.store import MemoryStore
from neospike.tests.test_notes import DIMENSIONS, BowEmbedder

# Minute-aligned fake epoch far after any real stamp, so rows the store stamps itself stay below every since.
BASE = datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp()
SKEW_SECONDS = 0.002  # a write after the lock can stamp ~2 ms before the backup's own clock read
INSIDE_LAG_SECONDS = 0.5  # < BOUNDARY_LAG_SECONDS before the last backup moment
MESSAGES = "conversation_messages"
NAMESPACE = "owner"
NOTE_BODY = "# Tides\nThe moon pulls the sea.\n\n# Springs\nSun and moon line up.\n\n# Neaps\nThey pull apart."
NOTE_EDIT = "# Tides\nThe moon and sun pull the sea.\n\n# Springs\nNew and full moon.\n\n# Neaps\nQuarter moons."
DANGLING_LINKS = ("SELECT COUNT(*) FROM item_links WHERE src_id NOT IN (SELECT id FROM items) "
                  "OR (dst_id IS NOT NULL AND dst_id NOT IN (SELECT id FROM items))")
DANGLING_FOLDER_ITEMS = ("SELECT COUNT(*) FROM folder_items WHERE item_id NOT IN (SELECT id FROM items) "
                         "OR folder_id NOT IN (SELECT id FROM folders)")


class _Clock:
    """Fake backup clock: returns self.at."""

    def __init__(self) -> None:
        self.at = BASE

    def __call__(self) -> float:
        return self.at


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    store = MemoryStore(tmp_path / "live" / "memory.db", dimensions=DIMENSIONS)
    clock = _Clock()
    monkeypatch.setattr(db_backup, "_now", clock)
    yield store, clock, tmp_path / "backups"
    store.close()


def _backup(store: MemoryStore, directory: Path, clock: _Clock, at: float) -> dict:
    clock.at = at
    return db_backup.run_backup(store.path, directory)


def _message_ids(store: MemoryStore) -> list[int]:
    return [int(row[0]) for row in store._conn.execute(f"SELECT id FROM {MESSAGES} ORDER BY id")]


def _set_message(store: MemoryStore, message_id: int, stamp: float, content: str | None = None) -> None:
    """Edit a message with an explicit updated_at (the tracking trigger leaves an explicit stamp alone)."""
    with store._lock, store._conn:
        if content is None:
            store._conn.execute(f"UPDATE {MESSAGES} SET updated_at=? WHERE id=?", (stamp, message_id))
        else:
            store._conn.execute(f"UPDATE {MESSAGES} SET content=?, updated_at=? WHERE id=?",
                                (content, stamp, message_id))


def _delete_message(store: MemoryStore, message_id: int, stamp: float) -> dict:
    """Delete a message (the delete trigger writes its tombstone), move the tombstone to *stamp*; returns the row."""
    with store._lock, store._conn:
        cursor = store._conn.execute(f"SELECT * FROM {MESSAGES} WHERE id=?", (message_id,))
        row = dict(zip([column[0] for column in cursor.description], cursor.fetchone()))
        store._conn.execute(f"DELETE FROM {MESSAGES} WHERE id=?", (message_id,))
        store._conn.execute("UPDATE tombstones SET deleted_at=? WHERE table_name=? AND row_key=?",
                            (stamp, MESSAGES, str(message_id)))
    return row


def _recreate_message(store: MemoryStore, row: dict, stamp: float, content: str) -> None:
    """Insert a message again under its old id with new content, stamped *stamp*."""
    values = {**row, "content": content}
    names = list(values)
    with store._lock, store._conn:
        store._conn.execute(f"INSERT INTO {MESSAGES}({','.join(names)}) VALUES ({','.join('?' * len(names))})",
                            [values[name] for name in names])
        store._conn.execute(f"UPDATE {MESSAGES} SET updated_at=? WHERE id=?", (stamp, row["id"]))


def _rows(path: str, table: str) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        return [json.loads(line) for line in archive.read(table + db_backup.JSONL_SUFFIX).decode("utf-8").splitlines()
                if line]


def _ids(path: str) -> list[int]:
    return sorted(int(row["id"]) for row in _rows(path, MESSAGES))


def _manifest(path: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(db_backup.MANIFEST_NAME))


def _meta_value(store: MemoryStore, key: str) -> str | None:
    row = store._conn.execute("SELECT value FROM meta_entries WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _backup_meta(store: MemoryStore) -> list[tuple]:
    return [tuple(row) for row in store._conn.execute(
        f"SELECT key,value FROM meta_entries WHERE key LIKE '{db_backup.BACKUP_META_PREFIX}%' ORDER BY key")]


def _files(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def _state(db: Path) -> dict[str, list[tuple]]:
    with closing(sqlite3.connect(db)) as connection:
        return {table: db_backup.table_rows(connection, table) for table in db_backup.data_tables(connection)}


def _restore(directory: Path, target: Path) -> dict[str, list[tuple]]:
    db_backup.restore(directory, target)
    return _state(target)


def _three_messages(store: MemoryStore) -> list[int]:
    conv.ConversationStore(store).append("s1", [("user", "Where do tides come from?"),
                                                ("assistant", "The moon pulls the sea."),
                                                ("user", "Why are spring tides larger?")])
    return _message_ids(store)


def _vector(seed: int) -> list[float]:
    vector = [0.0] * DIMENSIONS
    vector[seed % DIMENSIONS] = 1.0
    return vector


def test_boundary_is_one_second_back_rounded_down_to_the_minute() -> None:
    minute = datetime(2026, 11, 23, 2, 35, tzinfo=timezone.utc).timestamp()
    assert db_backup.boundary_at(minute + 23.4) == minute  # 02:35:23.4 -> 02:35:00
    assert db_backup.boundary_at(minute + 0.5) == minute - 60.0  # 02:35:00.5 -> 02:34:00


def test_full_snapshot_stores_boundary_as_since_and_the_moment(env) -> None:
    store, clock, directory = env
    _three_messages(store)
    assert _backup(store, directory, clock, BASE + 30.0)["kind"] == "full"
    assert float(_meta_value(store, db_backup.META_LAST_AT)) == BASE
    assert float(_meta_value(store, db_backup.META_LAST_MOMENT)) == BASE + 30.0


def test_write_stamped_before_the_backup_clock_reaches_the_next_increment(env) -> None:
    store, clock, directory = env
    late = _three_messages(store)[-1]
    taken = BASE + 30.0
    assert _backup(store, directory, clock, taken)["kind"] == "full"
    _set_message(store, late, taken - SKEW_SECONDS, "Written after the backup, stamped slightly before it.")
    # Former rule: watermark = the Python clock at the backup, next increment takes updated_at > watermark.
    old_rule = store._conn.execute(f"SELECT COUNT(*) FROM {MESSAGES} WHERE updated_at>?", (taken,)).fetchone()[0]
    assert old_rule == 0
    increment = _backup(store, directory, clock, BASE + 90.0)
    assert increment["kind"] == "incremental" and _ids(increment["path"]) == [late]


def test_overlap_rows_go_into_both_increments_and_restore_row_for_row(env, tmp_path: Path) -> None:
    store, clock, directory = env
    kept, deleted_early, deleted_late = _three_messages(store)
    _set_message(store, kept, BASE + 10.0, "Edited inside [B, N) of the full snapshot.")
    _delete_message(store, deleted_early, BASE + 15.0)
    assert _backup(store, directory, clock, BASE + 30.0)["kind"] == "full"  # B = BASE, N = BASE + 30
    _set_message(store, deleted_late, BASE + 40.0, "Edited before the first increment.")
    _set_message(store, kept, BASE + 75.0, "Edited inside [B, N) of the first increment.")
    first = _backup(store, directory, clock, BASE + 90.0)  # since BASE; B = BASE + 60, N = BASE + 90
    assert first["kind"] == "incremental" and _ids(first["path"]) == [kept, deleted_late]
    assert [row["row_key"] for row in _rows(first["path"], "tombstones")] == [str(deleted_early)]
    assert _manifest(first["path"])["since"] == BASE and _manifest(first["path"])["boundary"] == BASE + 60.0
    _delete_message(store, deleted_late, BASE + 110.0)
    second = _backup(store, directory, clock, BASE + 150.0)  # since BASE + 60
    assert second["kind"] == "incremental" and _ids(second["path"]) == [kept]  # kept: again from the overlap
    assert [row["row_key"] for row in _rows(second["path"], "tombstones")] == [str(deleted_late)]
    live = _state(store.path)
    restored = _restore(directory, tmp_path / "restored.db")
    assert restored[MESSAGES] == live[MESSAGES] and len(live[MESSAGES]) == 1
    assert restored["tombstones"] == live["tombstones"] and len(live["tombstones"]) == 2
    assert restored == live


def test_delete_and_recreate_across_the_overlap_restores_row_for_row(env, tmp_path: Path) -> None:
    store, clock, directory = env
    target, other = _three_messages(store)[:2]
    assert _backup(store, directory, clock, BASE + 30.0)["kind"] == "full"
    row = _delete_message(store, target, BASE + 70.0)  # inside [B, N) of the first increment
    first = _backup(store, directory, clock, BASE + 90.0)  # B = BASE + 60
    assert first["kind"] == "incremental" and _ids(first["path"]) == []
    _recreate_message(store, row, BASE + 100.0, "Same key, written again.")
    _set_message(store, other, BASE + 120.0, "Edited too.")
    second = _backup(store, directory, clock, BASE + 150.0)  # since BASE + 60: the tombstone again, then the row
    assert second["kind"] == "incremental" and _ids(second["path"]) == [target, other]
    assert [row["row_key"] for row in _rows(second["path"], "tombstones")] == [str(target)]
    assert _restore(directory, tmp_path / "restored.db") == _state(store.path)


def test_increment_right_after_writes_restores_links_and_folder_items_whole(tmp_path: Path, monkeypatch) -> None:
    store = MemoryStore(tmp_path / "live" / "memory.db", dimensions=DIMENSIONS)
    clock = _Clock()
    monkeypatch.setattr(db_backup, "_now", clock)
    directory = tmp_path / "backups"
    try:
        notes = NoteService(store, BowEmbedder())
        note = notes.save("Tides", NOTE_BODY, author="human", origin="human")
        folder = folder_family.create_folder(store._conn, "Sea")
        taken = time.time()
        assert _backup(store, directory, clock, taken)["kind"] == "full"
        since = float(_meta_value(store, db_backup.META_LAST_AT))
        links_before = _state(store.path)["item_links"]
        notes.save("Tides", NOTE_EDIT, author="human", origin="human", expected_rev=note["rev"], note_id=note["id"])
        memory = store.insert(NAMESPACE, "The user sails at high tide.", _vector(1), source="agent:s1")
        folder_family.move_items(store._conn, folder["id"], [("memory", memory), ("note", note["id"])])
        changed = [int(row[0]) for row in store._conn.execute("SELECT id FROM items WHERE updated_at>=?", (taken,))]
        earliest = store._conn.execute("SELECT MIN(updated_at) FROM items WHERE updated_at>=?", (taken,)).fetchone()[0]
        moment = earliest + db_backup.BOUNDARY_LAG_SECONDS  # B = earliest rounded down: every change is in [B, N)
        boundary = db_backup.boundary_at(moment)
        assert len(changed) >= 3 and boundary <= earliest and _state(store.path)["item_links"] != links_before
        # Former rule: [since, B) holds none of the changed rows, while item_links/folder_items are whole.
        old_rule = store._conn.execute("SELECT COUNT(*) FROM items WHERE updated_at>=? AND updated_at<? AND updated_at>=?",
                                       (since, boundary, taken)).fetchone()[0]
        assert old_rule == 0
        increment = _backup(store, directory, clock, moment)
        assert increment["kind"] == "incremental"
        assert {"item_links", "folder_items"} <= set(increment["replace_tables"])
        live = _state(store.path)
        target = tmp_path / "restored.db"
        restored = _restore(directory, target)
        with closing(sqlite3.connect(target)) as connection:
            assert connection.execute(DANGLING_LINKS).fetchone()[0] == 0
            assert connection.execute(DANGLING_FOLDER_ITEMS).fetchone()[0] == 0
        assert restored == live and len(live["folder_items"]) == 2
    finally:
        store.close()


def test_no_writes_and_overlap_rows_alone_write_nothing(env) -> None:
    store, clock, directory = env
    message = _three_messages(store)[0]
    _set_message(store, message, BASE + 10.0, "Inside [B, N) of the full snapshot.")
    assert _backup(store, directory, clock, BASE + 30.0)["kind"] == "full"
    files, meta = _files(directory), _backup_meta(store)
    assert _backup(store, directory, clock, BASE + 31.0) == {"created": False}
    assert _backup(store, directory, clock, BASE + 90.0) == {"created": False}  # BASE + 10 >= since, but overlap
    assert _files(directory) == files and _backup_meta(store) == meta


def test_row_stamped_within_the_lag_before_the_last_moment_is_taken_once(env) -> None:
    store, clock, directory = env
    message = _three_messages(store)[0]
    assert _backup(store, directory, clock, BASE + 30.0)["kind"] == "full"
    _set_message(store, message, BASE + 30.0 - INSIDE_LAG_SECONDS, "Committed after the backup, stamped early.")
    increment = _backup(store, directory, clock, BASE + 90.0)
    assert increment["kind"] == "incremental" and _ids(increment["path"]) == [message]
    files, meta = _files(directory), _backup_meta(store)
    assert _backup(store, directory, clock, BASE + 150.0) == {"created": False}
    assert _files(directory) == files and _backup_meta(store) == meta


def test_meta_without_the_moment_key_continues_the_chain(env, tmp_path: Path) -> None:
    store, clock, directory = env
    message, other = _three_messages(store)[:2]
    assert _backup(store, directory, clock, BASE + 30.0)["kind"] == "full"
    with store._lock, store._conn:  # a chain written before the moment key: last time = the old watermark
        store._conn.execute("DELETE FROM meta_entries WHERE key=?", (db_backup.META_LAST_MOMENT,))
        store._conn.execute("UPDATE meta_entries SET value=? WHERE key=?", (repr(BASE + 30.0), db_backup.META_LAST_AT))
    _set_message(store, other, BASE + 29.0, "Before the old watermark: no change on its own.")
    assert _backup(store, directory, clock, BASE + 90.0) == {"created": False}
    _set_message(store, message, BASE + 40.0, "After the old watermark.")
    increment = _backup(store, directory, clock, BASE + 90.0)
    assert increment["kind"] == "incremental" and _ids(increment["path"]) == [message]
    assert _manifest(increment["path"])["since"] == BASE + 30.0
    assert float(_meta_value(store, db_backup.META_LAST_AT)) == BASE + 60.0
    assert float(_meta_value(store, db_backup.META_LAST_MOMENT)) == BASE + 90.0
