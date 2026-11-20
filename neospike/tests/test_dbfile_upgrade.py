"""memory.sqlite -> memory.db upgrade path, crash safety, and embedding-metadata guards."""
from __future__ import annotations

import sqlite3
from contextlib import closing
import subprocess
import sys
from pathlib import Path

import pytest
from neospike.tests.support_provider import (
    OWNER_NS,
    boot_provider,
    integrity_of,
    make_legacy_db,
    seed_wal_db_and_exit,
    texts_found,
)

import neospike.database as database_module
from neospike.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from neospike.database import canonical_database_path
from neospike.store import MemoryStore

CANONICAL = "memory.db"
LEGACY = "memory.sqlite"
RACERS = 3
CHILD_TIMEOUT_S = 30
_RESOLVE_CODE = (
    "import sys\n"
    "from pathlib import Path\n"
    "from neospike.database import canonical_database_path\n"
    "print(canonical_database_path(Path(sys.argv[1]), 'memory'))\n"
)


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "neospike"
    path.mkdir()
    return path


def test_provider_startup_adopts_wal_legacy_file(tmp_path: Path) -> None:
    folder = tmp_path / "neospike"
    folder.mkdir()
    seed_wal_db_and_exit(folder / LEGACY, "curated only")
    assert (folder / f"{LEGACY}-wal").is_file()

    provider = boot_provider(tmp_path, "s1")

    assert provider._memory_db is not None
    assert provider._memory_db.path == folder / CANONICAL
    assert provider._memory_db.count(OWNER_NS) == 1
    assert integrity_of(folder / CANONICAL) == "ok"
    leftovers = [p.name for p in folder.iterdir() if (p.name.startswith(LEGACY) or (p.name.startswith(".memory.") and ".db" in p.name))]
    assert leftovers == []


def test_garbage_canonical_file_is_replaced_by_legacy_copy(data_dir: Path) -> None:
    make_legacy_db(data_dir, ("kept row",))
    (data_dir / CANONICAL).write_bytes(b"\x00garbage, not sqlite")

    resolved = canonical_database_path(data_dir, "memory")

    assert resolved == data_dir / CANONICAL
    assert integrity_of(resolved) == "ok"
    assert texts_found(resolved, "kept") == ["kept row"]
    assert not (data_dir / LEGACY).exists()


def test_legacy_content_wins_over_stale_canonical_sidecars(data_dir: Path) -> None:
    canonical = data_dir / CANONICAL
    seed_wal_db_and_exit(canonical, "stale canonical row")
    make_legacy_db(data_dir, ("legacy is the truth",))
    wal, shm, journal = (Path(f"{canonical}{suffix}") for suffix in ("-wal", "-shm", "-journal"))
    assert wal.exists()
    shm.write_bytes(b"junk shm")
    journal.write_bytes(b"junk journal")

    canonical_database_path(data_dir, "memory")

    assert texts_found(canonical, "row") == ["legacy is the truth"]
    assert [p.exists() for p in (wal, shm, journal)] == [False, False, False]


def test_several_processes_migrating_at_once_lose_nothing(data_dir: Path) -> None:
    make_legacy_db(data_dir, ("survives the race",))
    children = [
        subprocess.Popen(
            [sys.executable, "-c", _RESOLVE_CODE, str(data_dir)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(RACERS)
    ]
    outputs = [child.communicate(timeout=CHILD_TIMEOUT_S) for child in children]

    assert [child.returncode for child in children] == [0] * RACERS, outputs
    assert texts_found(data_dir / CANONICAL, "race") == ["survives the race"]
    assert not (data_dir / LEGACY).exists()


def test_failed_swap_keeps_legacy_and_leaves_no_temp(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_legacy_db(data_dir, ("must stay",))

    def refuse(src, dst) -> None:
        raise OSError("swap refused")

    monkeypatch.setattr(database_module.os, "replace", refuse)
    with pytest.raises(OSError, match="swap refused"):
        canonical_database_path(data_dir, "memory")

    assert (data_dir / LEGACY).is_file()
    assert not (data_dir / CANONICAL).exists()
    assert [p.name for p in data_dir.glob(".memory.*.db*")] == []


def test_integrity_failure_after_install_keeps_legacy(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_legacy_db(data_dir, ("must stay too",))
    real_check = database_module._assert_integrity_ok
    calls: list[Path] = []

    def check_then_break_second(connection: sqlite3.Connection, path: Path) -> None:
        calls.append(path)
        real_check(connection, path)
        if len(calls) == 2:
            raise RuntimeError("post-install check broke")

    monkeypatch.setattr(database_module, "_assert_integrity_ok", check_then_break_second)
    with pytest.raises(RuntimeError, match="post-install check broke"):
        canonical_database_path(data_dir, "memory")

    assert len(calls) == 2
    assert (data_dir / LEGACY).is_file()
    assert (data_dir / CANONICAL).is_file()


def test_second_run_finishes_after_legacy_removal_failed(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = make_legacy_db(data_dir, ("legacy still counts",))
    real_remove = database_module._remove_database_files
    state = {"armed": True}

    def flaky_remove(path: Path) -> None:
        if state["armed"] and path == legacy:
            state["armed"] = False
            raise OSError("cannot delete legacy")
        real_remove(path)

    monkeypatch.setattr(database_module, "_remove_database_files", flaky_remove)
    with pytest.raises(OSError, match="cannot delete legacy"):
        canonical_database_path(data_dir, "memory")
    assert legacy.is_file()
    assert (data_dir / CANONICAL).is_file()

    canonical_database_path(data_dir, "memory")  # the patched function no longer fails

    assert texts_found(data_dir / CANONICAL, "legacy") == ["legacy still counts"]
    assert not legacy.exists()


def test_migration_syncs_directory_around_each_durable_step(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    legacy = make_legacy_db(data_dir)
    timeline: list[str] = []
    real_replace = database_module.os.replace
    real_check = database_module._assert_integrity_ok
    real_remove = database_module._remove_database_files

    def note_sync(path: Path) -> None:
        timeline.append("sync")

    def note_replace(src, dst) -> None:
        timeline.append("install")
        real_replace(src, dst)

    def note_check(connection: sqlite3.Connection, path: Path) -> None:
        timeline.append("verify" if path == data_dir / CANONICAL else "pre-verify")
        real_check(connection, path)

    def note_remove(path: Path) -> None:
        if path == legacy:
            timeline.append("drop-legacy")
        real_remove(path)

    monkeypatch.setattr(database_module, "_sync_directory", note_sync)
    monkeypatch.setattr(database_module.os, "replace", note_replace)
    monkeypatch.setattr(database_module, "_assert_integrity_ok", note_check)
    monkeypatch.setattr(database_module, "_remove_database_files", note_remove)

    canonical_database_path(data_dir, "memory")

    install, verify, drop = (timeline.index(step) for step in ("install", "verify", "drop-legacy"))
    assert install < verify < drop
    assert "sync" in timeline[:install]
    assert "sync" in timeline[install + 1 : verify]
    assert "sync" in timeline[drop + 1 :]


def test_old_schema_file_is_copied_aside_before_upgrade(tmp_path: Path) -> None:
    path = tmp_path / LEGACY
    raw = sqlite3.connect(path)
    raw.execute(
        "CREATE TABLE memories(id INTEGER PRIMARY KEY, namespace TEXT, content_hash TEXT, text TEXT, "
        "source TEXT, embedding BLOB, created_at REAL)"
    )
    raw.commit()
    raw.close()

    MemoryStore(path, dimensions=3).close()

    assert (tmp_path / "memory.pre-v2.db").is_file()


def test_reopening_with_other_dimension_is_refused(tmp_path: Path) -> None:
    path = tmp_path / LEGACY
    MemoryStore(path, dimensions=3).close()

    with pytest.raises(RuntimeError, match="(?i)dimension"):
        MemoryStore(path, dimensions=4)


def test_pre_model_database_names_the_reembed_command(tmp_path: Path) -> None:
    """A 512-dim database without an embedding_model entry is refused and points to `reembed`."""
    home = tmp_path / "home"
    path = home / "neospike" / CANONICAL
    MemoryStore(path, dimensions=EMBEDDING_DIMENSIONS).close()
    with closing(sqlite3.connect(path)) as raw, raw:
        raw.execute("UPDATE meta_entries SET value = '512' WHERE key = 'embedding_dimensions'")
        raw.execute("DELETE FROM meta_entries WHERE key = 'embedding_model'")

    with pytest.raises(RuntimeError) as refused:
        MemoryStore(path, dimensions=EMBEDDING_DIMENSIONS)

    message = str(refused.value)
    assert f"python -m neospike.cli --home {home} reembed" in message
    assert EMBEDDING_MODEL in message
    relaxed = MemoryStore(path, dimensions=EMBEDDING_DIMENSIONS, verify_embedding=False)  # what reembed opens
    try:
        stored = relaxed._conn.execute("SELECT value FROM meta_entries WHERE key = 'embedding_dimensions'").fetchone()
    finally:
        relaxed.close()
    assert stored[0] == "512"
