from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Iterator

from ._wincompat import chmod as _wc_chmod

_IS_WINDOWS = os.name == "nt"
_SIDE_SUFFIXES = ("-wal", "-shm", "-journal")


def _assert_integrity_ok(sql_conn: sqlite3.Connection, db_path: Path) -> None:
    first = sql_conn.execute("PRAGMA integrity_check").fetchone()
    if not first or first[0] != "ok":
        raise RuntimeError(f"SQLite integrity check failed for {db_path}")


def _wal_companions(db_path: Path) -> tuple[Path, ...]:
    return tuple(Path(f"{db_path}{suffix}") for suffix in _SIDE_SUFFIXES)


def _remove_database_files(db_path: Path) -> None:
    for victim in (db_path, *_wal_companions(db_path)):
        victim.unlink(missing_ok=True)


def _sync_directory(folder: Path) -> None:
    if _IS_WINDOWS:  # directories cannot be opened for fsync on Windows
        return
    fd = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)


def _verify_readonly(db_path: Path) -> None:
    # closing(): an open handle blocks a later os.replace/unlink on Windows.
    with closing(_open_readonly(db_path)) as probe:
        _assert_integrity_ok(probe, db_path)


def _lock_file(fd: int) -> None:
    if _IS_WINDOWS:
        import msvcrt

        if os.fstat(fd).st_size == 0:
            os.write(fd, b"0")
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX)


def _unlock_file(fd: int) -> None:
    if _IS_WINDOWS:
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def _exclusive_migration_lock(folder: Path, stem: str) -> Iterator[None]:
    fd = os.open(folder / f".{stem}.migration.lock", os.O_RDWR | os.O_CREAT, 0o600)
    held = False
    try:
        _lock_file(fd)
        held = True
        yield
    finally:
        if held:
            _unlock_file(fd)
        os.close(fd)


def canonical_database_path(directory: Path, name: str) -> Path:
    """Path of the backup-safe .db file for `name`; a legacy .sqlite file is converted first.

    Only run this once no process of the old plugin is alive: the cross-process lock orders migrators of this
    version among themselves, but old versions never take it.
    """
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    final_path = folder / f"{name}.db"
    old_path = folder / f"{name}.sqlite"
    pre_v2 = folder / "memory.pre-v2.sqlite"

    with _exclusive_migration_lock(folder, name):
        if not old_path.exists():
            if name == "memory" and pre_v2.exists() and final_path.exists():
                _verify_readonly(final_path)
                _remove_database_files(pre_v2)
                _sync_directory(folder)
            return final_path

        fd, scratch_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".db", dir=folder)
        os.close(fd)
        scratch = Path(scratch_name)
        src_conn = dst_conn = None
        try:
            src_conn = _open_readonly(old_path)
            dst_conn = sqlite3.connect(scratch)
            src_conn.backup(dst_conn)
            _assert_integrity_ok(dst_conn, scratch)
            dst_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            dst_conn.execute("PRAGMA journal_mode=DELETE")
            dst_conn.close()
            dst_conn = None
            src_conn.close()
            src_conn = None

            # Windows os.fsync (FlushFileBuffers) needs a writable handle; POSIX accepts either.
            with scratch.open("r+b") as handle:
                os.fsync(handle.fileno())
            for leftover in _wal_companions(final_path):
                leftover.unlink(missing_ok=True)
            _sync_directory(folder)
            os.replace(scratch, final_path)
            _wc_chmod(final_path, 0o600)
            _sync_directory(folder)

            _verify_readonly(final_path)
            _remove_database_files(old_path)
            if name == "memory":
                _remove_database_files(pre_v2)
            _sync_directory(folder)
            return final_path
        finally:
            for conn in (dst_conn, src_conn):
                if conn is not None:
                    conn.close()
            _remove_database_files(scratch)
