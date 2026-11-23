"""Monthly incremental backup of memory.db.

- full-<UTC>.db: VACUUM INTO snapshot (first run, --full, or when the last backup file is missing).
- inc-<UTC>.zip: rows stamped at or after the previous boundary, one JSONL per table, BLOBs as
  base64, plus manifest.json (previous backup name, since/boundary, row counts, schema version).
  FTS tables are never backed up; restore rebuilds them.

Change detection per table:
- timestamp tables (TIMESTAMP_COLUMNS): rows whose timestamp >= since. items (memories,
  document heads and parts) and the conversation family get updated_at from store.py triggers; tombstones
  record every delete (a document save deletes its old part rows and inserts new ones). Read bookkeeping
  (items accessed_at/access_count) is not a change signal: a search hit leaves updated_at alone, so a row
  whose only change is a read is not in the increment; its read values are backed up by the next full
  snapshot or with the row's next edit.
- every other data table (item_links, folder_items, meta without backup_* keys, any unknown
  table) has no timestamp: the whole table is stored when its content hash differs from the last backup.
- folders is a timestamp table; trashing a folder only sets deleted_at.
- item_links has an id primary key: supersede-revert ids, topic names and next chains survive restore as-is.
- a schema_version change since the last backup (a migration) starts a new chain with a full snapshot;
  so does a change of meta embedding_model/embedding_dimensions (reembed keeps updated_at).
  items.invalid_at/superseded_by changes bump items.updated_at through the tracking trigger.

Overlapping increments: an increment takes every row stamped >= since up to the backup moment N (no upper
bound), so its timestamp rows and its hashed tables are one point in time. N = time.time() is taken after
BEGIN IMMEDIATE; B = (N - BOUNDARY_LAG_SECONDS) rounded down to the minute is stored as the last backup time
(the next increment's since) and N as the last backup moment. Writers stamp while holding the write lock, so
a write after this backup stamps at least N minus clock skew (the SQLite trigger clock can read ~2 ms behind
time.time()) > B and lands in the next increment. Rows of [B, N) come again in the next increment; restore
upserts them and skips a tombstone it already holds, so the result is the same.
No-file rule: the overlap alone is not a change. A backup writes a file only when a row is stamped
>= max(since, last moment - BOUNDARY_LAG_SECONDS) or a hashed table changed; otherwise file and meta stay.
A chain from before the moment key reads its last backup time as the last moment.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import sqlite3
import time
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .store import SQLITE_BUSY_TIMEOUT_SECONDS, install_change_triggers, remove_change_triggers

BACKUP_DIR_NAME = "backups"
FULL_PREFIX, FULL_SUFFIX = "full-", ".db"
INC_PREFIX, INC_SUFFIX = "inc-", ".zip"
MANIFEST_NAME = "manifest.json"
BACKUP_META_PREFIX = "backup_"
META_LAST_AT = "backup_last_at"
META_LAST_NAME = "backup_last_name"
# Wall clock N of the last backup run (META_LAST_AT holds its boundary B, the next since).
META_LAST_MOMENT = "backup_last_moment"
# schema_version of the last backup: a migration in between starts a new chain (restore refuses an
# increment whose schema_version differs from its full snapshot).
META_LAST_SCHEMA = "backup_last_schema"
# Text embedding model/dimensions of the last backup: a reembed in between rewrites every vector
# without moving updated_at, so the next backup is a full snapshot of a new chain, never an increment that
# mixes old-model vectors of the snapshot with the new meta.
META_LAST_EMBEDDING = "backup_last_embedding"
EMBEDDING_KEYS = ("embedding_model", "embedding_dimensions")
META_HASH_PREFIX = "backup_hash:"
# Boundary B = floor((now - BOUNDARY_LAG_SECONDS) / BOUNDARY_STEP_SECONDS) * BOUNDARY_STEP_SECONDS.
BOUNDARY_LAG_SECONDS = 1.0
BOUNDARY_STEP_SECONDS = 60.0
BLOB_KEY = "$b64"
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S%fZ"
JSONL_SUFFIX = ".jsonl"
PARTIAL_SUFFIX = ".partial"

# Row timestamp per table; rows stamped >= since go into the increment.
TIMESTAMP_COLUMNS = {
    "items": "updated_at",
    "item_revisions": "edited_at",  # append-only
    "tombstones": "deleted_at",
    # Conversation family: store.py triggers keep updated_at current.
    "conversations": "updated_at",
    "conversation_messages": "updated_at",
    "conversation_chunks": "updated_at",
    # Folders: store.py triggers keep updated_at current.
    "folders": "updated_at",
    # Preferences/constraints (v23): edited in place; store.py triggers keep updated_at current.
    "preferences": "updated_at",
}
# Child table -> (parent key column, parent table): replaced per changed parent row (none since v22: document
# parts are items rows of their own).
CHILD_TABLES: dict[str, tuple[str, str]] = {}
# Tombstone application in restore (key decoded from row_key). item_links/folder_items are hash-replaced.
TOMBSTONE_DELETES = {
    "items": ("DELETE FROM items WHERE id=?",),
    "conversations": ("DELETE FROM conversation_chunks WHERE conversation_id=?",
                      "DELETE FROM conversation_messages WHERE conversation_id=?",
                      "DELETE FROM conversations WHERE id=?"),
    "conversation_messages": ("DELETE FROM conversation_messages WHERE id=?",),
    "conversation_chunks": ("DELETE FROM conversation_chunks WHERE id=?",),
    "folders": ("DELETE FROM folder_items WHERE folder_id=?", "DELETE FROM folders WHERE id=?"),
    "preferences": ("DELETE FROM preferences WHERE id=?",),
}
# Revision tables: tombstone row_key is json_array(owner id, rev).
REVISION_DELETES = {
    "item_revisions": "DELETE FROM item_revisions WHERE item_id=? AND rev=?",
}
META_FILTER = f"key NOT LIKE '{BACKUP_META_PREFIX}%'"


class BackupError(RuntimeError):
    pass


# -- shared helpers ---------------------------------------------------------------------------

def backup_directory(hermes_home: Path) -> Path:
    return Path(hermes_home).expanduser() / "neospike" / BACKUP_DIR_NAME


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def data_tables(connection: sqlite3.Connection) -> list[str]:
    """User tables minus FTS virtual tables and their shadow tables."""
    rows = connection.execute("SELECT name,sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    virtual = [name for name, sql in rows if (sql or "").upper().startswith("CREATE VIRTUAL TABLE")]
    return [name for name, _ in rows
            if not name.startswith("sqlite_") and name not in virtual
            and not any(name.startswith(f"{table}_") for table in virtual)]


def fts_tables(connection: sqlite3.Connection) -> list[str]:
    return [row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND upper(sql) LIKE 'CREATE VIRTUAL TABLE%USING FTS5%'"
    )]


def _columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in connection.execute(f"PRAGMA table_info({_quote(table)})")]


def _select(connection: sqlite3.Connection, table: str, where: str = "", args: tuple = (), *,
            ordered: bool) -> sqlite3.Cursor:
    """All columns of *table* (optionally filtered); ordered=True sorts by every column for deterministic output,
    ordered=False leaves the planner free to use an index on the filter (increment rows are upserts, order-free)."""
    columns = _columns(connection, table)
    clause = f" WHERE {where}" if where else ""
    order = " ORDER BY " + ",".join(str(index + 1) for index in range(len(columns))) if ordered else ""
    return connection.execute(
        f"SELECT {','.join(_quote(column) for column in columns)} FROM {_quote(table)}{clause}{order}", args,
    )


def _ordered_select(connection: sqlite3.Connection, table: str, where: str = "", args: tuple = ()) -> sqlite3.Cursor:
    return _select(connection, table, where, args, ordered=True)


def table_rows(connection: sqlite3.Connection, table: str) -> list[tuple]:
    """Deterministic full content of one table (meta without backup bookkeeping)."""
    return list(_ordered_select(connection, table, META_FILTER if table == "meta_entries" else ""))


def _encode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {BLOB_KEY: base64.b64encode(value).decode("ascii")}
    return value


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        return base64.b64decode(value[BLOB_KEY])
    return value


def _jsonl(cursor: sqlite3.Cursor) -> tuple[bytes, int]:
    names = [item[0] for item in cursor.description]
    lines = []
    for row in cursor:
        lines.append(json.dumps({name: _encode_value(value) for name, value in zip(names, row)},
                                ensure_ascii=False, separators=(",", ":")))
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"), len(lines)


def _content_hash(connection: sqlite3.Connection, table: str) -> str:
    digest = hashlib.sha256()
    for row in table_rows(connection, table):
        digest.update(json.dumps([_encode_value(value) for value in row], ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _hashed_tables(connection: sqlite3.Connection) -> list[str]:
    return [table for table in data_tables(connection) if table not in TIMESTAMP_COLUMNS and table not in CHILD_TABLES]


def _stamp(moment: float) -> str:
    return datetime.fromtimestamp(moment, timezone.utc).strftime(TIMESTAMP_FORMAT)


def _meta(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute("SELECT value FROM meta_entries WHERE key=?", (key,)).fetchone()
    return None if row is None else str(row[0])


def _embedding_signature(connection: sqlite3.Connection) -> str:
    """meta embedding_model/embedding_dimensions as one comparable string (absent keys -> null)."""
    return json.dumps([_meta(connection, key) for key in EMBEDDING_KEYS])


def _set_meta(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute("INSERT OR REPLACE INTO meta_entries(key,value) VALUES (?,?)", (key, value))


def _fsync_file(path: Path) -> None:
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())


def _now() -> float:
    """Wall clock of a backup run (module level so tests can move it)."""
    return time.time()


def boundary_at(moment: float) -> float:
    """Increment boundary B for a run at *moment*: (moment - BOUNDARY_LAG_SECONDS) rounded down to the minute."""
    return math.floor((moment - BOUNDARY_LAG_SECONDS) / BOUNDARY_STEP_SECONDS) * BOUNDARY_STEP_SECONDS


# -- backup -----------------------------------------------------------------------------------

def run_backup(db_path: Path, backup_dir: Path, *, full: bool = False) -> dict:
    """One backup run: full snapshot or increment; no file when nothing changed."""
    db_path, backup_dir = Path(db_path), Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, isolation_level=None, timeout=SQLITE_BUSY_TIMEOUT_SECONDS)
    try:
        # IMMEDIATE (not a plain read transaction): blocks writers so every later commit stamps after
        # the clock read below; reads stay one consistent point in time.
        connection.execute("BEGIN IMMEDIATE")
        try:
            now = _now()
            boundary = boundary_at(now)
            schema_version = _meta(connection, "schema_version")
            last_name, last_at = _meta(connection, META_LAST_NAME), _meta(connection, META_LAST_AT)
            reason = "requested" if full else None
            if reason is None and (last_name is None or last_at is None):
                reason = "first"
            if reason is None and not (backup_dir / last_name).is_file():
                reason = "previous_missing"
            if reason is None and _meta(connection, META_LAST_SCHEMA) != schema_version:
                reason = "schema_changed"
            embedding = _embedding_signature(connection)
            if reason is None and _meta(connection, META_LAST_EMBEDDING) != embedding:
                reason = "embedding_changed"
            if reason is not None:
                result = _full_snapshot(connection, db_path, backup_dir, now)
                result["reason"] = reason
            else:
                since = float(last_at)
                last_moment = _meta(connection, META_LAST_MOMENT)
                changed_since = max(since, (since if last_moment is None else float(last_moment)) - BOUNDARY_LAG_SECONDS)
                result = _increment(connection, backup_dir, now, last_name, since, changed_since, boundary,
                                    schema_version)
            if result["created"]:
                path = backup_dir / result["name"]
                try:
                    _set_meta(connection, META_LAST_NAME, result["name"])
                    _set_meta(connection, META_LAST_AT, repr(boundary))
                    _set_meta(connection, META_LAST_MOMENT, repr(now))
                    if schema_version is not None:
                        _set_meta(connection, META_LAST_SCHEMA, schema_version)
                    _set_meta(connection, META_LAST_EMBEDDING, embedding)
                    connection.execute("COMMIT")
                except BaseException:
                    path.unlink(missing_ok=True)
                    raise
            else:
                connection.execute("ROLLBACK")
            return result
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
    finally:
        connection.close()


def _full_snapshot(connection: sqlite3.Connection, db_path: Path, backup_dir: Path, now: float) -> dict:
    name = f"{FULL_PREFIX}{_stamp(now)}{FULL_SUFFIX}"
    target = backup_dir / name
    if target.exists():
        raise BackupError(f"Backup already exists: {target}")
    temporary = backup_dir / f".{name}{PARTIAL_SUFFIX}"
    temporary.unlink(missing_ok=True)
    started = time.perf_counter()
    # VACUUM INTO cannot run inside a transaction: a second connection reads the last commit,
    # which is this transaction's view because our write lock keeps writers out.
    with closing(sqlite3.connect(db_path, isolation_level=None, timeout=SQLITE_BUSY_TIMEOUT_SECONDS)) as reader:
        reader.execute("VACUUM INTO ?", (str(temporary),))
    _fsync_file(temporary)
    os.replace(temporary, target)
    for table in _hashed_tables(connection):
        _set_meta(connection, META_HASH_PREFIX + table, _content_hash(connection, table))
    return {"created": True, "kind": "full", "name": name, "path": str(target), "bytes": target.stat().st_size,
            "seconds": round(time.perf_counter() - started, 4)}


def _from(column: str) -> str:
    """Open-ended time filter column >= since (no upper bound: rows up to the backup moment); binds (since,)."""
    return f"{_quote(column)}>=?"


def changed_rows(connection: sqlite3.Connection, table: str, since: float) -> sqlite3.Cursor:
    """Rows of a timestamp table stamped >= *since*, unordered so the timestamp index serves the filter."""
    return _select(connection, table, _from(TIMESTAMP_COLUMNS[table]), (since,), ordered=False)


def _has_rows_from(connection: sqlite3.Connection, table: str, since: float) -> bool:
    return connection.execute(f"SELECT EXISTS(SELECT 1 FROM {_quote(table)} WHERE {_from(TIMESTAMP_COLUMNS[table])})",
                              (since,)).fetchone()[0] == 1


def _increment(connection: sqlite3.Connection, backup_dir: Path, now: float, previous: str, since: float,
               changed_since: float, boundary: float, schema_version: str | None) -> dict:
    """Rows stamped >= *since* plus changed hashed tables; no file unless a row is stamped >= *changed_since*
    (past the overlap with the last backup) or a hashed table changed."""
    started = time.perf_counter()
    tables = set(data_tables(connection))
    stamped = [table for table in TIMESTAMP_COLUMNS if table in tables]
    replaced: list[str] = []
    hashes: dict[str, str] = {}
    for table in _hashed_tables(connection):
        current = _content_hash(connection, table)
        if current != _meta(connection, META_HASH_PREFIX + table):
            replaced.append(table)
            hashes[table] = current
    if not replaced and not any(_has_rows_from(connection, table, changed_since) for table in stamped):
        return {"created": False}
    parts: dict[str, bytes] = {}
    counts: dict[str, int] = {}
    for table in stamped:
        parts[table], counts[table] = _jsonl(changed_rows(connection, table, since))
    child_parents: dict[str, list[int]] = {}
    for table, (key, parent) in CHILD_TABLES.items():
        if table not in tables:
            continue
        parent_filter = _from(TIMESTAMP_COLUMNS[parent])
        ids = [int(row[0]) for row in connection.execute(
            f"SELECT id FROM {_quote(parent)} WHERE {parent_filter} ORDER BY id", (since,))]
        child_parents[table] = ids
        selected = f"{_quote(key)} IN (SELECT id FROM {_quote(parent)} WHERE {parent_filter})"
        parts[table], counts[table] = _jsonl(_select(connection, table, selected, (since,), ordered=False))
    for table in replaced:
        parts[table], counts[table] = _jsonl(
            _ordered_select(connection, table, META_FILTER if table == "meta_entries" else ""))
        _set_meta(connection, META_HASH_PREFIX + table, hashes[table])
    name = f"{INC_PREFIX}{_stamp(now)}{INC_SUFFIX}"
    target = backup_dir / name
    if target.exists():
        raise BackupError(f"Backup already exists: {target}")
    manifest = {
        "name": name, "previous": previous, "since": since, "boundary": boundary, "created_at": now,
        "created_at_utc": _stamp(now), "schema_version": None if schema_version is None else int(schema_version),
        "rows": counts, "replace_tables": replaced, "child_parents": child_parents,
    }
    temporary = backup_dir / f".{name}{PARTIAL_SUFFIX}"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, ensure_ascii=False, indent=2))
        for table, payload in parts.items():
            archive.writestr(table + JSONL_SUFFIX, payload)
    _fsync_file(temporary)
    os.replace(temporary, target)
    return {"created": True, "kind": "incremental", "name": name, "path": str(target), "previous": previous,
            "rows": {table: count for table, count in counts.items() if count}, "replace_tables": replaced,
            "bytes": target.stat().st_size, "seconds": round(time.perf_counter() - started, 4)}


# -- chain ------------------------------------------------------------------------------------

def _backup_files(backup_dir: Path) -> list[Path]:
    if not backup_dir.is_dir():
        return []
    files = [path for path in backup_dir.iterdir() if path.is_file() and (
        (path.name.startswith(FULL_PREFIX) and path.name.endswith(FULL_SUFFIX))
        or (path.name.startswith(INC_PREFIX) and path.name.endswith(INC_SUFFIX)))]
    # Names embed the UTC stamp after the prefix; sort by it, not by prefix.
    return sorted(files, key=_stamp_of)


def _stamp_of(path: Path) -> str:
    prefix = FULL_PREFIX if path.name.startswith(FULL_PREFIX) else INC_PREFIX
    return path.name[len(prefix):].rsplit(".", 1)[0]


def read_manifest(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))


def chain(backup_dir: Path) -> dict:
    """Latest full snapshot plus every later increment, with the first break if any."""
    files = _backup_files(Path(backup_dir))
    fulls = [index for index, path in enumerate(files) if path.name.startswith(FULL_PREFIX)]
    if not fulls:
        return {"full": None, "increments": [], "broken": "no full snapshot" if files else None}
    start = fulls[-1]
    increments, previous, broken = [], files[start].name, None
    for path in files[start + 1:]:
        manifest = read_manifest(path)
        if manifest.get("previous") != previous:
            broken = f"{path.name}: previous={manifest.get('previous')}, expected={previous}"
            break
        increments.append((path, manifest))
        previous = path.name
    return {"full": files[start], "increments": increments, "broken": broken}


def list_backups(backup_dir: Path, *, last_recorded: str | None = None) -> dict:
    backup_dir = Path(backup_dir)
    files = _backup_files(backup_dir)
    state = chain(backup_dir)
    in_chain = set()
    if state["full"] is not None:
        in_chain.add(state["full"].name)
        in_chain.update(path.name for path, _ in state["increments"])
    entries = []
    for path in files:
        entry = {"name": path.name, "kind": "full" if path.name.startswith(FULL_PREFIX) else "incremental",
                 "bytes": path.stat().st_size, "in_chain": path.name in in_chain}
        if entry["kind"] == "incremental":
            entry["previous"] = read_manifest(path).get("previous")
        entries.append(entry)
    tail = state["increments"][-1][0].name if state["increments"] else (state["full"].name if state["full"] else None)
    status = "empty" if not files else "broken" if state["broken"] else "ok"
    if status == "ok" and last_recorded is not None and last_recorded != tail:
        status = "tail_missing"
    return {"directory": str(backup_dir), "backups": entries, "chain": status, "broken": state["broken"],
            "chain_tail": tail, "last_recorded": last_recorded}


# -- restore ----------------------------------------------------------------------------------

def restore(backup_dir: Path, target: Path) -> dict:
    """Snapshot + increments in order into a NEW database file; never overwrites anything."""
    backup_dir, target = Path(backup_dir), Path(target).expanduser()
    if target.exists():
        raise BackupError(f"Refusing to restore over an existing file: {target}")
    state = chain(backup_dir)
    if state["full"] is None:
        raise BackupError(f"No full snapshot in {backup_dir}")
    if state["broken"]:
        raise BackupError(f"Backup chain is broken: {state['broken']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + PARTIAL_SUFFIX)
    if temporary.exists():
        raise BackupError(f"Refusing to overwrite an existing file: {temporary}")
    started = time.perf_counter()
    shutil.copyfile(state["full"], temporary)
    try:
        with closing(sqlite3.connect(temporary, isolation_level=None)) as connection:
            connection.execute("BEGIN IMMEDIATE")
            remove_change_triggers(connection)
            version = _meta(connection, "schema_version")
            for path, manifest in state["increments"]:
                if version is not None and manifest.get("schema_version") != int(version):
                    raise BackupError(f"{path.name}: schema_version {manifest.get('schema_version')} != snapshot {version}")
                _apply_increment(connection, path, manifest)
            install_change_triggers(connection)
            # The restored file starts its own backup chain.
            connection.execute(f"DELETE FROM meta_entries WHERE NOT ({META_FILTER})")
            rebuilt = fts_tables(connection)
            for table in rebuilt:
                connection.execute(f"INSERT INTO {_quote(table)}({_quote(table)}) VALUES ('rebuild')")
            connection.execute("COMMIT")
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise BackupError(f"integrity_check failed: {integrity}")
            connection.execute("PRAGMA journal_mode=DELETE")
        if target.exists():
            raise BackupError(f"Refusing to restore over an existing file: {target}")
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"restored": str(target), "full": state["full"].name,
            "increments": [path.name for path, _ in state["increments"]], "fts_rebuilt": rebuilt,
            "integrity": "ok", "seconds": round(time.perf_counter() - started, 4)}


def _read_rows(archive: zipfile.ZipFile, table: str) -> Iterable[dict]:
    try:
        payload = archive.read(table + JSONL_SUFFIX).decode("utf-8")
    except KeyError:
        return []
    return [json.loads(line) for line in payload.splitlines() if line]


def _insert(connection: sqlite3.Connection, table: str, rows: Iterable[dict], *, replace: bool) -> None:
    verb = "INSERT OR REPLACE" if replace else "INSERT"
    for row in rows:
        names = list(row)
        connection.execute(
            f"{verb} INTO {_quote(table)}({','.join(_quote(name) for name in names)}) VALUES ({','.join('?' * len(names))})",
            [_decode_value(row[name]) for name in names],
        )


def _apply_increment(connection: sqlite3.Connection, path: Path, manifest: dict) -> None:
    with zipfile.ZipFile(path) as archive:
        tombstones = list(_read_rows(archive, "tombstones"))
        # Deletes first: a key deleted and reused in the same period comes back with the upserts.
        for tombstone in tombstones:
            table, key = tombstone["table_name"], tombstone["row_key"]
            if table in REVISION_DELETES:
                connection.execute(REVISION_DELETES[table], tuple(json.loads(key)))
            for statement in TOMBSTONE_DELETES.get(table, ()):
                connection.execute(statement, (int(key),))
        for table in TIMESTAMP_COLUMNS:
            if table != "tombstones":
                _insert(connection, table, _read_rows(archive, table), replace=True)
        for table, parent_ids in manifest.get("child_parents", {}).items():
            key = CHILD_TABLES[table][0]
            connection.executemany(f"DELETE FROM {_quote(table)} WHERE {_quote(key)}=?", [(item,) for item in parent_ids])
            _insert(connection, table, _read_rows(archive, table), replace=True)
        for table in manifest.get("replace_tables", []):
            if table == "meta_entries":
                connection.execute(f"DELETE FROM meta_entries WHERE {META_FILTER}")
            else:
                connection.execute(f"DELETE FROM {_quote(table)}")
            _insert(connection, table, _read_rows(archive, table), replace=table == "meta_entries")
        _insert_new_tombstones(connection, tombstones)


def _insert_new_tombstones(connection: sqlite3.Connection, tombstones: Iterable[dict]) -> None:
    """Insert tombstones the restored file does not hold yet (every column identical): a tombstone stamped
    in [B, now) of a backup is in that backup and again in the next increment."""
    for tombstone in tombstones:
        names = list(tombstone)
        values = [_decode_value(tombstone[name]) for name in names]
        columns = ",".join(_quote(name) for name in names)
        same = " AND ".join(f"{_quote(name)} IS ?" for name in names)
        connection.execute(
            f"INSERT INTO tombstones({columns}) SELECT {','.join('?' * len(names))} "
            f"WHERE NOT EXISTS (SELECT 1 FROM tombstones WHERE {same})", values + values,
        )
