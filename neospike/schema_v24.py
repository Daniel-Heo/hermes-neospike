"""Schema v23 -> v24: the copy of Hermes conversations leaves memory.db.

Conversations are read from Hermes' own state.db (chat_history.py, chat_view.py), so the copied tables go.
In one BEGIN IMMEDIATE transaction, verified before COMMIT (any failed check or error rolls everything back and
raises schema_v22.SchemaError):
- conversation_chunks_fts (with its shadow tables), conversation_chunks, conversation_messages and conversations
  are dropped; their indexes and triggers go with them;
- the tombstones of those tables are deleted (no restore can apply them any more);
- meta schema_version = 24 and PRAGMA user_version = 24.
After COMMIT, outside the transaction, one VACUUM gives the freed pages back to the file system. A failed VACUUM
leaves the migration done (committed); it is logged as a warning and reported ("vacuum": "failed: ...").
Checks: none of the dropped names is left in sqlite_master, every other data table keeps its row count
(tombstones: minus the deleted ones), quick_check, FTS5 integrity-check of items_fts.
No other table points at these rows (conversations.summary_note_id pointed from them to notes).
Entry points: MemoryStore opening a v23 database in place (memory.pre-v24.db is made before, snapshot step) and
`schema-migrate --dry-run/--apply` (apply makes memory.pre-v<N+1>.db of a v<N> database first).
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from . import schema_v22
from .schema_v22 import SchemaError, check_write_allowed, table_counts
from .schema_v23 import SCHEMA_V23, migrate_v23, needs_v23

_LOG = logging.getLogger(__name__)
SCHEMA_V24 = 24
VACUUM_DONE = "done"
CONVERSATION_FTS_TABLE = "conversation_chunks_fts"
# Children first (conversation_messages/conversation_chunks reference conversations).
CONVERSATION_TABLES = ("conversation_chunks", "conversation_messages", "conversations")
DROPPED_NAMES = (CONVERSATION_FTS_TABLE, *CONVERSATION_TABLES)
TOMBSTONES_TABLE = "tombstones"
SNAPSHOT_PATTERN = "memory.pre-v{version}.db"
# The FTS sync triggers of the conversation chunks as v6..v23 created them; the v21 -> v22 migration re-creates
# them on a v21 database that still holds the table (dropped again here).
LEGACY_CONVERSATION_TRIGGERS_DDL = """
CREATE TRIGGER IF NOT EXISTS conversation_chunks_fts_ai AFTER INSERT ON conversation_chunks BEGIN
    INSERT INTO conversation_chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS conversation_chunks_fts_ad AFTER DELETE ON conversation_chunks BEGIN
    INSERT INTO conversation_chunks_fts(conversation_chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS conversation_chunks_fts_au AFTER UPDATE OF text ON conversation_chunks BEGIN
    INSERT INTO conversation_chunks_fts(conversation_chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO conversation_chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def _schema_version(conn: sqlite3.Connection) -> int | None:
    present = _names(conn, "table")
    if "meta_entries" not in present:
        return None
    row = conn.execute("SELECT value FROM meta_entries WHERE key='schema_version'").fetchone()
    try:
        return None if row is None else int(row[0])
    except (TypeError, ValueError):
        return None


def _names(conn: sqlite3.Connection, kind: str) -> set[str]:
    return {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type=?", (kind,))}


def needs_v24(conn: sqlite3.Connection) -> bool:
    """A v23 database (items, meta schema_version 23); a new database is created at v24 directly. A v22 database
    takes schema_v23 first."""
    if "items" not in _names(conn, "table"):
        return False
    version = _schema_version(conn)
    return version is not None and SCHEMA_V23 <= version < SCHEMA_V24


def _leftovers(conn: sqlite3.Connection) -> list[str]:
    """sqlite_master entries (tables, indexes, triggers) named after or on a dropped table."""
    marks = ",".join("?" * len(DROPPED_NAMES))
    rows = conn.execute(f"SELECT type, name FROM sqlite_master WHERE name IN ({marks}) OR tbl_name IN ({marks})",
                        (*DROPPED_NAMES, *DROPPED_NAMES)).fetchall()
    return [f"{row[0]} {row[1]}" for row in rows]


def vacuum(conn: sqlite3.Connection) -> str:
    """One VACUUM outside any transaction; VACUUM_DONE, or "failed: ..." (logged as a warning, never raised)."""
    try:
        if conn.in_transaction:
            conn.commit()
        conn.execute("VACUUM")
    except sqlite3.Error as exc:
        _LOG.warning("neospike: VACUUM after the v24 migration failed (the migration stays done): %s: %s",
                     type(exc).__name__, exc)
        return f"failed: {type(exc).__name__}: {exc}"
    return VACUUM_DONE


def migrate_v24(conn: sqlite3.Connection) -> dict[str, Any]:
    """v23 -> v24 in one verified transaction (see the module text). Returns the report."""
    check_write_allowed(conn)
    if not needs_v24(conn):
        return {"migrated": False, "reason": "not a v23 database"}
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    # Another process may have migrated while this one waited for the write lock.
    if not needs_v24(conn):
        conn.rollback()
        return {"migrated": False, "reason": "not a v23 database"}
    try:
        before = table_counts(conn)
        present = _names(conn, "table")
        dropped_rows = {table: before.get(table, 0) for table in CONVERSATION_TABLES if table in present}
        tombstones_dropped = 0
        if TOMBSTONES_TABLE in present:
            marks = ",".join("?" * len(CONVERSATION_TABLES))
            tombstones_dropped = conn.execute(f"DELETE FROM {TOMBSTONES_TABLE} WHERE table_name IN ({marks})",
                                              CONVERSATION_TABLES).rowcount
        for table in DROPPED_NAMES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        after = table_counts(conn)
        expected = {table: count for table, count in before.items() if table not in DROPPED_NAMES}
        if TOMBSTONES_TABLE in expected:
            expected[TOMBSTONES_TABLE] -= tombstones_dropped
        problems = []
        left = _leftovers(conn)
        if left:
            problems.append(f"left in sqlite_master: {left}")
        if after != expected:
            problems.append(f"row counts: expected {expected}, got {after}")
        quick = [str(row[0]) for row in conn.execute("PRAGMA quick_check").fetchall()]
        if quick != ["ok"]:
            problems.append(f"quick_check: {quick[:3]}")
        fts = "absent"
        if "items_fts" in _names(conn, "table"):
            try:
                conn.execute("INSERT INTO items_fts(items_fts) VALUES ('integrity-check')")
                fts = "ok"
            except sqlite3.DatabaseError as exc:
                fts = f"{type(exc).__name__}: {exc}"
                problems.append(f"fts items_fts: {exc}")
        if problems:
            raise SchemaError("v24 migration checks failed; nothing was changed: " + "; ".join(problems))
        conn.execute("INSERT OR REPLACE INTO meta_entries(key, value) VALUES ('schema_version', ?)",
                     (str(SCHEMA_V24),))
        conn.execute(f"PRAGMA user_version={SCHEMA_V24}")
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    return {"migrated": True, "dropped_tables": sorted(dropped_rows), "dropped_rows": dropped_rows,
            "tombstones_dropped": tombstones_dropped, "rows_after": after,
            "quick_check": quick[0] if quick else "", "fts_integrity": fts, "vacuum": vacuum(conn)}


def migrate_to_v24(conn: sqlite3.Connection) -> dict[str, Any]:
    """A v22 or v23 database to v24: schema_v23 first where needed, then this step."""
    report: dict[str, Any] = {}
    if needs_v23(conn):
        report["v23"] = migrate_v23(conn)
    report["v24"] = migrate_v24(conn)
    return report


# --- schema-migrate command (explicit, one home) ---------------------------------------------------------------

def dry_run(db_path: Path) -> dict[str, Any]:
    """The v22/v23 -> v24 steps on an in-memory copy (the file is opened mode=ro); the report, nothing written."""
    source = sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        copy = sqlite3.connect(":memory:")
        source.backup(copy)
    finally:
        source.close()
    try:
        try:
            report = migrate_to_v24(copy)
        except SchemaError as exc:
            return {"dry_run": True, "ok": False, "error": str(exc)}
        return {"dry_run": True, "ok": True, **report}
    finally:
        copy.close()


def apply(db_path: Path) -> dict[str, Any]:
    """Checkpoint (busy -> stop), memory.pre-v<N+1>.db of the v<N> database (checked), then the steps to v24."""
    probe = sqlite3.connect(db_path)
    try:
        check_write_allowed(probe)
        if not (needs_v23(probe) or needs_v24(probe)):
            return {"applied": False, "reason": "already v24"}
        version = _schema_version(probe)
        busy = probe.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if busy is None or int(busy[0]) != 0:
            raise SchemaError("wal_checkpoint busy: stop every neospike process (gateway, dashboard, cron, MCP) first")
    finally:
        probe.close()
    snapshot = schema_v22.snapshot_database(Path(db_path), SNAPSHOT_PATTERN.format(version=int(version or 0) + 1))
    conn = sqlite3.connect(db_path)
    try:
        report = migrate_to_v24(conn)
    finally:
        conn.close()
    return {"applied": True, "backup": snapshot.name, **report}
