"""Schema v22 -> v23: preferences and constraints leave items for their own table (preferences.py).

In one BEGIN IMMEDIATE transaction, verified before COMMIT (any failed check or error rolls everything back and
raises schema_v22.SchemaError):
- every items memory row labelled preference/constraint (metadata_json memory_kind) that is valid (invalid_at
  NULL) and not archived is copied into preferences under its own id (pref:N = the old mem:N); type = memory_kind;
  situation: scope global -> '' (common), project/session -> metadata subject, else the text; basis: the stored one,
  else agent_inferred (existing rows are not checked against the common-needs-user_said rule); the text vector
  is kept, the situation vector is left empty for embed-pending. Two rows with the same type and text in one
  owner keep the first (common first, then the lower id);
- every labelled row (valid or not) leaves items, with the item_links on either end, its item_revisions,
  memory_pairs, memory_checks and folder_items rows;
- meta schema_version = 23 and PRAGMA user_version = 23.
Checks: no labelled row left in items, items rows = before - labelled, preferences rows = moved, quick_check,
FTS5 integrity-check of items_fts. The memory.pre-v23.db copy is made by MemoryStore before (snapshot step).
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from . import preferences as prefs
from .schema_v22 import SchemaError, check_write_allowed

SCHEMA_V23 = 23
LABEL_KINDS = prefs.PREF_TYPES
_SCOPE_RANK = {"global": 0, "project": 1, "session": 2}
_LABELLED_SQL = ("type='memory' AND (CASE WHEN json_valid(metadata_json) "
                 "THEN COALESCE(json_extract(metadata_json,'$.memory_kind'),'') ELSE '' END) "
                 f"IN ({','.join(repr(kind) for kind in LABEL_KINDS)})")


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _schema_version(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT value FROM meta_entries WHERE key='schema_version'").fetchone()
    try:
        return None if row is None else int(row[0])
    except (TypeError, ValueError):
        return None


def needs_v23(conn: sqlite3.Connection) -> bool:
    """A v22 database (items and meta schema_version below 23); a new database is created at v23 directly."""
    present = _tables(conn)
    if "items" not in present or "meta_entries" not in present:
        return False
    version = _schema_version(conn)
    return version is not None and version < SCHEMA_V23


def situation_of(scope: Any, metadata: dict[str, Any], text: str) -> str:
    """Migration rule: global -> common (''); project/session -> the subject, else the text."""
    if scope == "global":
        return ""
    subject = metadata.get("subject")
    situation = subject.strip() if isinstance(subject, str) and subject.strip() else text
    return " ".join(situation.split())[:prefs.SITUATION_CEILING]


def _metadata(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _plan(conn: sqlite3.Connection) -> tuple[list[int], list[dict[str, Any]], int]:
    """(every labelled id, the rows to insert, the duplicates dropped)."""
    # A database migrated from v21 in this same open has no items.updated_at yet (store._run_migrations adds it).
    has_stamp = "updated_at" in {str(row[1]) for row in conn.execute("PRAGMA table_info(items)")}
    rows = conn.execute(
        "SELECT id,owner,text,scope,metadata_json,importance,origin,embedding,created_at,"
        f"{'updated_at' if has_stamp else 'NULL'},invalid_at,"
        f"archived_at FROM items WHERE {_LABELLED_SQL} ORDER BY id").fetchall()
    labelled = [int(row[0]) for row in rows]
    candidates = []
    for row in rows:
        if row[10] is not None or row[11] is not None:
            continue
        metadata = _metadata(row[4])
        pref_type = metadata.get("memory_kind")
        basis = metadata.get("basis") if metadata.get("basis") in prefs.BASIS_VALUES else prefs.BASIS_AGENT_INFERRED
        text = str(row[2]).strip()
        subject = metadata.get("subject")
        candidates.append({
            "id": int(row[0]), "owner": str(row[1] or ""), "type": pref_type, "text": text,
            "text_hash": prefs.text_hash(pref_type, text), "situation": situation_of(row[3], metadata, text),
            "basis": basis, "subject": subject.strip() if isinstance(subject, str) else "",
            "importance": float(row[5]), "embedding": bytes(row[7] or b""),
            "situation_embedding": prefs.EMPTY_VECTOR, "created_at": float(row[8]),
            "updated_at": float(row[9]) if row[9] is not None else float(row[8]),
            "_rank": (_SCOPE_RANK.get(str(row[3]), 3), int(row[0]))})
    candidates.sort(key=_candidate_order)
    kept: dict[tuple[str, str], dict[str, Any]] = {}
    for item in candidates:
        kept.setdefault((item["owner"], item["text_hash"]), item)
    moved = sorted(kept.values(), key=_candidate_id)
    return labelled, moved, len(candidates) - len(moved)


def _candidate_order(item: dict[str, Any]) -> tuple[int, int]:
    return item["_rank"]


def _candidate_id(item: dict[str, Any]) -> int:
    return item["id"]


_INSERT_COLUMNS = ("id", "owner", "type", "text", "text_hash", "situation", "basis", "subject", "importance",
                   "embedding", "situation_embedding", "created_at", "updated_at")


def _drop_item_rows(conn: sqlite3.Connection, ids: list[int]) -> dict[str, int]:
    """Delete the labelled items rows and every row that points at them; returns the deleted counts."""
    counts: dict[str, int] = {}
    if not ids:
        return counts
    present = _tables(conn)
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _v23_ids (id INTEGER PRIMARY KEY)")
    conn.execute("DELETE FROM _v23_ids")
    conn.executemany("INSERT INTO _v23_ids(id) VALUES (?)", [(item_id,) for item_id in ids])
    statements = (
        ("item_links", "DELETE FROM item_links WHERE src_id IN (SELECT id FROM _v23_ids) "
                       "OR dst_id IN (SELECT id FROM _v23_ids)"),
        ("item_revisions", "DELETE FROM item_revisions WHERE item_id IN (SELECT id FROM _v23_ids)"),
        ("memory_pairs", "DELETE FROM memory_pairs WHERE low_id IN (SELECT id FROM _v23_ids) "
                         "OR high_id IN (SELECT id FROM _v23_ids)"),
        ("memory_checks", "DELETE FROM memory_checks WHERE memory_id IN (SELECT id FROM _v23_ids)"),
        ("folder_items", "DELETE FROM folder_items WHERE item_type='memory' AND item_id IN (SELECT id FROM _v23_ids)"),
        ("items", "DELETE FROM items WHERE id IN (SELECT id FROM _v23_ids)"),
    )
    for table, statement in statements:
        if table in present:
            counts[table] = conn.execute(statement).rowcount
    conn.execute("DROP TABLE _v23_ids")
    return counts


def migrate_v23(conn: sqlite3.Connection) -> dict[str, Any]:
    """v22 -> v23 in one verified transaction (see the module text). Returns the report."""
    check_write_allowed(conn)
    if not needs_v23(conn):
        return {"migrated": False, "reason": "not a v22 database"}
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        if prefs.table_present(conn) and conn.execute(f"SELECT COUNT(*) FROM {prefs.PREFERENCES_TABLE}").fetchone()[0]:
            raise SchemaError("v23 migration refused: preferences already holds rows")
        for statement in prefs.PREFERENCES_DDL.split(";"):
            if statement.strip():
                conn.execute(statement)
        items_before = int(conn.execute("SELECT COUNT(*) FROM items").fetchone()[0])
        labelled, moved, merged = _plan(conn)
        conn.executemany(
            f"INSERT INTO {prefs.PREFERENCES_TABLE}({','.join(_INSERT_COLUMNS)}) "
            f"VALUES ({','.join('?' * len(_INSERT_COLUMNS))})",
            [tuple(item[column] for column in _INSERT_COLUMNS) for item in moved])
        dropped = _drop_item_rows(conn, labelled)
        problems = []
        left = int(conn.execute(f"SELECT COUNT(*) FROM items WHERE {_LABELLED_SQL}").fetchone()[0])
        if left:
            problems.append(f"{left} preference/constraint rows left in items")
        items_after = int(conn.execute("SELECT COUNT(*) FROM items").fetchone()[0])
        if items_after != items_before - len(labelled):
            problems.append(f"items: expected {items_before - len(labelled)}, got {items_after}")
        pref_count = int(conn.execute(f"SELECT COUNT(*) FROM {prefs.PREFERENCES_TABLE}").fetchone()[0])
        if pref_count != len(moved):
            problems.append(f"preferences: expected {len(moved)}, got {pref_count}")
        quick = [str(row[0]) for row in conn.execute("PRAGMA quick_check").fetchall()]
        if quick != ["ok"]:
            problems.append(f"quick_check: {quick[:3]}")
        fts = "absent"
        if "items_fts" in _tables(conn):
            try:
                conn.execute("INSERT INTO items_fts(items_fts) VALUES ('integrity-check')")
                fts = "ok"
            except sqlite3.DatabaseError as exc:
                fts = f"{type(exc).__name__}: {exc}"
                problems.append(f"fts items_fts: {exc}")
        if problems:
            raise SchemaError("v23 migration checks failed; nothing was changed: " + "; ".join(problems))
        conn.execute("INSERT OR REPLACE INTO meta_entries(key, value) VALUES ('schema_version', ?)",
                     (str(SCHEMA_V23),))
        conn.execute(f"PRAGMA user_version={SCHEMA_V23}")
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    return {"migrated": True, "items_before": items_before, "items_after": items_after,
            "labelled": len(labelled), "moved": len(moved), "merged_duplicates": merged,
            "dropped_rows": dropped, "quick_check": quick[0] if quick else "", "fts_integrity": fts}
