"""Test helpers over the v22 items layout: the row shapes older tests read from the v21 tables
(notes with a body, note parts in order, typed links, topic membership), computed from items / item_links."""
from __future__ import annotations

import sqlite3
from typing import Any

from neospike.notes import head_row, note_body
from neospike.store import NOTE_CHUNKS_SQL, NOTE_HEAD_SQL, topic_handles

LINK_TYPE_SQL = "(CASE {alias}.type WHEN 'memory' THEN 'memory' ELSE 'note' END)"
LINK_ROWS_SQL = (
    "SELECT l.id, " + LINK_TYPE_SQL.format(alias="s") + " AS src_type, l.src_id, "
    "CASE WHEN l.dst_id IS NULL THEN (CASE WHEN l.dst_title GLOB 'mem:[0-9]*' THEN 'memory' ELSE 'note' END) "
    "ELSE " + LINK_TYPE_SQL.format(alias="d") + " END AS dst_type, l.dst_id, l.dst_title, l.relation, l.created_by, "
    "l.origin, l.session_id, l.evidence, l.created_at, l.reverted_at FROM item_links l "
    "LEFT JOIN items s ON s.id=l.src_id LEFT JOIN items d ON d.id=l.dst_id "
    "WHERE l.relation IN ('refers','supersedes')")


def note_rows(connection: sqlite3.Connection, *, include_deleted: bool = True) -> list[dict[str, Any]]:
    """Every document as its old notes row (body included), id order."""
    ids = [int(row[0]) for row in connection.execute(f"SELECT id FROM items WHERE {NOTE_HEAD_SQL} ORDER BY id")]
    rows = [head_row(connection, note_id=note_id) for note_id in ids]
    return [row for row in rows if row is not None and (include_deleted or row["deleted_at"] is None)]


def note_row(connection: sqlite3.Connection, note_id: int) -> dict[str, Any] | None:
    return head_row(connection, note_id=note_id)


def body(connection: sqlite3.Connection, note_id: int) -> str:
    row = connection.execute("SELECT author FROM items WHERE id=?", (note_id,)).fetchone()
    return note_body(connection, note_id, str(row[0] or "")) if row else ""


def chunk_rows(connection: sqlite3.Connection, note_id: int | None = None) -> list[sqlite3.Row]:
    """The document parts in the v21 chunk row shape (id, note_id, ord, heading_path, text, text_hash, embedding,
    source_json), note then ord order."""
    where, args = ("WHERE note_id=?", (note_id,)) if note_id is not None else ("", ())
    return connection.execute(f"SELECT * FROM {NOTE_CHUNKS_SQL} c {where} ORDER BY note_id, ord", args).fetchall()


def link_rows(connection: sqlite3.Connection, where: str = "", args: tuple = ()) -> list[sqlite3.Row]:
    """refers/supersedes links in the old typed shape (src_type/dst_type derived from the items)."""
    extra = f" AND {where}" if where else ""
    return connection.execute(f"SELECT * FROM ({LINK_ROWS_SQL}) x WHERE 1{extra} ORDER BY id", args).fetchall()


def topic_labels(connection: sqlite3.Connection) -> dict[int, str]:
    """memory id -> topic name."""
    return {int(row[0]): str(row[1]) for row in connection.execute(
        "SELECT src_id,label FROM item_links WHERE relation='topic' ORDER BY id")}


def topic_names(connection: sqlite3.Connection) -> list[str]:
    """Topic names in creation order."""
    handles = topic_handles(connection)
    return sorted(handles, key=handles.__getitem__)


def members_of(connection: sqlite3.Connection, name: str) -> list[int]:
    return [int(row[0]) for row in connection.execute(
        "SELECT src_id FROM item_links WHERE relation='topic' AND label=? ORDER BY src_id", (name,))]


def memory_count(connection: sqlite3.Connection) -> int:
    return int(connection.execute("SELECT COUNT(*) FROM items WHERE type='memory'").fetchone()[0])


def note_count(connection: sqlite3.Connection, *, live: bool = False) -> int:
    alive = " AND deleted_at IS NULL" if live else ""
    return int(connection.execute(f"SELECT COUNT(*) FROM items WHERE {NOTE_HEAD_SQL}{alive}").fetchone()[0])
