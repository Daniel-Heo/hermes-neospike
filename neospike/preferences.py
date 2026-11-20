"""Preferences and constraints (schema v23): their own table, apart from the memory items.

A row is a preference or a constraint (type) of one namespace (owner). situation decides where it goes:
- common (situation ''): loaded once per conversation in the <neospike_profile> block;
- situational (a phrase saying when it applies): injected in the <neospike_situation> block of a turn whose
  message is close to the situation (cosine of the query vector and situation_embedding >= the recall cutoff,
  store.DEFAULT_SEMANTIC_CUTOFF), once per conversation.
Only a preference the user said (basis user_said) may be common: an agent_inferred row and every constraint
need a situation (check_situation; insert_preference/update_preference refuse a common constraint). No revision history: an edit changes the row in place. Ids are their own number space,
written pref:N in injected lines and tool results.

Plain SQL on a caller's connection, stdlib only: the dashboard (no numpy) reads through this module too.
The caller holds the lock and the write transaction.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from typing import Any

PREFERENCES_TABLE = "preferences"
PREF_PREFERENCE, PREF_CONSTRAINT = "preference", "constraint"
PREF_TYPES = (PREF_PREFERENCE, PREF_CONSTRAINT)
BASIS_USER_SAID, BASIS_AGENT_INFERRED = "user_said", "agent_inferred"
BASIS_VALUES = (BASIS_USER_SAID, BASIS_AGENT_INFERRED)
PREF_REF_PREFIX = "pref:"
PREF_TEXT_CEILING = 2000  # one short item, as a memory item (policy.ITEM_TEXT_CEILING)
SITUATION_CEILING = 300
DEFAULT_IMPORTANCE = 0.6  # policy.check_memory_item default
EMPTY_VECTOR = b""  # store.PENDING_EMBED_MARK: filled by the daily embed-pending stage

PREFERENCES_DDL = f"""
CREATE TABLE IF NOT EXISTS {PREFERENCES_TABLE} (
    id INTEGER PRIMARY KEY,
    owner TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL CHECK (type IN ('{PREF_PREFERENCE}','{PREF_CONSTRAINT}')),
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    situation TEXT NOT NULL DEFAULT '',
    basis TEXT NOT NULL DEFAULT '{BASIS_AGENT_INFERRED}' CHECK (basis IN ('{BASIS_USER_SAID}','{BASIS_AGENT_INFERRED}')),
    subject TEXT NOT NULL DEFAULT '',
    importance REAL NOT NULL DEFAULT 0.5,
    embedding BLOB NOT NULL DEFAULT x'',
    situation_embedding BLOB NOT NULL DEFAULT x'',
    created_at REAL NOT NULL,
    updated_at REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS preferences_hash ON {PREFERENCES_TABLE}(owner, text_hash);
CREATE INDEX IF NOT EXISTS preferences_owner ON {PREFERENCES_TABLE}(owner, situation);
"""

MSG_SITUATION_REQUIRED = ("Not saved: an agent_inferred preference/constraint needs a situation. Write when it "
                          "applies in situation (one phrase) and save again; only what the user said directly "
                          "(basis user_said) may apply to every situation.")
MSG_CONSTRAINT_SITUATION_REQUIRED = ("Not saved: a constraint always needs a situation, whoever said it. Write when "
                                     "it applies in situation (one phrase) and save again; only a preference may "
                                     "apply to every situation.")
MSG_KIND_CHANGE = ("A memory cannot become a preference/constraint or the reverse: delete it and save it again "
                   "with the other kind.")
RETIRED_ORIGIN_COLUMN = "origin"  # an earlier layout of this table recorded who saved a row; nothing reads it
_LIST_COLUMNS = ("id,owner,type,text,situation,basis,subject,importance,created_at,updated_at,"
                 "length(embedding)>0 AS has_vector,"
                 "(situation='' OR length(situation_embedding)>0) AS has_situation_vector")


class PreferenceError(ValueError):
    """A refused preference write; code: situation_required, invalid, kind_change."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

    def as_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": str(self)}


def pref_ref(pref_id: int) -> str:
    """pref:N of a preference id."""
    return f"{PREF_REF_PREFIX}{int(pref_id)}"


def text_hash(pref_type: str, text: str) -> str:
    """Duplicate key inside one owner: the type and the whitespace/case-normalized text."""
    flat = " ".join(str(text).casefold().split())
    return hashlib.sha256(f"{pref_type}\0{flat}".encode("utf-8")).hexdigest()


def check_constraint_situation(pref_type: str, situation: str) -> None:
    """A constraint is never common: it always needs a situation, whatever its basis."""
    if pref_type == PREF_CONSTRAINT and not situation.strip():
        raise PreferenceError("situation_required", MSG_CONSTRAINT_SITUATION_REQUIRED)


def check_situation(basis: str, situation: str, pref_type: str = PREF_PREFERENCE) -> None:
    """Who may be common (empty situation): a constraint never; a preference only with basis user_said."""
    check_constraint_situation(pref_type, situation)
    if basis != BASIS_USER_SAID and not situation.strip():
        raise PreferenceError("situation_required", MSG_SITUATION_REQUIRED)


def clean_fields(raw: dict[str, Any], *, partial: bool = False) -> dict[str, Any]:
    """Validated type/text/situation/basis/subject/importance of a write; partial: only the given ones."""
    out: dict[str, Any] = {}
    if "type" in raw or not partial:
        if raw.get("type") not in PREF_TYPES:
            raise PreferenceError("invalid", f"type must be one of {', '.join(PREF_TYPES)}")
        out["type"] = raw["type"]
    if "text" in raw or not partial:
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            raise PreferenceError("invalid", "text must be a non-empty string")
        if len(text.strip()) > PREF_TEXT_CEILING:
            raise PreferenceError("invalid", f"text exceeds the {PREF_TEXT_CEILING}-character limit")
        out["text"] = text.strip()
    if "situation" in raw or not partial:
        situation = raw.get("situation", "")
        if situation is None:
            situation = ""
        if not isinstance(situation, str) or len(situation.strip()) > SITUATION_CEILING:
            raise PreferenceError("invalid", f"situation must be a string of at most {SITUATION_CEILING} characters")
        out["situation"] = " ".join(situation.split())
    if "basis" in raw or not partial:
        if raw.get("basis", BASIS_USER_SAID) not in BASIS_VALUES:
            raise PreferenceError("invalid", f"basis must be one of {', '.join(BASIS_VALUES)}")
        out["basis"] = raw.get("basis", BASIS_USER_SAID)
    if "subject" in raw:
        subject = raw.get("subject")
        if not isinstance(subject, str):
            raise PreferenceError("invalid", "subject must be a string")
        out["subject"] = subject.strip()
    if "importance" in raw or not partial:
        number = raw.get("importance", DEFAULT_IMPORTANCE)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 <= number <= 1:
            raise PreferenceError("invalid", "importance must be a number from 0 to 1")
        out["importance"] = float(number)
    return out


def _row_dict(row: Any) -> dict[str, Any]:
    keys = ("id", "owner", "type", "text", "situation", "basis", "subject", "importance", "created_at",
            "updated_at", "has_vector", "has_situation_vector")
    item = {key: row[index] for index, key in enumerate(keys)}
    item["id"] = int(item["id"])
    item["ref"] = pref_ref(item["id"])
    item["common"] = item["situation"] == ""
    item["has_vector"] = bool(item["has_vector"])
    item["has_situation_vector"] = bool(item["has_situation_vector"])
    return item


def table_present(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (PREFERENCES_TABLE,)).fetchone() is not None


def drop_origin_column(conn: sqlite3.Connection) -> bool:
    """Remove the retired origin column from a table made by the earlier layout; True when it was dropped.
    Idempotent (a table without the column is left alone); for a write open only. No index or trigger
    names the column, so the DROP is not blocked; ids and the other values stay as stored."""
    columns = {str(rec[1]) for rec in conn.execute(f"PRAGMA table_info({PREFERENCES_TABLE})")}
    if RETIRED_ORIGIN_COLUMN not in columns:
        return False
    conn.execute(f"ALTER TABLE {PREFERENCES_TABLE} DROP COLUMN {RETIRED_ORIGIN_COLUMN}")
    return True


def list_preferences(conn: sqlite3.Connection, owner: str | None = None) -> list[dict[str, Any]]:
    """Rows without their vectors, newest change first (the dashboard list)."""
    where, args = ("WHERE owner=?", (owner,)) if owner is not None else ("", ())
    rows = conn.execute(f"SELECT {_LIST_COLUMNS} FROM {PREFERENCES_TABLE} {where} "
                        "ORDER BY COALESCE(updated_at, created_at) DESC, id DESC", args).fetchall()
    return [_row_dict(row) for row in rows]


def get_preference(conn: sqlite3.Connection, pref_id: int, owner: str | None = None) -> dict[str, Any] | None:
    where, args = ("AND owner=?", (owner,)) if owner is not None else ("", ())
    row = conn.execute(f"SELECT {_LIST_COLUMNS} FROM {PREFERENCES_TABLE} WHERE id=? {where}",
                       (int(pref_id), *args)).fetchone()
    return None if row is None else _row_dict(row)


def insert_preference(conn: sqlite3.Connection, owner: str, fields: dict[str, Any], *,
                      embedding: bytes, situation_embedding: bytes, now: float | None = None) -> int | None:
    """New row id, or None when the owner already has this type and text (exact duplicate). A constraint
    without a situation raises PreferenceError (every write path ends here)."""
    check_constraint_situation(fields["type"], fields["situation"])
    stamp = time.time() if now is None else now
    try:
        cur = conn.execute(
            f"INSERT INTO {PREFERENCES_TABLE}(owner,type,text,text_hash,situation,basis,subject,importance,"
            "embedding,situation_embedding,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (owner, fields["type"], fields["text"], text_hash(fields["type"], fields["text"]), fields["situation"],
             fields["basis"], fields.get("subject", ""), fields["importance"], embedding,
             situation_embedding if fields["situation"] else EMPTY_VECTOR, stamp))
    except sqlite3.IntegrityError:
        return None
    return int(cur.lastrowid)


def update_preference(conn: sqlite3.Connection, pref_id: int, changes: dict[str, Any], *,
                      embedding: bytes | None = None, situation_embedding: bytes | None = None) -> bool:
    """In-place change of the given fields; a new text/type gets a new text_hash, a new embedding when given
    (else pending); a new situation its vector when given (else pending; none when common). False when gone or
    when the change would duplicate another row of the owner. A change of type or situation that leaves a
    constraint without a situation raises PreferenceError (nothing written)."""
    row = conn.execute(f"SELECT type,text,situation FROM {PREFERENCES_TABLE} WHERE id=?", (int(pref_id),)).fetchone()
    if row is None:
        return False
    sets: dict[str, Any] = {key: value for key, value in changes.items()
                            if key in ("type", "text", "situation", "basis", "subject", "importance")}
    new_type, new_text = sets.get("type", row[0]), sets.get("text", row[1])
    if "type" in sets or "situation" in sets:
        check_constraint_situation(new_type, sets.get("situation", row[2]))
    if new_type != row[0] or new_text != row[1]:
        sets["text_hash"] = text_hash(new_type, new_text)
    if new_text != row[1]:
        sets["embedding"] = EMPTY_VECTOR if embedding is None else embedding
    new_situation = sets.get("situation", row[2])
    if new_situation != row[2]:
        sets["situation_embedding"] = (EMPTY_VECTOR if not new_situation or situation_embedding is None
                                       else situation_embedding)
    if not sets:
        return True
    try:
        cur = conn.execute(f"UPDATE {PREFERENCES_TABLE} SET {','.join(f'{key}=?' for key in sets)} WHERE id=?",
                           (*sets.values(), int(pref_id)))
    except sqlite3.IntegrityError:
        return False
    return cur.rowcount == 1


def delete_preference(conn: sqlite3.Connection, pref_id: int, owner: str | None = None) -> bool:
    where, args = ("AND owner=?", (owner,)) if owner is not None else ("", ())
    cur = conn.execute(f"DELETE FROM {PREFERENCES_TABLE} WHERE id=? {where}", (int(pref_id), *args))
    return cur.rowcount == 1


def common_rows(conn: sqlite3.Connection, owner: str) -> list[tuple]:
    """(id, text, created_at, importance, embedding, type, basis) of the common rows of owner. A
    constraint is never common: one stored without a situation before that rule is left out (still listed)."""
    return conn.execute(
        f"SELECT id,text,created_at,importance,embedding,type,basis FROM {PREFERENCES_TABLE} "
        "WHERE owner=? AND situation='' AND type<>?", (owner, PREF_CONSTRAINT)).fetchall()


def situation_rows(conn: sqlite3.Connection, owner: str) -> list[tuple]:
    """(id, type, situation, text, situation_embedding) of the situational rows of owner that have their
    situation vector (a row still waiting for embed-pending is not injected)."""
    return conn.execute(
        f"SELECT id,type,situation,text,situation_embedding FROM {PREFERENCES_TABLE} "
        "WHERE owner=? AND situation<>'' AND length(situation_embedding)>0 ORDER BY id", (owner,)).fetchall()


def missing_vectors(conn: sqlite3.Connection) -> list[tuple[int, str, str]]:
    """(id, column, text to embed) of every vector the daily embed-pending stage still has to fill."""
    out = [(int(row[0]), "embedding", str(row[1])) for row in conn.execute(
        f"SELECT id,text FROM {PREFERENCES_TABLE} WHERE length(embedding)=0 ORDER BY id")]
    out += [(int(row[0]), "situation_embedding", str(row[1])) for row in conn.execute(
        f"SELECT id,situation FROM {PREFERENCES_TABLE} WHERE situation<>'' AND length(situation_embedding)=0 "
        "ORDER BY id")]
    return out


_VECTOR_SOURCE = {"embedding": "text", "situation_embedding": "situation"}


def write_missing_vector(conn: sqlite3.Connection, pref_id: int, column: str, expected: str, blob: bytes) -> bool:
    """Fill one pending vector when its text is still the one embedded."""
    source = _VECTOR_SOURCE[column]
    cur = conn.execute(f"UPDATE {PREFERENCES_TABLE} SET {column}=? WHERE id=? AND {source}=? AND length({column})=0",
                       (blob, int(pref_id), expected))
    return cur.rowcount == 1
