"""Schema v22: the v21-or-older layout becomes the items layout in ONE transaction that is
verified before COMMIT (any failed check rolls everything back).

v21 -> v22:
- memories, notes, note_chunks -> items (type memory | note | doc). A memory keeps its id (so [mem:N] in old
  conversations and notes stays right); a note gets a new id above every memory id, in old note id order; a
  document is a head row (title, tags, rev, model) plus part rows along item_links 'next'. An automatic (formula)
  note's chunks become its parts one to one (text, vector and source_json kept). A human/agent note body is split
  again without overlap (notes.NoteService.split_parts) and each part embedded again: with the embedder given to
  the migration, else the part is left embedding-pending for the daily embed-pending stage.
- links -> item_links (ids kept; the endpoints mapped to the new ids), memories.next_piece_id -> 'next' links,
  live topics + topic_members -> 'topic' links (label = topic name, a repeated name gets ' (N)'); deleted topics
  and their members are dropped; the hand marks (meta topic_manual) name topics by name.
- memory_revisions + note_revisions -> item_revisions; one FTS table items_fts.
- every other table keeps its rows; ids pointing at notes (conversations.summary_note_id, folder_items note rows)
  and [[mem:N]] text in bodies are rewritten through the id maps (identity for memories); [[Title]] stays.
- plus the v21 renames kept from before: meta -> meta_entries, topic_discovery_links -> memory_pairs, the two check
  tables -> memory_checks, dropped llm_usage and dead meta keys, renamed columns of folders/folder_items/
  conversation_chunks.

Version marks after v22: PRAGMA user_version = SCHEMA_V22, PRAGMA application_id = APPLICATION_ID, and
meta_entries.schema_version (read by older code). A database whose user_version is above SUPPORTED_USER_VERSION,
or whose application_id belongs to another program, is never opened for writing (check_write_allowed).

Checks before COMMIT: items rows per type (memory = old memories, note/doc heads = old notes by author, automatic
parts = old automatic-note chunks), every human/agent body rebuilt from its parts equals the old body, item_links
per relation (refers+supersedes = old links that still have both ends, next = piece pointers + document chains,
topic = live topic members + topic pages), item_revisions = the two old revision tables, the other tables' rows,
quick_check, foreign_key_check, FTS5 integrity-check of items_fts and conversation_chunks_fts, count and total
length of the memory and automatic-part vectors, and the top-5 ids of three fixed memory queries (the vectors of
three stored memories) equal before/after.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA_V22 = 22
SUPPORTED_USER_VERSION = 23  # schema_v23.SCHEMA_V23: the newest layout this code writes (v22 -> v23 in schema_v23.py)
APPLICATION_ID = 0x4E53504B  # "NSPK"
LEGACY_MEMORY_TABLE = "memories"
SNAPSHOT_NAME = "memory.pre-v22.db"
PRE_V22_SUFFIX = ".pre-v22"
TOP_K = 5
FIXED_QUERY_COUNT = 3

TABLE_RENAMES = {"meta": "meta_entries", "topic_discovery_links": "memory_pairs"}
# (table after the table renames, old column, new column)
COLUMN_RENAMES = (
    ("conversation_chunks", "next_chunk_id", "next_id"),
    ("folders", "origin", "created_by"),
    ("folder_items", "lane", "item_type"),
)
OLD_ITEM_TABLES = ("memories", "notes", "note_chunks", "links", "memory_revisions", "note_revisions", "topics",
                   "topic_members")
OLD_FTS = ("memories_fts", "note_chunks_fts", "memory_items_fts")
OLD_INDEXES = ("topic_discovery_links_high",)
DROPPED_TABLES = ("llm_usage",)
CHECK_SOURCES = (("topic_discovery_checked", "topic"), ("note_absorb_checked", "absorb"))
DROPPED_META_KEYS = ("review_pending", "topic_pending", "weekly_cron_job_ids",
                     "backup_hash:inbox_decisions", "backup_hash:lint_state", "backup_hash:llm_usage",
                     "backup_hash:topic_discovery_links", "backup_hash:topic_discovery_checked",
                     "backup_hash:note_absorb_checked", "backup_hash:meta", "backup_hash:links",
                     "backup_hash:topics", "backup_hash:topic_members")
RENAMED_META_KEYS = {"live_namespace": "live_owner"}
FTS_TABLES = ("items_fts", "conversation_chunks_fts")
NEW_ITEM_TABLES = ("items", "item_links", "item_revisions")
MANUAL_META_KEY = "topic_manual"
FORMULA = "formula"
DEFAULT_OWNER = "default:local"
_MEMORY_REF = re.compile(r"\[\[([ \t]*)mem:(\d{1,19})([ \t]*)(?=\]\]|[#|])")
_EMPTY_HASH = hashlib.sha256(b"").hexdigest()


class SchemaError(RuntimeError):
    """A database this code must not write, or a v22 migration whose checks failed (nothing was changed)."""


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _pragma_int(conn: sqlite3.Connection, name: str) -> int:
    return int(conn.execute(f"PRAGMA {name}").fetchone()[0])


def check_write_allowed(conn: sqlite3.Connection) -> None:
    """Refuse a database written by a newer schema or by another program (SchemaError)."""
    user_version = _pragma_int(conn, "user_version")
    if user_version > SUPPORTED_USER_VERSION:
        raise SchemaError(f"memory database schema v{user_version} is newer than this neospike "
                          f"(supports v{SUPPORTED_USER_VERSION}); not opened for writing. Update neospike.")
    application_id = _pragma_int(conn, "application_id")
    if application_id not in (0, APPLICATION_ID):
        raise SchemaError(f"not a neospike database (application_id={application_id:#x}); not opened for writing")


def needs_v22(conn: sqlite3.Connection) -> bool:
    """True for a v21-or-older layout (any table of the old memory/note/link/topic family)."""
    return bool(_tables(conn) & set(OLD_ITEM_TABLES))


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    """Data tables: not sqlite_*, not FTS virtual tables or their shadow tables."""
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table'").fetchall()
    virtual = {name for name, sql in rows if (sql or "").upper().startswith("CREATE VIRTUAL TABLE")}
    return sorted(name for name, _ in rows if not name.startswith("sqlite_") and name not in virtual
                  and not any(name.startswith(fts + "_") for fts in virtual))


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in _user_tables(conn)}


def rewrite_memory_refs(text: str, memory_ids: dict[int, int]) -> str:
    """[[mem:N]] (also [[mem:N#x]] / [[mem:N|x]]) with N mapped through memory_ids; an unmapped N stays."""
    def _swap(found: re.Match) -> str:
        old = int(found.group(2))
        return f"[[{found.group(1)}mem:{memory_ids.get(old, old)}{found.group(3)}"
    return _MEMORY_REF.sub(_swap, text)


# --- reading the old layout (any column may be missing in an old file) ------------------------------------------
def _pick(conn: sqlite3.Connection, table: str, wanted: dict[str, str]) -> str:
    """SELECT list of wanted {column: default SQL}: the column when the table has it, else the default AS column."""
    present = _columns(conn, table)
    return ",".join(name if name in present else f"{default} AS {name}" for name, default in wanted.items())


_OLD_MEMORY_COLUMNS = {
    "id": "NULL", "namespace": f"'{DEFAULT_OWNER}'", "content_hash": "''", "text": "''", "source": "''",
    "embedding": "X''", "created_at": "0.0", "kind": "'episodic'", "confidence": "1.0", "importance": "0.5",
    "project": "''", "metadata_json": "'{}'", "accessed_at": "NULL", "access_count": "0", "invalid_at": "NULL",
    "superseded_by": "NULL", "archived_at": "NULL", "next_piece_id": "NULL", "updated_at": "NULL"}
_OLD_NOTE_COLUMNS = {
    "id": "NULL", "title": "''", "body": "''", "tags_json": "'[]'", "project": "''", "author": "'agent'",
    "created_at": "0.0", "updated_at": "0.0", "rev": "1", "deleted_at": "NULL", "accessed_at": "NULL",
    "access_count": "0", "model": "''"}
_OLD_CHUNK_COLUMNS = {"id": "NULL", "note_id": "NULL", "ord": "0", "heading_path": "''", "text": "''",
                      "text_hash": "''", "embedding": "X''", "source_json": "NULL"}
_OLD_LINK_COLUMNS = {"id": "rowid", "src_type": "'note'", "src_id": "NULL", "dst_type": "'note'", "dst_id": "NULL",
                     "dst_title": "NULL", "relation": "'refers'", "created_by": "'code'", "origin": "''",
                     "session_id": "''", "evidence": "''", "created_at": "0.0", "reverted_at": "NULL"}


def _rows(conn: sqlite3.Connection, table: str, columns: dict[str, str], order: str) -> list[dict]:
    if table not in _tables(conn):
        return []
    cursor = conn.execute(f"SELECT {_pick(conn, table, columns)} FROM {table} ORDER BY {order}")
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def _metadata(raw: Any) -> dict:
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


# --- vectors and fixed queries (memory rows of either layout) -----------------------------------------------------
def _memory_source(conn: sqlite3.Connection) -> str | None:
    present = _tables(conn)
    if LEGACY_MEMORY_TABLE in present:
        return f"SELECT id, embedding FROM {LEGACY_MEMORY_TABLE} WHERE length(embedding)>0 ORDER BY id"
    if "items" in present:
        return "SELECT id, embedding FROM items WHERE type='memory' AND length(embedding)>0 ORDER BY id"
    return None


def _vector_rows(conn: sqlite3.Connection) -> tuple[list[int], list[bytes]]:
    sql = _memory_source(conn)
    if sql is None:
        return [], []
    rows = conn.execute(sql).fetchall()
    return [int(row[0]) for row in rows], [bytes(row[1]) for row in rows]


def fixed_query_ids(conn: sqlite3.Connection) -> list[int]:
    """Ids of the stored memories whose vectors are the fixed queries: lowest, middle and highest id."""
    ids, _ = _vector_rows(conn)
    if not ids:
        return []
    picks = [ids[0], ids[len(ids) // 2], ids[-1]]
    return list(dict.fromkeys(picks))[:FIXED_QUERY_COUNT]


def top_ids(conn: sqlite3.Connection, query_ids: list[int], k: int = TOP_K,
            id_map: dict[int, int] | None = None) -> dict[str, list[int]]:
    """'mem:<query id>' -> the k memory ids with the highest dot product to that stored vector (ids through
    id_map, so a before/after pair compares the same items)."""
    import array
    from .store import dot_similarities
    ids, blobs = _vector_rows(conn)
    if not ids or not query_ids:
        return {}
    mapping = id_map or {}
    by_id = dict(zip(ids, blobs))
    result = {}
    for query_id in query_ids:
        blob = by_id.get(mapping.get(query_id, query_id))
        if blob is None:
            result[f"mem:{query_id}"] = []
            continue
        scores = dot_similarities(array.array("f", blob), blobs)
        ranked = sorted(range(len(ids)), key=scores.__getitem__, reverse=True)[:k]
        result[f"mem:{query_id}"] = [ids[index] for index in ranked]
    return result


def _blob_stats(blobs: list[bytes]) -> list[int]:
    kept = [blob for blob in blobs if blob]
    return [len(kept), sum(len(blob) for blob in kept)]


# --- the migration steps (inside the caller's transaction) ------------------------------------------------------
def _split_statements(script: str) -> list[str]:
    statements, current = [], ""
    for line in script.splitlines(keepends=True):
        current += line
        if sqlite3.complete_statement(current):
            if current.strip().strip(";").strip():
                statements.append(current.strip())
            current = ""
    if current.strip():
        statements.append(current.strip())
    return statements


def _run_script(conn: sqlite3.Connection, script: str) -> None:
    for statement in _split_statements(script):
        conn.execute(statement)


def _run_triggers(conn: sqlite3.Connection, script: str, table: str) -> None:
    """The CREATE TRIGGER statements of script on table (the FTS sync triggers of one family)."""
    for statement in _split_statements(script):
        if statement.startswith("CREATE TRIGGER") and f" ON {table} " in statement:
            conn.execute(statement)


class _Plan:
    """What the migration read and wrote, for the checks and the report."""

    def __init__(self) -> None:
        self.memories: list[dict] = []
        self.notes: list[dict] = []
        self.chunks: dict[int, list[dict]] = {}
        self.note_ids: dict[int, int] = {}
        self.memory_ids: dict[int, int] = {}
        self.chunk_ids: dict[int, int] = {}
        self.expected_links: dict[str, int] = {"refers": 0, "supersedes": 0, "next": 0, "topic": 0}
        self.dropped_links = 0
        self.dropped_topics: list[str] = []
        self.dropped_topic_members = 0
        self.renamed_topics: dict[str, str] = {}
        self.parts_embedded = 0
        self.parts_pending = 0
        self.parts_reused = 0
        self.body_mismatch: list[int] = []
        self.formula_vectors_before: list[int] = [0, 0]
        self.old_parts = {"formula": 0, "other": 0}
        self.new_parts = {"formula": 0, "other": 0}


def _base_renames(conn: sqlite3.Connection) -> None:
    """The v21 renames kept from the earlier v22 step (meta, memory pairs, checks, dropped tables and keys)."""
    from . import store as layout
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
        conn.execute(f"DROP TRIGGER IF EXISTS {name}")
    for name in OLD_INDEXES:
        conn.execute(f"DROP INDEX IF EXISTS {name}")
    for name in OLD_FTS:
        conn.execute(f"DROP TABLE IF EXISTS {name}")
    present = _tables(conn)
    for old, new in TABLE_RENAMES.items():
        if old in present and new not in present:
            conn.execute(f"ALTER TABLE {old} RENAME TO {new}")
    present = _tables(conn)
    for table, old, new in COLUMN_RENAMES:
        if table in present:
            columns = _columns(conn, table)
            if old in columns and new not in columns:
                conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
    _run_script(conn, layout._CHECKS_DDL)
    for table, stage in CHECK_SOURCES:
        if table not in present:
            continue
        columns = _columns(conn, table)
        digest = "content_hash" if "content_hash" in columns else ("text_hash" if "text_hash" in columns else "NULL")
        chunk = "chunk_id" if "chunk_id" in columns else "NULL"
        checked = "checked_at" if "checked_at" in columns else "NULL"
        conn.execute(f"INSERT INTO memory_checks(memory_id,stage,text_hash,chunk_id,checked_at) "
                     f"SELECT memory_id,?,{digest},{chunk},{checked} FROM {table} ORDER BY memory_id", (stage,))
        conn.execute(f"DROP TABLE {table}")
    for table in DROPPED_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute("CREATE TABLE IF NOT EXISTS meta_entries (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    marks = ",".join("?" * len(DROPPED_META_KEYS))
    conn.execute(f"DELETE FROM meta_entries WHERE key IN ({marks})", DROPPED_META_KEYS)
    for old, new in RENAMED_META_KEYS.items():
        conn.execute("UPDATE OR REPLACE meta_entries SET key=? WHERE key=?", (new, old))
    if "memory_pairs" in _tables(conn):
        conn.execute("CREATE INDEX IF NOT EXISTS memory_pairs_high ON memory_pairs(high_id)")


def _read_old(conn: sqlite3.Connection, plan: _Plan) -> None:
    plan.memories = _rows(conn, LEGACY_MEMORY_TABLE, _OLD_MEMORY_COLUMNS, "id")
    plan.notes = _rows(conn, "notes", _OLD_NOTE_COLUMNS, "id")
    for chunk in _rows(conn, "note_chunks", _OLD_CHUNK_COLUMNS, "note_id, ord, id"):
        plan.chunks.setdefault(int(chunk["note_id"]), []).append(chunk)
    plan.memory_ids = {int(row["id"]): int(row["id"]) for row in plan.memories}
    authors = {int(note["id"]): str(note["author"]) for note in plan.notes}
    for note_id, chunks in plan.chunks.items():
        lane = "formula" if authors.get(note_id) == FORMULA else "other"
        plan.old_parts[lane] += len(chunks)
        if lane == "formula":
            stats = _blob_stats([bytes(chunk["embedding"] or b"") for chunk in chunks])
            plan.formula_vectors_before = [plan.formula_vectors_before[0] + stats[0],
                                           plan.formula_vectors_before[1] + stats[1]]


def _insert_memories(conn: sqlite3.Connection, plan: _Plan) -> None:
    rows = []
    for memory in plan.memories:
        metadata = _metadata(memory["metadata_json"])
        scope = metadata.pop("scope", None)
        rows.append((int(memory["id"]), str(memory["namespace"]), str(memory["content_hash"]), str(memory["text"]),
                     str(memory["project"] or ""), scope if isinstance(scope, str) else None,
                     json.dumps(metadata, ensure_ascii=False), str(memory["source"] or ""),
                     str(memory["kind"] or "episodic"), float(memory["confidence"]), float(memory["importance"]),
                     bytes(memory["embedding"] or b""), float(memory["created_at"]),
                     memory["updated_at"] if memory["updated_at"] is not None else float(memory["created_at"]),
                     memory["accessed_at"], int(memory["access_count"] or 0), memory["archived_at"],
                     memory["invalid_at"], memory["superseded_by"]))
    conn.executemany(
        "INSERT INTO items(id,type,owner,title,text,text_hash,project,scope,metadata_json,origin,kind,confidence,"
        "importance,embedding,created_at,updated_at,accessed_at,access_count,archived_at,invalid_at,superseded_by) "
        "VALUES (?,'memory',?,NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [(row[0], row[1], row[3], row[2], *row[4:]) for row in rows])


def _pack(embedder: Any, text: str, dimensions: int | None) -> bytes | None:
    """The part's vector, or None (no embedder, or the embedding failed: the part stays embedding-pending)."""
    if embedder is None:
        return None
    from .store import normalize_rows
    try:
        vector = normalize_rows(embedder.embed_document(text))
    except (RuntimeError, OSError, ValueError):
        return None
    if (dimensions is not None and len(vector) != dimensions) or not all(math.isfinite(value) for value in vector) \
            or not any(vector):
        return None
    return vector.tobytes()


def _insert_documents(conn: sqlite3.Connection, plan: _Plan, embedder: Any, splitter: Any) -> None:
    """Heads in old note id order (new ids above every memory id), each followed by its parts."""
    from .notes import _text_hash, part_embed_text
    from .store import document_type, write_chain
    next_id = max([0, *plan.memory_ids.values(), *plan.note_ids.values()]) + 1
    dims_row = conn.execute("SELECT value FROM meta_entries WHERE key='embedding_dimensions'").fetchone()
    dimensions = int(dims_row[0]) if dims_row else None
    formula_top = 0
    for note in plan.notes:
        head_id = plan.note_ids[int(note["id"])]
        author = str(note["author"] or "agent")
        kind = document_type(author)
        body = rewrite_memory_refs(str(note["body"] or ""), plan.memory_ids)
        conn.execute(
            "INSERT INTO items(id,type,owner,title,text,text_hash,project,tags_json,author,kind,embedding,created_at,"
            "updated_at,accessed_at,access_count,rev,deleted_at,metadata_json) "
            "VALUES (?,?,'',?,'',?,?,?,?,?,X'',?,?,?,?,?,?,?)",
            (head_id, kind, str(note["title"]), _EMPTY_HASH, str(note["project"] or ""), str(note["tags_json"] or "[]"),
             author, kind, float(note["created_at"]), float(note["updated_at"]), note["accessed_at"],
             int(note["access_count"] or 0), int(note["rev"] or 1), note["deleted_at"],
             json.dumps({"model": str(note["model"] or "")}, ensure_ascii=False)))
        chain = [head_id]
        chunks = plan.chunks.get(int(note["id"]), [])
        stamp = float(note["updated_at"])
        if author == FORMULA:
            for chunk in chunks:
                metadata = {} if chunk["source_json"] is None else {"source_json": chunk["source_json"]}
                text = rewrite_memory_refs(str(chunk["text"]), plan.memory_ids)
                conn.execute(
                    "INSERT INTO items(id,type,owner,title,text,text_hash,project,author,kind,embedding,created_at,"
                    "updated_at,metadata_json) VALUES (?,?,'',NULL,?,?,?,?,?,?,?,?,?)",
                    (next_id, kind, text, _text_hash(text), str(note["project"] or ""), author, kind,
                     bytes(chunk["embedding"] or b""), stamp, stamp, json.dumps(metadata, ensure_ascii=False)))
                plan.chunk_ids[int(chunk["id"])] = next_id
                chain.append(next_id)
                formula_top = next_id
                next_id += 1
            plan.new_parts["formula"] += len(chunks)
        else:
            known = {str(chunk["text_hash"]): bytes(chunk["embedding"] or b"") for chunk in chunks if chunk["embedding"]}
            parts = splitter(body)
            if "".join(text for _, text in parts) != body:
                plan.body_mismatch.append(int(note["id"]))
            for heading, text in parts:
                digest = _text_hash(part_embed_text(text))
                blob = known.get(digest)
                if blob:
                    plan.parts_reused += 1
                else:
                    blob = _pack(embedder, part_embed_text(text), dimensions)
                    if blob is None:
                        plan.parts_pending += 1
                        blob = b""
                    else:
                        plan.parts_embedded += 1
                metadata = {"heading_path": heading} if heading else {}
                conn.execute(
                    "INSERT INTO items(id,type,owner,title,text,text_hash,project,author,kind,embedding,created_at,"
                    "updated_at,metadata_json) VALUES (?,?,'',NULL,?,?,?,?,?,?,?,?,?)",
                    (next_id, kind, text, digest, str(note["project"] or ""), author, kind, blob, stamp, stamp,
                     json.dumps(metadata, ensure_ascii=False)))
                chain.append(next_id)
                next_id += 1
            plan.new_parts["other"] += len(parts)
        write_chain(conn, chain, now=stamp)
        plan.expected_links["next"] += len(chain) - 1
    if formula_top:
        conn.execute("INSERT OR REPLACE INTO meta_entries(key,value) VALUES ('formula_chunk_hwm',?)", (str(formula_top),))
    else:
        conn.execute("DELETE FROM meta_entries WHERE key='formula_chunk_hwm'")


def _map_note_ids(plan: _Plan) -> None:
    """Old note id -> head id: in old note id order, above every memory id (the parts come after all heads)."""
    next_id = max([0, *plan.memory_ids.values()]) + 1
    for note in plan.notes:
        plan.note_ids[int(note["id"])] = next_id
        next_id += 1


def _map_end(plan: _Plan, kind: str, item_id: Any) -> int | None:
    if item_id is None:
        return None
    lane = plan.memory_ids if kind == "memory" else plan.note_ids
    return lane.get(int(item_id))


def _insert_links(conn: sqlite3.Connection, plan: _Plan) -> None:
    """refers/supersedes rows with their ids; an end that no longer exists: refers keeps dst NULL (its text
    stays), every other row whose source or supersedes target is gone is dropped (counted)."""
    rows = []
    for link in _rows(conn, "links", _OLD_LINK_COLUMNS, "id"):
        relation = str(link["relation"])
        if relation not in ("refers", "supersedes"):
            plan.dropped_links += 1
            continue
        src = _map_end(plan, str(link["src_type"]), link["src_id"])
        dst = _map_end(plan, str(link["dst_type"]), link["dst_id"])
        if src is None or (relation == "supersedes" and dst is None):
            plan.dropped_links += 1
            continue
        title = link["dst_title"]
        if str(link["dst_type"]) == "memory" and dst is not None:
            title = f"mem:{dst}"
        rows.append((int(link["id"]), src, dst, title, relation, str(link["created_by"] or "code"),
                     str(link["origin"] or ""), str(link["session_id"] or ""), str(link["evidence"] or ""),
                     float(link["created_at"] or 0.0), link["reverted_at"]))
        plan.expected_links[relation] += 1
    conn.executemany(
        "INSERT INTO item_links(id,src_id,dst_id,dst_title,relation,created_by,origin,session_id,evidence,created_at,"
        "reverted_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)


def _insert_piece_chains(conn: sqlite3.Connection, plan: _Plan) -> None:
    rows = [(int(memory["id"]), int(memory["next_piece_id"]), float(memory["created_at"]))
            for memory in plan.memories
            if memory["next_piece_id"] is not None and int(memory["next_piece_id"]) in plan.memory_ids
            and int(memory["next_piece_id"]) != int(memory["id"])]
    conn.executemany("INSERT INTO item_links(src_id,dst_id,relation,created_at) VALUES (?,?,'next',?)", rows)
    plan.expected_links["next"] += len(rows)


def _insert_topics(conn: sqlite3.Connection, plan: _Plan) -> dict[int, str]:
    """Live topics -> 'topic' links (members in memory id order, topics in old id order, so the new handles keep
    the old order); returns old topic id -> name."""
    present = _tables(conn)
    if "topics" not in present:
        return {}
    columns = _columns(conn, "topics")
    live_sql = "deleted_at IS NULL" if "deleted_at" in columns else "1"
    page = "note_id" if "note_id" in columns else "NULL"
    taken: set[str] = set()
    names: dict[int, str] = {}
    pages: dict[int, int] = {}
    for topic_id, name, note_id in conn.execute(f"SELECT id,name,{page} FROM topics WHERE {live_sql} ORDER BY id"):
        base = str(name or "topic")
        chosen, number = base, 2
        while chosen in taken:
            chosen, number = f"{base} ({number})", number + 1
        taken.add(chosen)
        names[int(topic_id)] = chosen
        if chosen != base:
            plan.renamed_topics[str(topic_id)] = chosen
        if note_id is not None and int(note_id) in plan.note_ids:
            pages[int(topic_id)] = plan.note_ids[int(note_id)]
    plan.dropped_topics = [str(row[0]) for row in conn.execute(
        f"SELECT name FROM topics WHERE NOT ({live_sql}) ORDER BY id")]
    members: list[tuple[int, int, float]] = []
    if "topic_members" in present:
        assigned = "assigned_at" if "assigned_at" in _columns(conn, "topic_members") else "0.0"
        for memory_id, topic_id, stamp in conn.execute(
                f"SELECT memory_id,topic_id,{assigned} FROM topic_members ORDER BY topic_id,memory_id"):
            if int(topic_id) in names and int(memory_id) in plan.memory_ids:
                members.append((int(memory_id), int(topic_id), float(stamp or 0.0)))
            else:
                plan.dropped_topic_members += 1
    rows = []
    for topic_id in sorted(names):
        if topic_id in pages:
            rows.append((pages[topic_id], names[topic_id], 0.0))
        rows.extend((memory_id, names[topic_id], stamp) for memory_id, owner, stamp in members if owner == topic_id)
    conn.executemany("INSERT INTO item_links(src_id,relation,label,created_at) VALUES (?,'topic',?,?)", rows)
    plan.expected_links["topic"] += len(rows)
    return names


def _rewrite_manual_marks(conn: sqlite3.Connection, names: dict[int, str]) -> None:
    row = conn.execute("SELECT value FROM meta_entries WHERE key=?", (MANUAL_META_KEY,)).fetchone()
    if row is None:
        return
    raw = _metadata(row[0])
    pinned = {str(key): names[int(value)] for key, value in (raw.get("pinned") or {}).items()
              if isinstance(value, int) and int(value) in names}
    excluded = {}
    for key, values in (raw.get("excluded") or {}).items():
        kept = sorted({names[int(value)] for value in values if isinstance(value, int) and int(value) in names})
        if kept:
            excluded[str(key)] = kept
    conn.execute("UPDATE meta_entries SET value=? WHERE key=?",
                 (json.dumps({"pinned": pinned, "excluded": excluded}, sort_keys=True, ensure_ascii=False),
                  MANUAL_META_KEY))


def _insert_revisions(conn: sqlite3.Connection, plan: _Plan) -> int:
    present = _tables(conn)
    count = 0
    if "memory_revisions" in present:
        session = "session_id" if "session_id" in _columns(conn, "memory_revisions") else "''"
        rows = [(plan.memory_ids.get(int(row[0]), int(row[0])), *row[1:]) for row in conn.execute(
            f"SELECT memory_id,rev,text,metadata_json,edited_by,edited_at,{session} FROM memory_revisions "
            f"ORDER BY memory_id,rev")]
        rows = [row for row in rows if int(row[0]) in plan.memory_ids.values()]
        conn.executemany("INSERT INTO item_revisions(item_id,rev,title,text,metadata_json,edited_by,edited_at,"
                         "session_id) VALUES (?,?,NULL,?,?,?,?,?)", rows)
        count += len(rows)
    if "note_revisions" in present:
        model = "model" if "model" in _columns(conn, "note_revisions") else "''"
        rows = []
        for note_id, rev, title, body, edited_by, edited_at, model_name in conn.execute(
                f"SELECT note_id,rev,title,body,edited_by,edited_at,{model} FROM note_revisions ORDER BY note_id,rev"):
            if int(note_id) not in plan.note_ids:
                continue
            rows.append((plan.note_ids[int(note_id)], rev, title, rewrite_memory_refs(str(body), plan.memory_ids),
                         json.dumps({"model": str(model_name or "")}), edited_by, edited_at))
        conn.executemany("INSERT INTO item_revisions(item_id,rev,title,text,metadata_json,edited_by,edited_at) "
                         "VALUES (?,?,?,?,?,?,?)", rows)
        count += len(rows)
    return count


def _remap_references(conn: sqlite3.Connection, plan: _Plan) -> dict[str, int]:
    """Ids of notes held by other tables, the absorb marks, and tombstones of the dropped tables."""
    present = _tables(conn)
    changed = {"conversations": 0, "folder_items_dropped": 0, "tombstones_dropped": 0}
    if "conversations" in present and "summary_note_id" in _columns(conn, "conversations"):
        for conversation_id, note_id in conn.execute(
                "SELECT id,summary_note_id FROM conversations WHERE summary_note_id IS NOT NULL").fetchall():
            conn.execute("UPDATE conversations SET summary_note_id=? WHERE id=?",
                         (plan.note_ids.get(int(note_id)), conversation_id))
            changed["conversations"] += 1
    if "folder_items" in present:
        lane = "item_type" if "item_type" in _columns(conn, "folder_items") else "lane"
        rows = conn.execute(f"SELECT rowid,{lane},item_id FROM folder_items").fetchall()
        conn.execute("UPDATE folder_items SET item_id=-item_id-1")  # two steps: the primary key must hold
        for rowid, item_type, item_id in rows:
            mapped = (plan.memory_ids if item_type == "memory" else plan.note_ids).get(int(item_id))
            if mapped is None:
                conn.execute("DELETE FROM folder_items WHERE rowid=?", (rowid,))
                changed["folder_items_dropped"] += 1
            else:
                conn.execute("UPDATE folder_items SET item_id=? WHERE rowid=?", (mapped, rowid))
    if plan.chunk_ids:
        ordered = sorted(plan.chunk_ids.items())
        for memory_id, mark in conn.execute(
                "SELECT memory_id,chunk_id FROM memory_checks WHERE stage='absorb' AND chunk_id IS NOT NULL").fetchall():
            new_mark = max([0, *(new for old, new in ordered if old <= int(mark))])
            conn.execute("UPDATE memory_checks SET chunk_id=? WHERE memory_id=? AND stage='absorb'", (new_mark, memory_id))
    else:
        conn.execute("UPDATE memory_checks SET chunk_id=0 WHERE stage='absorb' AND chunk_id IS NOT NULL")
    if "tombstones" in present:
        marks = ",".join("?" * len(OLD_ITEM_TABLES + ("memory_items",)))
        changed["tombstones_dropped"] = conn.execute(f"DELETE FROM tombstones WHERE table_name IN ({marks})",
                                                     OLD_ITEM_TABLES + ("memory_items",)).rowcount
    return changed


def _expected_counts(before: dict[str, int], conn: sqlite3.Connection) -> dict[str, int]:
    """Rows of the tables that keep their rows (renamed ones under the new name)."""
    expected: dict[str, int] = {}
    checks = 0
    for table, count in before.items():
        if table in DROPPED_TABLES or table in OLD_ITEM_TABLES or table in NEW_ITEM_TABLES or table in (
                "tombstones", "folder_items"):
            continue
        if table in dict(CHECK_SOURCES):
            checks += count
            continue
        expected[TABLE_RENAMES.get(table, table)] = count
    if any(table in before for table, _ in CHECK_SOURCES) or "memory_checks" in before:
        expected["memory_checks"] = checks + before.get("memory_checks", 0)
    if "meta" in before:
        keys = {str(row[0]) for row in conn.execute("SELECT key FROM meta")}
        expected["meta_entries"] = before["meta"] - len(keys & set(DROPPED_META_KEYS))
    return expected


def _type_counts(conn: sqlite3.Connection) -> dict[str, int]:
    found = {"memory": 0, "note": 0, "doc": 0, "formula_parts": 0, "other_parts": 0}
    for kind, head, formula, count in conn.execute(
            "SELECT type, title IS NOT NULL, author='formula', COUNT(*) FROM items GROUP BY 1,2,3"):
        if kind == "memory":
            found["memory"] += int(count)
        elif head:
            found[kind] += int(count)
        else:
            found["formula_parts" if formula else "other_parts"] += int(count)
    return found


def _verify(conn: sqlite3.Connection, plan: _Plan, before: dict[str, int], expected: dict[str, int],
            memory_vectors_before: list[int], top_before: dict, query_ids: list[int], revisions: int,
            tombstones_left: int | None, folder_items_left: int | None) -> dict[str, Any]:
    problems = []
    after = table_counts(conn)
    for table, count in expected.items():
        if after.get(table) != count:
            problems.append(f"rows {table}: expected {count}, got {after.get(table)}")
    for table in OLD_ITEM_TABLES:
        if table in after:
            problems.append(f"old table {table} still there")
    types = _type_counts(conn)
    authors = [str(note["author"]) for note in plan.notes]
    want_types = {"memory": len(plan.memories), "doc": authors.count("human"),
                  "note": len(authors) - authors.count("human"), "formula_parts": plan.old_parts["formula"],
                  "other_parts": plan.new_parts["other"]}
    for kind, count in want_types.items():
        if types[kind] != count:
            problems.append(f"items {kind}: expected {count}, got {types[kind]}")
    links = {name: int(count) for name, count in conn.execute(
        "SELECT relation, COUNT(*) FROM item_links GROUP BY relation")}
    for relation, count in plan.expected_links.items():
        if links.get(relation, 0) != count:
            problems.append(f"item_links {relation}: expected {count}, got {links.get(relation, 0)}")
    if after.get("item_revisions", 0) != revisions:
        problems.append(f"item_revisions: expected {revisions}, got {after.get('item_revisions', 0)}")
    if tombstones_left is not None and after.get("tombstones") != tombstones_left:
        problems.append(f"tombstones: expected {tombstones_left}, got {after.get('tombstones')}")
    if folder_items_left is not None and after.get("folder_items") != folder_items_left:
        problems.append(f"folder_items: expected {folder_items_left}, got {after.get('folder_items')}")
    if plan.body_mismatch:
        problems.append(f"bodies not rebuilt from their parts: notes {plan.body_mismatch[:10]}")
    rebuilt = _rebuilt_bodies(conn, plan)
    if rebuilt:
        problems.append(f"document bodies differ after the move: notes {rebuilt[:10]}")
    quick = [str(row[0]) for row in conn.execute("PRAGMA quick_check").fetchall()]
    if quick != ["ok"]:
        problems.append(f"quick_check: {quick[:3]}")
    foreign = conn.execute("PRAGMA foreign_key_check").fetchall()
    if foreign:
        problems.append(f"foreign_key_check: {len(foreign)} rows")
    fts = {}
    present = _tables(conn)
    for table in FTS_TABLES:
        if table not in present:
            continue
        try:
            conn.execute(f"INSERT INTO {table}({table}) VALUES ('integrity-check')")
            fts[table] = "ok"
        except sqlite3.DatabaseError as exc:
            fts[table] = f"{type(exc).__name__}: {exc}"
            problems.append(f"fts {table}: {exc}")
    memory_vectors = _blob_stats([bytes(row[0]) for row in conn.execute(
        "SELECT embedding FROM items WHERE type='memory'")])
    if memory_vectors != memory_vectors_before:
        problems.append(f"memory vectors: before {memory_vectors_before}, after {memory_vectors}")
    formula_vectors = _blob_stats([bytes(row[0]) for row in conn.execute(
        "SELECT embedding FROM items WHERE type='note' AND title IS NULL AND author='formula'")])
    if formula_vectors != plan.formula_vectors_before:
        problems.append(f"automatic-part vectors: before {plan.formula_vectors_before}, after {formula_vectors}")
    top_after = top_ids(conn, query_ids, id_map=plan.memory_ids)
    mapped_before = {key: [plan.memory_ids.get(item, item) for item in value] for key, value in top_before.items()}
    if top_after != mapped_before:
        problems.append(f"top-{TOP_K}: before {mapped_before}, after {top_after}")
    return {"rows_after": after, "items_by_type": types, "items_expected": want_types, "item_links": links,
            "item_links_expected": dict(plan.expected_links), "quick_check": quick[0] if quick else "",
            "foreign_key_check": len(foreign), "fts_integrity": fts,
            "vectors": {"memory": memory_vectors, "formula_parts": formula_vectors}, "top5": top_after,
            "problems": problems}


def _rebuilt_bodies(conn: sqlite3.Connection, plan: _Plan) -> list[int]:
    """Old note ids whose body read back through the new rows differs from the old body ([[mem:N]] mapped)."""
    from .notes import note_body
    differ = []
    for note in plan.notes:
        author = str(note["author"] or "agent")
        want = rewrite_memory_refs(str(note["body"] or ""), plan.memory_ids)
        if author == FORMULA:
            continue  # the body of an automatic note is its parts joined (checked by the part count and vectors)
        if note_body(conn, plan.note_ids[int(note["id"])], author) != want:
            differ.append(int(note["id"]))
    return differ


def _default_splitter(text: str) -> list[tuple[str, str]]:
    from .chunking import split_document_parts
    from .notes import default_token_counter, note_chunk_max_tokens
    counter = default_token_counter()
    return split_document_parts(text, counter.count, max_tokens=note_chunk_max_tokens(), cut_tokens=counter.cut)


def migrate_v22(conn: sqlite3.Connection, *, embedder: Any = None, splitter: Any = None) -> dict[str, Any]:
    """v21-or-older layout -> v22 in one BEGIN IMMEDIATE transaction, verified before COMMIT; on any failed
    check (or error) everything is rolled back and SchemaError is raised. embedder: embeds the re-split parts of
    human/agent notes (None: they stay embedding-pending). splitter: body -> [(heading, part)] (tests).
    Returns the report."""
    from . import store as layout
    check_write_allowed(conn)
    if not needs_v22(conn):
        return {"migrated": False, "reason": "already v22"}
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        before = table_counts(conn)
        filled = [table for table in NEW_ITEM_TABLES if before.get(table)]
        if filled:
            raise SchemaError(f"v22 migration refused: {', '.join(filled)} already hold rows beside the old tables")
        expected = _expected_counts(before, conn)
        _, memory_blobs = _vector_rows(conn)
        memory_vectors_before = _blob_stats(memory_blobs)
        query_ids = fixed_query_ids(conn)
        top_before = top_ids(conn, query_ids)
        plan = _Plan()
        _read_old(conn, plan)
        _base_renames(conn)
        _run_script(conn, layout.ITEMS_DDL + layout.ITEM_LINKS_DDL + layout.ITEM_REVISIONS_DDL)
        _insert_memories(conn, plan)
        _map_note_ids(plan)
        _insert_links(conn, plan)  # first: the old link ids are kept, every new link takes an id after them
        _insert_documents(conn, plan, embedder, splitter or _default_splitter)
        _insert_piece_chains(conn, plan)
        names = _insert_topics(conn, plan)
        _rewrite_manual_marks(conn, names)
        revisions = _insert_revisions(conn, plan)
        tombstones_before = before.get("tombstones")
        folder_before = before.get("folder_items")
        remapped = _remap_references(conn, plan)
        for table in OLD_ITEM_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        present = _tables(conn)
        conn.execute(layout.NEXT_CHAIN_TRIGGER_DDL)
        if "conversation_chunks" in present:
            _run_triggers(conn, layout._CONVERSATION_DDL, "conversation_chunks")
        conn.execute("INSERT INTO items_fts(items_fts) VALUES ('rebuild')")
        if "conversation_chunks_fts" in present:
            conn.execute("INSERT INTO conversation_chunks_fts(conversation_chunks_fts) VALUES ('rebuild')")
        if all(table in present for table in layout.STAMP_TRACKED_TABLES) and "tombstones" in present \
                and all("updated_at" in _columns(conn, table) for table in layout.STAMP_TRACKED_TABLES):
            layout.install_change_triggers(conn)
        conn.execute("INSERT OR REPLACE INTO meta_entries(key, value) VALUES ('schema_version', ?)",
                     (str(SCHEMA_V22),))  # schema_v23.migrate_v23 takes it on to v23
        conn.execute(f"PRAGMA user_version={SCHEMA_V22}")
        conn.execute(f"PRAGMA application_id={APPLICATION_ID}")
        tombstones_left = None if tombstones_before is None else tombstones_before - remapped["tombstones_dropped"]
        folder_left = None if folder_before is None else folder_before - remapped["folder_items_dropped"]
        report = _verify(conn, plan, before, expected, memory_vectors_before, top_before, query_ids, revisions,
                         tombstones_left, folder_left)
        report.update({
            "rows_before": before, "rows_expected": expected, "fixed_queries": query_ids, "top5_before": top_before,
            "id_map": {"notes": {str(old): new for old, new in plan.note_ids.items()}, "memories": "kept"},
            "parts": {"old_chunks": plan.old_parts, "new_parts": plan.new_parts, "embedded": plan.parts_embedded,
                      "reused_vector": plan.parts_reused, "embedding_pending": plan.parts_pending},
            "dropped": {"links": plan.dropped_links, "deleted_topics": plan.dropped_topics,
                        "topic_members": plan.dropped_topic_members, **remapped},
            "renamed_topics": plan.renamed_topics})
        if report["problems"]:
            raise SchemaError("v22 migration checks failed; nothing was changed: " + "; ".join(report["problems"]))
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    report["migrated"] = True
    return report


def mark_v22(conn: sqlite3.Connection) -> None:
    """A database created by v22 code (or already migrated) carries the v22 marks (inside the caller's write)."""
    if _pragma_int(conn, "user_version") != SUPPORTED_USER_VERSION:
        conn.execute(f"PRAGMA user_version={SUPPORTED_USER_VERSION}")
    if _pragma_int(conn, "application_id") != APPLICATION_ID:
        conn.execute(f"PRAGMA application_id={APPLICATION_ID}")


# --- schema-migrate command (explicit, one home) ---------------------------------------------------------------

def _connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _copy_database(source: sqlite3.Connection, target_path: Path | str) -> sqlite3.Connection:
    target = sqlite3.connect(target_path)
    source.backup(target)
    return target


def dry_run(db_path: Path, *, embedder: Any = None) -> dict[str, Any]:
    """The migration on an in-memory copy (the file is opened mode=ro); the report, nothing written."""
    source = _connect_ro(db_path)
    try:
        copy = _copy_database(source, ":memory:")
    finally:
        source.close()
    try:
        try:
            report = migrate_v22(copy, embedder=embedder)
        except SchemaError as exc:
            return {"dry_run": True, "ok": False, "error": str(exc)}
        return {"dry_run": True, "ok": True, **report}
    finally:
        copy.close()


def _snapshot(db_path: Path) -> Path:
    """memory.pre-v22.db through the backup API, checked with quick_check and per-table row counts."""
    target_path = db_path.with_name(SNAPSHOT_NAME)
    if target_path.exists():
        raise SchemaError(f"{target_path.name} already exists; move it away before --apply")
    source = sqlite3.connect(db_path)
    try:
        target = _copy_database(source, target_path)
        try:
            quick = target.execute("PRAGMA quick_check").fetchone()[0]
            if quick != "ok" or table_counts(target) != table_counts(source):
                raise SchemaError(f"backup {target_path.name} check failed (quick_check={quick})")
        finally:
            target.close()
    finally:
        source.close()
    return target_path


def _side_files(home: Path) -> list[tuple[Path, Path]]:
    folder = home / "neospike"
    return [(folder / name, folder / (Path(name).stem + PRE_V22_SUFFIX + Path(name).suffix))
            for name in ("config.json", "MEMORY_RULE.md")]


def apply(home: Path, db_path: Path, *, embedder: Any = None) -> dict[str, Any]:
    """Migration steps 1-6 for one home: checkpoint (busy -> stop), backup + check, config/rule copies, the migration,
    then the config move (config.migrate_config_files)."""
    from .config import migrate_config_files
    probe = sqlite3.connect(db_path)
    try:
        check_write_allowed(probe)
        if not needs_v22(probe):
            return {"applied": False, "reason": "already v22"}
        busy = probe.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if busy is None or int(busy[0]) != 0:
            raise SchemaError("wal_checkpoint busy: stop every neospike process (gateway, dashboard, cron, MCP) first")
    finally:
        probe.close()
    snapshot = _snapshot(db_path)
    copied = copy_side_files(home)
    conn = sqlite3.connect(db_path)
    try:
        report = migrate_v22(conn, embedder=embedder)
    finally:
        conn.close()
    report["config"] = migrate_config_files(home)
    return {"applied": True, "backup": snapshot.name, "copied": copied, **report}


def copy_side_files(home: Path) -> list[str]:
    """Migration step 4: config.json / MEMORY_RULE.md -> their .pre-v22 copies (names of the copies made)."""
    copied = []
    for live, saved in _side_files(home):
        if live.is_file():
            shutil.copy2(live, saved)
            copied.append(saved.name)
    return copied


def home_of_database(db_path: Path) -> Path | None:
    """The Hermes home that owns <home>/neospike/memory.db; None for a database elsewhere (tests, copies)."""
    path = Path(db_path)
    return path.parent.parent if path.name == "memory.db" and path.parent.name == "neospike" else None


def move_config_after_migration(db_path: Path) -> dict[str, Any] | None:
    """Migration steps 4 and 6 for the in-place migration (MemoryStore opening a v21 database), which never ran `apply`:
    the .pre-v22 copies, then config.migrate_config_files. None when the database is not in a Hermes home."""
    from .config import migrate_config_files
    home = home_of_database(db_path)
    if home is None:
        return None
    return {"copied": copy_side_files(home), "config": migrate_config_files(home)}


def restore(home: Path, db_path: Path) -> dict[str, Any]:
    """Put memory.pre-v22.db back as memory.db (the replaced file is kept as memory.v22-replaced-<time>.db) and the
    .pre-v22 copies of config.json / MEMORY_RULE.md back."""
    snapshot = db_path.with_name(SNAPSHOT_NAME)
    if not snapshot.is_file():
        raise SchemaError(f"no {SNAPSHOT_NAME} next to {db_path.name}")
    probe = sqlite3.connect(db_path) if db_path.exists() else None
    if probe is not None:
        try:
            busy = probe.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if busy is None or int(busy[0]) != 0:
                raise SchemaError("wal_checkpoint busy: stop every neospike process first")
        finally:
            probe.close()
    kept = None
    if db_path.exists():
        kept = db_path.with_name(f"memory.v22-replaced-{time.strftime('%Y%m%dT%H%M%S')}.db")
        db_path.replace(kept)
    for suffix in ("-wal", "-shm"):
        side = db_path.with_name(db_path.name + suffix)
        if side.exists():
            side.replace(kept.with_name(kept.name + suffix) if kept else side.with_suffix(side.suffix + ".old"))
    shutil.copy2(snapshot, db_path)
    restored = []
    for live, saved in _side_files(home):
        if saved.is_file():
            shutil.copy2(saved, live)
            restored.append(live.name)
    return {"restored": True, "from": snapshot.name, "replaced_kept_as": kept.name if kept else None,
            "config_restored": restored}
