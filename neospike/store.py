from __future__ import annotations

import hashlib
import heapq
import json
import logging
import math
import operator
import re
import sqlite3
import threading
import time
import array as std_array
import contextlib
import dataclasses
import pathlib
from typing import Iterable, Sequence

from .config import EMBEDDING_MODEL, NeospikeConfig
from .folders import FOLDERS_SCHEMA
from . import schema_v22
from . import schema_v23
from . import vector_index
from . import preferences
from .preferences import PREFERENCES_DDL, PREFERENCES_TABLE

_LOG = logging.getLogger(__name__)


SCHEMA_VERSION = 23  # current layout: schema_v22.py + the preferences table of schema_v23.py

# Namespace a standalone tool (CLI, MCP, eval) uses when nothing else is known.
DEFAULT_NAMESPACE = "default:local"
# meta key: namespace of the last primary provider initialize; written only on change.
META_LIVE_NAMESPACE = "live_owner"  # LEGACY_META_KEYS maps it to the key a v21 database stores
LEGACY_META_KEYS = {META_LIVE_NAMESPACE: "live_namespace"}
AGENT_EDITOR, HUMAN_EDITOR = "agent", "human"  # item_revisions.edited_by of correct / dashboard+CLI
# How widely a memory item is visible (a wider old duplicate lifts the new memory to its scope).
SCOPE_ORDER = {"session": 0, "project": 1, "global": 2}

# Hybrid ranking shared by memories and document parts (documents omit the kind bonus).
# Admission is cosine-only: a shared query word adds no score, so FTS is not
# consulted for ranking. The FTS tables stay for the dashboard's lexical memory filter.
SEMANTIC_WEIGHT, IMPORTANCE_WEIGHT, FRESHNESS_WEIGHT = 0.68, 0.07, 0.05
DURABLE_BOOST = 0.08
# 관련 없는 기억 주입을 막는 코사인 하한(cos 단독 문턱).
# 회상 주입과 MCP mem_search가 함께 쓰는 하나의 상수(mcp_server.MEMORY_SEARCH_COSINE_THRESHOLD가 이 값을 import).
DEFAULT_SEMANTIC_CUTOFF = 0.70
RECENCY_HALF_LIFE_DAYS = 90.0
FTS_TERM_CAP = 16
_FTS_WORD = re.compile(r"[\w-]{2,}", flags=re.UNICODE)


def freshness(now: float, made_ts: float) -> float:
    days_old = max(0.0, (now - made_ts) / 86400)
    return 1.0 / (1.0 + days_old / RECENCY_HALF_LIFE_DAYS)


def blend_score(sem: float, importance: float, recency: float) -> float:
    """Ranking only; admission is the caller's `semantic >= semantic_threshold` gate."""
    return SEMANTIC_WEIGHT * sem + IMPORTANCE_WEIGHT * importance + FRESHNESS_WEIGHT * recency


def build_fts_match(query: str) -> str:
    """OR of quoted terms for FTS5 MATCH; empty when the query has no terms."""
    word_list = _FTS_WORD.findall(query)
    return " OR ".join(f'"{token_str}"' for token_str in word_list[:FTS_TERM_CAP])
# Explicit busy wait for concurrent writers (CLI subprocess, agent, dashboard reads).
SQLITE_BUSY_TIMEOUT_SECONDS = 10.0


_SQLITE_ERRORS = (sqlite3.Error,)
_DUPLICATE_ERRORS = (sqlite3.IntegrityError,)
# Item labels in metadata_json (policy.ITEM_LABELS); a correction that omits one keeps the stored value.
_ITEM_LABEL_KEYS = ("memory_kind", "basis")


def _carried_labels(item: dict, previous: dict) -> dict:
    """memory_kind/basis of item, else of the previous metadata; a label neither has is left out."""
    return {key: item.get(key, previous.get(key)) for key in _ITEM_LABEL_KEYS if key in item or key in previous}


def split_scope(metadata: dict | None) -> tuple[str | None, str]:
    """(scope column, metadata_json without scope) of a memory's metadata dict (v22: scope is a column)."""
    rest = dict(metadata or {})
    scope = rest.pop("scope", None)
    return (scope if isinstance(scope, str) else None), json.dumps(rest)


def memory_metadata(row) -> dict:
    """A memory row's metadata as one dict, the scope column included (the shape callers and revisions keep)."""
    try:
        metadata = json.loads(row["metadata_json"] or "{}")
    except (TypeError, ValueError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    if row["scope"] is not None:
        metadata = {"scope": row["scope"], **metadata}
    return metadata


def revision_metadata(row) -> str:
    """metadata_json a memory revision keeps: the metadata with the scope column folded in."""
    return json.dumps(memory_metadata(row))


_ANY_ABORT = BaseException
_new_rlock = threading.RLock
_BEGIN_WRITE_SQL = "BEGIN IMMEDIATE"

# Column fragments shared by the DDL strings below (the generated SQL text is unchanged).
_PK_COL = "id INTEGER PRIMARY KEY"
_EMBED_COL = "embedding BLOB NOT NULL"
_CREATED_COL = "created_at REAL NOT NULL"
_TEXT_WITH_DEFAULT = "TEXT NOT NULL DEFAULT {}"
_REAL_WITH_DEFAULT = "REAL NOT NULL DEFAULT {}"
_INT_WITH_DEFAULT = "INTEGER NOT NULL DEFAULT {}"
_REAL_NULLABLE = "REAL"


# items (v22): memories, notes and docs in one table. type memory = one short memory (owner =
# namespace, title NULL); type note = a document written by an agent or an automatic (formula) note; type doc = a
# document written by a person. A document is a head row (the document id: title, tags, project, author, rev,
# deleted_at, model in metadata_json; text '' and no vector) followed through item_links relation 'next' by its
# part rows (title NULL; the searchable text and vector; metadata_json heading_path, and for an automatic-note part
# source_json = the absorbed memory it came from). Human/agent body = the parts concatenated (split without
# overlap); automatic-note body = the parts joined by a blank line (FORMULA_CHUNK_SEPARATOR).
ITEM_MEMORY, ITEM_NOTE, ITEM_DOC = "memory", "note", "doc"
ITEM_TYPES = (ITEM_MEMORY, ITEM_NOTE, ITEM_DOC)
DOCUMENT_TYPES = (ITEM_NOTE, ITEM_DOC)
# items.author: who created the document (memories: NULL). FORMULA_AUTHOR marks an automatic note.
FORMULA_AUTHOR = "formula"
NOTE_AUTHOR_NAMES = ("human", "agent", FORMULA_AUTHOR)
# Highest items id ever handed to an automatic-note part; ids above it are new (note-absorb).
FORMULA_HWM_META_KEY = "formula_chunk_hwm"
# metadata_json keys of a part row: its heading trail, and (automatic notes) the memory it was made from.
PART_HEADING_KEY, PART_SOURCE_KEY = "heading_path", "source_json"
_AUTHOR_CHECK_SQL = "CHECK (author IS NULL OR author IN (" + ",".join(f"'{value}'" for value in NOTE_AUTHOR_NAMES) + "))"


def document_type(author: str) -> str:
    """items.type of a document by its creator: a person's document is a doc, every other one a note."""
    return ITEM_DOC if author == "human" else ITEM_NOTE


ITEMS_DDL = f"""
CREATE TABLE IF NOT EXISTS items (
    {_PK_COL},
    type TEXT NOT NULL CHECK (type IN ('{ITEM_MEMORY}','{ITEM_NOTE}','{ITEM_DOC}')),
    owner TEXT NOT NULL DEFAULT '',
    title TEXT,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    project TEXT NOT NULL DEFAULT '',
    scope TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{{}}',
    author TEXT {_AUTHOR_CHECK_SQL},
    origin TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'episodic',
    confidence REAL NOT NULL DEFAULT 1.0,
    importance REAL NOT NULL DEFAULT 0.5,
    rev INTEGER NOT NULL DEFAULT 1,
    {_EMBED_COL},
    {_CREATED_COL},
    updated_at REAL,
    accessed_at REAL,
    access_count INTEGER NOT NULL DEFAULT 0,
    archived_at REAL,
    deleted_at REAL,
    invalid_at REAL,
    superseded_by INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS items_memory_hash ON items(owner, text_hash) WHERE type='{ITEM_MEMORY}';
CREATE UNIQUE INDEX IF NOT EXISTS items_title ON items(title) WHERE title IS NOT NULL;
CREATE INDEX IF NOT EXISTS items_type_owner ON items(type, owner);
CREATE INDEX IF NOT EXISTS items_text_hash ON items(text_hash);
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    title, text, content='items', content_rowid='id', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS items_fts_ai AFTER INSERT ON items BEGIN
    INSERT INTO items_fts(rowid, title, text) VALUES (new.id, COALESCE(new.title, ''), new.text);
END;
CREATE TRIGGER IF NOT EXISTS items_fts_ad AFTER DELETE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, title, text) VALUES ('delete', old.id, COALESCE(old.title, ''), old.text);
END;
CREATE TRIGGER IF NOT EXISTS items_fts_au AFTER UPDATE OF title, text ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, title, text) VALUES ('delete', old.id, COALESCE(old.title, ''), old.text);
    INSERT INTO items_fts(rowid, title, text) VALUES (new.id, COALESCE(new.title, ''), new.text);
END;
CREATE TABLE IF NOT EXISTS meta_entries (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _topic_page_tables():
    """topics.py on first use: it needs numpy, which plain store readers (dashboard) must not load."""
    from . import topics
    return topics


# item_links (v22): every connection between items, with a stable id (supersede-revert --link <id>; the id
# survives backup restore).
#   next       a document row -> the row after it (head -> first part -> ...); a split memory piece -> the next.
#   topic      an item -> the topic named by label (dst_id NULL; a topic has no row of its own).
#   refers     [[Title]] / [[mem:N]] text (dst_id NULL while unresolved; dst_title keeps the target text).
#   supersedes a judged replacement (new memory -> old memory).
RELATION_NEXT, RELATION_TOPIC = "next", "topic"
RELATION_REFERS, RELATION_SUPERSEDES = "refers", "supersedes"
LINK_RELATION_NAMES = (RELATION_NEXT, RELATION_TOPIC, RELATION_REFERS, RELATION_SUPERSEDES)
LINK_CREATOR_VALUES = ("code", "llm", "human")
LINK_ORIGIN_VALUES = ("", "conversation", "automation", "human", "migration")
CREATED_BY_CODE, CREATED_BY_HUMAN = "code", "human"
ORIGIN_CONVERSATION, ORIGIN_MIGRATION, ORIGIN_AUTOMATION = "conversation", "migration", "automation"
EVIDENCE_CHAR_LIMIT = 500
ITEM_LINKS_DDL = (
    "CREATE TABLE IF NOT EXISTS item_links (id INTEGER PRIMARY KEY, "
    "src_id INTEGER NOT NULL, "
    "dst_id INTEGER, "
    "dst_title TEXT, "
    f"relation TEXT NOT NULL CHECK (relation IN ({','.join(repr(v) for v in LINK_RELATION_NAMES)})), "
    "label TEXT NOT NULL DEFAULT '', "
    f"created_by TEXT NOT NULL DEFAULT 'code' CHECK (created_by IN ({','.join(repr(v) for v in LINK_CREATOR_VALUES)})), "
    f"origin TEXT NOT NULL DEFAULT '' CHECK (origin IN ({','.join(repr(v) for v in LINK_ORIGIN_VALUES)})), "
    "session_id TEXT NOT NULL DEFAULT '', "
    "evidence TEXT NOT NULL DEFAULT '', "
    "created_at REAL NOT NULL DEFAULT ((julianday('now') - 2440587.5) * 86400.0), "  # = SQL_NOW_EXPR
    "reverted_at REAL);\n"
    "CREATE INDEX IF NOT EXISTS item_links_src ON item_links(src_id, relation);\n"
    "CREATE INDEX IF NOT EXISTS item_links_dst ON item_links(dst_id, relation);\n"
    "CREATE INDEX IF NOT EXISTS item_links_title ON item_links(relation, dst_title);\n"
    "CREATE INDEX IF NOT EXISTS item_links_label ON item_links(relation, label);\n"
)
# Deleting a row of a next chain re-links its predecessor to its successor (a removed memory piece leaves the
# chain whole); a pointer left without a target is dropped.
NEXT_CHAIN_TRIGGER_DDL = (
    f"CREATE TRIGGER IF NOT EXISTS items_next_ad AFTER DELETE ON items "
    f"WHEN EXISTS (SELECT 1 FROM item_links WHERE relation='{RELATION_NEXT}' AND (src_id=old.id OR dst_id=old.id)) "
    f"BEGIN "
    f"UPDATE item_links SET dst_id=(SELECT n.dst_id FROM item_links n WHERE n.relation='{RELATION_NEXT}' "
    f"AND n.src_id=old.id ORDER BY n.id LIMIT 1) WHERE relation='{RELATION_NEXT}' AND dst_id=old.id; "
    f"DELETE FROM item_links WHERE relation='{RELATION_NEXT}' AND (src_id=old.id OR dst_id IS NULL OR dst_id=src_id); "
    f"END"
)

# item_revisions (v22: the memory and note revision tables of v21 in one): an item's title/text/metadata before an in-place change.
# A document revision keeps its whole body in text and its model in metadata_json.
ITEM_REVISIONS_DDL = """
CREATE TABLE IF NOT EXISTS item_revisions (
    item_id INTEGER NOT NULL,
    rev INTEGER NOT NULL,
    title TEXT,
    text TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    edited_by TEXT NOT NULL,
    edited_at REAL NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(item_id, rev)
);
"""

# SQL building blocks over items ---------------------------------------------------------------------------------
def memory_sql(alias: str = "") -> str:
    """SQL condition: a memory row."""
    prefix = f"{alias}." if alias else ""
    return f"{prefix}type='{ITEM_MEMORY}'"


DOC_TYPES_SQL = f"('{ITEM_NOTE}','{ITEM_DOC}')"
# Head rows of notes and docs (one per document; the document id).
NOTE_HEAD_SQL = f"type IN {DOC_TYPES_SQL} AND title IS NOT NULL"


def note_head_sql(alias: str) -> str:
    """NOTE_HEAD_SQL on a table alias."""
    return f"{alias}.type IN {DOC_TYPES_SQL} AND {alias}.title IS NOT NULL"


# Topics (relation 'topic', identified by label = name). Creation order = the id of each topic's oldest link. The
# handle is the integer the dashboard and the map use for a topic (its points, panel and drop targets): a hash of the
# name, so it stays the same whatever members come and go, and nothing is stored for it. The name is the identity
# everything else uses.
TOPIC_ORDER_SQL = f"SELECT label, MIN(id) FROM item_links WHERE relation='{RELATION_TOPIC}' GROUP BY label"
TOPIC_HANDLE_HEX_DIGITS = 13  # 52 bits: an exact integer in JavaScript (Number.MAX_SAFE_INTEGER is 2**53 - 1)


def topic_handle(name: str) -> int:
    """The dashboard/map integer of a topic name."""
    return int(hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:TOPIC_HANDLE_HEX_DIGITS], 16)


def topic_order(sql_conn: sqlite3.Connection) -> dict[str, int]:
    """topic name -> the id of its oldest link (creation order) of every topic."""
    return {str(row[0]): int(row[1]) for row in sql_conn.execute(TOPIC_ORDER_SQL)}


def topic_handles(sql_conn: sqlite3.Connection) -> dict[str, int]:
    """topic name -> handle of every topic, in creation order (numpy-free; the dashboard reads it)."""
    order = topic_order(sql_conn)
    return {name: topic_handle(name) for name in sorted(order, key=order.__getitem__)}


def topic_name_of(sql_conn: sqlite3.Connection, handle: int) -> str | None:
    """The name of the topic with this handle (None when no topic has it)."""
    for name, value in topic_handles(sql_conn).items():
        if value == int(handle):
            return name
    return None


# A refers link to a memory: its target text is mem:N ([[mem:N]]); every other refers link names a document title.
LINK_TO_MEMORY_SQL = "{alias}.dst_title GLOB 'mem:[0-9]*'"
# The old notes row shape without the body (notes.note_body gives it).
NOTE_COLUMNS_SQL = ("id,type,title,tags_json,project,author,created_at,updated_at,rev,deleted_at,accessed_at,"
                    "access_count,COALESCE(json_extract(metadata_json,'$.model'),'') AS model")
NOTES_SQL = f"(SELECT {NOTE_COLUMNS_SQL} FROM items WHERE {NOTE_HEAD_SQL})"
# Part rows of every document: (id, note_id, ord, heading_path, text, text_hash, embedding, source_json), ord =
# position along the next chain from the head (0 = first part).
NOTE_CHUNKS_SQL = (
    "(WITH RECURSIVE parts(note_id, id, ord) AS ("
    f"SELECT h.id, l.dst_id, 0 FROM items h JOIN item_links l ON l.relation='{RELATION_NEXT}' AND l.src_id=h.id "
    f"WHERE h.type IN {DOC_TYPES_SQL} AND h.title IS NOT NULL "
    f"UNION ALL SELECT p.note_id, l.dst_id, p.ord+1 FROM parts p JOIN item_links l ON l.relation='{RELATION_NEXT}' "
    "AND l.src_id=p.id WHERE p.ord<1000000) "
    "SELECT r.id AS id, p.note_id AS note_id, p.ord AS ord, "
    "COALESCE(json_extract(r.metadata_json,'$.heading_path'),'') AS heading_path, r.text AS text, "
    "r.text_hash AS text_hash, r.embedding AS embedding, json_extract(r.metadata_json,'$.source_json') AS source_json "
    "FROM parts p JOIN items r ON r.id=p.id)"
)

# memory_pairs (logic in topics.py): neighbour pairs between live memories (topic or not) with their cosine;
# not item_links (it holds only drawn connections). Joining a topic keeps them; a deleted memory loses them in
# _drop_memory_links / link-prune.
_DISCOVERY_DDL = f"""
CREATE TABLE IF NOT EXISTS memory_pairs (
    low_id INTEGER NOT NULL,
    high_id INTEGER NOT NULL,
    cosine REAL NOT NULL,
    {_CREATED_COL},
    PRIMARY KEY(low_id, high_id)
);
CREATE INDEX IF NOT EXISTS memory_pairs_high ON memory_pairs(high_id);
"""

# memory_checks (v22: topic_discovery_checked + note_absorb_checked, logic in topics.py): one row per memory and
# stage. stage 'topic' (topic-assign): the text_hash the neighbour check was made with (NULL = checked again) and
# checked_at. stage 'absorb' (note-absorb): the text_hash compared and the highest automatic-note part id compared
# against (chunk_id). Another body (hash) checks again; a deleted memory loses its rows.
CHECK_STAGE_TOPIC, CHECK_STAGE_ABSORB = "topic", "absorb"
_CHECKS_DDL = f"""
CREATE TABLE IF NOT EXISTS memory_checks (
    memory_id INTEGER NOT NULL,
    stage TEXT NOT NULL CHECK (stage IN ('{CHECK_STAGE_TOPIC}','{CHECK_STAGE_ABSORB}')),
    text_hash TEXT,
    chunk_id INTEGER,
    checked_at REAL,
    PRIMARY KEY(memory_id, stage)
);
"""

# dangling connection rows (prune_orphan_links) --------------------------------------------
_ITEM_GONE = "NOT EXISTS (SELECT 1 FROM items i WHERE i.id={id})"
_DOC_TRASHED = (f"EXISTS (SELECT 1 FROM items i WHERE i.id={{id}} AND i.type IN {DOC_TYPES_SQL} "
                f"AND i.deleted_at IS NOT NULL)")
# Gone source: the row does not exist, or a trashed document (a restore re-links its outgoing links).
_LINK_FROM_MISSING = f"({_ITEM_GONE.format(id='item_links.src_id')} OR {_DOC_TRASHED.format(id='item_links.src_id')})"
_LINK_TO_MISSING = f"(dst_id IS NOT NULL AND {_ITEM_GONE.format(id='item_links.dst_id')})"
_REFERS_OR_SUPERSEDES = f"relation IN ('{RELATION_REFERS}','{RELATION_SUPERSEDES}')"
_ORPHAN_REFERS_SQL = (f"UPDATE item_links SET dst_id=NULL WHERE relation=? AND {_LINK_TO_MISSING} "
                      f"AND NOT {_LINK_FROM_MISSING}")
_ORPHAN_LINKS_SQL = f"DELETE FROM item_links WHERE {_REFERS_OR_SUPERSEDES} AND ({_LINK_FROM_MISSING} OR {_LINK_TO_MISSING})"
_ORPHAN_TOPIC_LINKS_SQL = (f"DELETE FROM item_links WHERE relation='{RELATION_TOPIC}' AND "
                           f"{_ITEM_GONE.format(id='item_links.src_id')}")
_ORPHAN_NEXT_SQL = (f"DELETE FROM item_links WHERE relation='{RELATION_NEXT}' AND ({_ITEM_GONE.format(id='item_links.src_id')} "
                    f"OR dst_id IS NULL OR {_ITEM_GONE.format(id='item_links.dst_id')} OR dst_id=src_id)")
_ORPHAN_FOLDER_ITEMS_SQL = f"DELETE FROM folder_items WHERE {_ITEM_GONE.format(id='folder_items.item_id')}"
_ORPHAN_DISCOVERY_LINKS_SQL = (
    f"DELETE FROM memory_pairs WHERE {_ITEM_GONE.format(id='memory_pairs.low_id')} "
    f"OR {_ITEM_GONE.format(id='memory_pairs.high_id')}")
_ORPHAN_CHECKS_SQL = f"DELETE FROM memory_checks WHERE {_ITEM_GONE.format(id='memory_checks.memory_id')}"
_ORPHAN_REVISIONS_SQL = f"DELETE FROM item_revisions WHERE {_ITEM_GONE.format(id='item_revisions.item_id')}"

# Per Hermes session, the hash (config.criteria_text_digest) of the memory creation rules that the
# conversation was told, in its system prompt or in a recall notice. No row = nothing recorded for the conversation.
# pending_marker: the marker of a notice returned by prefetch whose delivery (state.db api_content) is not yet
# confirmed; NULL = confirmed (or told by the system prompt).
_SESSION_CRITERIA_DDL = """
CREATE TABLE IF NOT EXISTS session_save_criteria (
    session_id TEXT PRIMARY KEY,
    criteria_hash TEXT NOT NULL,
    recorded_at REAL NOT NULL,
    pending_marker TEXT
);
"""


# conversation transcripts (v6): copied from Hermes turns, masked before INSERT.
# conversation_chunks.embedding is NULL until the background indexer fills it (conversations.py).
_CONVERSATION_DDL = f"""
CREATE TABLE IF NOT EXISTS conversations (
    {_PK_COL},
    session_id TEXT NOT NULL UNIQUE,
    started_at REAL NOT NULL,
    ended_at REAL,
    last_message_at REAL NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    summary_note_id INTEGER,
    indexed_seq INTEGER NOT NULL DEFAULT 0,
    updated_at REAL,
    synced_through REAL,
    source_max_id INTEGER
);
CREATE TABLE IF NOT EXISTS conversation_messages (
    {_PK_COL},
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    seq INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user','assistant')),
    content TEXT NOT NULL,
    {_CREATED_COL},
    masked_count INTEGER NOT NULL DEFAULT 0,
    updated_at REAL,
    UNIQUE(conversation_id, seq)
);
CREATE TABLE IF NOT EXISTS conversation_chunks (
    {_PK_COL},
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    seq_start INTEGER NOT NULL,
    seq_end INTEGER NOT NULL,
    part INTEGER NOT NULL DEFAULT 0,
    text TEXT NOT NULL,
    prefix_chars INTEGER NOT NULL DEFAULT 0,
    text_hash TEXT NOT NULL,
    embedding BLOB,
    updated_at REAL,
    next_id INTEGER,
    UNIQUE(conversation_id, seq_start, part)
);
CREATE INDEX IF NOT EXISTS conversation_chunks_hash ON conversation_chunks(text_hash);
CREATE INDEX IF NOT EXISTS conversation_chunks_pending ON conversation_chunks(id) WHERE embedding IS NULL;
CREATE VIRTUAL TABLE IF NOT EXISTS conversation_chunks_fts USING fts5(
    text, content='conversation_chunks', content_rowid='id', tokenize='unicode61'
);
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
CONVERSATION_TABLE_NAMES = ("conversations", "conversation_messages", "conversation_chunks")

# change tracking (v5) for incremental backups (backup.py). Triggers, not call sites, so every
# write path is covered. SQL time is taken while the writer holds the write lock; backup.py reads
# its watermark under BEGIN IMMEDIATE, so a commit is either before it or strictly after it.
# REPLACE conflict deletes do not fire delete triggers (recursive_triggers is off): delete rows
# explicitly instead.
SQL_NOW_EXPR = "((julianday('now') - 2440587.5) * 86400.0)"
TABLES_WITH_UPDATE_STAMP = ("items",)
# Tables of the removed review inbox: a v20 database still has them; _run_migrations drops them with their tombstones.
RETIRED_TABLES = ("candidates", "inbox_decisions", "lint_state")
# Tables whose updated_at the triggers maintain: the v5 table plus the v6 conversation family
# (created with the column, so the v5 ALTER/backfill step does not apply to them).
# v7 folders is created with updated_at as well; folder_items is hash-replaced by backup.py.
# v23 preferences (preferences.py) as well: edited in place, so updated_at is its change signal.
STAMP_TRACKED_TABLES = TABLES_WITH_UPDATE_STAMP + CONVERSATION_TABLE_NAMES + ("folders", PREFERENCES_TABLE)
# Read bookkeeping (a search hit) is not a change; an UPDATE of only these columns leaves
# updated_at alone on every tracked table that has them.
READ_TRACKING_COLUMNS = ("accessed_at", "access_count")

# Embedding pending: memories.embedding is NOT NULL, so a memory saved while the embedder failed keeps an
# empty BLOB until the daily embed-pending stage fills it. Every vector reader skips it (PENDING / HAS_VECTOR SQL);
# FTS still lists it.
PENDING_EMBED_MARK = b""


def normalize_rows(vals: Iterable[float]) -> std_array.array:
    """float32 array of values scaled once, at write time, to unit L2 norm, so every reader's
    similarity is a plain dot product. A zero or non-finite norm leaves the values as they are (callers
    keep rejecting zero / non-finite vectors). The norm is taken of the float32 values, so a vector that
    underflows to zero or overflows to inf in float32 stays invalid."""
    vec = std_array.array("f", vals)
    unit_len = math.hypot(*vec)  # Python 3.11 has no math.sumprod (requires-python >= 3.11)
    if unit_len > 0 and math.isfinite(unit_len):
        vec = std_array.array("f", [value / unit_len for value in vec])
    return vec


F32_WIDTH = 4


def dot_similarities(query: Sequence[float], raw_blobs: Sequence[bytes | None]) -> list[float]:
    """MemoryStore._cosine of query against every float32 blob, as one numpy matrix product.
    Unit vectors, so the score is the dot product, computed in float64 like _cosine. A blob
    whose width differs from the query (embedding-pending empty BLOB, NULL, a stale width) scores 0.0, as
    _cosine does. NaN / inf values give a non-finite score, as before; callers keep their isfinite checks."""
    if not raw_blobs:
        return []
    import numpy as np
    vec = np.asarray(query, dtype=np.float64)
    vec_width = vec.size * F32_WIDTH
    retained = [index for index, raw_blob in enumerate(raw_blobs) if raw_blob is not None and len(raw_blob) == vec_width]
    if not retained or not vec_width:
        return [0.0] * len(raw_blobs)
    if len(retained) == len(raw_blobs):
        merged = b"".join(raw_blobs)
    else:
        merged = b"".join(raw_blobs[index] for index in retained)
    mat = np.frombuffer(merged, dtype=np.float32).reshape(len(retained), vec.size)
    with np.errstate(invalid="ignore", over="ignore"):
        dots = mat.astype(np.float64) @ vec
    if len(retained) == len(raw_blobs):
        return dots.tolist()
    sims = [0.0] * len(raw_blobs)
    for index, value in zip(retained, dots.tolist()):
        sims[index] = value
    return sims


def is_embed_pending(raw_blob: bytes | None) -> bool:
    """True for the empty BLOB of a memory whose embedding the daily batch fills later."""
    return not raw_blob


def memory_time_expr(alias_name: str = "m") -> str:
    """SQL value: when a memory's body was last written = max(created_at, its last item_revisions.edited_at)
    (updated_at moves only on other changes)."""
    return (f"MAX({alias_name}.created_at,COALESCE((SELECT MAX(r.edited_at) FROM item_revisions r "
            f"WHERE r.item_id={alias_name}.id),{alias_name}.created_at))")


def vector_present_sql(col: str = "embedding") -> str:
    """SQL condition: the row has a stored vector (not an embedding-pending memory)."""
    return f"length({col})>0"

# Tombstone row_key per table: the primary key, or the natural key when there is none.
TOMBSTONE_FIELDS = {
    "items": "CAST(old.id AS TEXT)",
    "item_revisions": "json_array(old.item_id, old.rev)",
    "item_links": "CAST(old.id AS TEXT)",
    "conversations": "CAST(old.id AS TEXT)",
    "conversation_messages": "CAST(old.id AS TEXT)",
    "conversation_chunks": "CAST(old.id AS TEXT)",
    "folders": "CAST(old.id AS TEXT)",
    "folder_items": "json_array(old.item_type, old.item_id)",
    PREFERENCES_TABLE: "CAST(old.id AS TEXT)",
}
TOMBSTONES_DDL = """
CREATE TABLE IF NOT EXISTS tombstones (
    table_name TEXT NOT NULL,
    row_key TEXT NOT NULL,
    deleted_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS tombstones_deleted_at ON tombstones(deleted_at);
"""


# Chunk chain (v10): the parts of one message split over the chunk budget point forward, part
# k -> part k+1; the last part, and every chunk of an unsplit message, has next_chunk_id NULL.
CHUNK_CHAIN_FIELD = "next_id"  # v22: was next_chunk_id
# Batch start time T up to which conv-import matched the conversation to state.db (NULL = never).
SYNCED_UPTO_FIELD = "synced_through"
# Largest state.db messages.id matched; a larger id is a late message even with an old timestamp.
SOURCE_LAST_ID_FIELD = "source_max_id"

# Piece chain: the pieces of one split memory text (neospike_memory_update, neospike_memory_reconcile,
# dashboard and CLI edits) point forward through item_links relation 'next', piece k -> piece k+1, like
# document parts. Deleting a piece re-links its predecessor to its successor (NEXT_CHAIN_TRIGGER_DDL);
# prune_orphan_links drops a pointer whose end is gone (a restore without the trigger).


def next_of(sql_conn: sqlite3.Connection, item_id: int) -> int | None:
    """The item after item_id along its next chain (None at the end)."""
    rec = sql_conn.execute(f"SELECT dst_id FROM item_links WHERE relation='{RELATION_NEXT}' AND src_id=? "
                           f"ORDER BY id LIMIT 1", (int(item_id),)).fetchone()
    return None if rec is None or rec[0] is None else int(rec[0])


def write_chain(sql_conn: sqlite3.Connection, ids: Sequence[int], *, now: float | None = None) -> None:
    """Inside the caller's write transaction: the next links ids[0] -> ids[1] -> ... (the old next links of
    these rows are replaced)."""
    order = [int(item_id) for item_id in ids]
    if order:
        marks = ",".join("?" * len(order))
        sql_conn.execute(f"DELETE FROM item_links WHERE relation='{RELATION_NEXT}' AND src_id IN ({marks})", order)
    stamp = time.time() if now is None else now
    sql_conn.executemany(
        f"INSERT INTO item_links(src_id,dst_id,relation,created_at) VALUES (?,?,'{RELATION_NEXT}',?)",
        [(low, high, stamp) for low, high in zip(order, order[1:])])


def link_piece_chain(sql_conn: sqlite3.Connection, head_id: int, part_ids: Sequence[int]) -> None:
    """Inside the caller's write transaction: first_id -> piece_ids[0] -> ... -> piece_ids[-1] -> the item
    first_id pointed to before (a re-split piece stays inside its chain). No-op without pieces."""
    if not part_ids:
        return
    rest_tail = next_of(sql_conn, head_id)
    order = [int(head_id), *(int(fragment) for fragment in part_ids)]
    if rest_tail is not None and rest_tail not in order:
        order.append(rest_tail)
    write_chain(sql_conn, order)


def chain_message_chunks(sql_conn: sqlite3.Connection, conv_id: int | None = None, seq_begin: int = 0) -> int:
    """Re-link next_chunk_id of the chunks with seq_start >= seq_from (of one conversation, or of
    all when conversation_id is None). Runs in the caller's transaction; only changed rows are
    written. Returns the number of rows updated."""
    where, args = ("conversation_id=? AND seq_start>=?", (conv_id, seq_begin)) if conv_id is not None         else ("seq_start>=?", (seq_begin,))
    recs = sql_conn.execute(
        f"SELECT id,conversation_id,seq_start,seq_end,part,{CHUNK_CHAIN_FIELD} FROM conversation_chunks "
        f"WHERE {where} ORDER BY conversation_id,seq_start,part", args,
    ).fetchall()
    pending_sets: list[tuple[int | None, int]] = []
    for index, rec in enumerate(recs):
        successors = recs[index + 1] if index + 1 < len(recs) else None
        desired = None
        if (successors is not None and rec[2] == rec[3] and successors[1] == rec[1] and successors[2] == rec[2]
                and successors[3] == rec[3] and successors[4] == rec[4] + 1):
            desired = int(successors[0])
        if rec[5] != desired:
            pending_sets.append((desired, int(rec[0])))
    sql_conn.executemany(f"UPDATE conversation_chunks SET {CHUNK_CHAIN_FIELD}=? WHERE id=?", pending_sets)
    return len(pending_sets)

def _update_stamp_trigger_sql(tbl: str, cols: Sequence[str] = ()) -> str:
    """{table}_track_au: bump updated_at when the statement did not set it itself. columns: the columns
    whose UPDATE counts as a change (UPDATE OF ...); empty = any UPDATE (tables without read bookkeeping)."""
    of = f" OF {','.join(cols)}" if cols else ""
    return (f"CREATE TRIGGER IF NOT EXISTS {tbl}_track_au AFTER UPDATE{of} ON {tbl} "
            f"WHEN new.updated_at IS old.updated_at BEGIN "
            f"UPDATE {tbl} SET updated_at={SQL_NOW_EXPR} WHERE id=new.id; END")


def _change_trigger_sql() -> dict[str, str]:
    """name -> CREATE TRIGGER statement of every change-tracking trigger (the update trigger of a table with
    read bookkeeping columns is rebuilt per database in create_tracking_triggers)."""
    trigger_list: dict[str, str] = {}
    for tbl in STAMP_TRACKED_TABLES:
        trigger_list[f"{tbl}_track_ai"] = (
            f"CREATE TRIGGER IF NOT EXISTS {tbl}_track_ai AFTER INSERT ON {tbl} WHEN new.updated_at IS NULL BEGIN "
            f"UPDATE {tbl} SET updated_at={SQL_NOW_EXPR} WHERE id=new.id; END"
        )
        trigger_list[f"{tbl}_track_au"] = _update_stamp_trigger_sql(tbl)
    for tbl, key in TOMBSTONE_FIELDS.items():
        trigger_list[f"{tbl}_track_ad"] = (
            f"CREATE TRIGGER IF NOT EXISTS {tbl}_track_ad AFTER DELETE ON {tbl} BEGIN "
            f"INSERT INTO tombstones(table_name,row_key,deleted_at) VALUES ('{tbl}', {key}, {SQL_NOW_EXPR}); END"
        )
    return trigger_list


CHANGE_TRIGGER_SPECS = _change_trigger_sql()


def _content_column_names(sql_conn: sqlite3.Connection, tbl: str) -> list[str]:
    """Columns of table whose UPDATE is a change: all but updated_at and READ_BOOKKEEPING_COLUMNS; [] when the
    table has no read bookkeeping column (then every UPDATE is a change)."""
    cols = [str(rec[1]) for rec in sql_conn.execute(f"PRAGMA table_info({tbl})")]
    if not set(cols) & set(READ_TRACKING_COLUMNS):
        return []
    return [name for name in cols if name not in READ_TRACKING_COLUMNS and name != "updated_at"]


def install_change_triggers(sql_conn: sqlite3.Connection) -> None:
    """Create the tracking triggers. On a table with read bookkeeping (memories: a search hit
    sets accessed_at/access_count) the update trigger fires only for an UPDATE of another column, so a recall
    leaves updated_at alone. An existing trigger with other SQL (or a table that gained a
    column) is replaced; values already stored are not changed."""
    for name, stmt in CHANGE_TRIGGER_SPECS.items():
        tbl = name[:-len("_track_au")] if name.endswith("_track_au") else None
        if tbl is not None:
            stmt = _update_stamp_trigger_sql(tbl, _content_column_names(sql_conn, tbl))
            cur_row = sql_conn.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?",
                                         (name,)).fetchone()
            desired = stmt.replace("IF NOT EXISTS ", "", 1)
            if cur_row is not None and cur_row[0] != desired:
                sql_conn.execute(f"DROP TRIGGER {name}")
        sql_conn.execute(stmt)


def remove_change_triggers(sql_conn: sqlite3.Connection) -> None:
    """Restore applies backed-up rows verbatim; tracking must not rewrite them."""
    for name in CHANGE_TRIGGER_SPECS:
        sql_conn.execute(f"DROP TRIGGER IF EXISTS {name}")


def memory_db_path(hermes_home: pathlib.Path) -> pathlib.Path:
    """The memory DB file of a home without creating folders or migrating (read-only callers)."""
    return pathlib.Path(hermes_home).expanduser() / "neospike" / "memory.db"


def peek_meta(dbfile: pathlib.Path, key: str) -> str | None:
    """One meta value through a mode=ro connection; None when the file, table, or key is missing."""
    dbfile = pathlib.Path(dbfile)
    if not dbfile.is_file():
        return None
    try:
        sql_conn = sqlite3.connect(f"{dbfile.resolve().as_uri()}?mode=ro", uri=True, timeout=SQLITE_BUSY_TIMEOUT_SECONDS)
    except _SQLITE_ERRORS:
        return None
    rec = None
    try:
        # v22 names first; a v21-or-older file not migrated yet still answers (table meta, LEGACY_META_KEYS).
        for table, wanted in (("meta_entries", key), ("meta", LEGACY_META_KEYS.get(key, key))):
            try:
                rec = sql_conn.execute(f"SELECT value FROM {table} WHERE key=?", (wanted,)).fetchone()
            except _SQLITE_ERRORS:
                continue
            if rec is not None:
                break
    finally:
        sql_conn.close()
    return None if rec is None else str(rec[0])


def pick_namespace(declared: str | None, dbfile: pathlib.Path) -> str:
    """Explicit namespace, else the live provider namespace recorded in meta, else DEFAULT_NAMESPACE."""
    if declared:
        return declared
    return peek_meta(dbfile, META_LIVE_NAMESPACE) or DEFAULT_NAMESPACE


@dataclasses.dataclass(frozen=True)
class SearchResult:
    id: int
    text: str
    origin: str
    score: float
    kind: str = dataclasses.field(default="episodic")
    confidence: float = dataclasses.field(default=1.0)
    superseded_by: int | None = None  # history mode: this row is invalid, replaced by that id
    replaced_id: int | None = None  # replace mode: this active row took the slot of that invalid id
    archived: bool | None = None  # direct search: True for an archived row

    def as_dict(self) -> dict:
        """Result fields for JSON; the optional history fields only when set."""
        data = dict(self.__dict__)
        for name in _EXTRA_RESULT_FIELDS:
            if data[name] is None:
                del data[name]
        return data


_EXTRA_RESULT_FIELDS = ("superseded_by", "replaced_id", "archived")


class SearchResultList(list):
    """search() results; `superseded` holds the invalid hits that lost their slot (replace mode),
    for the recall log. A plain list to every caller that does not look."""

    def __init__(self, items: Iterable[SearchResult] = (), old_gen: Iterable[SearchResult] = ()) -> None:
        super().__init__(items)
        self.superseded = list(old_gen)

# Search treatment of invalid (superseded) memories.
HISTORY_OMIT = "exclude"  # direct search default: invalid rows are left out
HISTORY_SHOW = "include"  # include_history=true: invalid rows scored x OLD_VERSION_SCORE_FACTOR
HISTORY_COLLAPSE = "replace"  # prefetch: an invalid hit's slot goes to all of its active replacers
HISTORY_CHOICES = (HISTORY_OMIT, HISTORY_SHOW, HISTORY_COLLAPSE)
OLD_VERSION_SCORE_FACTOR = 0.5
SUPERSEDE_CHAIN_LIMIT = 256  # chain walk bound; a longer chain is treated as broken


def read_embedding_meta(sql_conn: sqlite3.Connection) -> tuple[int | None, str | None]:
    """(dimensions, model) recorded in meta; None where a key is absent."""
    recs = dict(sql_conn.execute(
        "SELECT key, value FROM meta_entries WHERE key IN ('embedding_dimensions', 'embedding_model')").fetchall())
    dimensions = recs.get("embedding_dimensions")
    return (int(dimensions) if dimensions is not None else None), recs.get("embedding_model")


def store_embedding_meta(sql_conn: sqlite3.Connection, dimensions: int) -> None:
    sql_conn.execute("INSERT OR REPLACE INTO meta_entries(key, value) VALUES ('embedding_dimensions', ?)", (str(dimensions),))
    sql_conn.execute("INSERT OR REPLACE INTO meta_entries(key, value) VALUES ('embedding_model', ?)", (EMBEDDING_MODEL,))


def _rank_sort_key(item: SearchResult) -> float:
    return item.score


def _neighbor_rank(duo: tuple[int, float]) -> float:
    return duo[1]


class MemoryStore:
    def __init__(self, fpath: pathlib.Path, *, dimensions: int, semantic_threshold: float = DEFAULT_SEMANTIC_CUTOFF,
                 verify_embedding: bool = True) -> None:
        # A similarity floor for admission, separate from the lexical/metadata ranking bonuses: neither a
        # probability nor proof that a document answers. Default: DEFAULT_SEMANTIC_CUTOFF (user decision).
        threshold_ok = (
            not isinstance(semantic_threshold, bool)
            and math.isfinite(semantic_threshold)
            and 0 < semantic_threshold <= 1
        )
        if not threshold_ok:
            raise ValueError("semantic_threshold must be finite " "and in (0, 1]")
        self.path = pathlib.Path(fpath)
        self.semantic_threshold, self.dimensions = semantic_threshold, dimensions
        # False only for `reembed`, which opens a database of the old model to rewrite its vectors.
        self.verify_embedding = verify_embedding
        self.path.parent.mkdir(exist_ok=True, parents=True)
        self._snapshot_before_migrate()
        self._lock = _new_rlock()
        self._conn = sqlite3.connect(
            self.path, timeout=SQLITE_BUSY_TIMEOUT_SECONDS, check_same_thread=False
        )
        row_kind = sqlite3.Row
        self._conn.row_factory = row_kind
        try:
            # Refuse a newer schema; a v21-or-older layout is renamed in one verified transaction.
            schema_v22.check_write_allowed(self._conn)
            if schema_v22.needs_v22(self._conn):
                schema_v22.migrate_v22(self._conn)
                self._move_config_after_migration()
            # v22 -> v23: preferences/constraints leave items for their own table (one verified transaction).
            if schema_v23.needs_v23(self._conn):
                schema_v23.migrate_v23(self._conn)
        except _ANY_ABORT:
            self._conn.close()
            raise
        self._run_migrations()

    def _move_config_after_migration(self) -> None:
        """The config move `schema_v22.apply` does (old config keys, v21 tool names and the unedited previous default
        in MEMORY_RULE.md), also for this in-place migration. The database is already migrated and committed, so a
        failure here is logged and the store still opens; the move is re-runnable (`neospike.cli schema-migrate`)."""
        try:
            report = schema_v22.move_config_after_migration(self.path)
        except (OSError, ValueError) as exc:
            _LOG.warning("neospike: config move after the v22 migration failed: %s: %s", type(exc).__name__, exc)
            return
        if report is not None:
            _LOG.info("neospike: config move after the v22 migration: %s", report)

    def _snapshot_before_migrate(self) -> None:
        if not (self.path.exists() and self.path.stat().st_size):
            return
        source = sqlite3.connect(str(self.path))
        try:
            names = {str(rec[0]) for rec in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            tbl = next((name for name in ("items", "memories") if name in names), None)
            cols = {rec[1] for rec in source.execute(f"PRAGMA table_info({tbl})")} if tbl else set()
            bak_path = self.path.with_name("memory.pre-v2.db")
            if tbl and "kind" not in cols and not bak_path.exists():
                self._snapshot_into(source, bak_path)
            # Each version step (v2 -> v3 notes family, v3 -> v4, ...) snapshots the state
            # before the next version once, named after that next version.
            meta_map = next((name for name in ("meta_entries", "meta") if name in names), None)
            schema_ver = source.execute(f"SELECT value FROM {meta_map} WHERE key='schema_version'").fetchone() if meta_map else None
            if tbl and schema_ver is not None and int(schema_ver[0]) < SCHEMA_VERSION:
                step_snapshot = self.path.with_name(f"memory.pre-v{int(schema_ver[0]) + 1}.db")
                if not step_snapshot.exists():
                    self._snapshot_into(source, step_snapshot)
        finally:
            source.close()

    @staticmethod
    def _snapshot_into(source: sqlite3.Connection, bak_path: pathlib.Path) -> None:
        target = sqlite3.connect(bak_path)
        try:
            source.backup(target, pages=-1)
        finally:
            target.close()

    def _run_migrations(self) -> None:
        with self._write_scope():
            self._conn.executescript(
                "PRAGMA journal_mode=WAL;\n"
                + ITEMS_DDL
                + ITEM_LINKS_DDL
                + ITEM_REVISIONS_DDL
                + _CONVERSATION_DDL
                + FOLDERS_SCHEMA
                + _DISCOVERY_DDL
                + _CHECKS_DDL
                + _SESSION_CRITERIA_DDL
                + PREFERENCES_DDL
            )
            preferences.drop_origin_column(self._conn)
            self._conn.execute(NEXT_CHAIN_TRIGGER_DDL)
            # v10: forward chain over the parts of one split message (filled for existing rows).
            chunk_columns = {rec[1] for rec in self._conn.execute("PRAGMA table_info(conversation_chunks)")}
            if CHUNK_CHAIN_FIELD not in chunk_columns:
                self._conn.execute(f"ALTER TABLE conversation_chunks ADD COLUMN {CHUNK_CHAIN_FIELD} INTEGER")
                chain_message_chunks(self._conn)
            # NULL on existing rows, so the next conv-import matches every conversation once.
            if SYNCED_UPTO_FIELD not in {rec[1] for rec in self._conn.execute("PRAGMA table_info(conversations)")}:
                self._conn.execute(f"ALTER TABLE conversations ADD COLUMN {SYNCED_UPTO_FIELD} REAL")
            if SOURCE_LAST_ID_FIELD not in {rec[1] for rec in self._conn.execute("PRAGMA table_info(conversations)")}:
                self._conn.execute(f"ALTER TABLE conversations ADD COLUMN {SOURCE_LAST_ID_FIELD} INTEGER")
            # v5: updated_at (existing rows take created_at) before the tracking triggers exist.
            for tbl in TABLES_WITH_UPDATE_STAMP:
                if "updated_at" not in {rec[1] for rec in self._conn.execute(f"PRAGMA table_info({tbl})")}:
                    self._conn.execute(f"ALTER TABLE {tbl} ADD COLUMN updated_at REAL")
                self._conn.execute(f"UPDATE {tbl} SET updated_at=created_at WHERE updated_at IS NULL")
            for stmt in TOMBSTONES_DDL.split(";"):
                if stmt.strip():
                    self._conn.execute(stmt)
            self._drop_retired_tables()
            install_change_triggers(self._conn)
            if self.verify_embedding:
                self._verify_embedding_meta()
            self._conn.execute("INSERT OR REPLACE INTO meta_entries(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
            schema_v22.mark_v22(self._conn)

    def _verify_embedding_meta(self) -> None:
        """Stop on vectors of another model or dimension; a new database records the current pair.
        A database without the model key predates it (EmbeddingGemma) and counts as another model."""
        persisted = read_embedding_meta(self._conn)
        if persisted[0] is not None and persisted != (self.dimensions, EMBEDDING_MODEL):
            raise RuntimeError(
                f"Embedding model mismatch: database={persisted[0]} dimensions/{persisted[1] or 'unknown model'}, "
                f"expected={self.dimensions} dimensions/{EMBEDDING_MODEL}. Back up the database, then re-embed: "
                f"python -m neospike.cli --home {self.path.parent.parent} reembed")
        store_embedding_meta(self._conn, self.dimensions)

    def _drop_retired_tables(self) -> None:
        """v20 -> v21: the review inbox is removed. Its tables go (their
        triggers with them) and so do their tombstones, which no restore can apply any more. Runs inside
        _run_migrations's transaction, after the tombstones table exists."""
        marks = ",".join("?" * len(RETIRED_TABLES))
        for tbl in RETIRED_TABLES:
            self._conn.execute(f"DROP TABLE IF EXISTS {tbl}")
        self._conn.execute(f"DELETE FROM tombstones WHERE table_name IN ({marks})", RETIRED_TABLES)

    def add(
        self, namespace: str, text: str, embedding: Iterable[float], *, source: str,
        kind: str = "episodic", confidence: float = 1.0, importance: float = 0.5,
        dedupe_key: None | str = None, project: str = "", metadata: dict | None = None,
    ) -> bool:
        new_id = self.insert(
            namespace, text, embedding, source=source, kind=kind, confidence=confidence,
            importance=importance, project=project, metadata=metadata,
            dedupe_key=dedupe_key,
        )
        return new_id is not None

    def insert(
        self, namespace: str, text: str, embedding: Iterable[float] | None, *, source: str,
        kind: str = "episodic", confidence: float = 1.0, importance: float = 0.5,
        dedupe_key: None | str = None, project: str = "", metadata: dict | None = None,
    ) -> int | None:
        """add() that returns the new row id, or None for a duplicate. embedding None stores the memory
        embedding-pending (PENDING_EMBED_MARK), for the daily embed-pending stage to fill."""
        vec = [] if embedding is None else list(embedding)
        if embedding is not None and len(vec) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(vec)}")
        scrubbed = text.strip()
        digest_src = dedupe_key if dedupe_key is not None else f"{source}\0{scrubbed}"
        fingerprint = hashlib.sha256(digest_src.encode()).hexdigest()
        packed_blob = PENDING_EMBED_MARK if embedding is None else normalize_rows(vec).tobytes()
        # A stored memory gets no topic; the daily topic-assign joins it through its links.
        scope, meta_json = split_scope(metadata)
        try:
            with self._write_scope():
                now = time.time()
                cur = self._conn.execute(
                    "INSERT INTO items(type, owner, text_hash, text, origin, embedding, created_at, "
                    "kind, confidence, importance, project, scope, metadata_json) "
                    f"VALUES ('{ITEM_MEMORY}', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (namespace, fingerprint, scrubbed, source, packed_blob, now,
                     kind, confidence, importance, project, scope, meta_json),
                )
                mid = int(cur.lastrowid)
            return mid
        except _DUPLICATE_ERRORS:
            return None

    @staticmethod
    def _scope_clause(project: str, session_id: str, *, prefix: str = "") -> tuple[str, tuple[str, ...]]:
        """SQL fragment + params deciding which rows a caller may see or change (legacy unscoped rows too).

        An empty project/session identity never grants access to explicit project/session rows, and a
        project column on a legacy or inconsistent row still acts as a boundary.
        """
        metadata, project_column = f"{prefix}metadata_json", f"{prefix}project"
        scope = f"{prefix}scope"
        own_session = f"json_extract({metadata},'$.session_id')"
        by_project = f"{project_column}='' OR (?<>'' AND {project_column}=?)"
        by_scope = " OR ".join((
            f"{scope} IS NULL",
            f"{scope}='global'",
            f"({scope}='project' AND ?<>'' AND {project_column}=?)",
            f"({scope}='session' AND ?<>'' AND {own_session}=?)",
        ))
        return (
            f" AND ({by_project}) AND ({by_scope})",
            (project, project, project, project, session_id, session_id),
        )

    def _fetch_editable_item(self, namespace: str, mid: int, *, project: str, session_id: str) -> sqlite3.Row | None:
        vis_clause, vals = self._scope_clause(project, session_id)
        rec = self._conn.execute(
            f"SELECT * FROM items WHERE owner=? AND id=? AND type='{ITEM_MEMORY}' AND kind='memory_item'" + vis_clause,
            (namespace, mid, *vals),
        ).fetchone()
        if rec is not None:
            metadata = json.loads(rec["metadata_json"])
            origin = metadata.get("session_id", None)
            valid_origin = isinstance(origin, str) and bool(origin)
            known_scope = rec["scope"] in ("global", "project", "session")
            if known_scope and valid_origin and rec["origin"] in (f"agent:{origin}", f"historical-session:{origin}"):
                return rec
        return None

    def fetch_item_for_update(self, namespace: str, mid: int, *, project: str, session_id: str) -> dict | None:
        """Preflight snapshot for internal use only; the model tool never gets it back."""
        with self._lock:
            rec = self._fetch_editable_item(namespace, mid, project=project, session_id=session_id)
            return dict(rec) if rec is not None else None

    def update_memory_item(
        self, namespace: str, mid: int, *, expected_text: str, expected_row: dict,
        item: dict, embedding: Iterable[float], project: str, session_id: str, edited_by: str = AGENT_EDITOR,
        split_items: Sequence[dict] = (), split_embeddings: Sequence[Iterable[float]] = (),
        split_ids: list[int] | None = None,
    ) -> str:
        """mode=correct: overwrite in place; the text/metadata before the change become the next rev.

        split_items are the later pieces of an over-long correction; they are inserted as
        new items in this same transaction and their ids are appended to split_ids.
        """
        # Materialize and check expensive/generator embeddings before a write lock. The stored
        # float32 form is checked too: nonzero floats can underflow to an unusable zero vector.
        packed_blob = self._pack_embedding(embedding)
        split_packed = self._pack_split_embeddings(split_items, split_embeddings)
        with self._lock:
            try:
                self._begin_write()
                rec = self._fetch_editable_item(namespace, mid, project=project, session_id=session_id)
                if rec is None:
                    self._abort()
                    return "unavailable"
                # Read bookkeeping may differ; any replacement, reassignment or concurrent semantic/metadata
                # write made while the embedding was being prepared is a conflict.
                agreed = set(expected_row) - {"accessed_at", "access_count"}
                if rec["text"] != expected_text or any(rec[key] != expected_row[key] for key in agreed):
                    self._abort()
                    return "conflict"
                metadata = json.loads(rec["metadata_json"])
                scope = rec["scope"]  # always present on a memory_item
                ident = rec["project"] if scope == "project" else metadata["session_id"] if scope == "session" else ""
                if self._find_visible_duplicate(namespace, mid, scope, ident, item["text"],
                                           project=project, session_id=session_id):
                    self._abort()
                    return "conflict"
                # The canonical UNIQUE key may belong to a hidden or legacy row; that row stays, and this
                # one gets an unused key so the outcome cannot confirm a guess at hidden text.
                fingerprint = self._unused_item_digest(namespace, mid, scope, ident, item["text"])
                self._push_revision(mid, rec["text"], revision_metadata(rec), edited_by=edited_by,
                                      session_id=session_id, now=time.time())
                for fld in ("subject", "durability", "confidence", "tags", *_ITEM_LABEL_KEYS):
                    if fld in item:
                        metadata[fld] = item[fld]
                metadata.update(updated_by_session=session_id, updated_at=time.time())
                # An UPDATE only (no INSERT/DELETE): identity, source and scope stay fixed, and the UPDATE
                # trigger refreshes FTS inside this same transaction.
                cur = self._conn.execute(
                    "UPDATE items SET text=?,text_hash=?,embedding=?,confidence=?,importance=?,metadata_json=? "
                    f"WHERE id=? AND owner=? AND text=? AND type='{ITEM_MEMORY}'",
                    (item["text"], fingerprint, packed_blob, item.get("confidence", rec["confidence"]),
                     item.get("importance", rec["importance"]), json.dumps(metadata), mid, namespace, expected_text),
                )
                if cur.rowcount != 1:
                    self._abort()
                    return "conflict"
                made_at = self._insert_extra_pieces(namespace, rec, metadata, split_items, split_packed,
                                                   project=project, session_id=session_id, now=time.time())
                if made_at is None:
                    self._abort()
                    return "conflict"
                link_piece_chain(self._conn, mid, made_at)
                _topic_page_tables().touch_memories(self._conn, (mid,), time.time())
                self._conn.commit()
                if split_ids is not None:
                    split_ids.extend(made_at)
            except BaseException:  # undo, then re-raise
                self._abort()
                raise
        return "updated"

    def _find_visible_duplicate(self, namespace: str, mid: int, scope: str, ident: str, text: str, *,
                           project: str, session_id: str) -> bool:
        """Historical items can have different hashes. Only a duplicate visible to this caller
        (same scope identity, same normalized text) may cause a semantic conflict."""
        norm_text = " ".join(text.casefold().split())
        vis_clause, vals = self._scope_clause(project, session_id)
        statement = (
            f"SELECT text,project,scope,metadata_json FROM items WHERE owner=? AND type='{ITEM_MEMORY}' "
            "AND kind='memory_item' AND id<>?"
            + vis_clause
        )
        for other in self._conn.execute(statement, (namespace, mid, *vals)):
            peer_meta = json.loads(other["metadata_json"])
            peer_ident = other["project"] if scope == "project" else peer_meta.get("session_id") if scope == "session" else ""
            if other["scope"] == scope and peer_ident == ident and " ".join(other["text"].casefold().split()) == norm_text:
                return True
        return False

    def search(
        self,
        namespace: str,
        query: str,
        q_vector: Iterable[float],
        *,
        limit: int = 5,
        session_id: None | str = None,
        project: None | str = None,
        history: str = HISTORY_COLLAPSE,
        skip_kinds: Sequence[str] = (),
    ) -> SearchResultList:
        """Hybrid search. `history` picks the invalid-row treatment (HISTORY_*). The default is the
        auto-injection one (replace); direct search passes exclude or include. In replace mode every
        invalid hit that lost its slot is listed in the result's `superseded` (recall log).
        skip_kinds: row kinds left out before the top-k cut (prefetch: the builtin mirror)."""
        if history not in HISTORY_CHOICES:
            raise ValueError(f"history must be one of {', '.join(HISTORY_CHOICES)}")
        skipped = tuple(skip_kinds)
        kind_clause = f" AND kind NOT IN ({','.join('?' * len(skipped))})" if skipped else ""
        query_vec = list(q_vector)
        if len(query_vec) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(query_vec)}")
        now = time.time()
        with self._lock:
            vis_clause = ""
            vis_params: tuple[str, ...] = ()
            if None not in (project, session_id):
                vis_clause, vis_params = self._scope_clause(project, session_id)
            # Prefetch (replace) leaves archived active rows out. An archived invalid row
            # stays: it is never injected itself, but still hands its slot to its replacer.
            archived = " AND (archived_at IS NULL OR invalid_at IS NOT NULL)" if history == HISTORY_COLLAPSE else ""
            recs = self._conn.execute(
                "SELECT id,text,origin,kind,confidence,importance,created_at,invalid_at,superseded_by,"
                f"archived_at FROM items WHERE owner=? AND type='{ITEM_MEMORY}' "
                f"AND {vector_present_sql()}" + archived + kind_clause + vis_clause,
                (namespace, *skipped, *vis_params),
            ).fetchall()
            sims = vector_index.similarities(self._conn, "items", [rec["id"] for rec in recs], query_vec,
                                             threshold=self.semantic_threshold)
        ordered: list[SearchResult] = []
        for rec, sem in zip(recs, sims):
            mid = int(rec["id"])
            invalid = rec["invalid_at"] is not None
            if invalid and history == HISTORY_OMIT:
                continue
            # Admission happens before bonuses, top-k, access bookkeeping and RRF. Sharing a word is not
            # required (bilingual paraphrases pass on semantic evidence alone): the cosine floor decides.
            if not math.isfinite(sem) or sem < self.semantic_threshold:
                continue
            boost = DURABLE_BOOST if rec["kind"] == "durable" else 0.0
            rank_score = blend_score(sem, float(rec["importance"]),
                                 freshness(now, float(rec["created_at"]))) + boost
            superseded_by = None
            if invalid:
                superseded_by = None if rec["superseded_by"] is None else int(rec["superseded_by"])
                if history == HISTORY_SHOW:
                    rank_score *= OLD_VERSION_SCORE_FACTOR
            ordered.append(SearchResult(mid, rec["text"], rec["origin"], rank_score, rec["kind"],
                                       float(rec["confidence"]), superseded_by=superseded_by if invalid else None,
                                       archived=True if rec["archived_at"] is not None else None))
        ordered.sort(key=_rank_sort_key, reverse=True)
        old_gen: list[SearchResult] = []
        if history == HISTORY_COLLAPSE:
            with self._lock:
                found = self._swap_superseded_hits(ordered, recs, limit, old_gen)
        else:
            found = ordered[:limit]
        if found:
            with self._write_scope():
                self._bulk_write(
                    "UPDATE items SET accessed_at=?, access_count=access_count+1 WHERE id=?",
                    [(now, item.id) for item in found],
                )
        return SearchResultList(found, old_gen)

    def _swap_superseded_hits(self, ordered: list[SearchResult], recs: list[sqlite3.Row], limit: int,
                            dropped_old: list[SearchResult]) -> list[SearchResult]:
        """Prefetch cut: an invalid hit gives its slot to the active end of its chain,
        unless that row is already in (or not visible to) the result; then the next rank fills it."""
        shown = {int(rec["id"]): rec for rec in recs}
        found: list[SearchResult] = []
        claimed: set[int] = set()
        for item in ordered:
            if len(found) >= limit:
                break
            if item.superseded_by is None:
                if item.id not in claimed:
                    found.append(item)
                    claimed.add(item.id)
                continue
            dropped_old.append(item)
            for end in self._chain_tips(item.id):
                if len(found) >= limit:
                    break
                rec = shown.get(end)
                if rec is None or end in claimed or rec["invalid_at"] is not None:
                    continue
                found.append(SearchResult(end, rec["text"], rec["origin"], item.score, rec["kind"],
                                            float(rec["confidence"]), replaced_id=item.id))
                claimed.add(end)
        return found

    def _successors_of(self, mid: int) -> list[int]:
        """Sources of the live (not reverted) supersedes links into memory_id, newest link first.
        Several memories may each replace a different part of one memory. Caller holds _lock."""
        return [int(rec[0]) for rec in self._conn.execute(
            "SELECT src_id FROM item_links WHERE relation=? AND dst_id=? "
            "AND reverted_at IS NULL ORDER BY id DESC", (RELATION_SUPERSEDES, mid))]

    def _chain_tips(self, mid: int) -> list[int]:
        """Active memories that replace memory_id: every live replacer, followed further only when the
        replacer is itself invalid (A->B, B->C gives C). Breadth first, newest link first, bounded by
        SUPERSEDE_CHAIN_LIMIT visited rows; cycles and missing rows are skipped. Caller holds _lock."""
        tips: list[int] = []
        met = {mid}
        queue_ = [mid]
        while queue_ and len(met) < SUPERSEDE_CHAIN_LIMIT:
            successors: list[int] = []
            for cur_row in queue_:
                for heir_row in self._successors_of(cur_row):
                    if heir_row in met:
                        continue
                    met.add(heir_row)
                    rec = self._conn.execute("SELECT invalid_at FROM items WHERE id=?", (heir_row,)).fetchone()
                    if rec is None:
                        continue
                    if rec["invalid_at"] is None:
                        tips.append(heir_row)
                    else:
                        successors.append(heir_row)
            queue_ = successors
        return tips

    def _chain_reaches(self, from_id: int, target: int) -> bool:
        """True when target is start or one of its (transitive) live replacers. Caller holds _lock."""
        met = {from_id}
        queue_ = [from_id]
        while queue_ and len(met) < SUPERSEDE_CHAIN_LIMIT:
            if target in queue_:
                return True
            successors: list[int] = []
            for cur_row in queue_:
                for heir_row in self._successors_of(cur_row):
                    if heir_row not in met:
                        met.add(heir_row)
                        successors.append(heir_row)
            queue_ = successors
        return target in queue_

    def _active_supersede_link(self, next_id: int, prev_id: int) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM item_links WHERE relation=? AND src_id=? "
            "AND dst_id=? AND reverted_at IS NULL", (RELATION_SUPERSEDES, next_id, prev_id)).fetchone() is not None

    def _insert_supersede_link(self, next_id: int, prev_id: int, *, created_by: str, origin: str, session_id: str,
                         evidence: str, now: float) -> dict:
        """One supersedes(new -> old) inside the caller's write transaction.

        An already superseded old row gets one more link; its other replacers stay active and
        superseded_by moves to the newest one (the full list is in item_links). A link that would close
        a cycle (A->B->A) is refused.
        """
        target = prev_id
        exists_sql = f"SELECT 1 FROM items WHERE id=? AND type='{ITEM_MEMORY}'"
        if (self._conn.execute(exists_sql, (target,)).fetchone() is None
                or self._conn.execute(exists_sql, (next_id,)).fetchone() is None):
            return {"linked": False, "reason": "unavailable"}
        if self._chain_reaches(next_id, target):
            return {"linked": False, "reason": "cycle"}
        self._conn.execute("UPDATE items SET invalid_at=COALESCE(invalid_at, ?), superseded_by=? WHERE id=?",
                           (now, next_id, target))
        cur = self._conn.execute(
            "INSERT INTO item_links(src_id,dst_id,dst_title,relation,created_by,origin,session_id,"
            "evidence,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (next_id, target, f"mem:{target}", RELATION_SUPERSEDES, created_by, origin, session_id,
             evidence[:EVIDENCE_CHAR_LIMIT], now),
        )
        _topic_page_tables().touch_memories(self._conn, (next_id, target), now)
        return {"linked": True, "link_id": int(cur.lastrowid), "old_id": target, "new_id": next_id}

    def replace_item_with_new(
        self, namespace: str, mid: int, *, expected_text: str, expected_row: dict,
        item: dict, embedding: Iterable[float], project: str, session_id: str, evidence: str = "",
        split_items: Sequence[dict] = (), split_embeddings: Sequence[Iterable[float]] = (),
    ) -> dict:
        """mode=supersede: INSERT the new item, mark the old one invalid, link supersedes(new->old).

        CAS and authority are those of update_memory_item. The new row keeps the old row's scope and
        project; it is written by the current session. One write transaction; no embedding inside.
        split_items: later pieces of an over-long text, inserted as new items (no link) and
        returned as split_ids.
        """
        packed_blob = self._pack_embedding(embedding)
        split_packed = self._pack_split_embeddings(split_items, split_embeddings)
        now = time.time()
        with self._lock:
            try:
                self._begin_write()
                rec = self._fetch_editable_item(namespace, mid, project=project, session_id=session_id)
                if rec is None:
                    self._abort()
                    return {"superseded": False, "reason": "unavailable"}
                agreed = set(expected_row) - {"accessed_at", "access_count"}
                if rec["text"] != expected_text or any(rec[key] != expected_row[key] for key in agreed):
                    self._abort()
                    return {"superseded": False, "reason": "conflict"}
                # An already superseded row is linked once more (several replacers).
                target, dest_row = mid, rec
                prev_meta = json.loads(dest_row["metadata_json"])
                scope = dest_row["scope"]
                metadata = {"subject": item.get("subject", prev_meta.get("subject")),
                            "durability": item.get("durability", prev_meta.get("durability")),
                            "confidence": item.get("confidence", dest_row["confidence"]),
                            "tags": item.get("tags", prev_meta.get("tags", [])), "session_id": session_id,
                            **_carried_labels(item, prev_meta)}
                ident = dest_row["project"] if scope == "project" else session_id if scope == "session" else ""
                if self._find_visible_duplicate(namespace, 0, scope, ident, item["text"], project=project,
                                           session_id=session_id):
                    self._abort()
                    return {"superseded": False, "reason": "duplicate"}
                # Same hidden-row rule as update_memory_item: never confirm a guess at hidden text.
                fingerprint = self._unused_item_digest(namespace, 0, scope, ident, item["text"])
                try:
                    cur = self._conn.execute(
                        "INSERT INTO items(type, owner, text_hash, text, origin, embedding, created_at, "
                        "kind, confidence, importance, project, scope, metadata_json) "
                        f"VALUES ('{ITEM_MEMORY}', ?, ?, ?, ?, ?, ?, 'memory_item', ?, ?, ?, ?, ?)",
                        (namespace, fingerprint, item["text"], f"agent:{session_id}", packed_blob, now,
                         metadata["confidence"], item.get("importance", dest_row["importance"]),
                         dest_row["project"], scope, json.dumps(metadata)),
                    )
                except _DUPLICATE_ERRORS:
                    self._abort()
                    return {"superseded": False, "reason": "duplicate"}
                next_id = int(cur.lastrowid)
                linked_ids = self._insert_supersede_link(next_id, target, created_by=CREATED_BY_HUMAN, origin=ORIGIN_CONVERSATION,
                                               session_id=session_id, evidence=evidence, now=now)
                if not linked_ids["linked"]:
                    self._abort()
                    return {"superseded": False, "reason": linked_ids["reason"]}
                made_at = self._insert_extra_pieces(namespace, dest_row, metadata, split_items, split_packed,
                                                   project=project, session_id=session_id, now=now)
                if made_at is None:
                    self._abort()
                    return {"superseded": False, "reason": "duplicate"}
                link_piece_chain(self._conn, next_id, made_at)
                self._conn.commit()
            except BaseException:  # undo, then re-raise
                self._abort()
                raise
        verdict = {"superseded": True, "id": next_id, "old_id": target, "link_id": linked_ids["link_id"]}
        if split_items:
            verdict["split_ids"] = made_at
        return verdict

    def undo_supersede(self, lid: int) -> dict:
        """Undo one supersedes link; the link row stays with reverted_at. The old memory is active again
        only when no live supersedes link into it is left; otherwise superseded_by is the newest one left."""
        now = time.time()
        with self._lock:
            try:
                self._begin_write()
                lnk = self._conn.execute(
                    "SELECT src_id,dst_id FROM item_links WHERE id=? AND relation=? AND reverted_at IS NULL",
                    (lid, RELATION_SUPERSEDES),
                ).fetchone()
                if lnk is None:
                    self._abort()
                    return {"reverted": False, "reason": "not_found", "link_id": lid}
                self._conn.execute("UPDATE item_links SET reverted_at=? WHERE id=?", (now, lid))
                lhs = self._successors_of(int(lnk["dst_id"]))
                if lhs:
                    self._conn.execute("UPDATE items SET superseded_by=? WHERE id=?", (lhs[0], lnk["dst_id"]))
                    revived = 0
                else:
                    revived = self._conn.execute(
                        "UPDATE items SET invalid_at=NULL, superseded_by=NULL WHERE id=?", (lnk["dst_id"],),
                    ).rowcount
                _topic_page_tables().touch_memories(self._conn, (int(lnk["src_id"]), int(lnk["dst_id"])), now)
                self._conn.commit()
            except BaseException:  # undo, then re-raise
                self._abort()
                raise
        return {"reverted": True, "link_id": lid, "old_id": int(lnk["dst_id"]), "new_id": int(lnk["src_id"]),
                "reactivated": revived == 1}

    # -- memory revisions -------------------------------------------------------

    def _push_revision(self, mid: int, text: str, meta_json: str, *, edited_by: str,
                         session_id: str, now: float) -> None:
        """Next rev of one memory inside the caller's transaction."""
        self._conn.execute(
            "INSERT INTO item_revisions(item_id,rev,text,metadata_json,edited_at,edited_by,session_id) "
            "SELECT ?, COALESCE(MAX(rev),0)+1, ?, ?, ?, ?, ? FROM item_revisions WHERE item_id=?",
            (mid, text, meta_json, now, edited_by, session_id, mid),
        )

    def memory_history(self, mid: int) -> dict | None:
        """Current text plus every stored earlier rev (newest first); None when the memory is missing."""
        with self._lock:
            cur_row = self._conn.execute(
                f"SELECT text,scope,metadata_json,invalid_at,superseded_by FROM items WHERE id=? AND type='{ITEM_MEMORY}'",
                (mid,)).fetchone()
            if cur_row is None:
                return None
            recs = self._conn.execute(
                "SELECT rev,text,metadata_json,edited_at,edited_by,session_id FROM item_revisions "
                "WHERE item_id=? ORDER BY rev DESC", (mid,),
            ).fetchall()
        return {"id": mid, "text": cur_row["text"], "metadata_json": revision_metadata(cur_row),
                "invalid_at": cur_row["invalid_at"], "superseded_by": cur_row["superseded_by"],
                "revisions": [dict(rec) for rec in recs]}

    def revision_at(self, mid: int, revision_no: int) -> dict | None:
        with self._lock:
            rec = self._conn.execute(
                "SELECT rev,text,metadata_json FROM item_revisions WHERE item_id=? AND rev=?", (mid, revision_no),
            ).fetchone()
        return None if rec is None else dict(rec)

    def rollback_memory_revision(self, mid: int, revision_no: int, *, expected_text: str, embedding: Iterable[float],
                                edited_by: str = HUMAN_EDITOR, session_id: str = "") -> dict:
        """Owner restore: the current text/metadata become a new rev, then rev R's text/metadata return.

        expected_text is the current text read before embedding (CAS). The memory id stays.
        """
        packed_blob = self._pack_embedding(embedding)
        now = time.time()
        with self._lock:
            try:
                self._begin_write()
                rec = self._conn.execute(f"SELECT * FROM items WHERE id=? AND type='{ITEM_MEMORY}'", (mid,)).fetchone()
                rev_row = self._conn.execute(
                    "SELECT text,metadata_json FROM item_revisions WHERE item_id=? AND rev=?", (mid, revision_no),
                ).fetchone()
                if rec is None or rev_row is None:
                    self._abort()
                    return {"restored": False, "reason": "not_found"}
                if rec["text"] != expected_text:
                    self._abort()
                    return {"restored": False, "reason": "conflict"}
                self._push_revision(mid, rec["text"], revision_metadata(rec), edited_by=edited_by,
                                      session_id=session_id, now=now)
                fingerprint = rec["text_hash"]
                restored = json.loads(rev_row["metadata_json"] or "{}")
                back_scope = restored.get("scope", rec["scope"])
                if rec["kind"] == "memory_item":
                    scope = restored.get("scope", "global")
                    ident = rec["project"] if scope == "project" else restored.get("session_id", "") if scope == "session" else ""
                    fingerprint = self._unused_item_digest(rec["owner"], mid, scope, ident, rev_row["text"])
                _, restored_json = split_scope(restored)
                self._conn.execute(
                    "UPDATE items SET text=?,text_hash=?,embedding=?,scope=?,metadata_json=? WHERE id=?",
                    (rev_row["text"], fingerprint, packed_blob, back_scope, restored_json, mid),
                )
                _topic_page_tables().touch_memories(self._conn, (mid,), now)
                self._conn.commit()
            except BaseException:  # undo, then re-raise
                self._abort()
                raise
        return {"restored": True, "id": mid, "rev": revision_no}

    def _pack_split_embeddings(self, items: Sequence[dict], embed_vecs: Sequence[Iterable[float]]) -> list[bytes]:
        if len(items) != len(embed_vecs):
            raise ValueError("Each split memory piece needs one embedding")
        return [self._pack_embedding(embedding) for embedding in embed_vecs]

    def _insert_extra_pieces(self, namespace: str, rec: sqlite3.Row, base_meta: dict, items: Sequence[dict],
                            packed_blob: Sequence[bytes], *, project: str, session_id: str, now: float) -> list[int] | None:
        """Later pieces of a split correction, inside the caller's write transaction.

        Each piece keeps the original row's namespace, scope, project, durability and tags, and is
        written by the current session (so it stays editable). No topic assignment: the daily
        topic-assign cron places them. None on a visible duplicate (the caller rolls back everything).
        """
        scope = rec["scope"]
        ident = rec["project"] if scope == "project" else session_id if scope == "session" else ""
        made_at: list[int] = []
        for item, vec in zip(items, packed_blob):
            metadata = {"subject": item["subject"],
                        "durability": item.get("durability", base_meta.get("durability")),
                        "confidence": item.get("confidence", rec["confidence"]),
                        "tags": item.get("tags", base_meta.get("tags", [])), "session_id": session_id,
                        **_carried_labels(item, base_meta)}
            if self._find_visible_duplicate(namespace, 0, scope, ident, item["text"], project=project,
                                       session_id=session_id):
                return None
            fingerprint = self._unused_item_digest(namespace, 0, scope, ident, item["text"])
            try:
                cur = self._conn.execute(
                    "INSERT INTO items(type, owner, text_hash, text, origin, embedding, created_at, "
                    "kind, confidence, importance, project, scope, metadata_json) "
                    f"VALUES ('{ITEM_MEMORY}', ?, ?, ?, ?, ?, ?, 'memory_item', ?, ?, ?, ?, ?)",
                    (namespace, fingerprint, item["text"], f"agent:{session_id}", vec, now,
                     metadata["confidence"], item.get("importance", rec["importance"]), rec["project"],
                     scope, json.dumps(metadata)),
                )
            except _DUPLICATE_ERRORS:
                return None
            made_at.append(int(cur.lastrowid))
        return made_at

    def _unused_item_digest(self, namespace: str, mid: int, scope: str, ident: str, text: str) -> str:
        """memory_item content hash of text, or a private replacement key when another row owns it."""
        norm_text = " ".join(text.casefold().split())
        fingerprint = canon = hashlib.sha256(f"memory_item\0{scope}\0{ident}\0{norm_text}".encode()).hexdigest()
        clash = 0
        taken_sql = f"SELECT 1 FROM items WHERE owner=? AND text_hash=? AND id<>? AND type='{ITEM_MEMORY}'"
        while self._conn.execute(taken_sql, (namespace, fingerprint, mid)).fetchone() is not None:
            clash += 1
            fingerprint = hashlib.sha256(f"memory_item_replacement\0{mid}\0{clash}\0{canon}".encode()).hexdigest()
        return fingerprint

    def _pack_embedding(self, embedding: Iterable[float]) -> bytes:
        vec = list(embedding)
        complaint = "Invalid replacement " "embedding"
        if len(vec) != self.dimensions or not all(math.isfinite(value) for value in vec):
            raise ValueError(complaint)
        encoded_bytes = normalize_rows(vec)
        if not (any(encoded_bytes) and all(math.isfinite(value) for value in encoded_bytes)):
            raise ValueError(complaint)
        return encoded_bytes.tobytes()

    def log_note_access(self, nids: Iterable[int], *, now: float | None = None) -> int:
        """Search/prefetch hit bookkeeping for notes, like memories in search(). Leaves updated_at alone."""
        ids = sorted({int(note_id) for note_id in nids})
        if not ids:
            return 0
        stamp_now = time.time() if now is None else now
        with self._write_scope():
            self._bulk_write(
                f"UPDATE items SET accessed_at=?, access_count=access_count+1 WHERE id=? AND {NOTE_HEAD_SQL}",
                [(stamp_now, note_id) for note_id in ids],
            )
        return len(ids)

    def read_neospike_config(self) -> NeospikeConfig:
        """neospike/config.json beside the canonical memory.db; defaults elsewhere or when unreadable."""
        fpath = self.path.parent / "config.json"
        try:
            if fpath.is_file():
                return NeospikeConfig.from_values(json.loads(fpath.read_text(encoding="utf-8-sig")))
        except (ValueError, OSError):
            pass
        return NeospikeConfig()

    def archive(self, namespace: str, mid: int, *, now: float | None = None) -> bool:
        """Human archive: out of prefetch, still in direct search and the dashboard."""
        with self._write_scope():
            cur = self._conn.execute(
                f"UPDATE items SET archived_at=? WHERE owner=? AND id=? AND archived_at IS NULL AND type='{ITEM_MEMORY}'",
                (time.time() if now is None else now, namespace, mid),
            )
        return cur.rowcount == 1

    def unarchive(self, namespace: str, mid: int) -> bool:
        """Back into prefetch. archived_at is cleared; access bookkeeping is left alone."""
        with self._write_scope():
            cur = self._conn.execute(
                f"UPDATE items SET archived_at=NULL WHERE owner=? AND id=? AND archived_at IS NOT NULL AND type='{ITEM_MEMORY}'",
                (namespace, mid),
            )
        return cur.rowcount == 1

    def closest_memories(self, namespace: str, mid: int, *, limit: int, project: str = "",
                             session_id: str = "") -> list[tuple[int, float]]:
        """The `limit` closest valid memory items to memory_id by cosine, best first.
        Valid: memory_item, not invalid/superseded, not archived, visible in this
        project/session scope (the same authority delete() checks). memory_id itself is excluded;
        [] when it is missing. Scores: one numpy product (dot_similarities)."""
        vis_clause, vals = self._scope_clause(project, session_id)
        with self._lock:
            anchor_row = self._conn.execute(
                f"SELECT embedding FROM items WHERE owner=? AND id=? AND type='{ITEM_MEMORY}'", (namespace, mid),
            ).fetchone()
            if anchor_row is None or is_embed_pending(anchor_row["embedding"]):
                return []
            recs = self._conn.execute(
                f"SELECT id FROM items WHERE owner=? AND id<>? AND type='{ITEM_MEMORY}' AND kind='memory_item' "
                f"AND invalid_at IS NULL AND superseded_by IS NULL AND archived_at IS NULL AND {vector_present_sql()}" + vis_clause,
                (namespace, mid, *vals),
            ).fetchall()
            query = std_array.array("f", anchor_row["embedding"])
            sims = vector_index.similarities(self._conn, "items", [rec["id"] for rec in recs], query,
                                             limit=limit)
        scored_hits = [(int(rec["id"]), rank_score) for rec, rank_score in zip(recs, sims)]
        scored_hits = [duo for duo in scored_hits if math.isfinite(duo[1])]
        return heapq.nlargest(limit, scored_hits, key=_neighbor_rank)

    def scopes_of_memories(self, namespace: str, mids: Sequence[int]) -> dict[int, tuple[str, str]]:
        """{id: (scope, project column)} of the given memories in namespace. A legacy row without a
        known metadata scope counts as 'project' when its project column is set (the visibility boundary of
        _scope_clause), else 'global'."""
        ids = sorted({int(mid) for mid in mids})
        if not ids:
            return {}
        with self._lock:
            recs = self._conn.execute(
                f"SELECT id,project,scope FROM items WHERE owner=? AND type='{ITEM_MEMORY}' "
                f"AND id IN ({','.join('?' * len(ids))})", (namespace, *ids)).fetchall()
        scope_list: dict[int, tuple[str, str]] = {}
        for mid, project, scope in recs:
            if scope not in SCOPE_ORDER:
                scope = "project" if project else "global"
            scope_list[int(mid)] = (str(scope), str(project or ""))
        return scope_list

    def widen_memory_scope(self, namespace: str, mid: int, scope: str, project: str) -> bool:
        """Duplicate across scopes: the new memory takes the wider scope of the old one it replaced
        (the scope and project columns; metadata session_id stays the writer for authority). The
        content hash is recomputed for the new scope (a private key when another row owns it). True when
        changed."""
        if scope not in SCOPE_ORDER:
            raise ValueError("Invalid " "memory scope")
        col = project if scope == "project" else ""
        with self._write_scope():
            rec = self._conn.execute(f"SELECT text FROM items WHERE owner=? AND id=? AND type='{ITEM_MEMORY}'",
                                     (namespace, mid)).fetchone()
            if rec is None:
                return False
            fingerprint = self._unused_item_digest(namespace, mid, scope, col, str(rec["text"]))
            cur = self._conn.execute("UPDATE items SET scope=?,project=?,text_hash=? WHERE id=?",
                                        (scope, col, fingerprint, mid))
        return cur.rowcount == 1

    def items_missing_embedding(self, namespace: str | None = None) -> list[dict]:
        """Embedding-pending memories, id order: {id, namespace, text, project, session_id}."""
        where, args = ("AND owner=?", (namespace,)) if namespace else ("", ())
        with self._lock:
            recs = self._conn.execute(
                f"SELECT id,owner,text,project,json_extract(metadata_json,'$.session_id') AS session_id "
                f"FROM items WHERE type='{ITEM_MEMORY}' AND NOT {vector_present_sql()} {where} ORDER BY id",
                args).fetchall()
        return [{"id": int(rec["id"]), "namespace": str(rec["owner"]), "text": str(rec["text"]),
                 "project": str(rec["project"] or ""), "session_id": str(rec["session_id"] or "")} for rec in recs]

    def parts_missing_embedding(self) -> list[dict]:
        """Document part rows without a vector, id order: {id, text} (the head rows never have one)."""
        with self._lock:
            recs = self._conn.execute(
                f"SELECT id,text FROM items WHERE type IN {DOC_TYPES_SQL} AND title IS NULL "
                f"AND NOT {vector_present_sql()} ORDER BY id").fetchall()
        return [{"id": int(rec["id"]), "text": str(rec["text"])} for rec in recs]

    def write_missing_embedding(self, mid: int, embedding: Iterable[float]) -> bool:
        """Store the vector of an embedding-pending memory; False when it is gone or already has one."""
        packed_blob = self._pack_embedding(embedding)
        with self._write_scope():
            cur = self._conn.execute(
                f"UPDATE items SET embedding=? WHERE id=? AND title IS NULL AND NOT {vector_present_sql()}",
                (packed_blob, mid))
        return cur.rowcount == 1

    def _redirect_deleted_successors(self, ids: list[int], now: float, heir: int | None) -> list[int]:
        """Inside the caller's transaction after the rows of ids are gone (links still there):
        memories that a deleted memory superseded (a live supersedes link from it and invalid_at set). heir (a duplicate deletion): the live supersedes links move
        to the heir and superseded_by follows. Otherwise such a memory is deleted with it, unless another live
        replacer remains (superseded_by -> that one) or a human archived it (kept; superseded_by cleared).
        Returns the ids deleted here (their refs are purged by the caller's recursion)."""
        marks = ",".join("?" * len(ids))
        after_ids = sorted({int(rec[0]) for rec in self._conn.execute(
            f"SELECT l.dst_id FROM item_links l JOIN items m ON m.id=l.dst_id WHERE l.relation=? AND m.invalid_at IS NOT NULL "
            f"AND l.reverted_at IS NULL AND l.src_id IN ({marks})",
            (RELATION_SUPERSEDES, *ids))})
        if not after_ids:
            return []
        remainder = after_ids
        if heir is not None and self._conn.execute("SELECT 1 FROM items WHERE id=?", (heir,)).fetchone():
            remainder = []
            for after_id in after_ids:
                if after_id == heir or self._chain_reaches(heir, after_id):  # never close a cycle (A->B->A)
                    remainder.append(after_id)
                    continue
                if not self._active_supersede_link(heir, after_id):
                    self._conn.execute(
                        f"UPDATE item_links SET src_id=? WHERE id=(SELECT MAX(id) FROM item_links WHERE relation=? "
                        f"AND reverted_at IS NULL AND dst_id=? "
                        f"AND src_id IN ({marks}))", (heir, RELATION_SUPERSEDES, after_id, *ids))
                self._conn.execute(f"UPDATE items SET superseded_by=? WHERE id=? AND superseded_by IN ({marks})",
                                   (heir, after_id, *ids))
        deleted: list[int] = []
        for after_id in remainder:
            other_rows = [int(rec[0]) for rec in self._conn.execute(
                f"SELECT src_id FROM item_links WHERE relation=? "
                f"AND reverted_at IS NULL AND dst_id=? AND src_id NOT IN ({marks}) "
                f"AND EXISTS (SELECT 1 FROM items m WHERE m.id=item_links.src_id) ORDER BY id DESC",
                (RELATION_SUPERSEDES, after_id, *ids))]
            archived = self._conn.execute("SELECT archived_at FROM items WHERE id=?", (after_id,)).fetchone()[0]
            if other_rows or archived is not None:
                self._conn.execute("UPDATE items SET superseded_by=? WHERE id=?",
                                   (other_rows[0] if other_rows else None, after_id))
                continue
            self._conn.execute("DELETE FROM items WHERE id=?", (after_id,))
            deleted.append(after_id)
        return deleted

    def _drop_memory_links(self, mids: Sequence[int], now: float, *, heir: int | None = None) -> list[str]:
        """Inside the caller's transaction after its memory rows are gone: drop their topic links and the
        links they start or end. A `refers` link into them keeps its row with dst_id NULL (the [[mem:N]] text
        stays). Returns the names of the topics they were in. Also their item_revisions and folder_items rows;
        memories they superseded follow them (_redirect_deleted_successors, heir = the duplicate that replaced
        them). A topic left without members is gone with its last link (a topic is its links)."""
        ids = sorted({int(mid) for mid in mids})
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        before_id = self._redirect_deleted_successors(ids, now, heir)
        self._conn.execute(f"DELETE FROM item_revisions WHERE item_id IN ({marks})", ids)
        self._conn.execute(f"DELETE FROM folder_items WHERE item_type='memory' AND item_id IN ({marks})", ids)
        topic_names = _topic_page_tables().member_topics(self._conn, ids)
        self._conn.execute(
            f"UPDATE item_links SET dst_id=NULL WHERE relation=? AND dst_id IN ({marks}) "
            f"AND src_id NOT IN ({marks})", (RELATION_REFERS, *ids, *ids))
        self._conn.execute(
            f"DELETE FROM item_links WHERE relation<>'{RELATION_NEXT}' AND (src_id IN ({marks}) "
            f"OR dst_id IN ({marks}))", (*ids, *ids))
        self._conn.execute(f"DELETE FROM memory_pairs WHERE low_id IN ({marks}) OR high_id IN ({marks})",
                           (*ids, *ids))
        self._conn.execute(f"DELETE FROM memory_checks WHERE memory_id IN ({marks})", ids)
        if before_id:
            topic_names = sorted({*topic_names, *self._drop_memory_links(before_id, now)})
        return topic_names

    def _transfer_pinned_topic(self, prev_id: int, successor_id: int, now: float) -> None:
        """A hand-pinned topic of a deleted duplicate passes to the memory that replaced it
        (membership and pin)."""
        page_list = _topic_page_tables()
        marks = page_list.manual_marks(self._conn)
        name = marks.pinned_to(prev_id)
        if name is None or not page_list.topic_exists(self._conn, name):
            return
        page_list.set_topic(self._conn, successor_id, name, now)
        marks.pin(successor_id, name)
        marks.pinned.pop(str(int(prev_id)), None)
        page_list.save_manual_marks(self._conn, marks)

    def delete(self, namespace: str, mid: int, *, project: str = "", session_id: str = "",
               heir: int | None = None) -> bool:
        """Delete one memory with its topic membership and links. heir: the memory that
        replaces it (post-save dedupe); a hand-pinned topic membership passes to it."""
        vis_clause, vals = self._scope_clause(project, session_id)
        now = time.time()
        with self._write_scope():
            cur = self._conn.execute(
                f"DELETE FROM items WHERE owner=? AND id=? AND type='{ITEM_MEMORY}'" + vis_clause,
                (namespace, mid, *vals),
            )
            if cur.rowcount == 1:
                if heir is not None:
                    self._transfer_pinned_topic(mid, heir, now)
                self._drop_memory_links((mid,), now, heir=heir)
        return cur.rowcount == 1

    def delete_sources_matching(self, namespace: str, source: str, probe: str) -> int:
        """Delete the mirrored rows whose text contains the old_text selector Hermes resolved."""
        value = probe.strip()
        if not value:
            return 0
        with self._write_scope():
            recs = self._conn.execute(
                f"SELECT id,text FROM items WHERE owner=? AND origin=? AND type='{ITEM_MEMORY}'",
                (namespace, source),
            ).fetchall()
            merged_ids = value.casefold()
            ids = [int(rec["id"]) for rec in recs if merged_ids in str(rec["text"]).casefold()]
            if ids:
                self._conn.executemany("DELETE FROM items WHERE id=?", [(mid,) for mid in ids])
                self._drop_memory_links(ids, time.time())
            return len(ids)

    def texts_by_id(self, namespace: str, mids: Sequence[int]) -> dict[int, str]:
        """{id: text} of the given memories that exist in namespace (word-count check)."""
        ids = sorted({int(mid) for mid in mids})
        if not ids:
            return {}
        with self._lock:
            recs = self._conn.execute(
                f"SELECT id,text FROM items WHERE owner=? AND type='{ITEM_MEMORY}' AND id IN ({','.join('?' * len(ids))})",
                (namespace, *ids)).fetchall()
        return {int(rec[0]): str(rec[1]) for rec in recs}

    def prune_orphan_links(self) -> dict:
        """Daily code stage: drop connection rows whose endpoint is gone; items themselves are never touched.
        Gone: an item row that does not exist; a source document that is trashed (a restore re-links its
        outgoing links). A target document that is trashed can come back, so links into it and its folder
        place stay. A `refers` link from a live source into a gone target keeps its row with dst_id NULL (its
        [[...]] text still stands); title-only unresolved links (dst_id NULL) are never touched. Topic links,
        memory pairs, check marks and revisions of a gone item go too; a next pointer to a gone row (or to
        itself) is dropped. Returns {"topic_links": n, "links": n, "links_unresolved": n, "folder_items": n,
        "memory_pairs": n, "memory_checks": n, "revisions": n, "piece_chain": n}."""
        with self._write_scope():
            member_rows = self._conn.execute(_ORPHAN_TOPIC_LINKS_SQL).rowcount
            pending_ids = self._conn.execute(_ORPHAN_REFERS_SQL, (RELATION_REFERS,)).rowcount
            link_rows = self._conn.execute(_ORPHAN_LINKS_SQL).rowcount
            folder_rows = self._conn.execute(_ORPHAN_FOLDER_ITEMS_SQL).rowcount
            disc_links = self._conn.execute(_ORPHAN_DISCOVERY_LINKS_SQL).rowcount
            checks = self._conn.execute(_ORPHAN_CHECKS_SQL).rowcount
            rev_rows = self._conn.execute(_ORPHAN_REVISIONS_SQL).rowcount
            chain_col = self._conn.execute(_ORPHAN_NEXT_SQL).rowcount
        return {"topic_links": member_rows, "links": link_rows, "links_unresolved": pending_ids,
                "folder_items": folder_rows, "memory_pairs": disc_links, "memory_checks": checks,
                "revisions": rev_rows, "piece_chain": chain_col}

    def count(self, namespace: str):
        with self._lock:
            total = self._conn.execute(f"SELECT COUNT(*) FROM items WHERE owner=? AND type='{ITEM_MEMORY}'",
                                       (namespace,)).fetchone()
            return int(total[0])

    def list_namespaces(self) -> list[dict]:
        with self._lock:
            recs = self._conn.execute(
                "SELECT owner,COUNT(*) entries,SUM(kind='durable') durable,SUM(kind='summary') summaries "
                f"FROM items WHERE type='{ITEM_MEMORY}' GROUP BY owner ORDER BY owner"
            ).fetchall()
        return [dict(rec) for rec in recs]

    # -- meta / namespace migration -----------------------------------------------

    def write_meta(self, key: str, value: str) -> bool:
        """Write one meta value only when it differs; True when written."""
        with self._write_scope():
            rec = self._conn.execute("SELECT value FROM meta_entries WHERE key=?", (key,)).fetchone()
            if rec is not None and str(rec[0]) == value:
                return False
            self._conn.execute("INSERT OR REPLACE INTO meta_entries(key, value) VALUES (?, ?)", (key, value))
        return True

    def remember_live_namespace(self, namespace: str) -> bool:
        return self.write_meta(META_LIVE_NAMESPACE, namespace)

    def criteria_for_session(self, session_id: str) -> tuple[str, str | None] | None:
        """(rules hash, pending notice marker) a conversation knows; None when nothing is recorded."""
        with self._lock:
            rec = self._conn.execute("SELECT criteria_hash, pending_marker FROM session_save_criteria "
                                     "WHERE session_id=?", (session_id,)).fetchone()
        return None if rec is None else (str(rec[0]), None if rec[1] is None else str(rec[1]))

    def store_session_criteria(self, session_id: str, crit_digest: str,
                                     pending_marker: str | None = None) -> None:
        with self._write_scope():
            self._conn.execute(
                "INSERT INTO session_save_criteria(session_id, criteria_hash, recorded_at, pending_marker) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET criteria_hash=excluded.criteria_hash, "
                "recorded_at=excluded.recorded_at, pending_marker=excluded.pending_marker",
                (session_id, crit_digest, time.time(), pending_marker))

    # -- preferences / constraints (v23, preferences.py; no revisions: edits are in place) ----------------

    def _pack_optional(self, embedding: Iterable[float] | None) -> bytes:
        """Packed unit vector, or the pending mark for None (the daily embed-pending stage fills it)."""
        return PENDING_EMBED_MARK if embedding is None else self._pack_embedding(embedding)

    def add_preference(self, namespace: str, fields: dict, *, embedding: Iterable[float] | None,
                       situation_embedding: Iterable[float] | None) -> int | None:
        """New preferences row (fields: preferences.clean_fields), or None for an exact duplicate."""
        text_blob = self._pack_optional(embedding)
        situation_blob = self._pack_optional(situation_embedding if fields.get("situation") else None)
        with self._write_scope():
            return preferences.insert_preference(self._conn, namespace, fields, embedding=text_blob,
                                                 situation_embedding=situation_blob)

    def preference(self, namespace: str | None, pref_id: int) -> dict | None:
        with self._lock:
            return preferences.get_preference(self._conn, pref_id, namespace)

    def change_preference(self, namespace: str, pref_id: int, changes: dict, *, expected_text: str | None = None,
                          embedding: Iterable[float] | None = None,
                          situation_embedding: Iterable[float] | None = None) -> str:
        """In-place edit: "updated", "unavailable" (no such row of namespace), "conflict" (expected_text is not
        the stored text) or "duplicate" (the owner already has that type and text)."""
        text_blob = None if embedding is None else self._pack_embedding(embedding)
        situation_blob = None if situation_embedding is None else self._pack_embedding(situation_embedding)
        with self._write_scope():
            row = self._conn.execute(f"SELECT text FROM {PREFERENCES_TABLE} WHERE id=? AND owner=?",
                                     (int(pref_id), namespace)).fetchone()
            if row is None:
                return "unavailable"
            if expected_text is not None and row[0] != expected_text:
                return "conflict"
            if not preferences.update_preference(self._conn, pref_id, changes, embedding=text_blob,
                                                 situation_embedding=situation_blob):
                return "duplicate"
        return "updated"

    def remove_preference(self, namespace: str, pref_id: int) -> bool:
        with self._write_scope():
            return preferences.delete_preference(self._conn, pref_id, namespace)

    @staticmethod
    def _cosine(lhs: Iterable[float], rhs: Iterable[float]) -> float:
        """Cosine as a plain dot product: stored and query vectors are unit vectors (normalize_rows).
        Unequal lengths (an embedding-pending empty BLOB, a stale width) score 0. Searches use dot_similarities;
        this is the one-pair form (Python 3.11 has no math.sumprod)."""
        a, b = list(lhs), list(rhs)
        if len(a) != len(b):
            return 0.0
        return sum(map(operator.mul, a, b))

    @contextlib.contextmanager
    def _write_scope(self):
        """Process lock plus a committing SQLite transaction scope."""
        with contextlib.ExitStack() as held:
            held.enter_context(self._lock)
            held.enter_context(self._conn)
            yield

    def _bulk_write(self, stmt: str, rows) -> None:
        self._conn.executemany(stmt, rows)

    def _begin_write(self) -> None:
        self._conn.execute(_BEGIN_WRITE_SQL)

    def _abort(self) -> None:
        conn = self._conn
        conn.rollback()

    def close(self):
        with self._lock:
            self._conn.close()
