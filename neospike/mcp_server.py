"""MCP stdio server for Claude Code: read notes, memories and past Hermes conversations.

Run: python -m neospike.mcp_server [--home <HERMES_HOME>] [--owner <owner>]  (v21 --namespace: accepted with a warning)
Register (install.py does this, not this module):
    claude mcp add --scope user neospike-mcp -- <runtime python> -I <plugins>/neospike/hermes_launch.py <hermes checkout> neospike.mcp_server

Reads open the database with mode=ro and never write (no access bookkeeping).
Query embeddings come from the shared embedding daemon (EmbeddingClient). Every
tool is read-only. Every tool answers with JSON text and never shortens note
bodies or chunk texts.

The `mcp` SDK is the optional extra `neospike[mcp]`; this module
imports without it, and main() reports the missing extra.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import sqlite3
import sys
import threading
from operator import itemgetter
from pathlib import Path
from typing import Any

from ._wincompat import default_hermes_home
from .config import EMBEDDING_DIMENSIONS, PAUSED_MESSAGE, NeospikeConfig, read_enabled
from .database import canonical_database_path
from .inference import EmbeddingClient
from .renames import forward_old_options
from . import vector_index
from . import chat_history
from .notes import NoteError, NoteNotFound, _NOT_TOPIC_PAGE_SQL, note_record
from .store import (DEFAULT_NAMESPACE, DEFAULT_SEMANTIC_CUTOFF, FORMULA_AUTHOR, ITEM_DOC, ITEM_MEMORY, ITEM_NOTE,
                    NOTE_CHUNKS_SQL, SQLITE_BUSY_TIMEOUT_SECONDS, OLD_VERSION_SCORE_FACTOR,
                    MemoryStore, note_head_sql, vector_present_sql, pick_namespace)

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # optional extra [mcp]
    MCPServer = None

    class ToolError(Exception):
        """Stand-in so the tool class imports without the SDK."""


SERVER_NAME = "neospike-mcp"
# DEFAULT_NAMESPACE (store.py) applies only when no live provider namespace is recorded.
MAX_RESULTS = 20
DEFAULT_RESULTS = 5
EXIT_MISSING_SDK = 2
# The version the server reports at initialize: plugin.yaml's top-level `version:` (the copy installed with it).
PLUGIN_MANIFEST = Path(__file__).resolve().with_name("plugin.yaml")
_VERSION_KEY = "version:"
# The store's admission floor; read from MemoryStore so the value lives in one place.
SEMANTIC_THRESHOLD = float(inspect.signature(MemoryStore.__init__).parameters["semantic_threshold"].default)
# mem_search keeps memories, note chunks and doc chunks with cosine >= this, highest cosine first, cut to
# limit. One constant, the store's (Hermes recall keeps SEMANTIC_THRESHOLD).
MEMORY_SEARCH_COSINE_THRESHOLD = DEFAULT_SEMANTIC_CUTOFF
KIND_NOTE_CHUNK = "note_chunk"  # search_memory_and_formula_readonly kind of an automatic-note chunk (evaluation lane)
# mem_search result kinds: a memory item, a chunk of a note (agent or automatic) and a chunk of a doc.
KIND_MEMORY, KIND_NOTE, KIND_DOC = "memory", "note", "doc"
DOC_AUTHOR = "human"  # a doc is a note the user wrote (notes.author); the table is the same

TOOL_DESCRIPTIONS = {
    "note_read": (
        "Read the full body of one note or doc (a TASK_ work order, an agent note, a note the user wrote) by id or "
        "exact title. id_or_title is a numeric id or an exact title (e.g. TASK_NNN). Returns the body (never "
        "shortened), rev, author, tags, project and outgoing links as JSON. Trashed notes are not found."
    ),
    "mem_search": (
        "Find past decisions, memories, notes and docs by meaning; use first before deciding or asking about anything "
        "that may already be decided. Covers memories (facts, decisions, constraints), notes (agent notes such as "
        "TASK_ orders, automatic notes) and docs (notes the user wrote) in one search, "
        f"cosine >= {MEMORY_SEARCH_COSINE_THRESHOLD}, best first, limit 1-20. "
        "Each result has kind memory, note or doc. note/doc: note_id, title, chunk_id, heading_path, text. memory: "
        "id, text, origin, confidence; superseded memories only with include_history, at half score with "
        "superseded_by=N: mem:N is the current decision. Optional tag keeps only items with that tag (e.g. 'task', "
        "'status:active')."
    ),
    "mem_period": (
        "List memories, notes and docs created or changed in a time range (e.g. the last 2 days), without a query. "
        "Range: since <= changed < until; changed = updated_at, else created_at. Newest first, limit 1-20 "
        "(default 10), no embedding. since is "
        "required, until defaults to now; both accept ISO time (local without a zone), epoch seconds or 24h/7d/2w. "
        "kind memory, note, doc or all; tag and include_history as in mem_search. Notes and docs come without "
        "the body (note_id, title, tags, rev): read one with note_read. truncated=true: call again with until = "
        "the last result's timestamp (order oldest: since)."
    ),
    "chat_search": (
        "Find exact words, numbers, file names, commands or error texts in past Hermes conversations, tool calls and "
        "tool results. Full-text search over the read-only chat history database, user and assistant messages and "
        "tool call arguments included. query is an FTS5 expression; column filters content:, tool_name:, "
        "tool_calls:. Deleted conversations are not found. "
        "exclude_notifications drops Hermes gateway notifications (background-process and delegation results) from "
        "user messages; default true for chat_period, false for chat_search."
    ),
    "chat_read": (
        "Read one past chat message in full with its tool calls, their results and neighbouring messages. "
        "message_id comes from chat_search or chat_period; context = how many neighbours on each side (default 0)."
    ),
    "chat_period": (
        "List what happened in a time range (e.g. yesterday) without a query: conversations with a summary, then "
        "messages. Messages: since <= time < until, newest first, limit 1-50 (default 20). Summary of the whole "
        "range regardless of limit (sessions: title, first and last message time, message count; up to 50, most "
        "recent first). since is required, until defaults to now; "
        "both accept ISO time (local without a zone), epoch seconds or 24h/7d/2w. Filters session, role, "
        "tool_name as in chat_search. truncated=true: call again with until = the last result's timestamp "
        "(order oldest: since). "
        "exclude_notifications drops Hermes gateway notifications (background-process and delegation results) from "
        "user messages; default true for chat_period, false for chat_search."
    ),
}
TOOL_NAMES = tuple(TOOL_DESCRIPTIONS)
# Claude Code loads a tool marked with this _meta key at session start, without a tool search.
TOOL_META = {"anthropic/alwaysLoad": True}
SERVER_INSTRUCTIONS = (
    "Neospike is the user's long-term memory for this machine: past decisions, memories, notes (TASK_ work orders, "
    "agent and automatic notes), docs (notes the user wrote, read-only) and the full Hermes chat history. All tools "
    "are read-only.\n"
    "\n"
    "Search here when you: start a task or a new session; are about to decide, or ask the user, something that may "
    "already be decided (settings, thresholds, names, designs, fixes, rules); see a work order mention an earlier "
    "TASK, decision or bug; need what happened before (yesterday, last week, an earlier attempt, an error seen "
    "before).\n"
    "\n"
    "Pick the tool by what you know:\n"
    "1. The meaning, not the exact words -> mem_search (memories, notes and docs by meaning). Always first for "
    "decisions; cite mem:N or the note title.\n"
    "2. An exact word, number, file name, command or error text from past work -> chat_search (full-text over "
    "conversations, tool calls and tool results).\n"
    "3. Only the time (\"yesterday\", \"the last 2 days\") -> chat_period for conversations (with a per-conversation "
    "summary) or mem_period for memories and notes created or changed then.\n"
    "4. Then read in full: note_read for a note or doc body, chat_read for one message with its tool results and "
    "neighbours.\n"
    "\n"
    "Order: mem_search first; go to chat_search or chat_period only for original wording or evidence; then "
    "chat_read. Start with a small limit and narrow with since/until, role or session instead of raising it. A newer "
    "decision overrides an older one; superseded memories appear only with include_history."
)


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _failure(code: str, message: str, **details: Any) -> ToolError:
    return ToolError(_json({"error": code, "message": message, **details}))


def _limit(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= MAX_RESULTS:
        raise _failure("invalid", f"limit must be an integer from 1 to {MAX_RESULTS}")
    return value


def _visible_memory_where(namespace: str, project: str, session_id: str,
                          tag: str | None = None) -> tuple[str, tuple]:
    """WHERE condition (no leading AND) and its parameters: memories of *namespace* visible from project/session
    (MemoryStore._scope_clause); with tag, only those whose metadata tags hold it."""
    visibility, visibility_args = MemoryStore._scope_clause(project, session_id)
    tag_sql, tag_args = "", ()
    if tag:
        tag_sql = (" AND json_valid(metadata_json) AND EXISTS (SELECT 1 FROM json_each(metadata_json,'$.tags') "
                   "WHERE json_each.value=?)")
        tag_args = (tag,)
    return (f"owner=? AND type='{ITEM_MEMORY}'" + visibility + tag_sql,
            (namespace, *visibility_args, *tag_args))


def _visible_memory_rows(connection: sqlite3.Connection, namespace: str, project: str,
                         session_id: str, tag: str | None = None) -> list[sqlite3.Row]:
    """Visible memories (_visible_memory_where) that have a vector."""
    where, args = _visible_memory_where(namespace, project, session_id, tag)
    return connection.execute(
        "SELECT id,text,origin,kind,confidence,importance,created_at,invalid_at,superseded_by "
        f"FROM items WHERE {where} AND {vector_present_sql()}",
        args,
    ).fetchall()


# mem_period: when an item last changed (updated_at; created_at when it has none).
_CHANGED_SQL = "COALESCE(updated_at,created_at)"
PERIOD_KINDS = (KIND_MEMORY, KIND_NOTE, KIND_DOC, "all")
DEFAULT_PERIOD_RESULTS = 10


def _period_order_sql(order: str, id_column: str = "id") -> str:
    direction = "ASC" if order == chat_history.ORDER_OLDEST else "DESC"
    return f" ORDER BY changed {direction}, {id_column} {direction} LIMIT ?"


def _period_item_times(row: sqlite3.Row) -> dict:
    changed = float(row["changed"])
    return {"time": chat_history._local_time(changed), "timestamp": changed, "created_at": row["created_at"],
            "updated_at": row["updated_at"]}


def _memory_period_items(connection: sqlite3.Connection, namespace: str, start: float, end: float, *, project: str,
                         session_id: str, include_history: bool, tag: str | None, order: str,
                         wanted: int) -> list[dict]:
    """Visible memories (embedded or not) changed in [start, end); superseded ones only with include_history."""
    where, args = _visible_memory_where(namespace, project, session_id, tag)
    history_sql = "" if include_history else " AND invalid_at IS NULL"
    rows = connection.execute(
        f"SELECT id,text,origin,kind,confidence,created_at,updated_at,invalid_at,superseded_by,"
        f"{_CHANGED_SQL} AS changed FROM items WHERE {where}{history_sql} "
        f"AND {_CHANGED_SQL}>=? AND {_CHANGED_SQL}<?" + _period_order_sql(order),
        (*args, start, end, wanted)).fetchall()
    items = []
    for row in rows:
        item = {"kind": KIND_MEMORY, "id": int(row["id"]), "text": row["text"], "origin": row["origin"],
                "item_kind": row["kind"], "confidence": float(row["confidence"]), **_period_item_times(row)}
        if row["invalid_at"] is not None and row["superseded_by"] is not None:
            item["superseded_by"] = int(row["superseded_by"])
        items.append(item)
    return items


def _tag_list(tags_json: Any) -> list:
    try:
        tags = json.loads(tags_json)
    except (TypeError, ValueError):
        return []
    return tags if isinstance(tags, list) else []


def _document_period_items(connection: sqlite3.Connection, start: float, end: float, *, types: tuple[str, ...],
                           tag: str | None, order: str, wanted: int) -> list[dict]:
    """Live notes and docs (no TOPIC pages) changed in [start, end): head rows without the body."""
    tag_sql, tag_args = "", ()
    if tag:
        tag_sql, tag_args = " AND EXISTS (SELECT 1 FROM json_each(n.tags_json) WHERE json_each.value=?)", (tag,)
    placeholders = ",".join("?" for _ in types)
    rows = connection.execute(
        f"SELECT n.id AS id,n.type AS type,n.title AS title,n.tags_json AS tags_json,n.rev AS rev,"
        f"n.created_at AS created_at,n.updated_at AS updated_at,COALESCE(n.updated_at,n.created_at) AS changed "
        f"FROM items n WHERE {note_head_sql('n')} AND n.type IN ({placeholders}) AND n.deleted_at IS NULL"
        + _NOT_TOPIC_PAGE_SQL + tag_sql
        + " AND COALESCE(n.updated_at,n.created_at)>=? AND COALESCE(n.updated_at,n.created_at)<?"
        + _period_order_sql(order, "n.id"),
        (*types, *tag_args, start, end, wanted)).fetchall()
    return [{"kind": KIND_DOC if row["type"] == ITEM_DOC else KIND_NOTE, "note_id": int(row["id"]),
             "title": row["title"], "tags": _tag_list(row["tags_json"]), "rev": int(row["rev"]),
             **_period_item_times(row)} for row in rows]


class _ChangedKey:
    """Sort key of mem_period results: change time, then id (note_id for notes and docs)."""

    @staticmethod
    def of(item: dict) -> tuple[float, int]:
        return item["timestamp"], item.get("id", item.get("note_id", 0))


def mem_period_readonly(connection: sqlite3.Connection, namespace: str, since: float, until: float, *, kind: str,
                        limit: int, project: str, session_id: str, include_history: bool = False,
                        tag: str | None = None, order: str = chat_history.ORDER_NEWEST) -> dict:
    """MCP mem_period: memories, notes and docs whose last change (updated_at, else created_at) lies in
    [since, until), sorted by that time, cut to limit; truncated tells more matched."""
    wanted = limit + 1
    items: list[dict] = []
    if kind in (KIND_MEMORY, "all"):
        items += _memory_period_items(connection, namespace, since, until, project=project, session_id=session_id,
                                      include_history=include_history, tag=tag, order=order, wanted=wanted)
    types = {KIND_NOTE: (ITEM_NOTE,), KIND_DOC: (ITEM_DOC,), "all": (ITEM_NOTE, ITEM_DOC)}.get(kind, ())
    if types:
        items += _document_period_items(connection, since, until, types=types, tag=tag, order=order, wanted=wanted)
    items.sort(key=_ChangedKey.of, reverse=order != chat_history.ORDER_OLDEST)
    results = items[:limit]
    return {"since": chat_history._local_time(since), "until": chat_history._local_time(until),
            "results": results, "count": len(results), "truncated": len(items) > limit}


def _memory_hits(connection: sqlite3.Connection, namespace: str, qvec: list[float], *, project: str,
                 session_id: str, include_history: bool, threshold: float, tag: str | None = None) -> list[dict]:
    """Visible memories with cosine >= threshold; score = cosine (x OLD_VERSION_SCORE_FACTOR for a superseded
    memory, include_history only). kind is the memory row's kind column."""
    hits = []
    rows = _visible_memory_rows(connection, namespace, project, session_id, tag)
    for row, cosine in zip(rows, vector_index.similarities(connection, "items", [row["id"] for row in rows],
                                                           qvec, threshold=threshold)):
        invalid = row["invalid_at"] is not None
        if invalid and not include_history:
            continue
        if not math.isfinite(cosine) or cosine < threshold:
            continue
        hit = {"id": int(row["id"]), "text": row["text"], "origin": row["origin"], "kind": row["kind"],
               "score": cosine * OLD_VERSION_SCORE_FACTOR if invalid else cosine,
               "confidence": float(row["confidence"])}
        if invalid and row["superseded_by"] is not None:
            hit["superseded_by"] = int(row["superseded_by"])
        hits.append(hit)
    return hits


def _query_vector(query_embedding: Any, dimensions: int) -> list[float]:
    qvec = list(query_embedding)
    if len(qvec) != dimensions:
        raise ValueError(f"Expected {dimensions} dimensions, got {len(qvec)}")
    return qvec


def search_memory_and_formula_readonly(connection: sqlite3.Connection, namespace: str, query_embedding: Any, *,
                                       dimensions: int, limit: int, project: str, session_id: str,
                                       include_history: bool = False,
                                       threshold: float = MEMORY_SEARCH_COSINE_THRESHOLD) -> list[dict]:
    """Evaluation memory lane (the former MCP memory search): memories (memory visibility rules) plus chunks of live automatic notes
    (author FORMULA_AUTHOR), cosine >= threshold, score = cosine (x OLD_VERSION_SCORE_FACTOR for a superseded
    memory, include_history only), highest first, cut to limit. Human-note chunks never come here."""
    qvec = _query_vector(query_embedding, dimensions)
    ranked = _memory_hits(connection, namespace, qvec, project=project, session_id=session_id,
                          include_history=include_history, threshold=threshold)
    chunks = connection.execute(
        f"SELECT c.id,c.note_id,c.text,n.title FROM {NOTE_CHUNKS_SQL} c JOIN items n ON n.id=c.note_id "
        f"WHERE n.deleted_at IS NULL AND n.author=? AND {vector_present_sql('c.embedding')}",
        (FORMULA_AUTHOR,)).fetchall()
    for row, cosine in zip(chunks, vector_index.similarities(connection, "items", [row["id"] for row in chunks],
                                                             qvec, threshold=threshold)):
        if not math.isfinite(cosine) or cosine < threshold:
            continue
        ranked.append({"kind": KIND_NOTE_CHUNK, "note_id": int(row["note_id"]), "title": row["title"],
                       "chunk_id": int(row["id"]), "text": row["text"], "score": cosine})
    ranked.sort(key=itemgetter("score"), reverse=True)
    return ranked[:limit]


def mem_search_readonly(connection: sqlite3.Connection, namespace: str, query_embedding: Any, *, dimensions: int,
                        limit: int, project: str, session_id: str, include_history: bool = False,
                        tag: str | None = None, threshold: float = MEMORY_SEARCH_COSINE_THRESHOLD) -> list[dict]:
    """MCP mem_search: memories (as search_memory_and_formula_readonly, kind memory, the row kind in item_kind)
    plus parts of every live document except TOPIC pages: kind = the document's items.type (doc: written by the
    user; note: agent and automatic notes). cosine >= threshold, score = cosine, highest first, cut to limit. tag keeps
    memories and notes carrying it."""
    qvec = _query_vector(query_embedding, dimensions)
    ranked = []
    for hit in _memory_hits(connection, namespace, qvec, project=project, session_id=session_id,
                            include_history=include_history, threshold=threshold, tag=tag):
        hit["item_kind"] = hit["kind"]
        hit["kind"] = KIND_MEMORY
        ranked.append(hit)
    tag_sql, tag_args = "", ()
    if tag:
        tag_sql, tag_args = " AND EXISTS (SELECT 1 FROM json_each(n.tags_json) WHERE json_each.value=?)", (tag,)
    chunks = connection.execute(
        f"SELECT c.id,c.note_id,c.heading_path,c.text,n.title,n.type FROM {NOTE_CHUNKS_SQL} c "
        f"JOIN items n ON n.id=c.note_id WHERE n.deleted_at IS NULL AND {vector_present_sql('c.embedding')}"
        + _NOT_TOPIC_PAGE_SQL + tag_sql,
        tag_args).fetchall()
    for row, cosine in zip(chunks, vector_index.similarities(connection, "items", [row["id"] for row in chunks],
                                                             qvec, threshold=threshold)):
        if not math.isfinite(cosine) or cosine < threshold:
            continue
        ranked.append({"kind": KIND_DOC if row["type"] == ITEM_DOC else KIND_NOTE,
                       "note_id": int(row["note_id"]), "title": row["title"], "chunk_id": int(row["id"]),
                       "heading_path": row["heading_path"], "text": row["text"], "score": cosine})
    ranked.sort(key=itemgetter("score"), reverse=True)
    return ranked[:limit]


class _ReadOnlyConnection:
    """`with _ReadOnlyConnection(path) as conn`: a mode=ro connection to the memory database."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise _failure("not_found", "Neospike database does not exist yet")
        self.connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True,
                                          timeout=SQLITE_BUSY_TIMEOUT_SECONDS)
        self.connection.row_factory = sqlite3.Row
        return self.connection

    def __exit__(self, *exc_info: Any) -> None:
        if self.connection is not None:
            self.connection.close()


class NeospikeTools:
    """The six MCP tools, all read-only. Each method returns JSON text or raises ToolError with a JSON error."""

    def __init__(self, home: Path, namespace: str | None = None) -> None:
        self.home = Path(home).expanduser()
        self.config = NeospikeConfig.load(self.home)
        self.database = canonical_database_path(self.home / "neospike", "memory")
        self.namespace = pick_namespace(namespace, self.database)
        self.chat_database = chat_history.chat_database_path(self.home)
        self._embedder: Any = None
        self._embedder_lock = threading.Lock()

    # -- helpers -----------------------------------------------------------------------------

    def _refuse_when_paused(self) -> None:
        """The on/off switch, read at every call: a memory search (it may start the embedding daemon) is
        refused while it is off; reading a note still works."""
        if not read_enabled(self.home):
            raise _failure("paused", PAUSED_MESSAGE)

    def _query_embedding(self, query: str) -> list[float]:
        with self._embedder_lock:
            if self._embedder is None:
                self._embedder = EmbeddingClient(self.home)
            return list(self._embedder.embed_query(query))

    @staticmethod
    def _resolve(connection: sqlite3.Connection, id_or_title: Any) -> dict:
        """Exact title first; a string of digits that is no title is an id. Trashed notes are not found."""
        if type(id_or_title) is int:
            return note_record(connection, note_id=id_or_title)
        if not isinstance(id_or_title, str) or not id_or_title.strip():
            raise NoteError("id_or_title must be a note id or a non-empty title")
        try:
            return note_record(connection, title=id_or_title)
        except NoteNotFound:
            if id_or_title.strip().isdigit():
                return note_record(connection, note_id=int(id_or_title.strip()))
            raise

    def _guarded(self, method: Any, *args: Any) -> str:
        try:
            with _ReadOnlyConnection(self.database) as connection:
                return _json(method(connection, *args))
        except NoteError as exc:
            raise ToolError(_json(exc.as_dict())) from None
        except sqlite3.OperationalError as exc:
            raise _failure("unavailable", f"Database read failed: {exc}") from None

    # -- tools ---------------------------------------------------------------------------------

    def note_read(self, id_or_title: str) -> str:
        return self._guarded(self._note_read, id_or_title)

    def _note_read(self, connection: sqlite3.Connection, id_or_title: str) -> dict:
        return {"note": self._resolve(connection, id_or_title)}

    def mem_search(self, query: str, limit: int = DEFAULT_RESULTS, include_history: bool = False,
                   tag: str | None = None) -> str:
        limit = _limit(limit)
        if not isinstance(query, str) or not query.strip():
            raise _failure("invalid", "query must be a non-empty string")
        if not isinstance(include_history, bool):
            raise _failure("invalid", "include_history must be a boolean")
        if tag is not None and not isinstance(tag, str):
            raise _failure("invalid", "tag must be a string")
        self._refuse_when_paused()
        return self._guarded(self._mem_search, query, limit, include_history, tag or None)

    def _mem_search(self, connection: sqlite3.Connection, query: str, limit: int, include_history: bool,
                    tag: str | None) -> dict:
        # Same authority as the standalone CLI search: the working directory, no session.
        results = mem_search_readonly(connection, self.namespace, self._query_embedding(query),
                                      dimensions=EMBEDDING_DIMENSIONS, limit=limit,
                                      project=str(Path.cwd().resolve()), session_id="",
                                      include_history=include_history, tag=tag)
        return {"results": results}

    def mem_period(self, since: float | str, until: float | str | None = None, kind: str = "all",
                   tag: str | None = None, include_history: bool = False, order: str = chat_history.ORDER_NEWEST,
                   limit: int = DEFAULT_PERIOD_RESULTS) -> str:
        limit = _limit(limit)
        if kind not in PERIOD_KINDS:
            raise _failure("invalid", f"kind must be one of: {', '.join(PERIOD_KINDS)}")
        if order not in chat_history.PERIOD_ORDERS:
            raise _failure("invalid", f"order must be one of: {', '.join(chat_history.PERIOD_ORDERS)}")
        if not isinstance(include_history, bool):
            raise _failure("invalid", "include_history must be a boolean")
        if tag is not None and not isinstance(tag, str):
            raise _failure("invalid", "tag must be a string")
        try:
            start, end = chat_history.parse_period(since, until)
        except chat_history.ChatHistoryError as exc:
            raise _failure(exc.code, exc.message) from None
        self._refuse_when_paused()
        return self._guarded(self._mem_period, start, end, kind, tag or None, include_history, order, limit)

    def _mem_period(self, connection: sqlite3.Connection, start: float, end: float, kind: str, tag: str | None,
                    include_history: bool, order: str, limit: int) -> dict:
        # The visibility of mem_search: the working directory, no session.
        return mem_period_readonly(connection, self.namespace, start, end, kind=kind, limit=limit,
                                   project=str(Path.cwd().resolve()), session_id="",
                                   include_history=include_history, tag=tag, order=order)

    # Hermes chat history: no embedding daemon, so like note_read these work while the switch is off.

    def chat_search(self, query: str, session: str | None = None, role: str | None = None,
                    tool_name: str | None = None, since: float | str | None = None, until: float | str | None = None,
                    order: str = chat_history.ORDER_RECENT, limit: int = chat_history.DEFAULT_SEARCH_RESULTS,
                    exclude_notifications: bool = chat_history.DEFAULT_EXCLUDE_NOTIFICATIONS_SEARCH) -> str:
        try:
            return _json(chat_history.search(self.chat_database, query, session=session, role=role,
                                             tool_name=tool_name, since=since, until=until, order=order, limit=limit,
                                             exclude_notifications=exclude_notifications))
        except chat_history.ChatHistoryError as exc:
            raise _failure(exc.code, exc.message) from None

    def chat_read(self, message_id: int, context: int = 0) -> str:
        try:
            return _json(chat_history.read(self.chat_database, message_id, context=context))
        except chat_history.ChatHistoryError as exc:
            raise _failure(exc.code, exc.message) from None

    def chat_period(self, since: float | str, until: float | str | None = None, session: str | None = None,
                    role: str | None = None, tool_name: str | None = None, order: str = chat_history.ORDER_NEWEST,
                    limit: int = chat_history.DEFAULT_PERIOD_RESULTS,
                    exclude_notifications: bool = chat_history.DEFAULT_EXCLUDE_NOTIFICATIONS_PERIOD) -> str:
        try:
            return _json(chat_history.period(self.chat_database, since, until, session=session, role=role,
                                             tool_name=tool_name, order=order, limit=limit,
                                             exclude_notifications=exclude_notifications))
        except chat_history.ChatHistoryError as exc:
            raise _failure(exc.code, exc.message) from None


def server_version(manifest: Path = PLUGIN_MANIFEST) -> str:
    """plugin.yaml's top-level version, unquoted; "" when the file or the key is missing."""
    try:
        lines = manifest.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return ""
    for line in lines:
        if line.startswith(_VERSION_KEY):
            return line[len(_VERSION_KEY):].split("#", 1)[0].strip().strip("'\"")
    return ""


def build_server(tools: NeospikeTools) -> Any:
    if MCPServer is None:
        raise RuntimeError("The MCP SDK is not installed; install the extra: pip install 'neospike[mcp]'")
    server = MCPServer(SERVER_NAME, version=server_version(), instructions=SERVER_INSTRUCTIONS)
    for name in TOOL_NAMES:
        server.add_tool(getattr(tools, name), name=name, description=TOOL_DESCRIPTIONS[name], meta=dict(TOOL_META),
                        structured_output=False)
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neospike.mcp_server", description=__doc__.splitlines()[0])
    parser.add_argument("--home", default=default_hermes_home())
    parser.add_argument("--owner", dest="namespace", default=None,
                        help=f"Owner of the memory items (v21: --namespace). Default: the live provider owner "
                             f"recorded in the DB, else {DEFAULT_NAMESPACE}")
    args = parser.parse_args(forward_old_options(sys.argv[1:] if argv is None else list(argv)))
    if MCPServer is None:
        sys.stderr.write("neospike.mcp_server needs the MCP SDK: pip install 'neospike[mcp]'\n")
        return EXIT_MISSING_SDK
    build_server(NeospikeTools(Path(args.home), args.namespace)).run("stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
