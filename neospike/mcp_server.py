"""MCP stdio server for Claude Code: read notes and memories, append TASK reports.

Run: python -m neospike.mcp_server [--home <HERMES_HOME>] [--owner <owner>]  (v21 --namespace: accepted with a warning)
Register (install.py does this, not this module):
    claude mcp add --scope user neospike -- <runtime python> -I <plugins>/neospike/hermes_launch.py <hermes checkout> neospike.mcp_server

Reads open the database with mode=ro and never write (no access bookkeeping).
Query embeddings come from the shared embedding daemon (EmbeddingClient). The
only write, note_append_report, runs the CLI in a subprocess: the single
writer path shared with the dashboard. Every tool answers with JSON text and
never shortens note bodies or chunk texts.

The `mcp` SDK is the optional extra `neospike[mcp]`; this module
imports without it, and main() reports the missing extra.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sqlite3
import subprocess
import sys
import threading
from operator import itemgetter
from pathlib import Path
from typing import Any

from ._wincompat import default_hermes_home
from .config import EMBEDDING_DIMENSIONS, PAUSED_MESSAGE, NeospikeConfig, read_enabled
from .database import canonical_database_path
from . import hermes_runtime
from .inference import EmbeddingClient
from .renames import forward_old_options
from . import vector_index
from .notes import NoteError, NoteNotFound, _NOT_TOPIC_PAGE_SQL, note_record, note_related
from .store import (DEFAULT_NAMESPACE, DEFAULT_SEMANTIC_CUTOFF, FORMULA_AUTHOR, ITEM_DOC, ITEM_MEMORY, NOTE_CHUNKS_SQL,
                    SQLITE_BUSY_TIMEOUT_SECONDS, OLD_VERSION_SCORE_FACTOR,
                    MemoryStore, vector_present_sql, pick_namespace)

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # optional extra [mcp]
    MCPServer = None

    class ToolError(Exception):
        """Stand-in so the tool class imports without the SDK."""


SERVER_NAME = "neospike"
# DEFAULT_NAMESPACE (store.py) applies only when no live provider namespace is recorded.
MAX_RESULTS = 20
DEFAULT_RESULTS = 5
CLI_TIMEOUT_SECONDS = 180
EXIT_MISSING_SDK = 2
CLI_MODULE = "neospike.cli"
# None: the CLI child runs this same copy of neospike through hermes_launch.py (hermes_runtime.launch_command);
# tests replace this prefix.
CLI_PREFIX: list[str] | None = None
_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)
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
        "Read one Neospike note or doc (e.g. a TASK_NNN work order) by numeric id or exact title. Returns the full body "
        "(never shortened), rev, author, tags, project and outgoing links as JSON. Trashed notes are not found."
    ),
    "mem_search": (
        "Search memories (facts, decisions, constraints), notes (agent notes such as TASK_ orders, automatic notes) "
        f"and docs (notes the user wrote) at once, cosine >= {MEMORY_SEARCH_COSINE_THRESHOLD}, best first, limit 1-20. "
        "Each result has kind memory, note or doc. note/doc: note_id, title, chunk_id, heading_path, text. memory: "
        "id, text, origin, confidence; superseded memories only with include_history, at half score with "
        "superseded_by=N: mem:N is the current decision. Optional tag keeps only items with that tag (e.g. 'task', "
        "'status:active')."
    ),
    "note_related": (
        "Graph neighbours of one note (id or exact title): backlinks from notes and memories, outgoing links, "
        "broken links and the most similar notes, as JSON."
    ),
    "note_append_report": (
        "Append a work report to the '## Report' section of a TASK_ note (title must start with 'TASK_'; '## 보고' "
        "is also recognised, and a missing section is created titled in the configured extraction language). Only "
        "that section grows; every other section stays byte-for-byte. The note rev rises by 1. The text must not contain "
        "'#' or '##' headings or unclosed code fences; secrets and prompt-injection text are rejected."
    ),
}
TOOL_NAMES = tuple(TOOL_DESCRIPTIONS)


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _failure(code: str, message: str, **details: Any) -> ToolError:
    return ToolError(_json({"error": code, "message": message, **details}))


def _limit(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= MAX_RESULTS:
        raise _failure("invalid", f"limit must be an integer from 1 to {MAX_RESULTS}")
    return value


def _visible_memory_rows(connection: sqlite3.Connection, namespace: str, project: str,
                         session_id: str, tag: str | None = None) -> list[sqlite3.Row]:
    """Memories of *namespace* visible from project/session (MemoryStore._scope_clause); with tag, only those
    whose metadata tags hold it."""
    visibility, visibility_args = MemoryStore._scope_clause(project, session_id)
    history_columns = ",invalid_at,superseded_by"
    tag_sql, tag_args = "", ()
    if tag:
        tag_sql = (" AND json_valid(metadata_json) AND EXISTS (SELECT 1 FROM json_each(metadata_json,'$.tags') "
                   "WHERE json_each.value=?)")
        tag_args = (tag,)
    return connection.execute(
        "SELECT id,text,origin,kind,confidence,importance,created_at" + history_columns + " "
        f"FROM items WHERE owner=? AND type='{ITEM_MEMORY}' AND {vector_present_sql()}" + visibility + tag_sql,
        (namespace, *visibility_args, *tag_args),
    ).fetchall()


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
    """The four MCP tools. Each method returns JSON text or raises ToolError with a JSON error."""

    def __init__(self, home: Path, namespace: str | None = None) -> None:
        self.home = Path(home).expanduser()
        self.config = NeospikeConfig.load(self.home)
        self.database = canonical_database_path(self.home / "neospike", "memory")
        self.namespace = pick_namespace(namespace, self.database)
        self._embedder: Any = None
        self._embedder_lock = threading.Lock()

    # -- helpers -----------------------------------------------------------------------------

    def _refuse_when_paused(self) -> None:
        """The on/off switch, read at every call: a search or a write (both may start the embedding daemon) is
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

    def note_related(self, id_or_title: str) -> str:
        return self._guarded(self._note_related, id_or_title)

    def _note_related(self, connection: sqlite3.Connection, id_or_title: str) -> dict:
        note = self._resolve(connection, id_or_title)
        return {"title": note["title"], **note_related(connection, note["id"])}

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

    def note_append_report(self, title: str, text: str) -> str:
        self._refuse_when_paused()
        code, payload = self._cli(["note-append-report"], {"title": title, "text": text})
        if code or "error" in payload:
            raise ToolError(_json(payload))
        return _json(payload)

    def _cli(self, arguments: list[str], payload: dict) -> tuple[int, dict]:
        """Run one CLI command with JSON on stdin; return (exit code, parsed JSON object)."""
        try:
            prefix = CLI_PREFIX or hermes_runtime.launch_command(self.home, [CLI_MODULE])
        except hermes_runtime.HermesRuntimeNotFound:
            raise _failure("unavailable", "Neospike write command could not run: Hermes runtime not found") from None
        command = [*prefix, "--home", str(self.home), "--owner", self.namespace, *arguments]
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (_PACKAGE_PARENT, env.get("PYTHONPATH"))))
        try:
            result = subprocess.run(command, input=_json(payload).encode("utf-8"), capture_output=True,
                                    timeout=CLI_TIMEOUT_SECONDS, env=env)
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise _failure("unavailable", f"Neospike write command could not run: {type(exc).__name__}") from None
        text = result.stdout.decode("utf-8", errors="replace").strip()
        for candidate in (text, text.rsplit("\n", 1)[-1]):
            try:
                parsed = json.loads(candidate)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return result.returncode, parsed
        raise _failure("unavailable", f"Neospike write command failed with status {result.returncode}")


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
    server = MCPServer(SERVER_NAME, version=server_version(), instructions=(
        "Neospike second brain. TASK_NNN notes hold work orders; read them with note_read and report with "
        "note_append_report. Docs (notes the user wrote) are read-only. Before you decide, or ask the user, about anything that may already be "
        "decided (settings, thresholds, designs, fixes), search mem_search (memories, notes and docs, the notes "
        "the user wrote) first; cite mem:N or the title. A work order may already list related decisions; search for the rest."
    ))
    for name in TOOL_NAMES:
        server.add_tool(getattr(tools, name), name=name, description=TOOL_DESCRIPTIONS[name], structured_output=False)
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
