"""Notes family: long human/agent documents (items type note/doc) beside the memories in one items table.

A document is a head row (title, tags, rev) followed along item_links 'next' by its part rows (the searchable
text and vectors; store.py). Writes prepare the parts and embeddings before a short BEGIN IMMEDIATE transaction
that does the rev CAS, revision copy, part replace and link recomputation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from array import array
from operator import itemgetter
from pathlib import Path
from typing import Any, Iterable, Sequence

from .chunking import pack_markdown_chunks, count_headings, split_document_parts
from .config import EMBEDDING_MAX_TOKENS
from . import vector_index
from .policy import ITEM_TEXT_CEILING, AdmitVerdict, flagged_line_numbers, screen_text
from .store import (DOC_TYPES_SQL, FORMULA_AUTHOR, FORMULA_HWM_META_KEY, ITEM_MEMORY, LINK_TO_MEMORY_SQL,
                    NOTE_CHUNKS_SQL, NOTE_COLUMNS_SQL, NOTE_HEAD_SQL, PART_HEADING_KEY, PART_SOURCE_KEY,
                    PENDING_EMBED_MARK, RELATION_NEXT, RELATION_TOPIC, SCOPE_ORDER, MemoryStore,
                    document_type, is_embed_pending, freshness, blend_score, next_of, normalize_rows, note_head_sql,
                    vector_present_sql, write_chain)

# A refers recompute must leave the other item_links relations (next, topic, supersedes) alone.
_REFERS = "relation='refers' AND "


TASK_TITLE_PREFIX = "TASK_"
# S5 Express: a note tagged DRAFT_TAG must cite at least one [[...]] source.
DRAFT_TAG = "draft"
AUTHORS = ("human", "agent")
# Who asked for a write. automation must not overwrite human work directly.
ORIGIN_CONVERSATION, ORIGIN_AUTOMATION, ORIGIN_HUMAN = "conversation", "automation", "human"
ORIGINS = (ORIGIN_CONVERSATION, ORIGIN_AUTOMATION, ORIGIN_HUMAN)
# An automation write to a human note is refused (NotePermissionError); nothing is proposed.
HUMAN_NOTE_READ_ONLY = "Automation cannot edit a human note"
MAX_BODY_BYTES = 10_000_000
MAX_TITLE_CHARS = 200
MAX_TAGS, MAX_TAG_CHARS = 16, 64
# Window 2048 - 48 = 2000-token note chunks.
CHUNK_TOKEN_MARGIN = 48
# Notes carry no importance column; use the memories column default.
NOTE_IMPORTANCE = 0.5
# Concurrent rewrites of notes that link a renamed note are retried this often.
MAX_SAVE_ATTEMPTS = 3
SUMMARY_MAX_CHARS = 300
MIN_STRUCTURED_HEADINGS = 2
# A SentencePiece token covers >= 1 character (>= 1 UTF-8 byte); BOS/EOS are extra. bge-m3 embeds the text
# as is (no document prompt), so the count is of the text alone.
_SPECIAL_TOKEN_ALLOWANCE = 2

_WIKI_LINK = re.compile(r"\[\[([^\[\]\n|#]+?)(?:[#|][^\[\]\n]*)?\]\]")
_MEMORY_TARGET = re.compile(r"^mem:(\d{1,19})$")
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_HEADING_LINE = re.compile(r"^#{1,6}[ \t]+")
_BLANK_LINE_RUN = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")
# Automatic (formula) notes: body = chunk texts joined by this separator, in ord order.
FORMULA_CHUNK_SEPARATOR = "\n\n"
_SQL_IN_BATCH = 500


class NoteError(ValueError):
    code = "invalid"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": str(self), **self.details}


class NoteNotFound(NoteError):
    code = "not_found"


class NoteConflict(NoteError):
    code = "conflict"


class NotePermissionError(NoteError):
    code = "forbidden"


class NotePolicyError(NoteError):
    code = "policy"


# --- embedding window and token counting --------------------------------------------------

def embedding_window() -> int:
    """Token window of the embedder (config.EMBEDDING_MAX_TOKENS = embedder.OnnxDenseEmbedder.sequence_length);
    read from config so runtimes without the model packages get the same number."""
    return EMBEDDING_MAX_TOKENS


def note_chunk_max_tokens() -> int:
    """Note chunk ceiling: the embedding window minus a margin for special tokens."""
    return embedding_window() - CHUNK_TOKEN_MARGIN


def _model_dir() -> Path:
    # Mirrors embedder.resolve_model_dir without importing the model packages.
    configured = os.environ.get("HERMES_BGE_M3_DIR")
    if configured:
        return Path(configured).expanduser()
    from ._wincompat import default_hermes_home, shared_inference_home
    return shared_inference_home(default_hermes_home()) / "models" / "bge-m3-onnx"


class TokenCounter:
    """Counts tokens of the embedded document string (special tokens included).

    Uses the embedder's tokenizer.json. Without it, UTF-8 byte length is a safe
    upper bound for a SentencePiece vocabulary.
    """

    def __init__(self, tokenizer_path: Path | None = None) -> None:
        self.source = "utf8-bytes-upper-bound"
        self._tokenizer = None
        path = tokenizer_path or _model_dir() / "tokenizer.json"
        try:
            from tokenizers import Tokenizer
            if path.is_file():
                self._tokenizer = Tokenizer.from_file(str(path))
                self._tokenizer.no_truncation()  # a capped count would hide an over-window chunk
                self._tokenizer.no_padding()
                self.source = str(path)
        except ImportError:
            pass
        self._lock = threading.Lock()

    def count(self, text: str) -> int:
        if self._tokenizer is None:
            return len(text.encode("utf-8")) + _SPECIAL_TOKEN_ALLOWANCE
        with self._lock:
            return len(self._tokenizer.encode(text, add_special_tokens=True).ids)

    def cut(self, text: str, max_tokens: int) -> list[str]:
        """Pieces of text at token offsets, each counting <= max_tokens; [] without a tokenizer."""
        if self._tokenizer is None:
            return []
        overhead = self.count("")
        size = max_tokens - overhead
        if size < 1:
            return []
        with self._lock:
            offsets = self._tokenizer.encode(text, add_special_tokens=False).offsets
        pieces, start_index = [], 0
        while start_index < len(offsets):
            end_index = min(start_index + size, len(offsets))
            while True:
                begin = offsets[start_index][0] if start_index else 0
                end = offsets[end_index][0] if end_index < len(offsets) else len(text)
                piece = text[begin:end]
                # Re-tokenizing a slice can merge differently; shrink until it fits.
                if self.count(piece) <= max_tokens or end_index - start_index <= 1:
                    break
                end_index -= max(1, (end_index - start_index) // 16)
            pieces.append(piece)
            start_index = end_index
        return pieces


_counter_lock = threading.Lock()
_counters: dict[str, TokenCounter] = {}


def default_token_counter() -> TokenCounter:
    key = str(_model_dir())
    with _counter_lock:
        if key not in _counters:
            _counters[key] = TokenCounter()
        return _counters[key]


# --- text helpers --------------------------------------------------------------------------

def is_note_text(text: str) -> bool:
    """remember routing: too long for a memory item, or structured."""
    return len(text) > ITEM_TEXT_CEILING or count_headings(text) >= MIN_STRUCTURED_HEADINGS


def summary_excerpt(text: str, *, max_chars: int = SUMMARY_MAX_CHARS) -> str:
    """First non-heading paragraph, whitespace-collapsed, at most max_chars."""
    fallback = ""
    for block in _PARAGRAPH_BREAK.split(text):
        lines = [line for line in block.splitlines() if line.strip()]
        content = [line for line in lines if not _HEADING_LINE.match(line)]
        if content:
            return _clip(" ".join(" ".join(content).split()), max_chars)
        if lines and not fallback:
            fallback = _HEADING_LINE.sub("", lines[0]).strip()
    return _clip(fallback, max_chars)


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    return (cut[:space] if space > max_chars // 2 else cut).rstrip()


def parse_links(text: str) -> tuple[list[str], list[int]]:
    """Distinct `[[Title]]` note targets and `[[mem:N]]` memory targets, in order."""
    titles: list[str] = []
    memories: list[int] = []
    for match in _WIKI_LINK.finditer(text):
        target = match.group(1).strip()
        memory = _MEMORY_TARGET.match(target)
        if memory:
            value = int(memory.group(1))
            if value not in memories:
                memories.append(value)
        elif target and target not in titles:
            titles.append(target)
    return titles, memories


def rewrite_title_links(text: str, old_title: str, new_title: str) -> str:
    pattern = re.compile(r"\[\[[ \t]*" + re.escape(old_title) + r"[ \t]*(?=\]\]|[#|])")
    return pattern.sub("[[" + new_title.replace("\\", "\\\\"), text)


def _text_hash(text: str) -> str:
    # Embeddings depend on the chunk text only; heading_path is stored beside it.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validated_tags(tags: Any) -> list[str]:
    if tags is None:
        return []
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise NoteError("Note tags must be an array of strings")
    cleaned = [tag.strip() for tag in tags if tag.strip()]
    if len(cleaned) > MAX_TAGS or any(len(tag) > MAX_TAG_CHARS for tag in cleaned):
        raise NoteError(f"Note tags must contain at most {MAX_TAGS} values of at most {MAX_TAG_CHARS} characters")
    return list(dict.fromkeys(cleaned))


def _validated_title(title: Any) -> str:
    if not isinstance(title, str):
        raise NoteError("Note title must be a string")
    cleaned = " ".join(title.split())
    if not cleaned or len(cleaned) > MAX_TITLE_CHARS:
        raise NoteError(f"Note title must contain 1 to {MAX_TITLE_CHARS} characters")
    if "[[" in cleaned or "]]" in cleaned or "|" in cleaned or "#" in cleaned:
        raise NoteError("Note title must not contain '[[', ']]', '|' or '#'")
    return cleaned


def check_draft_sources(body: str, tags: list[str]) -> None:
    """A draft note must cite at least one `[[Note title]]` or `[[mem:N]]` source."""
    if DRAFT_TAG not in tags:
        return
    titles, memories = parse_links(body)
    if not titles and not memories:
        raise NoteError("Draft note must cite at least one source link: add [[Note title]] or [[mem:N]] "
                        "to the body for each claim")


def check_note_policy(title: str, body: str, tags: list[str], author: str) -> None:
    """Secret/injection lines reject any note. Agent notes also follow the memory-item
    rule that the text must classify as INDEX (too-short noise is rejected)."""
    lines = flagged_line_numbers(body)
    fields = [name for name, value in (("title", title), *(("tags", tag) for tag in tags))
              if screen_text(value) is AdmitVerdict.BLOCK]
    if lines or fields:
        raise NotePolicyError("Note was rejected by ingestion policy (secret or prompt-injection pattern)",
                              lines=lines, fields=sorted(set(fields)))
    if author == "agent" and screen_text(body) is not AdmitVerdict.INDEX:
        raise NotePolicyError("Agent note was rejected by ingestion policy (too short to be useful)", lines=[], fields=["body"])


class _Prepared:
    """Chunks and embeddings computed outside any write transaction."""

    def __init__(self, chunks: list[tuple[str, str, str, bytes]], *, formula: bool = False) -> None:
        self.chunks = chunks  # (heading_path, text, text_hash, packed embedding)
        self.formula = formula  # automatic note: chunk ids come from insert_formula_chunks


# --- automatic (formula) notes ----------------------------------------------------------
# The chunks are the source: one chunk = one memory body (or one paragraph of a whole-body write).
# Functions taking a connection run inside the caller's write transaction (lock held).

def formula_chunk_text(text: str) -> str:
    """One chunk text: line ends unified, outer whitespace trimmed, a run of blank lines inside reduced
    to one line break (the blank line is the chunk separator of the body)."""
    flat = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return _BLANK_LINE_RUN.sub("\n", flat)


def formula_body(texts: Iterable[str]) -> str:
    return FORMULA_CHUNK_SEPARATOR.join(texts)


# The whitespace a part keeps at its ends (the body is rebuilt byte for byte) is not embedded: a part's vector and
# text_hash are of the text without it (the same characters as the trim of reembed.MEMORY_TABLES).
PART_EMBED_TRIM = "".join(map(chr, (32, 9, 10, 13, 12, 11)))  # space, tab, LF, CR, FF, VT


def part_embed_text(text: str) -> str:
    """The string a document part is embedded (and hashed) as: its text without outer ASCII whitespace."""
    return text.strip(PART_EMBED_TRIM) or text


def formula_chunk_watermark(connection: sqlite3.Connection) -> int:
    """Highest items id ever handed to an automatic-note part (ids are never reused for them)."""
    row = connection.execute("SELECT value FROM meta_entries WHERE key=?", (FORMULA_HWM_META_KEY,)).fetchone()
    top = connection.execute(f"SELECT COALESCE(MAX(id),0) FROM items WHERE title IS NULL AND author=?",
                             (FORMULA_AUTHOR,)).fetchone()[0]
    return max(int(row[0]) if row else 0, int(top))


# --- document rows (v22 items: a head row + part rows along item_links 'next') ---------------------------
# Functions taking a connection run inside the caller's lock (and write transaction for writes).
_CHAIN_LIMIT = 1_000_000
_PART_COLUMNS = "id,text,text_hash,embedding,metadata_json"
_HEAD_SELECT = f"SELECT {NOTE_COLUMNS_SQL},metadata_json FROM items WHERE {NOTE_HEAD_SQL}"


def part_ids(connection: sqlite3.Connection, note_id: int) -> list[int]:
    """Part row ids of one document in order (head excluded): along the next chain from the head."""
    ids: list[int] = []
    seen = {int(note_id)}
    current = next_of(connection, note_id)
    while current is not None and current not in seen and len(ids) < _CHAIN_LIMIT:
        ids.append(current)
        seen.add(current)
        current = next_of(connection, current)
    return ids


def part_rows(connection: sqlite3.Connection, note_id: int) -> list[sqlite3.Row]:
    """(id, text, text_hash, embedding, metadata_json) of the parts of one document, in order."""
    ids = part_ids(connection, note_id)
    found: dict[int, Any] = {}
    for start in range(0, len(ids), _SQL_IN_BATCH):
        batch = ids[start:start + _SQL_IN_BATCH]
        for row in connection.execute(f"SELECT {_PART_COLUMNS} FROM items WHERE id IN ({','.join('?' * len(batch))})",
                                      batch):
            found[int(row[0])] = row
    return [found[item_id] for item_id in ids if item_id in found]


def body_of_parts(texts: Iterable[str], author: str) -> str:
    """A document body from its part texts: concatenated, or joined by a blank line for an automatic note."""
    return formula_body(texts) if author == FORMULA_AUTHOR else "".join(texts)


def note_body(connection: sqlite3.Connection, note_id: int, author: str) -> str:
    return body_of_parts((str(row[1]) for row in part_rows(connection, note_id)), author)


def head_row(connection: sqlite3.Connection, *, note_id: int | None = None, title: str | None = None) -> dict | None:
    """The document as the old notes row (id, type, title, body, tags_json, project, author, created_at,
    updated_at, rev, deleted_at, accessed_at, access_count, model, metadata_json); None when missing."""
    if note_id is not None:
        row = connection.execute(_HEAD_SELECT + " AND id=?", (int(note_id),)).fetchone()
    else:
        row = connection.execute(_HEAD_SELECT + " AND title=?", (title,)).fetchone()
    if row is None:
        return None
    record = {key: row[key] for key in row.keys()}
    record["body"] = note_body(connection, int(row["id"]), str(row["author"] or ""))
    return record


def _metadata_with(metadata_json: Any, **values: Any) -> str:
    metadata = _metadata_dict(metadata_json)
    metadata.update({key: value for key, value in values.items()})
    return json.dumps(metadata, ensure_ascii=False)


def _drop_parts(connection: sqlite3.Connection, note_id: int) -> list[int]:
    """Delete the part rows of one document and every next link of the document (head included)."""
    ids = part_ids(connection, note_id)
    chain = [int(note_id), *ids]
    for start in range(0, len(chain), _SQL_IN_BATCH):
        batch = chain[start:start + _SQL_IN_BATCH]
        connection.execute(f"DELETE FROM item_links WHERE relation='{RELATION_NEXT}' AND src_id IN "
                           f"({','.join('?' * len(batch))})", batch)
    for start in range(0, len(ids), _SQL_IN_BATCH):
        batch = ids[start:start + _SQL_IN_BATCH]
        connection.execute(f"DELETE FROM items WHERE id IN ({','.join('?' * len(batch))})", batch)
    return ids


def _rechain(connection: sqlite3.Connection, note_id: int, ids: Sequence[int], now: float) -> None:
    """The next chain head -> ids[0] -> ... (every old next link from these rows replaced)."""
    chain = [int(note_id), *(int(item_id) for item_id in ids)]
    for start in range(0, len(chain), _SQL_IN_BATCH):
        batch = chain[start:start + _SQL_IN_BATCH]
        connection.execute(f"DELETE FROM item_links WHERE relation='{RELATION_NEXT}' AND src_id IN "
                           f"({','.join('?' * len(batch))})", batch)
    write_chain(connection, chain, now=now)


def _insert_part(connection: sqlite3.Connection, head: dict, text: str, digest: str, blob: bytes, now: float, *,
                 heading: str = "", source: str | None = None, item_id: int | None = None) -> int:
    metadata: dict[str, Any] = {}
    if heading:
        metadata[PART_HEADING_KEY] = heading
    if source is not None:
        metadata[PART_SOURCE_KEY] = source
    cursor = connection.execute(
        "INSERT INTO items(id,type,owner,title,text,text_hash,project,author,kind,embedding,created_at,updated_at,"
        "metadata_json) VALUES (?,?,'',NULL,?,?,?,?,?,?,?,?,?)",
        (item_id, head["type"], text, digest, head["project"], head["author"], head["type"], blob, now, now,
         json.dumps(metadata, ensure_ascii=False)))
    return int(cursor.lastrowid)


def _write_parts(connection: sqlite3.Connection, head: dict, prepared: "_Prepared", now: float) -> list[int]:
    """Replace the parts of one document by prepared.chunks (heading, text, digest, blob[, source])."""
    _drop_parts(connection, int(head["id"]))
    if prepared.formula:
        ids = insert_formula_chunks(connection, head, [(text, packed, *rest) for _, text, _, packed, *rest
                                                       in prepared.chunks], now)
    else:
        ids = [_insert_part(connection, head, text, digest, packed, now, heading=path)
               for path, text, digest, packed, *_ in prepared.chunks]
    _rechain(connection, int(head["id"]), ids, now)
    return ids


# Memory columns a chunk source is read from (chunk_source_json).
MEMORY_SOURCE_COLUMNS = "id,origin,created_at,importance,metadata_json"
_AGENT_ORIGIN = "agent:"


def _metadata_dict(metadata_json: Any) -> dict:
    try:
        metadata = json.loads(metadata_json or "{}")
    except (TypeError, ValueError):
        return {}
    return metadata if isinstance(metadata, dict) else {}


def chunk_source_json(memory: Any) -> str:
    """source_json of an automatic-note part made from one memory row (MEMORY_SOURCE_COLUMNS): memory_id, origin,
    session_id (metadata, else the agent:<id> origin), created_at, memory_kind, basis, subject, importance."""
    metadata = _metadata_dict(memory["metadata_json"])
    origin = str(memory["origin"] or "")
    session_id = metadata.get("session_id") or (origin[len(_AGENT_ORIGIN):] if origin.startswith(_AGENT_ORIGIN) else "")
    return json.dumps({"memory_id": int(memory["id"]), "origin": origin, "session_id": str(session_id),
                       "created_at": float(memory["created_at"]), "memory_kind": metadata.get("memory_kind"),
                       "basis": metadata.get("basis"), "subject": metadata.get("subject"),
                       "importance": float(memory["importance"])}, ensure_ascii=False)


def memory_chunk_source(connection: sqlite3.Connection, memory_id: int) -> str | None:
    """chunk_source_json of one stored memory (None when it is gone)."""
    row = connection.execute(f"SELECT {MEMORY_SOURCE_COLUMNS} FROM items WHERE id=? AND type='{ITEM_MEMORY}'",
                             (memory_id,)).fetchone()
    return None if row is None else chunk_source_json(row)


def _item_source(item: Sequence[Any]) -> str | None:
    """source_json of a (text, blob[, source_json]) chunk item."""
    return item[2] if len(item) > 2 else None


def insert_formula_chunks(connection: sqlite3.Connection, head: dict, items: Sequence[tuple[Any, ...]],
                          now: float) -> list[int]:
    """Insert (text, packed embedding[, source_json]) part rows of one automatic note (not chained yet) with ids
    above every id handed out so far, so a part id above a recorded watermark is always a part not compared yet
    (note-absorb). Returns the new ids in order."""
    if not items:
        return []
    top = connection.execute("SELECT COALESCE(MAX(id),0) FROM items").fetchone()[0]
    first = max(formula_chunk_watermark(connection), int(top)) + 1
    ids = [_insert_part(connection, head, item[0], _text_hash(item[0]), item[1], now, source=_item_source(item),
                        item_id=first + index) for index, item in enumerate(items)]
    connection.execute("INSERT OR REPLACE INTO meta_entries(key,value) VALUES (?,?)", (FORMULA_HWM_META_KEY, str(ids[-1])))
    return ids


def unused_note_title(connection: sqlite3.Connection, base: str) -> str:
    """base, or `base (N)` with the smallest free N (the existing duplicate-title rule)."""
    base = _validated_title(base)
    taken = {row[0] for row in connection.execute(
        f"SELECT title FROM items WHERE {NOTE_HEAD_SQL} AND (title=? OR title LIKE ? ESCAPE '\\')",
        (base, base.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + " (%)"),
    )}
    if base not in taken:
        return base
    number = 2
    while f"{base} ({number})" in taken:
        number += 1
    return f"{base} ({number})"


def write_revision(connection: sqlite3.Connection, row: dict, edited_by: str, now: float) -> None:
    """The document as it is now (title, whole body, model) becomes its revision row of the current rev."""
    connection.execute(
        "INSERT INTO item_revisions(item_id,rev,title,text,metadata_json,edited_by,edited_at) VALUES (?,?,?,?,?,?,?)",
        (row["id"], row["rev"], row["title"], row["body"], json.dumps({"model": row["model"] or ""}), edited_by, now),
    )


def relink(connection: sqlite3.Connection, src_id: int, text: str) -> None:
    """Recompute outgoing refers links of one source item; unknown targets keep dst_id NULL."""
    connection.execute("DELETE FROM item_links WHERE relation='refers' AND src_id=?", (src_id,))
    titles, memories = parse_links(text)
    rows = []
    for title in titles:
        target = connection.execute(f"SELECT id FROM items WHERE {NOTE_HEAD_SQL} AND title=? AND deleted_at IS NULL",
                                    (title,)).fetchone()
        rows.append((src_id, None if target is None else int(target[0]), title))
    for memory_id in memories:
        target = connection.execute(f"SELECT id FROM items WHERE id=? AND type='{ITEM_MEMORY}'", (memory_id,)).fetchone()
        rows.append((src_id, None if target is None else memory_id, f"mem:{memory_id}"))
    connection.executemany("INSERT INTO item_links(src_id,dst_id,dst_title,relation) VALUES (?,?,?,'refers')", rows)


def _resolve_waiting_links(connection: sqlite3.Connection, note_id: int, title: str) -> None:
    """Links by id follow the title; broken links waiting for this title resolve."""
    connection.execute("UPDATE item_links SET dst_title=? WHERE relation='refers' AND dst_id=?", (title, note_id))
    connection.execute(
        f"UPDATE item_links SET dst_id=? WHERE relation='refers' AND dst_id IS NULL AND dst_title=? "
        f"AND NOT ({LINK_TO_MEMORY_SQL.format(alias='item_links')})", (note_id, title),
    )


def trash_links_into(connection: sqlite3.Connection, note_id: int) -> None:
    """A trashed document: refers links into it become broken (dst_id NULL; the [[title]] text stays)."""
    connection.execute("UPDATE item_links SET dst_id=NULL WHERE " + _REFERS + "dst_id=?", (note_id,))


def _new_head(connection: sqlite3.Connection, *, title: str, author: str, tags: list[str], project: str, model: str,
              now: float) -> dict:
    kind = document_type(author)
    cursor = connection.execute(
        "INSERT INTO items(type,owner,title,text,text_hash,project,tags_json,author,kind,embedding,created_at,"
        "updated_at,rev,metadata_json) VALUES (?,'',?,'',?,?,?,?,?,?,?,?,1,?)",
        (kind, title, _text_hash(""), project, json.dumps(tags, ensure_ascii=False), author, kind, PENDING_EMBED_MARK,
         now, now, json.dumps({"model": model}, ensure_ascii=False)))
    return {"id": int(cursor.lastrowid), "type": kind, "title": title, "author": author, "project": project,
            "rev": 1, "model": model}


def create_formula_note(connection: sqlite3.Connection, title: str, items: Sequence[tuple[Any, ...]],
                        now: float) -> dict:
    """New automatic note (author formula, project '', no tags, rev 1) from (memory body, packed memory
    embedding[, source_json]) parts in this order; the title gets the duplicate rule. No embedding is computed."""
    title = unused_note_title(connection, title)
    texts = [formula_chunk_text(item[0]) for item in items]
    kept = [(text, item[1], _item_source(item)) for text, item in zip(texts, items) if text]
    body = formula_body(item[0] for item in kept)
    head = _new_head(connection, title=title, author=FORMULA_AUTHOR, tags=[], project="", model="", now=now)
    ids = insert_formula_chunks(connection, head, kept, now)
    _rechain(connection, head["id"], ids, now)
    relink(connection, head["id"], body)
    _resolve_waiting_links(connection, head["id"], title)
    return {"id": head["id"], "title": title, "rev": 1, "chunks": len(kept)}


def apply_formula_chunk_ops(connection: sqlite3.Connection, note_id: int, *,
                            add: Sequence[tuple[Any, ...]] = (), replace: dict[int, Any] | None = None,
                            delete: Iterable[int] = (), editor: str, now: float) -> dict:
    """Part-level edit of one live automatic note: delete parts, replace a part in place (a new part id, same
    position; a list of items replaces it with consecutive parts there), append parts at the end. Then the body
    is joined again from the parts, the previous rev goes to item_revisions once and the rev rises by one. Texts
    must already be chunk texts (formula_chunk_text) and blobs packed embeddings: nothing is embedded here."""
    replace = replace or {}
    delete = [int(chunk_id) for chunk_id in delete]
    row = head_row(connection, note_id=note_id)
    if row is None or row["deleted_at"] is not None:
        raise NoteNotFound("Note not found")
    if row["author"] != FORMULA_AUTHOR:
        raise NotePermissionError("Only automatic notes are edited chunk by chunk")
    order = part_ids(connection, note_id)
    present = set(order)
    missing = [chunk_id for chunk_id in (*delete, *replace) if int(chunk_id) not in present]
    if missing:
        raise NoteNotFound("Chunk not found in this note", chunk_ids=sorted(missing))
    gone = set(delete)
    replaced: dict[int, list[int]] = {}
    for chunk_id, item in sorted(replace.items()):
        items = list(item) if isinstance(item, list) else [item]
        replaced[int(chunk_id)] = insert_formula_chunks(connection, row, items, now)
        gone.add(int(chunk_id))
    new_order: list[int] = []
    for chunk_id in order:
        if chunk_id in replaced:
            new_order.extend(replaced[chunk_id])
        elif chunk_id not in gone:
            new_order.append(chunk_id)
    if add:
        new_order.extend(insert_formula_chunks(connection, row, list(add), now))
    chain = [int(note_id), *order]
    for start in range(0, len(chain), _SQL_IN_BATCH):
        batch = chain[start:start + _SQL_IN_BATCH]
        connection.execute(f"DELETE FROM item_links WHERE relation='{RELATION_NEXT}' AND src_id IN "
                           f"({','.join('?' * len(batch))})", batch)
    dropped = sorted(gone)
    for start in range(0, len(dropped), _SQL_IN_BATCH):
        batch = dropped[start:start + _SQL_IN_BATCH]
        connection.execute(f"DELETE FROM items WHERE id IN ({','.join('?' * len(batch))})", batch)
    _rechain(connection, note_id, new_order, now)
    body = note_body(connection, note_id, FORMULA_AUTHOR)
    write_revision(connection, row, editor, now)
    rev = int(row["rev"]) + 1
    connection.execute("UPDATE items SET updated_at=?,rev=?,metadata_json=? WHERE id=?",
                       (now, rev, _metadata_with(row["metadata_json"], model=""), note_id))
    relink(connection, note_id, body)
    return {"id": note_id, "title": row["title"], "rev": rev, "chunks": len(new_order), "replaced": replaced}


# Automatic-note chunks at save time (neospike_memory_save) and in neospike_memory_reconcile ------------
# Functions taking a connection: caller holds the store lock (and the write transaction for writes).
_FORMULA_PARTS_SQL = (f"SELECT c.id,c.note_id,c.ord,c.text,c.embedding FROM {NOTE_CHUNKS_SQL} c "
                      f"JOIN items n ON n.id=c.note_id WHERE n.author='{FORMULA_AUTHOR}' AND n.deleted_at IS NULL")


def nearest_formula_chunks(connection: sqlite3.Connection, blob: bytes, *, limit: int) -> list[tuple[int, float, str]]:
    """(chunk id, cos, text) of the `limit` parts of live automatic notes closest to one packed embedding,
    best first (ties: lower id). Human and agent note parts are never compared. Scores: one numpy product (dot_similarities); a part of another width is skipped."""
    anchor = array("f", blob)
    rows = [row for row in connection.execute(_FORMULA_PARTS_SQL + " ORDER BY c.id")
            if row[4] is not None and len(row[4]) == len(blob)]
    scored = []
    for row, similarity in zip(rows, vector_index.similarities(connection, "items", [row[0] for row in rows],
                                                               anchor, limit=limit)):
        if math.isfinite(similarity):
            scored.append((-similarity, int(row[0]), row[3]))
    scored.sort()
    return [(chunk_id, -negative, text) for negative, chunk_id, text in scored[:limit]]


class RememberTuning:
    """Save-time tidy: the neighbours compared right after a memory item is stored.
    Duplicate: cos >= DUPLICATE_COSINE and a word gap <= DUPLICATE_WORD_GAP (a wider gap means one side says more, so the pair goes to the contradiction check instead)."""
    NEIGHBORS = 3
    DUPLICATE_COSINE = 0.90
    DUPLICATE_WORD_GAP = 2
    CONTRADICTION_COSINE = 0.80


NEIGHBOR_MEMORY, NEIGHBOR_CHUNK = 0, 1  # save-time neighbour lanes; a memory sorts before a chunk at equal cos


def _neighbor_order(item: tuple[float, int, int, str]) -> tuple[float, int, int]:
    """Save-time neighbours: closest first, then memory before chunk, then the lower id."""
    return -item[0], item[1], item[2]


def tidy_saved_memory(store: MemoryStore, namespace: str, memory_id: int, *, project: str, session_id: str,
                      chunk_editor: str) -> dict:
    """Duplicate handling right after a memory item gets its vector (neospike_memory_save, and embed-pending for a
    memory saved without one). Its RememberTuning.NEIGHBORS closest valid visible memory items and live
    automatic-note chunks (together, closest first) are compared:
    - memory duplicate: the old memory is deleted (store.delete with heir = the new one). When the old
      one was visible more widely (global > project > session), the new memory first takes that scope;
    - chunk duplicate: the chunk takes the new body and embedding and the new memory is deleted;
    - otherwise cos >= CONTRADICTION_COSINE: listed as (lane, id, text) for the caller (the saving agent's
      reconcile); nothing is filed.
    Returns {"deleted": [ids], "replaced_chunk": {...} | None, "listed": [...], "scope": wider scope | None}.
    An embedding-pending memory is not compared (all empty)."""
    result: dict[str, Any] = {"deleted": [], "replaced_chunk": None, "listed": [], "scope": None}
    memories = store.closest_memories(namespace, memory_id, limit=RememberTuning.NEIGHBORS,
                                          project=project, session_id=session_id)
    with store._lock:
        row = store._conn.execute(f"SELECT text,embedding,metadata_json FROM items WHERE id=? AND type='{ITEM_MEMORY}'",
                                  (memory_id,)).fetchone()
        if row is None or is_embed_pending(row[1]):
            return result
        chunks = nearest_formula_chunks(store._conn, row[1], limit=RememberTuning.NEIGHBORS)
    new_text, new_blob = str(row[0]), row[1]
    new_words = len(new_text.split())
    texts = store.texts_by_id(namespace, [old_id for old_id, _ in memories])
    neighbors = sorted([*((similarity, NEIGHBOR_MEMORY, old_id, texts.get(old_id, "")) for old_id, similarity in memories),
                        *((similarity, NEIGHBOR_CHUNK, chunk_id, text) for chunk_id, similarity, text in chunks)],
                       key=_neighbor_order)[:RememberTuning.NEIGHBORS]
    duplicates = [similarity >= RememberTuning.DUPLICATE_COSINE
                  and abs(len(text.split()) - new_words) <= RememberTuning.DUPLICATE_WORD_GAP
                  for similarity, _, _, text in neighbors]
    # The closest duplicate chunk takes the new body (the new memory then goes away).
    duplicate_chunk = next((item_id for (_, lane, item_id, _), duplicate in zip(neighbors, duplicates)
                            if duplicate and lane == NEIGHBOR_CHUNK), None)
    duplicate_memories = [item_id for (_, lane, item_id, _), duplicate in zip(neighbors, duplicates)
                          if duplicate and lane == NEIGHBOR_MEMORY]
    scopes = store.scopes_of_memories(namespace, [memory_id, *duplicate_memories]) if duplicate_chunk is None else {}
    for (similarity, lane, item_id, text), duplicate in zip(neighbors, duplicates):
        if lane == NEIGHBOR_MEMORY and duplicate:
            heir = None if duplicate_chunk is not None else memory_id
            if store.delete(namespace, item_id, project=project, session_id=session_id, heir=heir):
                result["deleted"].append(item_id)
        elif item_id == duplicate_chunk and lane == NEIGHBOR_CHUNK:
            continue
        elif similarity >= RememberTuning.CONTRADICTION_COSINE:
            result["listed"].append((lane, item_id, text))
    if duplicate_chunk is not None:
        with store._lock, store._conn:
            result["replaced_chunk"] = replace_formula_chunk(store._conn, duplicate_chunk, new_text, new_blob,
                                                             editor=chunk_editor, now=time.time(),
                                                             source=memory_chunk_source(store._conn, memory_id))
        store.delete(namespace, memory_id, project=project, session_id=session_id)
        return result
    own = scopes.get(memory_id)
    wider = max((scopes[item_id] for item_id in result["deleted"] if item_id in scopes),
                key=_scope_rank, default=None)
    if own is not None and wider is not None and SCOPE_ORDER[wider[0]] > SCOPE_ORDER[own[0]]:
        if store.widen_memory_scope(namespace, memory_id, wider[0], wider[1]):
            result["scope"] = wider[0]
    return result


def _scope_rank(scope: tuple[str, str]) -> int:
    return SCOPE_ORDER[scope[0]]


def _pref_neighbor_order(item: tuple[float, int]) -> tuple[float, int]:
    return -item[0], item[1]


def tidy_saved_preference(store: MemoryStore, namespace: str, pref_id: int) -> dict:
    """The save-time duplicate rule of tidy_saved_memory inside the preferences table: the
    RememberTuning.NEIGHBORS closest other preferences/constraints of the namespace (any type, any situation) are
    compared; cos >= DUPLICATE_COSINE with a word gap <= DUPLICATE_WORD_GAP deletes the old row and keeps the new
    one. The scope rule becomes: when a deleted duplicate was common (situation ''), the new row becomes common too
    (a new constraint keeps its situation: a constraint is never common), and a deleted user_said basis carries
    over. No contradiction list (reconcile works on memories only).
    Returns {"deleted": [ids], "common": True when the new row was made common}. A row without its text
    vector is not compared."""
    from .preferences import BASIS_USER_SAID, PREF_CONSTRAINT, PREFERENCES_TABLE
    from .store import dot_similarities
    result: dict[str, Any] = {"deleted": [], "common": False}
    with store._lock:
        row = store._conn.execute(f"SELECT text,embedding,situation,basis,type FROM {PREFERENCES_TABLE} "
                                  "WHERE id=? AND owner=?", (int(pref_id), namespace)).fetchone()
        if row is None or is_embed_pending(row[1]):
            return result
        others = store._conn.execute(
            f"SELECT id,text,embedding,situation,basis FROM {PREFERENCES_TABLE} WHERE owner=? AND id<>? "
            f"AND {vector_present_sql()}", (namespace, int(pref_id))).fetchall()
    if not others:
        return result
    new_words = len(str(row[0]).split())
    scored = sorted(((similarity, index) for index, similarity in enumerate(
        dot_similarities(array("f", row[1]), [other[2] for other in others])) if math.isfinite(similarity)),
        key=_pref_neighbor_order)[:RememberTuning.NEIGHBORS]
    duplicates = [others[index] for similarity, index in scored
                  if similarity >= RememberTuning.DUPLICATE_COSINE
                  and abs(len(str(others[index][1]).split()) - new_words) <= RememberTuning.DUPLICATE_WORD_GAP]
    widen = bool(row[2]) and row[4] != PREF_CONSTRAINT and any(other[3] == "" for other in duplicates)
    stated = row[3] != BASIS_USER_SAID and any(other[4] == BASIS_USER_SAID for other in duplicates)
    with store._write_scope():
        for other in duplicates:
            if store._conn.execute(f"DELETE FROM {PREFERENCES_TABLE} WHERE id=?", (int(other[0]),)).rowcount:
                result["deleted"].append(int(other[0]))
        changes = {**({"situation": ""} if widen else {}), **({"basis": BASIS_USER_SAID} if stated else {})}
        if changes:
            from .preferences import update_preference
            update_preference(store._conn, pref_id, changes)
    result["common"] = widen
    return result


def _embed_pending_preferences(store: MemoryStore, embedder: Any, failures: list[dict], failure_log: Any) -> dict:
    """Preference text and situation vectors saved without one get theirs; a filled text vector then gets the
    save-time duplicate rule (tidy_saved_preference)."""
    from .preferences import missing_vectors, write_missing_vector
    from .recall_log import failure_cause
    with store._lock:
        pending = missing_vectors(store._conn) if store._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='preferences'").fetchone() else []
    filled: list[int] = []
    deleted: list[int] = []
    for pref_id, column, text in pending:
        try:
            blob = store._pack_embedding(embedder.embed_document(text))
        except (RuntimeError, OSError, ValueError) as exc:
            cause = failure_cause(exc)
            failures.append({"pref": pref_id, "cause": cause})
            if failure_log is not None:
                failure_log("embed-pending", cause)
            continue
        with store._write_scope():
            written = write_missing_vector(store._conn, pref_id, column, text, blob)
            owner = store._conn.execute("SELECT owner FROM preferences WHERE id=?", (pref_id,)).fetchone()
        if not written:
            continue
        filled.append(pref_id)
        if column == "embedding" and owner is not None:
            deleted.extend(tidy_saved_preference(store, str(owner[0]), pref_id)["deleted"])
    with store._lock:
        left = len(missing_vectors(store._conn)) if pending else 0
    return {"pending": len(pending), "filled": len(filled), "filled_ids": filled, "deleted_ids": sorted(deleted),
            "left": left}


def embed_pending(store: MemoryStore, embedder: Any, *, chunk_editor: str, failure_log: Any = None) -> dict:
    """embed-pending (daily): every embedding-pending memory gets its vector; a memory item then gets the
    save-time duplicate handling it skipped (tidy_saved_memory, scope rule included; contradictions are the
    conversation agent's, so the listed items are only counted). A failed embedding stays pending for the next run;
    failure_log(operation, cause) records it. Returns counts, the filled/deleted ids and the failure causes."""
    from .recall_log import failure_cause
    pending = store.items_missing_embedding()
    filled: list[int] = []
    deleted: list[int] = []
    raised: list[int] = []
    replaced = listed = 0
    failures: list[dict] = []
    for item in pending:
        try:
            vector = list(embedder.embed_document(item["text"]))
        except (RuntimeError, OSError, ValueError) as exc:
            cause = failure_cause(exc)
            failures.append({"id": item["id"], "cause": cause})
            if failure_log is not None:
                failure_log("embed-pending", cause)
            continue
        if not store.write_missing_embedding(item["id"], vector):
            continue
        filled.append(item["id"])
        with store._lock:
            kind = store._conn.execute(f"SELECT kind FROM items WHERE id=? AND type='{ITEM_MEMORY}'",
                                       (item["id"],)).fetchone()
        if kind is None or kind[0] != "memory_item":
            continue
        done = tidy_saved_memory(store, item["namespace"], item["id"], project=item["project"],
                                 session_id=item["session_id"], chunk_editor=chunk_editor)
        deleted.extend(done["deleted"])
        replaced += int(done["replaced_chunk"] is not None)
        listed += len(done["listed"])
        if done["scope"] is not None:
            raised.append(item["id"])
    return {"pending": len(pending), "filled": len(filled), "failed": len(failures), "filled_ids": filled,
            "deleted_ids": sorted(deleted), "replaced_chunks": replaced, "scope_raised_ids": raised,
            "contradiction_neighbors": listed, "failures": failures, "left": len(store.items_missing_embedding()),
            **_embed_pending_parts(store, embedder, failures, failure_log),
            "preferences": _embed_pending_preferences(store, embedder, failures, failure_log)}


def _embed_pending_parts(store: MemoryStore, embedder: Any, failures: list[dict], failure_log: Any) -> dict:
    """Document parts left without a vector (a v22 migration run without the embedder) get theirs."""
    from .recall_log import failure_cause
    pending = store.parts_missing_embedding()
    filled = 0
    for item in pending:
        try:
            vector = list(embedder.embed_document(item["text"]))
        except (RuntimeError, OSError, ValueError) as exc:
            cause = failure_cause(exc)
            failures.append({"id": item["id"], "cause": cause})
            if failure_log is not None:
                failure_log("embed-pending", cause)
            continue
        filled += int(store.write_missing_embedding(item["id"], vector))
    return {"parts_pending": len(pending), "parts_filled": filled}


def _part_places(connection: sqlite3.Connection, chunk_ids: Iterable[int], *,
                 formula_only: bool) -> dict[int, tuple[int, int, str]]:
    """part id -> (note id, ord, note author) of the parts of live documents among chunk_ids (others left out)."""
    ids = {int(chunk_id) for chunk_id in chunk_ids}
    if not ids:
        return {}
    author = f" AND n.author='{FORMULA_AUTHOR}'" if formula_only else ""
    return {int(row[0]): (int(row[1]), int(row[2]), str(row[3])) for row in connection.execute(
        f"SELECT c.id,c.note_id,c.ord,n.author FROM {NOTE_CHUNKS_SQL} c JOIN items n ON n.id=c.note_id "
        f"WHERE n.deleted_at IS NULL{author}") if int(row[0]) in ids}


def chunk_owners(connection: sqlite3.Connection, chunk_ids: Iterable[int]) -> dict[int, tuple[int, str]]:
    """part id -> (note id, note author) of the parts of live documents among chunk_ids (others left out)."""
    return {chunk_id: (place[0], place[2]) for chunk_id, place in
            _part_places(connection, chunk_ids, formula_only=False).items()}


def replace_formula_chunk(connection: sqlite3.Connection, chunk_id: int, text: str, blob: bytes, *, editor: str,
                          now: float, source: str | None = None) -> dict:
    """One automatic-note part takes a new body and embedding (same position, a new part id; source: its
    source_json); the note body is joined again. Returns {"note_id", "chunk_id"} with the new part id."""
    owner = chunk_owners(connection, (chunk_id,)).get(int(chunk_id))
    if owner is None or owner[1] != FORMULA_AUTHOR:
        raise NoteNotFound("Automatic-note chunk not found", chunk_ids=[int(chunk_id)])
    done = apply_formula_chunk_ops(connection, owner[0],
                                   replace={int(chunk_id): (formula_chunk_text(text), blob, source)},
                                   editor=editor, now=now)
    return {"note_id": owner[0], "chunk_id": int(done["replaced"][int(chunk_id)][0])}


def formula_chunk_places(connection: sqlite3.Connection, chunk_ids: Iterable[int]) -> dict[int, tuple[int, int]]:
    """part id -> (note id, ord) of the parts of live automatic notes among chunk_ids (others left out)."""
    return {chunk_id: (place[0], place[1]) for chunk_id, place in
            _part_places(connection, chunk_ids, formula_only=True).items()}


def reconcile_formula_chunks(connection: sqlite3.Connection, replace: dict[int, list[tuple[str, bytes]]],
                             delete: Iterable[int], *, editor: str, now: float) -> dict:
    """neospike_memory_reconcile on automatic-note parts, inside the caller's write transaction: per note one
    part edit (replaced parts keep their place with a new id, deleted ones go), so the body is joined once; a
    note left without parts goes to the trash. Every part must belong to a live automatic note (NoteNotFound
    otherwise, nothing changed by the caller's rollback). A replacement is a list of pieces (a body over
    the item limit, already split), written as consecutive parts in its place; the note order chains them.
    Returns {"changed": {old id: first new id}, "pieces": {old id: [new ids]}, "deleted": [ids],
    "trashed_notes": [note ids]}."""
    delete = [int(chunk_id) for chunk_id in delete]
    places = formula_chunk_places(connection, [*replace, *delete])
    missing = sorted({*replace, *delete} - places.keys())
    if missing:
        raise NoteNotFound("Automatic-note chunk not found", chunk_ids=missing)
    by_note: dict[int, tuple[dict[int, list[tuple[str, bytes]]], list[int]]] = {}
    for chunk_id, items in replace.items():
        pieces = [items] if isinstance(items, tuple) else list(items)
        by_note.setdefault(places[int(chunk_id)][0], ({}, []))[0][int(chunk_id)] = [
            (formula_chunk_text(text), blob) for text, blob in pieces]
    for chunk_id in delete:
        by_note.setdefault(places[chunk_id][0], ({}, []))[1].append(chunk_id)
    changed: dict[int, int] = {}
    pieces_of: dict[int, list[int]] = {}
    trashed: list[int] = []
    for note_id in sorted(by_note):
        replaced, deleted = by_note[note_id]
        done = apply_formula_chunk_ops(connection, note_id, replace=replaced, delete=deleted, editor=editor, now=now)
        for old_id, new_ids in done["replaced"].items():
            changed[old_id] = new_ids[0]
            pieces_of[old_id] = new_ids
        if done["chunks"] == 0:
            connection.execute("UPDATE items SET deleted_at=?,updated_at=? WHERE id=?", (now, now, note_id))
            trash_links_into(connection, note_id)
            trashed.append(note_id)
    return {"changed": changed, "pieces": pieces_of, "deleted": delete, "trashed_notes": trashed}


class NoteService:
    def __init__(self, store: MemoryStore, embedder: Any, *, token_counter: TokenCounter | None = None,
                 max_tokens: int | None = None) -> None:
        self.store = store
        self.embedder = embedder
        self.token_counter = token_counter or default_token_counter()
        self.max_tokens = max_tokens or note_chunk_max_tokens()

    # -- reads ---------------------------------------------------------------------------

    def _row(self, *, note_id: int | None = None, title: str | None = None) -> dict | None:
        return head_row(self.store._conn, note_id=note_id, title=title)

    def read(self, *, note_id: int | None = None, title: str | None = None, include_deleted: bool = False) -> dict:
        with self.store._lock:
            return note_record(self.store._conn, note_id=note_id, title=title, include_deleted=include_deleted)

    def chunks(self, note_id: int) -> list[dict]:
        with self.store._lock:
            rows = self.store._conn.execute(
                f"SELECT id,ord,heading_path,text,text_hash FROM {NOTE_CHUNKS_SQL} c WHERE note_id=? ORDER BY ord",
                (note_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def has_searchable(self) -> bool:
        with self.store._lock:
            return self.store._conn.execute(
                f"SELECT 1 FROM items p JOIN item_links l ON l.relation='{RELATION_NEXT}' AND l.src_id=p.id "
                f"WHERE {note_head_sql('p')} AND p.deleted_at IS NULL LIMIT 1"
            ).fetchone() is not None

    def unused_title(self, base: str) -> str:
        with self.store._lock:
            return unused_note_title(self.store._conn, base)

    # -- preparation (outside transactions) -------------------------------------------------

    def _known_embeddings(self, hashes: list[str], note_id: int | None = None) -> dict[str, bytes]:
        """text_hash -> stored part embedding: this note's own parts first (an automatic note keeps the
        copied memory embedding), then any document part with the same text."""
        known: dict[str, bytes] = {}
        unique = list(dict.fromkeys(hashes))
        with self.store._lock:
            if note_id is not None:
                for row in part_rows(self.store._conn, note_id):
                    if row[3]:
                        known.setdefault(row[2], row[3])
            unique = [digest for digest in unique if digest not in known]
            for start in range(0, len(unique), _SQL_IN_BATCH):
                batch = unique[start:start + _SQL_IN_BATCH]
                for row in self.store._conn.execute(
                    f"SELECT text_hash,embedding FROM items WHERE type IN {DOC_TYPES_SQL} AND title IS NULL "
                    f"AND length(embedding)>0 AND text_hash IN ({','.join('?' * len(batch))})", batch,
                ):
                    known.setdefault(row[0], row[1])
        return known

    def _embedded(self, texts: list[str], note_id: int | None = None) -> list[tuple[str, bytes]]:
        """(text_hash, packed embedding) per text; only a text whose hash has no stored chunk is embedded."""
        hashes = [_text_hash(text) for text in texts]
        known = self._known_embeddings(hashes, note_id)
        result = []
        for text, digest in zip(texts, hashes):
            if digest not in known:
                known[digest] = self._pack(self.embedder.embed_document(text))
            result.append((digest, known[digest]))
        return result

    def split_parts(self, body: str) -> list[tuple[str, str]]:
        """(heading trail, part) of a human/agent body: no overlap, "".join(parts) == body."""
        return split_document_parts(body, self.token_counter.count, max_tokens=self.max_tokens,
                                    cut_tokens=self.token_counter.cut)

    def _prepare(self, body: str) -> _Prepared:
        pieces = self.split_parts(body)
        embedded = self._embedded([part_embed_text(text) for _, text in pieces])
        return _Prepared([(path, text, digest, packed) for (path, text), (digest, packed) in zip(pieces, embedded)])

    def formula_texts(self, body: str) -> list[str]:
        """A whole-body write of an automatic note split back into chunk texts: one per blank-line
        paragraph; a paragraph over the chunk ceiling is split further by the note splitter (no overlap,
        so the joined body repeats nothing)."""
        texts: list[str] = []
        for block in _PARAGRAPH_BREAK.split(body.replace("\r\n", "\n").replace("\r", "\n")):
            text = formula_chunk_text(block)
            if not text:
                continue
            if self.token_counter.count(text) <= self.max_tokens:
                texts.append(text)
                continue
            for _, piece in pack_markdown_chunks(text, self.token_counter.count, max_tokens=self.max_tokens,
                                           overlap_tokens=0, cut_tokens=self.token_counter.cut):
                piece = formula_chunk_text(piece)
                if piece:
                    texts.append(piece)
        return texts

    def _prepare_formula(self, body: str, note_id: int) -> tuple[str, _Prepared]:
        """(normalized body, chunks) of a whole-body write of an automatic note; a chunk whose text did
        not change keeps its embedding (no embedding call)."""
        texts = self.formula_texts(body)
        embedded = self._embedded(texts, note_id)
        return formula_body(texts), _Prepared([("", text, digest, packed) for text, (digest, packed)
                                               in zip(texts, embedded)], formula=True)

    def _prepare_any(self, row: dict | None, body: str) -> tuple[str, _Prepared]:
        if row is not None and row["author"] == FORMULA_AUTHOR:
            return self._prepare_formula(body, int(row["id"]))
        return body, self._prepare(body)

    def _pack(self, embedding: Any) -> bytes:
        vector = normalize_rows(embedding)
        if len(vector) != self.store.dimensions or not all(math.isfinite(value) for value in vector) or not any(vector):
            raise RuntimeError("Invalid note chunk embedding")
        return vector.tobytes()

    # -- writes ----------------------------------------------------------------------------

    def save(self, title: Any, body: Any, *, tags: Any = None, project: Any = "", author: str = "agent",
             expected_rev: int | None = None, note_id: int | None = None,
             origin: str = ORIGIN_CONVERSATION, namespace: str | None = None, model: str = "") -> dict:
        """Create or update one note under rev CAS.

        expected_rev None (or 0) means create-only; updating an existing note
        requires its current rev. `author` is the editor (edited_by in item_revisions);
        notes.author records only who created the note and never changes (2026-11-09).
        origin: conversation and human writes apply to every note. An
        automation write to a note created or last edited by a human is refused
        (NotePermissionError); the note is not changed. `namespace` is unused.
        model: the LLM that wrote this rev ('' for people and tools); kept with the rev.
        """
        if origin not in ORIGINS:
            raise NoteError(f"Note origin must be one of {', '.join(ORIGINS)}")
        if origin == ORIGIN_HUMAN and author != "human":
            raise NoteError("origin 'human' requires author 'human'")
        title = _validated_title(title)
        if not isinstance(body, str):
            raise NoteError("Note body must be a string")
        if len(body.encode("utf-8")) > MAX_BODY_BYTES:
            raise NoteError(f"Note body exceeds the {MAX_BODY_BYTES}-byte limit")
        if author not in AUTHORS:
            raise NoteError("Note author must be 'human' or 'agent'")
        if not isinstance(project, str):
            raise NoteError("Note project must be a string")
        if expected_rev is not None and (type(expected_rev) is not int or expected_rev < 0):
            raise NoteError("expected_rev must be a non-negative integer")
        if note_id is not None and (type(note_id) is not int or note_id < 1):
            raise NoteError("Note id must be a positive integer")
        if not isinstance(model, str) or len(model) > MAX_TITLE_CHARS:
            raise NoteError(f"Note model must be a string of at most {MAX_TITLE_CHARS} characters")
        tags = _validated_tags(tags)
        check_note_policy(title, body, tags, author)
        check_draft_sources(body, tags)
        if origin == ORIGIN_AUTOMATION:
            self._refuse_if_human(title, note_id)
        for _attempt in range(MAX_SAVE_ATTEMPTS):
            result = self._save_once(title, body, tags, project, author, expected_rev, note_id, model)
            if result is not None:
                return result
        raise NoteConflict("Notes linking to this note changed concurrently; retry", current_rev=None)

    def _save_once(self, title: str, body: str, tags: list[str], project: str, author: str,
                   expected_rev: int | None, note_id: int | None, model: str = "") -> dict | None:
        with self.store._lock:
            existing = self._row(note_id=note_id) if note_id is not None else self._row(title=title)
            if note_id is not None and existing is None:
                raise NoteNotFound("Note not found")
            other = self._row(title=title) if existing is not None and existing["title"] != title else None
        if other is not None:
            raise NoteConflict("Another note already has this title", current_rev=None, title=title)
        self._check_cas(existing, author, expected_rev)
        new_author = author if existing is None else existing["author"]
        renamed_from = existing["title"] if existing is not None and existing["title"] != title else None
        if renamed_from is not None:
            body = rewrite_title_links(body, renamed_from, title)
        body, prepared = self._prepare_any(existing, body)
        referrers = self._prepare_referrers(existing, renamed_from, title) if renamed_from else []
        connection = self.store._conn
        with self.store._lock:
            try:
                connection.execute("BEGIN IMMEDIATE")
                # Timestamp under the write lock: backup.py's watermark relies on it (see store.py).
                now = time.time()
                current =self._row(note_id=existing["id"]) if existing is not None else self._row(title=title)
                if existing is None and current is not None:
                    connection.rollback()
                    raise NoteConflict("A note with this title was created concurrently", current_rev=int(current["rev"]))
                if existing is not None and (current is None or current["rev"] != existing["rev"]):
                    connection.rollback()
                    raise NoteConflict("Note changed since it was read", current_rev=None if current is None else int(current["rev"]))
                for referrer_id, referrer_rev, *_ in referrers:
                    row = connection.execute("SELECT rev FROM items WHERE id=?", (referrer_id,)).fetchone()
                    if row is None or row[0] != referrer_rev:
                        connection.rollback()
                        return None
                if existing is None:
                    head = _new_head(connection, title=title, author=new_author, tags=tags, project=project,
                                     model=model, now=now)
                    target_id, rev = head["id"], 1
                else:
                    target_id, rev = int(existing["id"]), int(existing["rev"]) + 1
                    self._write_revision(existing, author, now)
                    connection.execute(
                        "UPDATE items SET title=?,tags_json=?,project=?,author=?,updated_at=?,rev=?,deleted_at=NULL,"
                        "metadata_json=? WHERE id=?",
                        (title, json.dumps(tags, ensure_ascii=False), project, new_author, now, rev,
                         _metadata_with(existing["metadata_json"], model=model), target_id),
                    )
                    head = {**existing, "title": title, "project": project, "rev": rev}
                self._replace_chunks(head, prepared, now)
                self._relink(target_id, body)
                # Links by id follow the rename; broken links waiting for this title resolve.
                _resolve_waiting_links(connection, target_id, title)
                for referrer_id, referrer_rev, referrer_row, referrer_body, referrer_prepared in referrers:
                    self._write_revision(referrer_row, author, now)
                    connection.execute("UPDATE items SET updated_at=?,rev=? WHERE id=?",
                                       (now, referrer_rev + 1, referrer_id))
                    self._replace_chunks(referrer_row, referrer_prepared, now)
                    self._relink(referrer_id, referrer_body)
                connection.commit()
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise
        return {"id": target_id, "title": title, "rev": rev, "author": new_author, "chunks": len(prepared.chunks),
                "created": existing is None, "renamed_from": renamed_from,
                "rewritten_notes": [referrer[0] for referrer in referrers]}

    def _check_cas(self, existing: dict | None, author: str, expected_rev: int | None) -> None:
        if existing is None:
            if expected_rev not in (None, 0):
                raise NoteConflict("Note does not exist; create it without expected_rev", current_rev=0)
            return
        if expected_rev is None or expected_rev != existing["rev"]:
            raise NoteConflict("Note rev mismatch; read the note again", current_rev=int(existing["rev"]))

    def _human_owned(self, row: dict) -> bool:
        """Created by a human, or the latest edit (highest revision's edited_by) was a human."""
        if row["author"] == "human":
            return True
        last = self.store._conn.execute(
            "SELECT edited_by FROM item_revisions WHERE item_id=? ORDER BY rev DESC LIMIT 1", (row["id"],),
        ).fetchone()
        return last is not None and last[0] == "human"

    def _refuse_if_human(self, title: str, note_id: int | None) -> None:
        """Automation (origin automation) never edits a note created or last edited by a human."""
        with self.store._lock:
            existing = self._row(note_id=note_id) if note_id is not None else self._row(title=title)
            if existing is not None and self._human_owned(existing):
                raise NotePermissionError(HUMAN_NOTE_READ_ONLY, id=int(existing["id"]))

    def _prepare_referrers(self, existing: dict, old_title: str, new_title: str) -> list[tuple]:
        with self.store._lock:
            ids = [int(row[0]) for row in self.store._conn.execute(
                f"SELECT DISTINCT n.id FROM item_links l JOIN items n ON n.id=l.src_id AND {note_head_sql('n')} "
                "WHERE l.relation='refers' AND l.dst_id=? AND l.src_id<>? ORDER BY n.id",
                (existing["id"], existing["id"]),
            )]
            rows = [head_row(self.store._conn, note_id=note_id) for note_id in ids]
        referrers = []
        for row in rows:
            rewritten = rewrite_title_links(row["body"], old_title, new_title)
            if rewritten != row["body"]:
                rewritten, prepared = self._prepare_any(row, rewritten)
                referrers.append((int(row["id"]), int(row["rev"]), row, rewritten, prepared))
        return referrers

    def _write_revision(self, row: dict, edited_by: str, now: float) -> None:
        write_revision(self.store._conn, row, edited_by, now)

    def _replace_chunks(self, head: dict, prepared: _Prepared, now: float) -> None:
        _write_parts(self.store._conn, head, prepared, now)

    def _relink(self, src_id: int, text: str) -> None:
        """Recompute outgoing links of one source; unknown targets keep dst_id NULL."""
        relink(self.store._conn, src_id, text)

    def link_memory(self, memory_id: int, text: str) -> int:
        """memory -> note links from `[[Title]]` in a memory item's text."""
        connection = self.store._conn
        with self.store._lock:
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._relink(memory_id, text)
                count = connection.execute("SELECT COUNT(*) FROM item_links WHERE relation='refers' AND src_id=?",
                                           (memory_id,)).fetchone()[0]
                connection.commit()
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise
        return int(count)

    # -- chunk operations of automatic notes ---------------------------------------------

    def delete(self, note_id: int, *, author: str, expected_rev: int | None) -> dict:
        """Trash: set deleted_at, keep the row and a revision; links to it become broken."""
        if type(note_id) is not int or note_id < 1:
            raise NoteError("Note id must be a positive integer")
        if author not in AUTHORS:
            raise NoteError("Note author must be 'human' or 'agent'")
        connection = self.store._conn
        with self.store._lock:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = self._row(note_id=note_id)
                if row is None or row["deleted_at"] is not None:
                    connection.rollback()
                    raise NoteNotFound("Note not found")
                self._check_cas(row, author, expected_rev)
                now = time.time()
                self._write_revision(row, author, now)
                connection.execute("UPDATE items SET deleted_at=?,updated_at=?,rev=? WHERE id=?",
                                   (now, now, int(row["rev"]) + 1, note_id))
                trash_links_into(connection, note_id)
                connection.commit()
            except BaseException:
                if connection.in_transaction:
                    connection.rollback()
                raise
        return {"id": note_id, "deleted": True, "rev": int(row["rev"]) + 1}

    def restore(self, note_id: int, rev: int, *, author: str = "human", expected_rev: int | None = None) -> dict:
        """Save the title/body of an earlier rev as a new rev (history is never rewritten).

        Tags and project stay as they are now; revisions do not store them. A
        trashed note is restored too, because save clears deleted_at.
        """
        if type(note_id) is not int or note_id < 1:
            raise NoteError("Note id must be a positive integer")
        if type(rev) is not int or rev < 1:
            raise NoteError("rev must be a positive integer")
        with self.store._lock:
            current = self._row(note_id=note_id)
            if current is None:
                raise NoteNotFound("Note not found")
            source = self.store._conn.execute(
                "SELECT title,text AS body FROM item_revisions WHERE item_id=? AND rev=?", (note_id, rev),
            ).fetchone()
        if source is None:
            raise NoteNotFound("Revision not found")
        return self.save(source["title"], source["body"], tags=json.loads(current["tags_json"]),
                         project=current["project"], author=author,
                         expected_rev=int(current["rev"]) if expected_rev is None else expected_rev, note_id=note_id)

    # -- search ----------------------------------------------------------------------------

    def search(self, query: str, query_embedding: Any, *, limit: int = 5, project: str | None = None,
               tag: str | None = None, formula: bool | None = None) -> list[dict]:
        """Hybrid document-part search: the memories score without the kind bonus (formula: search_note_parts)."""
        with self.store._lock:
            results = search_note_parts(self.store._conn, query, query_embedding, dimensions=self.store.dimensions,
                                         semantic_threshold=self.store.semantic_threshold, limit=limit,
                                         project=project, tag=tag, formula=formula)
        self.store.log_note_access(hit["id"] for hit in results)  # recent ring
        return results


# --- TASK owned-file parsing -----------------------------------

OWNED_FILES_SECTIONS = ("소유 파일", "owned files")
# A bullet group whose label contains one of these lists files that are NOT owned.
OWNED_EXCLUDE_MARKERS = ("금지", "forbidden", "do not")
TASK_STATUS_ACTIVE = "status:active"
TASK_NUMBER_MIN_DIGITS = 3
_LINE = re.compile(r"[^\n]*\n|[^\n]+$")
_HEADING_TEXT = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*\r?$")
_FENCE_LINE = re.compile(r"^\s*(?:```|~~~)")
_BULLET = re.compile(r"^([ \t]*)(?:[-*+]|\d+[.)])[ \t]+(.*?)\s*$")
_CODE_SPAN = re.compile(r"`([^`\n]+)`")
_PATH_LIKE = re.compile(r"^[^\s`]*[/\\.][^\s`]*$")
_TASK_NUMBER = re.compile(r"^" + TASK_TITLE_PREFIX + r"(\d+)")


def _headings(body: str) -> list[tuple[int, int, int, str]]:
    """(line start, line end, level, text) of every `#` heading outside fenced code."""
    found, in_fence = [], False
    for match in _LINE.finditer(body):
        line = match.group(0)
        if _FENCE_LINE.match(line):
            in_fence = not in_fence
            continue
        heading = None if in_fence else _HEADING_TEXT.match(line.rstrip("\n"))
        if heading:
            found.append((match.start(), match.end(), len(heading.group(1)), heading.group(2).strip()))
    return found


def section_span(body: str, names: tuple[str, ...], *, level: int | None = None) -> tuple[int, int, int] | None:
    """(heading start, content start, section end) of the first heading named one of names.

    The section ends at the next heading of the same or a higher level (or the end).
    """
    wanted = {name.casefold() for name in names}
    headings = _headings(body)
    for index, (start, content, heading_level, text) in enumerate(headings):
        if text.casefold() in wanted and (level is None or heading_level == level):
            end = next((other[0] for other in headings[index + 1:] if other[2] <= heading_level), len(body))
            return start, content, end
    return None


def owned_paths(body: str) -> list[str]:
    """Paths listed in the owned-files section (`## Owned files` or `## 소유 파일`): code spans or path-like
    bullets, minus excluded groups (labels with an OWNED_EXCLUDE_MARKERS word such as `forbidden` or `금지`)."""
    span = section_span(body, OWNED_FILES_SECTIONS)
    if span is None:
        return []
    paths: list[str] = []
    excluded_indent: int | None = None
    in_fence = False
    for line in body[span[1]:span[2]].splitlines():
        if _FENCE_LINE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        bullet = _BULLET.match(line)
        if bullet is None:
            continue
        indent, label = len(bullet.group(1).expandtabs(4)), bullet.group(2)
        if excluded_indent is not None and indent <= excluded_indent:
            excluded_indent = None
        spans = _CODE_SPAN.findall(label)
        if not spans and any(marker in label.casefold() for marker in OWNED_EXCLUDE_MARKERS):
            excluded_indent = indent
            continue
        if excluded_indent is not None:
            continue
        candidates = spans or [label.split(":", 1)[0].strip()]
        for candidate in candidates:
            candidate = candidate.strip()
            if _PATH_LIKE.match(candidate):
                path = candidate.replace("\\", "/").removeprefix("./")
                if path not in paths:
                    paths.append(path)
    return paths


def task_owners(connection: sqlite3.Connection) -> dict:
    """Owned files of every live `status:active` TASK_ note, and the paths shared by two or more."""
    escaped = TASK_TITLE_PREFIX.replace("_", "\\_")
    rows = connection.execute(
        f"SELECT id,title,author FROM items WHERE {NOTE_HEAD_SQL} AND deleted_at IS NULL AND title LIKE ? ESCAPE '\\' "
        "AND EXISTS (SELECT 1 FROM json_each(items.tags_json) WHERE json_each.value=?) ORDER BY title",
        (escaped + "%", TASK_STATUS_ACTIVE),
    ).fetchall()
    tasks, holders = [], {}
    for row in rows:
        files = owned_paths(note_body(connection, int(row[0]), str(row[2] or "")))
        tasks.append({"id": int(row[0]), "title": row[1], "files": files})
        for path in files:
            # Windows paths are case-insensitive; compare casefolded, report the first spelling.
            entry = holders.setdefault(path.casefold(), {"path": path, "tasks": []})
            if row[1] not in entry["tasks"]:
                entry["tasks"].append(row[1])
    overlaps = sorted((entry for entry in holders.values() if len(entry["tasks"]) > 1), key=itemgetter("path"))
    return {"tasks": tasks, "overlaps": overlaps}


def next_task_title(connection: sqlite3.Connection) -> dict:
    """TASK_ number one above the largest of the documents (trashed ones included, so numbers never repeat)."""
    escaped = TASK_TITLE_PREFIX.replace("_", "\\_")
    largest, digits = 0, TASK_NUMBER_MIN_DIGITS
    for (title,) in connection.execute(f"SELECT title FROM items WHERE {NOTE_HEAD_SQL} AND title LIKE ? ESCAPE '\\'",
                                       (escaped + "%",)):
        match = _TASK_NUMBER.match(title)
        if match and int(match.group(1)) >= largest:
            largest, digits = int(match.group(1)), max(digits, len(match.group(1)))
    number = largest + 1
    return {"number": number, "title": f"{TASK_TITLE_PREFIX}{number:0{digits}d}", "max": largest}


# A TOPIC summary page (an agent note attached to a topic by a topic link, or titled "TOPIC: " with the topic
# tag; the rule of topics.topic_summary_notes) never comes up in search or recall: a topic has no content.
_TOPIC_PAGE_TITLE_PREFIX, _TOPIC_PAGE_TAG = "TOPIC: ", "topic"
_NOT_TOPIC_PAGE_SQL = (
    f" AND NOT (n.author='agent' AND (EXISTS (SELECT 1 FROM item_links tl WHERE tl.relation='{RELATION_TOPIC}' "
    "AND tl.src_id=n.id) "
    f"OR (substr(n.title,1,{len(_TOPIC_PAGE_TITLE_PREFIX)})='{_TOPIC_PAGE_TITLE_PREFIX}' "
    f"AND EXISTS (SELECT 1 FROM json_each(n.tags_json) WHERE value='{_TOPIC_PAGE_TAG}'))))")


def formula_chunk_source(note_id: int, chunk_id: int, title: str) -> str:
    """Result source of an automatic-note part: one source per part (the two-per-source recall cap) showing
    the part id."""
    return f"note:{int(note_id)}.{int(chunk_id)} #{title}"


def search_note_parts(connection: sqlite3.Connection, query: str, query_embedding: Any, *, dimensions: int,
                       semantic_threshold: float, limit: int = 5, project: str | None = None,
                       tag: str | None = None, formula: bool | None = None) -> list[dict]:
    """Cosine-gated document-part search on any connection (read-only ones included); never writes.
    formula: True only automatic-note parts, False none of them, None all. TOPIC pages never. Each hit carries
    kind = the document's items.type (note | doc)."""
    qvec = list(query_embedding)
    if len(qvec) != dimensions:
        raise ValueError(f"Expected {dimensions} dimensions, got {len(qvec)}")
    now = time.time()
    clauses, values = [_NOT_TOPIC_PAGE_SQL], []
    if formula is not None:
        clauses.append(" AND n.author=?" if formula else " AND n.author<>?")
        values.append(FORMULA_AUTHOR)
    if project is not None:
        clauses.append(" AND n.project=?")
        values.append(project)
    if tag:
        clauses.append(" AND EXISTS (SELECT 1 FROM json_each(n.tags_json) WHERE json_each.value=?)")
        values.append(tag)
    note_filter = "".join(clauses)
    rows = connection.execute(
        f"SELECT c.id,c.note_id,c.heading_path,c.text,n.title,n.updated_at,n.author,n.type FROM {NOTE_CHUNKS_SQL} c "
        f"JOIN items n ON n.id=c.note_id WHERE n.deleted_at IS NULL AND {vector_present_sql('c.embedding')}"
        + note_filter,
        values,
    ).fetchall()
    ranked = []
    for row, semantic in zip(rows, vector_index.similarities(connection, "items", [row["id"] for row in rows],
                                                             qvec, threshold=semantic_threshold)):
        if not math.isfinite(semantic) or semantic < semantic_threshold:
            continue
        score = blend_score(semantic, NOTE_IMPORTANCE, freshness(now, float(row["updated_at"])))
        heading = row["heading_path"] or row["title"]
        ranked.append({
            "id": int(row["note_id"]), "kind": "note", "type": row["type"], "chunk_id": int(row["id"]),
            "title": row["title"], "heading_path": row["heading_path"], "text": row["text"], "score": score,
            "origin": formula_chunk_source(row["note_id"], row["id"], row["title"]) if row["author"] == FORMULA_AUTHOR
            else f"note:{int(row['note_id'])} #{heading}",
        })
    ranked.sort(key=itemgetter("score"), reverse=True)
    return ranked[:limit]


# --- read-only queries on any connection (the dashboard opens the DB with mode=ro) ------------
# Each takes a sqlite3.Connection whose row_factory is sqlite3.Row and never writes.

NOTE_LIST_LIMIT = 200
TITLE_SUGGESTION_LIMIT = 20
SIMILAR_NOTES_LIMIT = 5
BACKLINK_EXCERPT_CHARS = 160
_SQL_BATCH = 500
# A refers link's target kind: a [[mem:N]] link points at a memory, every other one at a document title.
_LINK_TARGET_TYPE = f"(CASE WHEN {LINK_TO_MEMORY_SQL.format(alias='l')} THEN 'memory' ELSE 'note' END)"


def _outgoing_links(connection: sqlite3.Connection, note_id: int) -> list[sqlite3.Row]:
    return connection.execute(
        f"SELECT {_LINK_TARGET_TYPE} AS dst_type,l.dst_id,l.dst_title FROM item_links l WHERE l.relation='refers' "
        "AND l.src_id=? ORDER BY l.id", (note_id,)).fetchall()


def note_record(connection: sqlite3.Connection, *, note_id: int | None = None, title: str | None = None,
                include_deleted: bool = False) -> dict:
    if (note_id is None) == (title is None):
        raise NoteError("Provide exactly one of note id or title")
    if note_id is not None and (type(note_id) is not int or note_id < 1):
        raise NoteError("Note id must be a positive integer")
    if note_id is not None:
        row = head_row(connection, note_id=note_id)
    else:
        row = head_row(connection, title=" ".join(str(title).split()))
    if row is None or (row["deleted_at"] is not None and not include_deleted):
        raise NoteNotFound("Note not found")
    links = _outgoing_links(connection, int(row["id"]))
    return {
        "id": int(row["id"]), "title": row["title"], "body": row["body"], "tags": json.loads(row["tags_json"]),
        "project": row["project"], "author": row["author"], "type": row["type"], "rev": int(row["rev"]),
        "created_at": row["created_at"], "updated_at": row["updated_at"], "deleted_at": row["deleted_at"],
        "model": row["model"], "links": [dict(link) for link in links],
    }


_SUMMARY_COLUMNS = "id,type,title,tags_json,project,author,rev,updated_at"


def _summary(row: sqlite3.Row) -> dict:
    return {"id": int(row["id"]), "title": row["title"], "tags": json.loads(row["tags_json"]), "project": row["project"],
            "author": row["author"], "type": row["type"], "rev": int(row["rev"]), "updated_at": row["updated_at"]}


def _filters(tag: str | None, project: str | None) -> tuple[str, list[Any]]:
    clauses, values = [NOTE_HEAD_SQL, "deleted_at IS NULL"], []
    if tag:
        clauses.append("EXISTS (SELECT 1 FROM json_each(items.tags_json) WHERE json_each.value=?)")
        values.append(tag)
    if project:
        clauses.append("project=?")
        values.append(project)
    return " AND ".join(clauses), values


def list_notes(connection: sqlite3.Connection, *, tag: str | None = None, project: str | None = None,
               ids: list[int] | None = None, limit: int = NOTE_LIST_LIMIT) -> list[dict]:
    """Live (not trashed) documents, newest first; with ids, keep the given order instead."""
    where, values = _filters(tag, project)
    if ids is None:
        rows = connection.execute(
            f"SELECT {_SUMMARY_COLUMNS} FROM items WHERE {where} ORDER BY updated_at DESC, id DESC LIMIT ?",
            (*values, limit),
        ).fetchall()
        return [_summary(row) for row in rows]
    found: dict[int, dict] = {}
    for start in range(0, len(ids), _SQL_BATCH):
        batch = ids[start:start + _SQL_BATCH]
        for row in connection.execute(
            f"SELECT {_SUMMARY_COLUMNS} FROM items WHERE {where} AND id IN ({','.join('?' * len(batch))})",
            (*values, *batch),
        ):
            found[int(row["id"])] = _summary(row)
    return [found[note_id] for note_id in ids if note_id in found][:limit]


def note_titles(connection: sqlite3.Connection, prefix: str, *, limit: int = TITLE_SUGGESTION_LIMIT) -> list[str]:
    """`[[` completion: live titles starting with prefix (ASCII case-insensitive)."""
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    rows = connection.execute(
        f"SELECT title FROM items WHERE {NOTE_HEAD_SQL} AND deleted_at IS NULL AND title LIKE ? ESCAPE '\\' "
        "ORDER BY title LIMIT ?",
        (escaped + "%", limit),
    ).fetchall()
    return [row[0] for row in rows]


def note_history(connection: sqlite3.Connection, note_id: int) -> dict:
    """Current rev plus every stored earlier rev, newest first. Bodies are left out."""
    current = connection.execute(f"SELECT rev FROM items WHERE id=? AND {NOTE_HEAD_SQL}", (note_id,)).fetchone()
    if current is None:
        raise NoteNotFound("Note not found")
    rows = connection.execute(
        "SELECT rev,title,edited_by,edited_at,length(text) AS body_chars FROM item_revisions "
        "WHERE item_id=? ORDER BY rev DESC", (note_id,),
    ).fetchall()
    return {"id": note_id, "current_rev": int(current[0]), "revisions": [dict(row) for row in rows]}


def _unit_vector(total: list[float]) -> list[float] | None:
    norm = math.sqrt(sum(value * value for value in total))
    return [value / norm for value in total] if norm > 0 and math.isfinite(norm) else None


def _note_centroids(connection: sqlite3.Connection) -> dict[int, list[float]]:
    """Normalized mean of the stored part vectors of each live document. Computes no embedding."""
    sums: dict[int, list[float]] = {}
    for note_id, blob in connection.execute(
        f"SELECT c.note_id,c.embedding FROM {NOTE_CHUNKS_SQL} c JOIN items n ON n.id=c.note_id "
        f"WHERE n.deleted_at IS NULL AND {vector_present_sql('c.embedding')}"
    ):
        vector = array("f", blob)
        total = sums.get(note_id)
        if total is None:
            sums[note_id] = list(vector)
        elif len(total) == len(vector):
            for index, value in enumerate(vector):
                total[index] += value
    centroids = {}
    for note_id, total in sums.items():
        unit = _unit_vector(total)
        if unit is not None:
            centroids[int(note_id)] = unit
    return centroids


def similar_notes(connection: sqlite3.Connection, note_id: int, *, limit: int = SIMILAR_NOTES_LIMIT) -> list[dict]:
    centroids = _note_centroids(connection)
    target = centroids.pop(note_id, None)
    if target is None:
        return []
    scored = sorted(((math.fsum(a * b for a, b in zip(target, vector)), other)
                     for other, vector in centroids.items() if len(vector) == len(target)), reverse=True)[:limit]
    if not scored:
        return []
    titles = dict(connection.execute(
        f"SELECT id,title FROM items WHERE id IN ({','.join('?' * len(scored))})", [other for _, other in scored],
    ).fetchall())
    return [{"id": other, "title": titles.get(other, ""), "score": round(score, 4)} for score, other in scored]


def note_related(connection: sqlite3.Connection, note_id: int, *, similar_limit: int = SIMILAR_NOTES_LIMIT) -> dict:
    """Backlinks (live documents and memories pointing here), outgoing and broken links, similar documents."""
    if connection.execute(f"SELECT 1 FROM items WHERE id=? AND {NOTE_HEAD_SQL}", (note_id,)).fetchone() is None:
        raise NoteNotFound("Note not found")
    backlinks = []
    for row in connection.execute(
        "SELECT DISTINCT s.type,l.src_id,s.title,s.text,s.deleted_at FROM item_links l JOIN items s ON s.id=l.src_id "
        "WHERE l.relation='refers' AND l.dst_id=? AND l.src_id<>? "
        f"ORDER BY CASE s.type WHEN '{ITEM_MEMORY}' THEN 0 ELSE 1 END,l.src_id",
        (note_id, note_id),
    ):
        if row["type"] != ITEM_MEMORY and row["title"] is not None and row["deleted_at"] is None:
            backlinks.append({"type": "note", "id": int(row["src_id"]), "title": row["title"]})
        elif row["type"] == ITEM_MEMORY:
            backlinks.append({"type": "memory", "id": int(row["src_id"]),
                              "title": _clip(row["text"], BACKLINK_EXCERPT_CHARS)})
    outgoing, broken = [], []
    for row in _outgoing_links(connection, note_id):
        entry = {"type": row["dst_type"], "id": None if row["dst_id"] is None else int(row["dst_id"]),
                 "title": row["dst_title"]}
        (broken if row["dst_id"] is None else outgoing).append(entry)
    return {"id": note_id, "backlinks": backlinks, "outgoing": outgoing, "broken": broken,
            "similar": similar_notes(connection, note_id, limit=similar_limit)}
