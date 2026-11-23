"""Save-time older pairs that the saving agent did not settle, and the write path both settle them with.

neospike_memory_save lists earlier similar items in its reply ("older"). Each (new memory, older item) pair is
put in a waiting list (meta_entries, PENDING_META_KEY) when the reply is made; a neospike_memory_reconcile call
that changes, deletes or keeps either side takes it off in the same write transaction. Pairs still waiting are
judged once a day by an LLM inside the preference tidy-up idle task (pref_reconcile.run): keep both, update
(rewrite one, delete the other) or delete one. Pairs judged "keep" are remembered with both body hashes
(KEPT_META_KEY), so the same pair is never asked again.

one_write() is the write scope of both: every memory correction, memory deletion, chunk edit and waiting-list
change of one call commit together or not at all.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
import time
from typing import Any, Callable, Iterable, Iterator

from .config import EXTRACTION_LANGUAGE_DEFAULT, LANGUAGE_NAMES
from .notes import NoteError, formula_chunk_places, formula_chunk_text, reconcile_formula_chunks
from .policy import AdmitVerdict, check_memory_item, screen_text
from .service import cut_memory_text, numbered_subject
from .store import AGENT_EDITOR, FORMULA_AUTHOR, ITEM_MEMORY, NOTE_CHUNKS_SQL, MemoryStore, normalize_rows
from .topics import FORMULA_EDITOR

PENDING_META_KEY = "memory_reconcile_pending"
KEPT_META_KEY = "memory_reconcile_kept"
LANE_MEMORY, LANE_CHUNK = "id", "chunk_id"
LANES = (LANE_MEMORY, LANE_CHUNK)
_REF_PREFIX = {LANE_MEMORY: "id:", LANE_CHUNK: "chunk:"}
BASIS_USER_SAID = "user_said"
_BEGIN_WRITE_SQL = "BEGIN IMMEDIATE"
_SAVEPOINT_PREFIX = "neospike_nested_"
logger = logging.getLogger(__name__)


class MemoryReconcileTuning:
    PAIRS_PER_CALL = 32
    MAX_CALLS = 4  # pairs beyond MAX_CALLS * PAIRS_PER_CALL wait for the next day (oldest first)
    PENDING_MAX = 1000  # waiting pairs kept; past it the oldest are dropped (one meta value, rewritten per save)
    KEPT_MAX = 2000  # remembered keep pairs; past it the oldest are forgotten
    HASH_CHARS = 32


ACTION_KEEP, ACTION_UPDATE, ACTION_DELETE = "keep", "update", "delete"
ACTIONS = (ACTION_KEEP, ACTION_UPDATE, ACTION_DELETE)
SKIP_BAD_JSON = "bad_json"
SKIP_UNKNOWN_PAIR = "unknown_pair"
SKIP_BAD_DECISION = "bad_decision"
SKIP_NO_ANSWER = "no_answer"
SKIP_PROTECTED = "user_said_protected"
SKIP_REFUSED = "write_refused"
SKIP_LLM_ERROR = "llm_error"
SKIP_EMBED_ERROR = "embedding_error"
# Skips that leave the pair waiting for the next day; every other outcome takes it off the list.
_RETRY_SKIPS = frozenset({SKIP_BAD_JSON, SKIP_NO_ANSWER, SKIP_LLM_ERROR, SKIP_EMBED_ERROR})

SYSTEM_PROMPT = (
    "You tidy a user's stored memory items. Each input item is a pair of similar items of the same user; one is "
    'marked "newer" (saved later) and one "older". For every pair return one decision:\n'
    '- "keep": both stay unchanged (no conflict: they say different things, or both are true).\n'
    '- "update": keep_id is the item that stays, text is its corrected full body; the other item is deleted.\n'
    '- "delete": keep_id is the item that stays unchanged; the other item is deleted.\n'
    "Rules: the newer item wins a conflict. Never merge: an update only removes from the kept item what the "
    "newer item made stale, and adds nothing from the deleted item. If both mean the same, keep the newer one "
    "(delete). Never delete or rewrite an item with basis user_said because of an item with another basis. "
    "When unsure, choose keep.\n"
    "Write every new text in {language}.\n"
    'Answer with a JSON array only, one object per pair: {{"pair":[ref_a,ref_b],"action":"keep|update|delete",'
    '"keep_id":ref or null,"text":"new text" or null}}. Refs are the strings given. No other words.'
)


def system_prompt(language: str) -> str:
    return SYSTEM_PROMPT.format(language=LANGUAGE_NAMES.get(language, LANGUAGE_NAMES[EXTRACTION_LANGUAGE_DEFAULT]))


def body_hash(text: str) -> str:
    """Hash of a body as an automatic-note chunk would hold it, so a memory and the chunk note-absorb made of
    it hash alike."""
    digest = hashlib.sha256(formula_chunk_text(str(text)).encode("utf-8")).hexdigest()
    return digest[:MemoryReconcileTuning.HASH_CHARS]


# -- one write transaction --------------------------------------------------------------------------------------

class _NestedConnection:
    """Stands in for store._conn inside one_write(): the store's own BEGIN, commit, rollback and `with conn`
    become savepoints of the outer transaction, so an inner "conflict" undo stays local and an inner commit
    commits nothing. A rollback with no savepoint open marks the whole write for rollback."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._real = connection
        self.depth = 0
        self.aborted = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def _open(self) -> sqlite3.Cursor:
        self.depth += 1
        return self._real.execute(f"SAVEPOINT {_SAVEPOINT_PREFIX}{self.depth}")

    def _release(self) -> None:
        self._real.execute(f"RELEASE {_SAVEPOINT_PREFIX}{self.depth}")
        self.depth -= 1

    def _undo(self) -> None:
        name = f"{_SAVEPOINT_PREFIX}{self.depth}"
        self._real.execute(f"ROLLBACK TO {name}")
        self._real.execute(f"RELEASE {name}")
        self.depth -= 1

    def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
        if isinstance(sql, str) and sql.lstrip().upper().startswith("BEGIN"):
            return self._open()
        return self._real.execute(sql, *args)

    def executemany(self, sql: str, *args: Any) -> sqlite3.Cursor:
        return self._real.executemany(sql, *args)

    def commit(self) -> None:
        if self.depth:
            self._release()

    def rollback(self) -> None:
        if self.depth:
            self._undo()
        else:
            self.aborted = True

    def __enter__(self) -> "_NestedConnection":
        self._open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is None:
            self._release()
        else:
            self._undo()
        return False


@contextlib.contextmanager
def one_write(store: MemoryStore) -> Iterator[sqlite3.Connection]:
    """One BEGIN IMMEDIATE transaction under the store lock for several store writes (update_memory_item, delete,
    the chunk and meta functions on the yielded connection): committed when the block ends, rolled back
    entirely on any exception. The store's own transactions inside become savepoints."""
    with store._lock:
        real = store._conn
        real.execute(_BEGIN_WRITE_SQL)
        nested = _NestedConnection(real)
        store._conn = nested
        try:
            yield real
            if nested.aborted or nested.depth:
                raise RuntimeError("A store write was undone inside the transaction")
        except BaseException:
            store._conn = real
            real.rollback()
            raise
        store._conn = real
        real.commit()


# -- waiting list and keep memory (meta_entries) ------------------------------------------------------------

def _read_list(connection: sqlite3.Connection, key: str) -> list:
    rec = connection.execute("SELECT value FROM meta_entries WHERE key=?", (key,)).fetchone()
    try:
        values = json.loads(rec[0]) if rec else []
    except (TypeError, ValueError):
        return []
    return values if isinstance(values, list) else []


def _write_list(connection: sqlite3.Connection, key: str, values: list) -> None:
    connection.execute("INSERT OR REPLACE INTO meta_entries(key, value) VALUES (?, ?)",
                       (key, json.dumps(values, ensure_ascii=False, separators=(",", ":"))))


def _is_entry(value: Any) -> bool:
    return (isinstance(value, dict) and type(value.get("memory_id")) is int and value.get("lane") in LANES
            and type(value.get("other_id")) is int and isinstance(value.get("memory_hash"), str)
            and isinstance(value.get("other_hash"), str))


def read_pending(connection: sqlite3.Connection) -> list[dict]:
    """The waiting pairs, oldest first (caller holds the store lock)."""
    return [value for value in _read_list(connection, PENDING_META_KEY) if _is_entry(value)]


def _entry_identity(entry: dict) -> tuple[int, str, int]:
    return entry["memory_id"], entry["lane"], entry["other_id"]


def side_key(lane: str, item_id: int, digest: str) -> str:
    return f"{lane}:{item_id}:{digest}"


def pair_key(first: str, second: str) -> str:
    """Identity of a pair and of both bodies: a changed side makes it a new question."""
    return "|".join(sorted((first, second)))


def _entry_key(entry: dict) -> str:
    return pair_key(side_key(LANE_MEMORY, entry["memory_id"], entry["memory_hash"]),
                    side_key(entry["lane"], entry["other_id"], entry["other_hash"]))


def read_kept(connection: sqlite3.Connection) -> list[str]:
    return [value for value in _read_list(connection, KEPT_META_KEY) if isinstance(value, str)]


def add_pending(store: MemoryStore, memory_id: int, memory_text: str, older: Iterable[dict],
                now: float | None = None) -> int:
    """Put (new memory, older item) pairs of one save reply on the waiting list; older items are the reply's
    {"id" | "chunk_id", "text"}. A pair already waiting or remembered as kept is not added again. Returns the
    number added."""
    stamp = time.time() if now is None else now
    memory_hash = body_hash(memory_text)
    fresh = []
    for item in older:
        lane = LANE_MEMORY if LANE_MEMORY in item else LANE_CHUNK
        fresh.append({"memory_id": int(memory_id), "lane": lane, "other_id": int(item[lane]),
                      "memory_hash": memory_hash, "other_hash": body_hash(item.get("text", "")), "at": stamp})
    if not fresh:
        return 0
    with one_write(store) as connection:
        pending = read_pending(connection)
        kept = set(read_kept(connection))
        waiting = {_entry_identity(entry) for entry in pending}
        added = [entry for entry in fresh if _entry_identity(entry) not in waiting and _entry_key(entry) not in kept]
        if not added:
            return 0
        pending.extend(added)
        overflow = len(pending) - MemoryReconcileTuning.PENDING_MAX
        if overflow > 0:
            logger.info("Neospike memory reconcile: %d oldest waiting pairs dropped (list full)", overflow)
            pending = pending[overflow:]
        _write_list(connection, PENDING_META_KEY, pending)
    return len(added)


def drop_pending(connection: sqlite3.Connection, items: Iterable[tuple[str, int]]) -> int:
    """Inside the caller's write transaction: take off every waiting pair with one of the (lane, id) items
    on either side (the new memory is lane "id"). Returns the number taken off."""
    touched = {(lane, int(item_id)) for lane, item_id in items}
    if not touched:
        return 0
    pending = read_pending(connection)
    rest = [entry for entry in pending if (LANE_MEMORY, entry["memory_id"]) not in touched
            and (entry["lane"], entry["other_id"]) not in touched]
    if len(rest) != len(pending):
        _write_list(connection, PENDING_META_KEY, rest)
    return len(pending) - len(rest)


def _drop_entries(connection: sqlite3.Connection, done: set[tuple[int, str, int]]) -> None:
    pending = read_pending(connection)
    rest = [entry for entry in pending if _entry_identity(entry) not in done]
    if len(rest) != len(pending):
        _write_list(connection, PENDING_META_KEY, rest)


def _remember_kept(connection: sqlite3.Connection, keys: Iterable[str]) -> None:
    kept = read_kept(connection)
    known = set(kept)
    kept.extend(key for key in keys if key not in known)
    _write_list(connection, KEPT_META_KEY, kept[-MemoryReconcileTuning.KEPT_MAX:])


# -- shared write helpers ---------------------------------------------------------------------------------

def checked_pieces(text: str) -> list[str] | None:
    """A new body split like neospike_memory_update (paragraph, sentence, limit); None when empty or when a
    piece fails the save policy."""
    pieces = cut_memory_text(text)
    if not pieces or any(screen_text(piece) is not AdmitVerdict.INDEX for piece in pieces):
        return None
    return pieces


def correction_items(new_text: str, subject: str, durability: str) -> list[dict]:
    """Validated memory items of a correction: one, or numbered pieces over the item limit (the
    NeospikeService.update_memory_item preparation)."""
    parts = cut_memory_text(new_text)
    total = len(parts)
    return [check_memory_item({"text": part, "durability": durability,
                               "subject": numbered_subject(subject, index, total) if total > 1 else subject},
                              partial_defaults=True)
            for index, part in enumerate(parts, 1)]


def correct_memory(store: MemoryStore, namespace: str, row: dict, items: list[dict], embeddings: list[list[float]],
                   *, project: str, session_id: str, edited_by: str = AGENT_EDITOR) -> tuple[str, list[int]]:
    """store.update_memory_item in place (the old body becomes a revision by edited_by): (status, [id, piece
    ids...])."""
    split_ids: list[int] = []
    status = store.update_memory_item(namespace, int(row["id"]), expected_text=row["text"], expected_row=row,
                                      item=items[0], embedding=embeddings[0], project=project,
                                      session_id=session_id, edited_by=edited_by, split_items=items[1:],
                                      split_embeddings=embeddings[1:], split_ids=split_ids)
    return status, [int(row["id"]), *split_ids]


def chunk_blob(embedder: Any, text: str) -> bytes:
    return normalize_rows(embedder.embed_document(text)).tobytes()


# -- daily judgement --------------------------------------------------------------------------------------

_MEMORY_SQL = (f"SELECT * FROM items WHERE id=? AND type='{ITEM_MEMORY}' AND kind='memory_item' "
               "AND archived_at IS NULL AND invalid_at IS NULL")
_CHUNK_SQL = "SELECT text, json_extract(metadata_json,'$.source_json') FROM items WHERE id=?"
# Live automatic-note chunks made from one memory (note-absorb, a save-time duplicate chunk).
_ABSORBED_SQL = (f"SELECT c.id FROM {NOTE_CHUNKS_SQL} c JOIN items n ON n.id=c.note_id "
                 f"WHERE n.author='{FORMULA_AUTHOR}' AND n.deleted_at IS NULL AND json_valid(c.source_json) "
                 "AND json_extract(c.source_json,'$.memory_id')=? ORDER BY c.id")


def _json_dict(text: Any) -> dict:
    try:
        value = json.loads(text or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _memory_side(connection: sqlite3.Connection, memory_id: int) -> dict | None:
    row = connection.execute(_MEMORY_SQL, (memory_id,)).fetchone()
    if row is None:
        return None
    record = dict(row)
    metadata = _json_dict(record.get("metadata_json"))
    return {"lane": LANE_MEMORY, "id": memory_id, "text": str(record["text"]),
            "basis": str(metadata.get("basis") or ""), "row": record, "owner": record["owner"],
            "project": str(record.get("project") or ""), "session_id": str(metadata.get("session_id") or ""),
            "subject": str(metadata.get("subject") or "memory"),
            "durability": str(metadata.get("durability") or "ongoing")}


def _chunk_side(connection: sqlite3.Connection, chunk_id: int) -> dict | None:
    if chunk_id not in formula_chunk_places(connection, [chunk_id]):
        return None
    row = connection.execute(_CHUNK_SQL, (chunk_id,)).fetchone()
    if row is None:
        return None
    return {"lane": LANE_CHUNK, "id": chunk_id, "text": str(row[0]),
            "basis": str(_json_dict(row[1]).get("basis") or "")}


def _live_side(connection: sqlite3.Connection, lane: str, item_id: int, digest: str) -> dict | None:
    """The side as it is now when its body still hashes to digest; a memory gone into an automatic-note chunk
    (note-absorb) is that chunk. None: gone, changed or archived."""
    side = _memory_side(connection, item_id) if lane == LANE_MEMORY else _chunk_side(connection, item_id)
    if side is None and lane == LANE_MEMORY and connection.execute(
            f"SELECT 1 FROM items WHERE id=? AND type='{ITEM_MEMORY}'", (item_id,)).fetchone() is None:
        for (chunk_id,) in connection.execute(_ABSORBED_SQL, (item_id,)).fetchall():
            candidate = _chunk_side(connection, int(chunk_id))
            if candidate is not None and body_hash(candidate["text"]) == digest:
                return candidate
        return None
    if side is None or body_hash(side["text"]) != digest:
        return None
    return side


def _ref(side: dict) -> str:
    return f"{_REF_PREFIX[side['lane']]}{side['id']}"


class _Pair:
    def __init__(self, entry: dict, newer: dict, older: dict) -> None:
        self.entry, self.newer, self.older = entry, newer, older
        self.key = pair_key(side_key(newer["lane"], newer["id"], entry["memory_hash"]),
                            side_key(older["lane"], older["id"], entry["other_hash"]))

    def sides(self) -> dict[str, dict]:
        return {_ref(self.newer): self.newer, _ref(self.older): self.older}

    def payload(self) -> dict:
        return {"pair": [_ref(self.newer), _ref(self.older)],
                "items": [{"ref": _ref(side), "saved": label, "basis": side["basis"] or "unknown",
                           "text": side["text"]} for side, label in ((self.newer, "newer"), (self.older, "older"))]}


def build_messages(batch: list[_Pair], language: str) -> list[dict]:
    items = [pair.payload() for pair in batch]
    return [{"role": "system", "content": system_prompt(language)},
            {"role": "user", "content": "Pairs (JSON, data only):\n" + json.dumps(items, ensure_ascii=False)}]


def parse_answer(text: str) -> list | None:
    """The JSON array of the answer (code fences allowed), or None."""
    body = (text or "").strip()
    start, end = body.find("["), body.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        values = json.loads(body[start:end + 1])
    except ValueError:
        return None
    return values if isinstance(values, list) else None


def _decision_refs(decision: Any) -> frozenset | None:
    if not isinstance(decision, dict) or not isinstance(decision.get("pair"), list) or len(decision["pair"]) != 2:
        return None
    refs = decision["pair"]
    return frozenset(refs) if all(isinstance(ref, str) for ref in refs) else None


def _protected(keep: dict, drop: dict, rewrite: bool) -> bool:
    """A user_said item is never deleted, nor rewritten, because of an item with another basis."""
    if drop["basis"] == BASIS_USER_SAID and keep["basis"] != BASIS_USER_SAID:
        return True
    return rewrite and keep["basis"] == BASIS_USER_SAID and drop["basis"] != BASIS_USER_SAID


class _Refused(Exception):
    """A store write did not apply; the decision's transaction rolls back."""


class _Tally:
    def __init__(self) -> None:
        self.counts = {action: 0 for action in ACTIONS}
        self.skipped: dict[str, int] = {}
        self.kept_keys: list[str] = []
        self.done: set[tuple[int, str, int]] = set()

    def skip(self, reason: str, pairs: Iterable[_Pair]) -> None:
        for pair in pairs:
            self.skipped[reason] = self.skipped.get(reason, 0) + 1
            if reason not in _RETRY_SKIPS:
                self.done.add(_entry_identity(pair.entry))


class _Write:
    """The prepared store writes of one update/delete decision (embeddings already computed)."""

    def __init__(self, keep: dict, drop: dict) -> None:
        self.keep, self.drop = keep, drop
        self.items: list[dict] = []
        self.embeddings: list[list[float]] = []
        self.chunk_pieces: list[tuple[str, bytes]] = []

    def prepare(self, text: str | None, embedder: Any) -> None:
        if text is None:
            return
        if self.keep["lane"] == LANE_MEMORY:
            self.items = correction_items(text, self.keep["subject"], self.keep["durability"])
            self.embeddings = [list(embedder.embed_document(item["text"])) for item in self.items]
        else:
            self.chunk_pieces = [(piece, chunk_blob(embedder, piece)) for piece in cut_memory_text(text)]

    def apply(self, store: MemoryStore, connection: sqlite3.Connection) -> None:
        replace: dict[int, list[tuple[str, bytes]]] = {}
        deleted_chunks: list[int] = []
        if self.items:
            status, _ = correct_memory(store, self.keep["owner"], self.keep["row"], self.items, self.embeddings,
                                       project=self.keep["project"], session_id=self.keep["session_id"],
                                       edited_by=FORMULA_EDITOR)
            if status != "updated":
                raise _Refused(status)
        if self.chunk_pieces:
            replace[self.keep["id"]] = self.chunk_pieces
        if self.drop["lane"] == LANE_MEMORY:
            if not store.delete(self.drop["owner"], self.drop["id"], project=self.drop["project"],
                                session_id=self.drop["session_id"]):
                raise _Refused("delete")
        else:
            deleted_chunks.append(self.drop["id"])
        if replace or deleted_chunks:
            reconcile_formula_chunks(connection, replace, deleted_chunks, editor=FORMULA_EDITOR, now=time.time())


def _apply(store: MemoryStore, embedder: Any, pair: _Pair, decision: dict, tally: _Tally) -> None:
    action = decision.get("action")
    if action == ACTION_KEEP:
        tally.counts[ACTION_KEEP] += 1
        tally.kept_keys.append(pair.key)
        tally.done.add(_entry_identity(pair.entry))
        return
    sides = pair.sides()
    keep_ref = decision.get("keep_id")
    if action not in (ACTION_UPDATE, ACTION_DELETE) or keep_ref not in sides:
        tally.skip(SKIP_BAD_DECISION, [pair])
        return
    keep = sides[keep_ref]
    drop = pair.older if keep is pair.newer else pair.newer
    text = None
    if action == ACTION_UPDATE:
        raw = decision.get("text")
        if not isinstance(raw, str) or checked_pieces(raw) is None:
            tally.skip(SKIP_BAD_DECISION, [pair])
            return
        text = raw.strip() if raw.strip() != keep["text"] else None
    if _protected(keep, drop, text is not None):
        tally.skip(SKIP_PROTECTED, [pair])
        return
    write = _Write(keep, drop)
    try:
        write.prepare(text, embedder)
    except ValueError:
        tally.skip(SKIP_BAD_DECISION, [pair])
        return
    except Exception:  # noqa: BLE001 - an unreachable embedder: the pair waits for the next day
        tally.skip(SKIP_EMBED_ERROR, [pair])
        return
    try:
        with one_write(store) as connection:
            write.apply(store, connection)
            _drop_entries(connection, {_entry_identity(pair.entry)})
    except (_Refused, sqlite3.Error, NoteError):
        tally.skip(SKIP_REFUSED, [pair])
        return
    tally.counts[action] += 1
    tally.done.add(_entry_identity(pair.entry))


def _judge_batch(store: MemoryStore, embedder: Any, batch: list[_Pair], llm: Callable[[list[dict]], str],
                 language: str, tally: _Tally, errors: list[str]) -> None:
    try:
        answer = llm(build_messages(batch, language))
    except Exception as exc:  # any provider failure: the batch waits, the run reports it
        errors.append(f"{type(exc).__name__}: {exc}")
        tally.skip(SKIP_LLM_ERROR, batch)
        return
    decisions = parse_answer(answer)
    if decisions is None:
        tally.skip(SKIP_BAD_JSON, batch)
        return
    by_refs = {frozenset(pair.sides()): pair for pair in batch}
    answered: set[frozenset] = set()
    for decision in decisions:
        refs = _decision_refs(decision)
        if refs is None or refs not in by_refs or refs in answered:
            tally.skipped[SKIP_UNKNOWN_PAIR] = tally.skipped.get(SKIP_UNKNOWN_PAIR, 0) + 1
            continue
        answered.add(refs)
        _apply(store, embedder, by_refs[refs], decision, tally)
    tally.skip(SKIP_NO_ANSWER, [pair for refs, pair in by_refs.items() if refs not in answered])


def _live_pairs(store: MemoryStore, pending: list[dict], kept: set[str]) -> tuple[list[_Pair], set, int]:
    """(pairs to judge, oldest first, each item in at most one; identities to take off unasked: gone, changed,
    archived or kept before; number deferred to the next day)."""
    chosen: list[_Pair] = []
    used: set[tuple[str, int]] = set()
    stale: set[tuple[int, str, int]] = set()
    deferred = 0
    with store._lock:
        for entry in pending:
            newer = _live_side(store._conn, LANE_MEMORY, entry["memory_id"], entry["memory_hash"])
            older = _live_side(store._conn, entry["lane"], entry["other_id"], entry["other_hash"])
            pair = None if newer is None or older is None else _Pair(entry, newer, older)
            if pair is None or (newer["lane"], newer["id"]) == (older["lane"], older["id"]) or pair.key in kept:
                stale.add(_entry_identity(entry))
                continue
            ids = {(newer["lane"], newer["id"]), (older["lane"], older["id"])}
            if used & ids:
                deferred += 1
                continue
            used |= ids
            chosen.append(pair)
    return chosen, stale, deferred


def judge(store: MemoryStore, llm: Callable[[list[dict]], str], embedder: Any, *, language: str) -> dict:
    """One daily run over the waiting pairs. Returns counts: pending, judged, applied, keep/update/delete,
    skipped (total) and skip_reasons, dropped (gone, changed or kept before: taken off unasked), deferred,
    calls, errors (LLM call failures)."""
    with store._lock:
        pending = read_pending(store._conn)
        kept = set(read_kept(store._conn))
    chosen, stale, deferred = _live_pairs(store, pending, kept)
    limit = MemoryReconcileTuning.MAX_CALLS * MemoryReconcileTuning.PAIRS_PER_CALL
    deferred += max(0, len(chosen) - limit)
    chosen = chosen[:limit]
    tally, errors, calls = _Tally(), [], 0
    for start in range(0, len(chosen), MemoryReconcileTuning.PAIRS_PER_CALL):
        calls += 1
        _judge_batch(store, embedder, chosen[start:start + MemoryReconcileTuning.PAIRS_PER_CALL], llm, language,
                     tally, errors)
    finished = tally.done | stale
    if finished or tally.kept_keys:
        with one_write(store) as connection:
            _drop_entries(connection, finished)
            if tally.kept_keys:
                _remember_kept(connection, tally.kept_keys)
    applied = tally.counts[ACTION_UPDATE] + tally.counts[ACTION_DELETE]
    return {"pending": len(pending), "judged": len(chosen), "applied": applied, **tally.counts,
            "skipped": sum(tally.skipped.values()), "skip_reasons": tally.skipped, "dropped": len(stale),
            "deferred": deferred, "calls": calls, "errors": errors}


def summary_part(result: dict) -> str:
    return (f"memory pairs: judged {result['judged']} of {result['pending']} in {result['calls']} calls, "
            f"applied {result['applied']}, keep {result[ACTION_KEEP]}, skipped {result['skipped']}, "
            f"dropped {result['dropped']}, deferred {result['deferred']}")


class LazyEmbedder:
    """embed_document that starts the inference client of a home on first use (none when nothing is rewritten)."""

    def __init__(self, home: Any) -> None:
        self._home = home
        self._client: Any = None

    def embed_document(self, text: str) -> list[float]:
        if self._client is None:
            from .inference import EmbeddingClient
            self._client = EmbeddingClient(self._home)
        return self._client.embed_document(text)
