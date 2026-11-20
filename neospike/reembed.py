"""`reembed`: rewrite every stored text vector with the current text model.

The store refuses a database whose meta embedding dimension/model differ from the code (EMBEDDING_DIMENSIONS,
EMBEDDING_MODEL). This command opens it without that check, re-embeds each table in id order and commits every
BATCH_ROWS rows together with the table's progress (meta key PROGRESS_PREFIX + table = last id done), so a stopped
run continues where it left off. Only when every table is done are meta embedding_dimensions/embedding_model
updated and the progress keys removed. Rows without a vector (embedding-pending memories, unindexed conversation
chunks) are left to their own fill stages. Row timestamps (updated_at, accessed_at) are kept as they were.
The database backup is the operator's job; this command makes none.
"""
from __future__ import annotations

import math
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

from .config import EMBEDDING_MODEL
from .store import (
    MemoryStore,
    normalize_rows,
    read_embedding_meta,
    store_embedding_meta,
)

BATCH_ROWS = 32
PROGRESS_PREFIX = "reembed_progress:"
TARGET_KEY = "reembed_target"  # model/dimensions the stored progress belongs to


@dataclass(frozen=True)
class _Table:
    name: str
    vector: str
    text_sql: str  # SQL expression of the embedded text (same text the writer embedded)
    has_vector_sql: str
    progress_key: str = ""  # meta progress name when one table has two vector columns (default: the table)

    @property
    def progress_name(self) -> str:
        return self.progress_key or self.name


# Text vector columns of memory.db; each writer embeds the row's text column with embed_document.
MEMORY_TABLES = (
    # A document part is embedded without its outer whitespace (notes.part_embed_text); memory text has none.
    _Table("items", "embedding", "trim(text, char(32,9,10,13,12,11))", "length(embedding)>0"),
    _Table("conversation_chunks", "embedding", "text", "embedding IS NOT NULL AND length(embedding)>0"),
    # v23 preferences: the text vector and the situation vector (common rows have none).
    _Table("preferences", "embedding", "text", "length(embedding)>0"),
    _Table("preferences", "situation_embedding", "situation", "length(situation_embedding)>0",
           "preferences.situation"),
)


# Tables computed from the old vectors: emptied in the transaction that records the new model, so the
# next daily note-absorb / topic-assign / topic-discover compute them again from the new vectors. item_links
# (topic links and hand edits included) are kept; a topic has no stored vector.
DERIVED_TABLES = ("memory_pairs", "memory_checks")

# Change-tracking column a vector rewrite must leave as it was (store.py tracking triggers).
TRACKED_COLUMN = "updated_at"


def _column_names(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


class ReembedChanged(RuntimeError):
    """A row's text changed between reading and writing; rerun to continue from the last committed batch."""


def _target(dimensions: int) -> str:
    return f"{EMBEDDING_MODEL}/{dimensions}"


def _stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _pack(vector: Any, dimensions: int, *, allow_zero: bool) -> bytes:
    packed = normalize_rows(vector)
    if len(packed) != dimensions:
        raise RuntimeError(f"Embedder returned {len(packed)} dimensions, expected {dimensions} "
                           "(an inference daemon of the old model may still be running; restart it)")
    if not all(math.isfinite(value) for value in packed) or not (allow_zero or any(packed)):
        raise RuntimeError("Invalid embedding (non-finite or zero)")
    return packed.tobytes()


class _Run:
    def __init__(self, store: MemoryStore, embedder: Any, *, budget_seconds: float | None,
                 batch_rows: int, log: Callable[[str], None]) -> None:
        self.store = store
        self.embedder = embedder
        self.dimensions = store.dimensions
        self.deadline = None if budget_seconds is None else time.monotonic() + budget_seconds
        self.batch_rows = batch_rows
        self.log = log
        self.started = time.monotonic()
        self.embedded = 0

    def _meta(self, key: str) -> str | None:
        row = self.store._conn.execute("SELECT value FROM meta_entries WHERE key=?", (key,)).fetchone()
        return None if row is None else row[0]

    def progress(self, table: str) -> int:
        value = self._meta(PROGRESS_PREFIX + table)
        return int(value) if value is not None else 0

    def out_of_time(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline

    def reset_stale_progress(self) -> None:
        """Progress of a run toward another model/dimension is void."""
        with self.store._lock, self.store._conn:
            if self._meta(TARGET_KEY) != _target(self.dimensions):
                self.store._conn.execute("DELETE FROM meta_entries WHERE key LIKE ?", (PROGRESS_PREFIX + "%",))
                self.store._conn.execute("INSERT OR REPLACE INTO meta_entries(key, value) VALUES (?, ?)",
                                         (TARGET_KEY, _target(self.dimensions)))

    def table(self, table: _Table, connection: sqlite3.Connection, *, allow_zero: bool) -> bool:
        """Re-embed one table from its stored progress; True when it is done, False when the budget ran out."""
        last = self.progress(table.progress_name)
        total = connection.execute(f"SELECT COUNT(*) FROM {table.name} WHERE {table.has_vector_sql}").fetchone()[0]
        done = connection.execute(f"SELECT COUNT(*) FROM {table.name} WHERE {table.has_vector_sql} AND id<=?",
                                  (last,)).fetchone()[0]
        while True:
            if done < total and self.out_of_time():
                self.log(f"reembed {table.name} {done}/{total} stopped (budget) {self._elapsed()}")
                return False
            rows = connection.execute(
                f"SELECT id, {table.text_sql} FROM {table.name} WHERE id>? AND {table.has_vector_sql} "
                f"ORDER BY id LIMIT ?", (last, self.batch_rows)).fetchall()
            if not rows:
                self.log(f"reembed {table.name} {done}/{total} done {self._elapsed()}")
                return True
            updates = []
            for row_id, text in rows:
                vector = (self.embedder.embed_document(text) if text
                          else [0.0] * self.dimensions)
                updates.append((_pack(vector, self.dimensions, allow_zero=allow_zero), row_id, text))
            self.embedded += len(updates)
            last = rows[-1][0]
            self._write(table, connection, updates, last)
            done += len(rows)
            self.log(f"reembed {table.name} {done}/{total} {self._elapsed()}")

    def _write(self, table: _Table, connection: sqlite3.Connection, updates: list, last: int) -> None:
        """One batch and its progress in one transaction."""
        tracked = TRACKED_COLUMN in _column_names(connection, table.name)
        # The tracking trigger ({table}_track_au) bumps updated_at when the statement leaves it equal to the old
        # value, so writing the old value back fails. Instead the vector UPDATE parks updated_at on a value that
        # differs from the stored one (trigger condition false), and a second UPDATE restores the original (also
        # different from the parked value). A new vector is not a user-visible change.
        parked = (f", {TRACKED_COLUMN}=CASE WHEN {TRACKED_COLUMN} IS NULL OR {TRACKED_COLUMN}<>0 THEN 0 ELSE 1 END"
                  if tracked else "")
        sql = f"UPDATE {table.name} SET {table.vector}=?{parked} WHERE id=? AND {table.text_sql}=?"
        restore = f"UPDATE {table.name} SET {TRACKED_COLUMN}=? WHERE id=?"
        with self.store._lock:
            with connection:
                original = (dict(connection.execute(
                    f"SELECT id, {TRACKED_COLUMN} FROM {table.name} WHERE id IN ({','.join('?' * len(updates))})",
                    [u[1] for u in updates]).fetchall()) if tracked else {})
                for update in updates:
                    if connection.execute(sql, update).rowcount != 1:
                        raise ReembedChanged(f"{table.name} id {update[1]} changed during reembed; rerun reembed")
                    if tracked:
                        connection.execute(restore, (original[update[1]], update[1]))
                self._save_progress(table.progress_name, last)

    def _save_progress(self, table: str, last: int) -> None:
        self.store._conn.execute("INSERT OR REPLACE INTO meta_entries(key, value) VALUES (?, ?)",
                                 (PROGRESS_PREFIX + table, str(last)))

    def finish(self) -> None:
        with self.store._lock, self.store._conn:
            store_embedding_meta(self.store._conn, self.dimensions)
            for table in DERIVED_TABLES:
                self.store._conn.execute(f"DELETE FROM {table}")
            self.store._conn.execute("DELETE FROM meta_entries WHERE key LIKE ? OR key=?", (PROGRESS_PREFIX + "%", TARGET_KEY))

    def _elapsed(self) -> str:
        return f"{time.monotonic() - self.started:.1f}s"


def reembed(store: MemoryStore, embedder: Any, *, budget_seconds: float | None = None, batch_rows: int = BATCH_ROWS,
            log: Callable[[str], None] = _stderr) -> dict:
    """Re-embed every text vector of store (opened with verify_embedding=False). Returns {"complete", "embedded", "elapsed_seconds", "before", "after"}."""
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    if budget_seconds is not None and not (math.isfinite(budget_seconds) and budget_seconds > 0):
        raise ValueError("budget_seconds must be a positive number")
    run = _Run(store, embedder, budget_seconds=budget_seconds, batch_rows=batch_rows, log=log)
    before = read_embedding_meta(store._conn)
    run.reset_stale_progress()
    complete = all(run.table(table, store._conn, allow_zero=False) for table in MEMORY_TABLES)
    if complete:
        run.finish()
    return {"complete": complete, "embedded": run.embedded,
            "elapsed_seconds": round(time.monotonic() - run.started, 1),
            "before": {"dimensions": before[0], "model": before[1]},
            "after": dict(zip(("dimensions", "model"), read_embedding_meta(store._conn)))}
