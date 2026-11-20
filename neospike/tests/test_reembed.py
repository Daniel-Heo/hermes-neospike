"""`reembed` rewrites every stored text vector of an EmbeddingGemma-era (512) database with the current
model (1024), resumably, and only then records the new dimension/model in meta."""
from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

from neospike import backup as db_backup
from neospike import cli
from neospike import conversations as conv
from neospike.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL
from neospike.reembed import DERIVED_TABLES, PROGRESS_PREFIX, TARGET_KEY, reembed
from neospike.service import NeospikeService
from neospike.store import MemoryStore, read_embedding_meta

OLD_DIMENSIONS = 512
FLOAT_BYTES = 4
NAMESPACE = "test:reembed"
VECTOR_TABLES = ("items", "conversation_chunks")
BATCH = 2


class AxisEmbedder:
    """Deterministic non-zero unit vector of any size; counts embed_document calls."""

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS, *, fail_after: int | None = None) -> None:
        self.dimensions = dimensions
        self.fail_after = fail_after
        self.documents = 0

    def embed_document(self, text: str, **_kwargs) -> list[float]:
        if self.fail_after is not None and self.documents >= self.fail_after:
            raise RuntimeError("simulated stop")
        self.documents += 1
        vector = [0.0] * self.dimensions
        vector[sum(map(ord, text)) % self.dimensions] = 1.0
        return vector

    embed_query = embed_document


class _Factory:
    """cli.EmbeddingClient stand-in: EmbeddingClient(home) -> the shared embedder."""

    def __init__(self, embedder: AxisEmbedder) -> None:
        self.embedder = embedder

    def __call__(self, *_args, **_kwargs) -> AxisEmbedder:
        return self.embedder


def _old_home(tmp_path: Path) -> Path:
    """A home whose memory.db holds 512-dimension vectors and no embedding_model key (before reembed)."""
    home = tmp_path / "home"
    path = home / "neospike" / "memory.db"
    store = MemoryStore(path, dimensions=OLD_DIMENSIONS)
    try:
        old = AxisEmbedder(OLD_DIMENSIONS)
        service = NeospikeService(store=store, embedder=old, namespace=NAMESPACE)
        for index in range(5):
            fact = f"Fact number {index} about the lighthouse keeper."
            store.add(NAMESPACE, fact, old.embed_document(fact), source=f"test:{index}")
        service.save_note("Lighthouse", "# Lighthouse\n\nForty meters tall.\n\n# Keeper\n\nLives nearby.", author="human")
        conversations = conv.ConversationStore(store)
        conversations.append("ended", [("user", "How tall is the lighthouse?"), ("assistant", "About forty meters.")])
        conversations.end_session("ended")
        conv.ConversationIndexer(store, old).index_pending()
        with store._conn:
            # One embedding-pending memory: left for embed-pending, never re-embedded here.
            store._conn.execute("UPDATE items SET embedding=x'' WHERE id=(SELECT MAX(id) FROM items WHERE type='memory')")
            store._conn.execute("DELETE FROM meta_entries WHERE key='embedding_model'")
    finally:
        store.close()
    return home


def _database(home: Path) -> Path:
    return home / "neospike" / "memory.db"


def _vector_rows(path: Path) -> dict[str, list[int]]:
    """Byte length of every stored vector per table (pending rows excluded)."""
    with closing(sqlite3.connect(path)) as connection:
        return {table: [row[0] for row in connection.execute(
            f"SELECT length(embedding) FROM {table} WHERE length(embedding)>0 ORDER BY id")] for table in VECTOR_TABLES}


def _open_unchecked(home: Path) -> MemoryStore:
    return MemoryStore(_database(home), dimensions=EMBEDDING_DIMENSIONS, verify_embedding=False)


def test_old_database_is_refused_until_reembed_completes(tmp_path: Path) -> None:
    home = _old_home(tmp_path)
    with pytest.raises(RuntimeError, match="reembed"):
        MemoryStore(_database(home), dimensions=EMBEDDING_DIMENSIONS)
    before = _vector_rows(_database(home))
    total = sum(len(lengths) for lengths in before.values())
    assert all(before.values()) and {length for lengths in before.values() for length in lengths} == {
        OLD_DIMENSIONS * FLOAT_BYTES}

    embedder = AxisEmbedder()
    store = _open_unchecked(home)
    try:
        result = reembed(store, embedder, batch_rows=BATCH, log=_quiet)
    finally:
        store.close()
    assert result["complete"] is True and result["embedded"] == total == embedder.documents
    assert result["before"] == {"dimensions": OLD_DIMENSIONS, "model": None}
    assert result["after"] == {"dimensions": EMBEDDING_DIMENSIONS, "model": EMBEDDING_MODEL}
    after = _vector_rows(_database(home))
    assert {table: len(rows) for table, rows in after.items()} == {table: len(rows) for table, rows in before.items()}
    assert {length for lengths in after.values() for length in lengths} == {EMBEDDING_DIMENSIONS * FLOAT_BYTES}
    with closing(sqlite3.connect(_database(home))) as connection:
        assert connection.execute("SELECT COUNT(*) FROM items WHERE type='memory' AND length(embedding)=0").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM meta_entries WHERE key LIKE ? OR key=?",
                                  (PROGRESS_PREFIX + "%", TARGET_KEY)).fetchone()[0] == 0
    MemoryStore(_database(home), dimensions=EMBEDDING_DIMENSIONS).close()  # the normal open accepts it now


def test_stopped_run_resumes_without_redoing_committed_rows(tmp_path: Path) -> None:
    home = _old_home(tmp_path)
    total = sum(len(lengths) for lengths in _vector_rows(_database(home)).values())
    stop_after = 2 * BATCH + 1  # two batches committed, the third fails mid-batch
    store = _open_unchecked(home)
    try:
        with pytest.raises(RuntimeError, match="simulated stop"):
            reembed(store, AxisEmbedder(fail_after=stop_after), batch_rows=BATCH, log=_quiet)
        assert read_embedding_meta(store._conn) == (OLD_DIMENSIONS, None)  # meta waits for the whole run
        assert store._conn.execute("SELECT value FROM meta_entries WHERE key=?",
                                   (PROGRESS_PREFIX + "items",)).fetchone() is not None
        second = AxisEmbedder()
        result = reembed(store, second, batch_rows=BATCH, log=_quiet)
    finally:
        store.close()
    assert result["complete"] is True
    assert second.documents == total - 2 * BATCH  # the committed rows are not embedded again
    assert {length for lengths in _vector_rows(_database(home)).values() for length in lengths} == {
        EMBEDDING_DIMENSIONS * FLOAT_BYTES}


def test_budget_stops_cleanly_and_keeps_the_old_meta(tmp_path: Path) -> None:
    home = _old_home(tmp_path)
    store = _open_unchecked(home)
    try:
        result = reembed(store, AxisEmbedder(), batch_rows=BATCH, budget_seconds=1e-9, log=_quiet)
        assert result["complete"] is False and read_embedding_meta(store._conn) == (OLD_DIMENSIONS, None)
    finally:
        store.close()


def test_wrong_size_vectors_from_an_old_daemon_are_refused(tmp_path: Path) -> None:
    home = _old_home(tmp_path)
    store = _open_unchecked(home)
    try:
        with pytest.raises(RuntimeError, match="expected 1024"):
            reembed(store, AxisEmbedder(OLD_DIMENSIONS), batch_rows=BATCH, log=_quiet)
        assert read_embedding_meta(store._conn) == (OLD_DIMENSIONS, None)
    finally:
        store.close()


def test_cli_reembed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    home = _old_home(tmp_path)
    monkeypatch.setattr(cli, "EmbeddingClient", _Factory(AxisEmbedder()))
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(home), "--owner", NAMESPACE, "reembed"])
    assert cli.main() == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["complete"] is True and output["after"]["dimensions"] == EMBEDDING_DIMENSIONS
    assert "reembed items" in captured.err and "/" in captured.err  # N/M progress with elapsed time
    MemoryStore(_database(home), dimensions=EMBEDDING_DIMENSIONS).close()


# A new vector is not a user-visible change. Every column but the vector (updated_at, accessed_at,
# access_count, ...) stays as it was, so incremental backups, topic-assign, folder brightness and 'Recent only'
# do not see every row as just changed.
NON_VECTOR_TABLES = VECTOR_TABLES + ("conversations", "conversation_messages")


def _non_vector_columns(path: Path) -> dict[str, list[tuple]]:
    with closing(sqlite3.connect(path)) as connection:
        result = {}
        for table in NON_VECTOR_TABLES:
            columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})") if row[1] != "embedding"]
            result[table] = connection.execute(f"SELECT {','.join(columns)} FROM {table} ORDER BY id").fetchall()
        return result


def test_reembed_keeps_row_timestamps(tmp_path: Path) -> None:
    home = _old_home(tmp_path)
    with closing(sqlite3.connect(_database(home))) as connection, connection:
        # Distinct past values (the triggers would put "now" there): a read hit and an old update time.
        connection.execute("UPDATE items SET accessed_at=1000.5+id, access_count=3, updated_at=2000.25+id")
        connection.execute("UPDATE conversation_chunks SET updated_at=3000.125+id")
        connection.execute("UPDATE conversation_chunks SET updated_at=0 WHERE id=(SELECT MIN(id) FROM conversation_chunks)")
    before = _non_vector_columns(_database(home))
    assert before["items"] and before["conversation_chunks"]
    store = _open_unchecked(home)
    try:
        assert reembed(store, AxisEmbedder(), batch_rows=BATCH, log=_quiet)["complete"] is True
    finally:
        store.close()
    assert _non_vector_columns(_database(home)) == before
    assert {length for lengths in _vector_rows(_database(home)).values() for length in lengths} == {
        EMBEDDING_DIMENSIONS * FLOAT_BYTES}


# Tables computed from the old vectors are emptied with the meta update, so daily recomputes them from
# the new vectors; item_links (topic links and hand edits included) stay.
KEPT_TABLES = ("item_links",)


def _seed_derived(path: Path) -> None:
    """Rows in every derived table and in the kept tables, as an old daily run would have left them."""
    with closing(sqlite3.connect(path)) as connection, connection:
        low, high = connection.execute("SELECT MIN(id), MAX(id) FROM items WHERE type='memory'").fetchone()
        connection.execute("INSERT INTO item_links(src_id, relation, label, created_at) VALUES (?, 'topic', "
                           "'Lighthouse', 2.0)", (low,))
        connection.execute("INSERT INTO item_links(src_id, dst_id, dst_title, relation) VALUES (?, ?, ?, 'refers')",
                           (high, low, f"mem:{low}"))
        connection.execute("INSERT INTO memory_pairs VALUES (?, ?, 0.9, 3.0)", (low, high))
        connection.executemany("INSERT INTO memory_checks(memory_id,stage,text_hash,checked_at) VALUES (?, 'topic', 'h', 3.0)", [(low,), (high,)])
        connection.execute("INSERT INTO memory_checks(memory_id,stage,text_hash,chunk_id) VALUES (?, 'absorb', 'h', 7)", (low,))


def _counts(path: Path, tables: tuple[str, ...]) -> dict[str, int]:
    with closing(sqlite3.connect(path)) as connection:
        return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables}


def _rows(path: Path, tables: tuple[str, ...]) -> dict[str, list[tuple]]:
    with closing(sqlite3.connect(path)) as connection:
        return {table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall() for table in tables}


def test_complete_reembed_empties_derived_tables(tmp_path: Path) -> None:
    home = _old_home(tmp_path)
    _seed_derived(_database(home))
    kept = _rows(_database(home), KEPT_TABLES)
    assert all(kept.values())
    store = _open_unchecked(home)
    try:
        assert reembed(store, AxisEmbedder(), batch_rows=BATCH, budget_seconds=1e-9, log=_quiet)["complete"] is False
        assert all(_counts(_database(home), DERIVED_TABLES).values())  # a stopped run keeps them
        assert reembed(store, AxisEmbedder(), batch_rows=BATCH, log=_quiet)["complete"] is True
    finally:
        store.close()
    assert _counts(_database(home), DERIVED_TABLES) == dict.fromkeys(DERIVED_TABLES, 0)
    assert _rows(_database(home), KEPT_TABLES) == kept


def test_backup_after_reembed_restores_the_empty_derived_tables(tmp_path: Path) -> None:
    """The backup after reembed is full; a later increment that empties a derived table (replace
    table) restores it empty as well."""
    home = _old_home(tmp_path)
    _seed_derived(_database(home))
    directory = tmp_path / "backups"
    assert db_backup.run_backup(_database(home), directory)["kind"] == "full"
    store = _open_unchecked(home)
    try:
        assert reembed(store, AxisEmbedder(), batch_rows=BATCH, log=_quiet)["complete"] is True
    finally:
        store.close()
    result = db_backup.run_backup(_database(home), directory)
    assert result["kind"] == "full" and result["reason"] == "embedding_changed"
    _assert_restores(directory, _database(home), tmp_path / "restored1" / "memory.db")
    _seed_derived_again(_database(home))
    assert db_backup.run_backup(_database(home), directory)["kind"] == "incremental"
    with closing(sqlite3.connect(_database(home))) as connection, connection:
        for table in DERIVED_TABLES:
            connection.execute(f"DELETE FROM {table}")
    increment = db_backup.run_backup(_database(home), directory)
    assert increment["kind"] == "incremental"
    assert set(DERIVED_TABLES) <= set(db_backup.read_manifest(Path(increment["path"]))["replace_tables"])
    _assert_restores(directory, _database(home), tmp_path / "restored2" / "memory.db")


def _seed_derived_again(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        low, high = connection.execute("SELECT MIN(id), MAX(id) FROM items WHERE type='memory'").fetchone()
        connection.execute("INSERT INTO memory_pairs VALUES (?, ?, 0.8, 4.0)", (low, high))
        connection.execute("INSERT INTO memory_checks(memory_id,stage,text_hash,checked_at) VALUES (?, 'topic', 'n', 4.0)", (low,))
        connection.execute("INSERT INTO memory_checks(memory_id,stage,text_hash,chunk_id) VALUES (?, 'absorb', 'n', 9)", (low,))


def _assert_restores(directory: Path, live: Path, target: Path) -> None:
    assert db_backup.restore(directory, target)["integrity"] == "ok"
    assert _rows(target, DERIVED_TABLES + KEPT_TABLES) == _rows(live, DERIVED_TABLES + KEPT_TABLES)


def _quiet(_line: str) -> None:
    pass
