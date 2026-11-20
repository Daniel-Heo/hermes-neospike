"""conv-import increments: new and changed sessions up to the batch start T, rows only once."""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest

from neospike import cli
from neospike import conversations as conv
from neospike.store import MemoryStore
from neospike.tests.test_conv_import import STATE_SCHEMA
from neospike.tests.test_notes import DIMENSIONS, BowEmbedder

NAMESPACE = "owner"
# Source times lie before the live rows' created_at (time.time() at append), as in Hermes.
BASE = time.time() - 100_000.0
T1 = BASE + 1_000.0
T2 = BASE + 2_000.0
T3 = BASE + 3_000.0
FAILING_SESSION = "gamma"


class _Source:
    """A writable synthetic Hermes state.db (the test plays Hermes)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with closing(sqlite3.connect(path)) as connection:
            connection.executescript(STATE_SCHEMA)

    def add(self, session: str, role: str, content: str, offset: float) -> None:
        with closing(sqlite3.connect(self.path)) as connection:
            if connection.execute("SELECT 1 FROM sessions WHERE id=?", (session,)).fetchone() is None:
                connection.execute("INSERT INTO sessions(id,origin,started_at,title) VALUES (?,?,?,NULL)",
                                   (session, "cli", BASE + offset))
            connection.execute("INSERT INTO messages(session_id,role,content,timestamp) VALUES (?,?,?,?)",
                               (session, role, content, BASE + offset))
            connection.commit()


@pytest.fixture
def store(tmp_path: Path):
    opened = MemoryStore(tmp_path / "home" / "neospike" / "memory.db", dimensions=DIMENSIONS)
    yield opened
    opened.close()


@pytest.fixture
def source(tmp_path: Path) -> _Source:
    (tmp_path / "hermes").mkdir()
    return _Source(tmp_path / "hermes" / "state.db")


def _sync(store: MemoryStore, source: _Source, until: float, **options) -> dict:
    return conv.import_state_db(source.path, store=store, until=until, **options)


def _messages(store: MemoryStore, session: str) -> list[tuple[int, str, str]]:
    with store._lock:
        return [(int(row[0]), str(row[1]), str(row[2])) for row in store._conn.execute(
            "SELECT m.seq,m.role,m.content FROM conversation_messages m JOIN conversations c ON c.id=m.conversation_id "
            "WHERE c.session_id=? ORDER BY m.seq", (session,))]


def _chunk_ids(store: MemoryStore, session: str) -> list[int]:
    with store._lock:
        return [int(row[0]) for row in store._conn.execute(
            "SELECT ch.id FROM conversation_chunks ch JOIN conversations c ON c.id=ch.conversation_id "
            "WHERE c.session_id=? ORDER BY ch.id", (session,))]


def _turn(source: _Source, session: str, number: int, offset: float) -> None:
    source.add(session, "user", f"{session} question {number} about the harbor ferry", offset)
    source.add(session, "assistant", f"{session} answer {number} about the harbor ferry", offset + 1)


def _live(store: MemoryStore, session: str, number: int) -> None:
    """sync_turn's write (ConversationStore.append) of the same turn."""
    conv.ConversationStore(store).append(session, (("user", f"{session} question {number} about the harbor ferry"),
                                                   ("assistant", f"{session} answer {number} about the harbor ferry")))


def test_new_session_after_the_last_run_is_imported(store, source) -> None:
    _turn(source, "alpha", 1, 10)
    first = _sync(store, source, T1)
    assert first["imported"] == 1 and first["processed"] == 1 and first["until"] == T1
    _turn(source, "beta", 1, 1_500)
    second = _sync(store, source, T2)
    assert second["imported"] == 1 and second["processed"] == 1 and second["skipped_existing"] == 1
    assert [row[2] for row in _messages(store, "beta")] == ["beta question 1 about the harbor ferry",
                                                            "beta answer 1 about the harbor ferry"]


def test_appended_rows_go_in_once_and_live_rows_are_not_doubled(store, source) -> None:
    _turn(source, "alpha", 1, 10)
    _sync(store, source, T1)
    _turn(source, "alpha", 2, 1_100)
    _live(store, "alpha", 2)  # sync_turn stored turn 2 already
    _turn(source, "alpha", 3, 1_200)  # turn 3 was never stored live
    result = _sync(store, source, T2)
    assert result["updated"] == 1 and result["messages"] == 2 and result["reordered"] == 0 and result["ours_only"] == 0
    contents = [row[2] for row in _messages(store, "alpha")]
    assert len(contents) == len(set(contents)) == 6
    assert [row[0] for row in _messages(store, "alpha")] == [1, 2, 3, 4, 5, 6]
    assert contents[-2:] == ["alpha question 3 about the harbor ferry", "alpha answer 3 about the harbor ferry"]


def test_messages_after_the_boundary_wait_for_the_next_run(store, source) -> None:
    _turn(source, "alpha", 1, 10)
    source.add("alpha", "user", "alpha question 2 typed after the batch started", T1 - BASE + 5)
    first = _sync(store, source, T1)
    assert first["messages"] == 2 and len(_messages(store, "alpha")) == 2
    second = _sync(store, source, T2)
    assert second["processed"] == 1 and second["messages"] == 1
    assert _messages(store, "alpha")[-1][2] == "alpha question 2 typed after the batch started"


def test_a_missing_middle_row_is_put_in_order_and_only_that_conversation_is_reindexed(store, source) -> None:
    _turn(source, "alpha", 1, 10)
    _turn(source, "beta", 1, 20)
    _sync(store, source, T1)
    source.add("alpha", "user", "alpha interrupted question that never finished", 1_100)  # no sync_turn
    _turn(source, "alpha", 2, 1_200)
    _live(store, "alpha", 2)
    indexer = conv.ConversationIndexer(store, BowEmbedder())
    indexer.index_pending()
    beta_chunks = _chunk_ids(store, "beta")
    assert _chunk_ids(store, "alpha") and beta_chunks
    result = _sync(store, source, T2)
    assert result["messages"] == 1 and result["reordered"] == 1
    assert [(row[0], row[2]) for row in _messages(store, "alpha")] == [
        (1, "alpha question 1 about the harbor ferry"), (2, "alpha answer 1 about the harbor ferry"),
        (3, "alpha interrupted question that never finished"),
        (4, "alpha question 2 about the harbor ferry"), (5, "alpha answer 2 about the harbor ferry")]
    with store._lock:  # marked stale (negated seqs) for a full re-chunk; beta untouched
        assert store._conn.execute(
            "SELECT COUNT(*) FROM conversation_chunks ch JOIN conversations c ON c.id=ch.conversation_id "
            "WHERE c.session_id='alpha' AND ch.seq_start>0").fetchone()[0] == 0
    again = indexer.index_pending()
    assert again["conversations"] == 1 and again["pending"] == 0
    assert _chunk_ids(store, "beta") == beta_chunks
    with store._lock:
        seq_end = store._conn.execute(
            "SELECT MAX(ch.seq_end) FROM conversation_chunks ch JOIN conversations c ON c.id=ch.conversation_id "
            "WHERE c.session_id='alpha'").fetchone()[0]
    assert seq_end == 5


def test_rows_only_we_hold_are_kept_and_reported(store, source) -> None:
    _turn(source, "alpha", 1, 10)
    _sync(store, source, T1)
    with store._lock, store._conn:
        conversation_id = store._conn.execute("SELECT id FROM conversations WHERE session_id='alpha'").fetchone()[0]
        store._conn.execute(
            "INSERT INTO conversation_messages(conversation_id,seq,role,content,created_at) VALUES (?,?,?,?,?)",
            (conversation_id, 3, "user", "a row the source does not have", BASE + 20))
    source.add("alpha", "assistant", "alpha late answer", 1_100)
    result = _sync(store, source, T2)
    assert result["ours_only"] == 1 and result["messages"] == 1
    assert [row[2] for row in _messages(store, "alpha")][2:] == ["a row the source does not have", "alpha late answer"]


class _FailingStore(conv.ConversationStore):
    def sync_session(self, session_id, *args, **kwargs):
        if session_id == FAILING_SESSION:
            raise sqlite3.OperationalError("database is locked")
        return super().sync_session(session_id, *args, **kwargs)


def test_a_run_stopped_by_an_error_leaves_the_rest_for_the_next_run(store, source, monkeypatch) -> None:
    for number, session in enumerate(("alpha", "beta", FAILING_SESSION, "delta")):
        _turn(source, session, 1, 10 + number * 10)
    monkeypatch.setattr(conv, "ConversationStore", _FailingStore)
    stopped = _sync(store, source, T1)
    assert stopped["stopped"] == "error" and stopped["error"] == "sqlite_error"
    assert stopped["processed"] == 2 and stopped["remaining"] == 2
    monkeypatch.undo()
    rest = _sync(store, source, T1)
    assert rest["candidates"] == 2 and rest["processed"] == 2 and rest["imported"] == 2 and rest["remaining"] == 0


class _BudgetAfter:
    """Deadline whose budget runs out after `calls` checks."""

    calls = 2

    def __init__(self, seconds) -> None:
        self.left = self.calls

    def passed(self) -> bool:
        self.left -= 1
        return self.left < 0


def test_a_run_stopped_by_the_budget_continues_next_time(store, source, monkeypatch) -> None:
    for number, session in enumerate(("alpha", "beta", "gamma", "delta")):
        _turn(source, session, 1, 10 + number * 10)
    monkeypatch.setattr(conv, "Deadline", _BudgetAfter)
    stopped = _sync(store, source, T1, budget_seconds=1)
    assert stopped["stopped"] == "budget" and stopped["processed"] == 2 and stopped["remaining"] == 2
    monkeypatch.undo()
    rest = _sync(store, source, T2)
    assert rest["processed"] == 2 and rest["skipped_existing"] == 2


def test_an_unchanged_second_run_processes_nothing(store, source) -> None:
    _turn(source, "alpha", 1, 10)
    _turn(source, "beta", 1, 20)
    _live(store, "alpha", 2)  # a live-only conversation part is matched once, then left alone
    first = _sync(store, source, T1)
    assert first["processed"] == 2
    second = _sync(store, source, T2)
    assert second["processed"] == 0 and second["remaining"] == 0 and second["skipped_existing"] == 2


class _TimingOutEmbedder(BowEmbedder):
    """Times out on the embed call after `allowed` successful ones."""

    def __init__(self, allowed: int) -> None:
        super().__init__()
        self.allowed = allowed

    def embed_document(self, text, **kwargs):
        if self.allowed == 0:
            raise TimeoutError("timed out")
        self.allowed -= 1
        return super().embed_document(text, **kwargs)


def _pending(store: MemoryStore) -> int:
    return conv.ConversationIndexer(store, BowEmbedder()).pending_embeddings()


def test_an_embedding_timeout_keeps_the_vectors_done_so_far(store, source) -> None:
    for number in range(12):
        _turn(source, f"s{number:02d}", 1, 10 + number * 10)
    _sync(store, source, T1)
    indexer = conv.ConversationIndexer(store, _TimingOutEmbedder(allowed=5))
    with pytest.raises(TimeoutError):
        indexer.index_pending()
    total = 12
    assert _pending(store) == total - 5
    done = conv.ConversationIndexer(store, BowEmbedder()).index_pending()
    assert done["embedded"] == total - 5 and done["pending"] == 0


def test_the_index_budget_stops_and_the_next_run_finishes(store, source, monkeypatch) -> None:
    for number in range(4):
        _turn(source, f"s{number}", 1, 10 + number * 10)
    _sync(store, source, T1)
    monkeypatch.setattr(conv, "Deadline", _BudgetAfter)
    stopped = conv.ConversationIndexer(store, BowEmbedder()).index_pending(budget_seconds=1)
    assert stopped["stopped"] == "budget" and stopped["processed"] == 2 and stopped["remaining_conversations"] == 2
    assert stopped["pending"] == 2 and stopped["remaining"] == 4  # 2 conversations to chunk + 2 chunks to embed
    monkeypatch.undo()
    rest = conv.ConversationIndexer(store, BowEmbedder()).index_pending()
    assert rest["processed"] == 2 and rest["pending"] == 0 and rest["remaining"] == 0


# CLI: result JSON (processed / remaining / until) and the progress log -----------------------------------
def _cli(home: Path, monkeypatch, capsys, *arguments: str) -> dict:
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(home), "--owner", NAMESPACE, *arguments])
    assert cli.main() == 0
    return json.loads(capsys.readouterr().out)


def test_cli_result_json_and_progress_log(tmp_path, source, monkeypatch, capsys) -> None:
    home = tmp_path / "cli-home"
    (home / "neospike").mkdir(parents=True)
    (home / "neospike" / "config.json").write_text(json.dumps({"embedding_dimensions": DIMENSIONS}), encoding="utf-8")
    monkeypatch.setattr(cli, "EmbeddingClient", BowEmbedder)
    for number in range(12):
        _turn(source, f"s{number:02d}", 1, 10 + number * 10)
    imported = _cli(home, monkeypatch, capsys, "conv-import", "--state-db", str(source.path), "--until", repr(T3))
    assert imported["processed"] == 12 and imported["remaining"] == 0 and imported["until"] == T3
    indexed = _cli(home, monkeypatch, capsys, "conv-index")
    assert indexed["processed"] == 12 and indexed["remaining"] == 0
    again = _cli(home, monkeypatch, capsys, "conv-import", "--state-db", str(source.path), "--until", repr(T3))
    assert again["processed"] == 0
    lines = [json.loads(line) for path in sorted((home / "neospike" / "logs").glob("conv-*.jsonl"))
             for line in path.read_text(encoding="utf-8").splitlines()]
    progress = [line["progress"] for line in lines if line["stage"] == "conv-import" and "progress" in line]
    assert progress == ["10/12", "12/12"]
    assert [line["until"] for line in lines if "progress" in line and line["stage"] == "conv-import"] == [T3, T3]
    results = [line["result"] for line in lines if "result" in line]
    assert [result.get("processed") for result in results] == [12, 12, 0]
    assert [line["progress"] for line in lines if line["stage"] == "conv-index" and "progress" in line] == ["10/12", "12/12"]


def test_daily_passes_one_batch_start_to_every_home() -> None:
    import importlib.util
    path = Path(conv.__file__).resolve().parent / "cron" / "neospike_daily.py"
    spec = importlib.util.spec_from_file_location("daily_t416_under_test", path)
    daily = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(daily)
    args = dict(daily.stages(Path("home"), T1))["conv-import"]
    assert args[args.index("--until") + 1] == repr(T1)


def test_a_live_row_holding_a_turn_correction_matches_both_source_messages(store, source) -> None:
    source.add("alpha", "user", "continue", 10)
    source.add("alpha", "user", "why ask? just put it in at install", 11)  # typed while the turn ran
    source.add("alpha", "assistant", "Right, install will do it.", 12)
    conv.ConversationStore(store).append("alpha", (
        ("user", "continue" + conv.TURN_CORRECTION_SEPARATOR + "why ask? just put it in at install"),
        ("assistant", "Right, install will do it.")))
    result = _sync(store, source, T1)
    assert result["messages"] == 0 and result["ours_only"] == 0 and result["unchanged"] == 1
    assert len(_messages(store, "alpha")) == 2


DISPLAY_SCHEMA = """
CREATE TABLE sessions (id TEXT PRIMARY KEY, origin TEXT NOT NULL, started_at REAL NOT NULL, ended_at REAL, title TEXT);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
    timestamp REAL NOT NULL, _compressed_summary INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0, display_identity TEXT, display_order INTEGER
);
"""


def test_source_order_follows_display_order_and_one_row_per_identity(store, tmp_path) -> None:
    """Compaction re-stamps kept messages and may keep a compacted original next to an active copy."""
    path = tmp_path / "display-state.db"
    rows = (  # (id, role, content, timestamp offset, active, compacted, identity, display_order)
        (1, "user", "first question", 10, 0, 1, "q1", 1),
        (2, "assistant", "first answer", 4_000, 1, 0, "a1", 2),  # re-stamped by compaction, still shown 2nd
        (3, "user", "second question", 20, 1, 0, "q2", 3),
        (4, "user", "first question", 4_000, 1, 0, "q1", 1),  # active copy of row 1
    )
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(DISPLAY_SCHEMA)
        connection.executemany(
            "INSERT INTO messages(id,session_id,role,content,timestamp,active,compacted,display_identity,display_order) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [(row[0], "alpha", row[1], row[2], BASE + row[3], row[4], row[5], row[6], row[7]) for row in rows])
        connection.commit()
    result = conv.import_state_db(path, store=store, until=BASE + 5_000)
    assert result["messages"] == 3
    assert [row[2] for row in _messages(store, "alpha")] == ["first question", "first answer", "second question"]


def test_rows_stored_in_another_order_are_moved_not_stored_twice(store, source) -> None:
    """Rows stored in timestamp order, which compaction rewrites: same rows, other order."""
    _turn(source, "alpha", 1, 10)
    _turn(source, "alpha", 2, 20)
    with store._lock, store._conn:
        conversation_id = store._conn.execute(
            "INSERT INTO conversations(session_id,started_at,last_message_at) VALUES ('alpha',?,?)", (BASE, BASE)).lastrowid
        for seq, (role, number) in enumerate((("user", 1), ("user", 2), ("assistant", 1), ("assistant", 2)), 1):
            word = "question" if role == "user" else "answer"
            store._conn.execute(
                "INSERT INTO conversation_messages(conversation_id,seq,role,content,created_at) VALUES (?,?,?,?,?)",
                (conversation_id, seq, role, f"alpha {word} {number} about the harbor ferry", BASE + seq))
    conv.ConversationIndexer(store, BowEmbedder()).index_pending()
    result = _sync(store, source, T1)
    assert result["messages"] == 0 and result["reordered"] == 1 and result["ours_only"] == 0
    assert [row[2].split(" about")[0] for row in _messages(store, "alpha")] == [
        "alpha question 1", "alpha answer 1", "alpha question 2", "alpha answer 2"]
    assert _sync(store, source, T2)["processed"] == 0


class _CountingEmbedder(BowEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def embed_document(self, text, **kwargs):
        self.calls += 1
        return super().embed_document(text, **kwargs)


def test_a_reordered_conversation_reuses_the_vectors_of_unchanged_chunks(store, source) -> None:
    for number in range(1, 16):
        _turn(source, "alpha", number, number * 10)
    _sync(store, source, T1)
    indexer = conv.ConversationIndexer(store, BowEmbedder(), max_tokens=80, context_tokens=0)
    indexer.index_pending()
    before = len(_chunk_ids(store, "alpha"))
    source.add("alpha", "user", "alpha interrupted question that never finished", 1_100)  # no sync_turn
    _turn(source, "alpha", 16, 1_200)
    _live(store, "alpha", 16)  # stored live after the interrupted one, so that one lands mid-conversation
    assert _sync(store, source, T2)["reordered"] == 1
    counting = _CountingEmbedder()
    done = conv.ConversationIndexer(store, counting, max_tokens=80, context_tokens=0).index_pending()
    assert done["pending"] == 0 and before > 2 and 0 < counting.calls < len(_chunk_ids(store, "alpha"))


def test_a_late_message_with_an_old_timestamp_is_picked_up_by_its_new_id(store, source) -> None:
    """Written after the run with a timestamp before the previous boundary: the larger messages.id finds it."""
    _turn(source, "alpha", 1, 10)
    _turn(source, "beta", 1, 20)
    assert _sync(store, source, T1)["processed"] == 2
    source.add("alpha", "user", "alpha question typed before T1 but written late", 900)  # 900 < T1 - BASE
    late = _sync(store, source, T2)
    assert late["processed"] == 1 and late["messages"] == 1 and late["skipped_existing"] == 1
    assert _messages(store, "alpha")[-1][2] == "alpha question typed before T1 but written late"
    assert _sync(store, source, T3)["processed"] == 0
    source.add("alpha", "user", "alpha message after the boundary", T3 - BASE + 50)
    assert _sync(store, source, T3)["processed"] == 0  # timestamp > T is still the next day's


def test_the_live_indexer_embeds_one_chunk_per_batch_and_the_cli_eight() -> None:
    live = conv.ConversationIndexer.__init__.__kwdefaults__
    assert conv.LIVE_EMBED_BATCH == 1 and conv.EMBED_BATCH == 8 and live["embed_batch"] == conv.LIVE_EMBED_BATCH
