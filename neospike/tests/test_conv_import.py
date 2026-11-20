"""conv-import: past Hermes sessions from a (synthetic) state.db, read-only."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

from neospike import cli
from neospike import conversations as conv
from neospike.store import MemoryStore
from neospike.tests.test_notes import DIMENSIONS, BowEmbedder

NAMESPACE = "owner"
FAKE_KEY = "sk-FAKEKEY1234567890abcdefXYZ"
BASE_TIME = 1_760_000_000.0

# Subset of the real Hermes state.db schema (column names read from %LOCALAPPDATA%\hermes\state.db).
STATE_SCHEMA = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY, origin TEXT NOT NULL, started_at REAL NOT NULL, ended_at REAL, title TEXT
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL, content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
    timestamp REAL NOT NULL, _compressed_summary INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1, compacted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX idx_messages_session ON messages(session_id, timestamp);
"""

# (session, role, content, offset seconds, active, compacted, compressed_summary, tool_name)
STATE_MESSAGES = (
    ("noisy", "system", "You are a helpful agent.", 0, 1, 0, 0, None),
    ("noisy", "user", f"My deploy key is {FAKE_KEY} keep it safe.", 1, 1, 0, 0, None),
    ("noisy", "assistant", "", 2, 1, 0, 0, None),
    ("noisy", "tool", "exit code 0\nfile list", 3, 1, 0, 0, "terminal"),
    ("noisy", "user", "[CONTEXT COMPACTION] earlier turns summarized", 4, 1, 0, 0, None),
    ("noisy", "assistant", "[IMPORTANT: Background process finished]", 5, 1, 0, 0, None),
    ("noisy", "user", "Rewound question that was undone.", 6, 0, 0, 0, None),
    ("noisy", "assistant", "Synthetic compaction summary text.", 7, 1, 0, 1, None),
    ("noisy", "assistant", "Stored the deploy notes without repeating the key.", 8, 1, 0, 0, None),
    ("lighthouse", "user", "How tall is the old lighthouse on the cape?", 100, 1, 1, 0, None),
    ("lighthouse", "assistant", "The old lighthouse is about forty meters tall.", 101, 1, 1, 0, None),
    ("orphan", "user", "Which ferry leaves the northern harbor first?", 200, 1, 0, 0, None),
    ("orphan", "assistant", "The seven o'clock ferry leaves first.", 201, 1, 0, 0, None),
)
STATE_SESSIONS = (("noisy", "cli", -10, None), ("lighthouse", "cli", 90, "Lighthouse height"))  # "orphan" has no row
EXPECTED_ROWS = 6


def _state_db(path: Path) -> Path:
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(STATE_SCHEMA)
        connection.executemany(
            "INSERT INTO sessions(id,origin,started_at,title) VALUES (?,?,?,?)",
            [(sid, source, BASE_TIME + offset, title) for sid, source, offset, title in STATE_SESSIONS],
        )
        connection.executemany(
            "INSERT INTO messages(session_id,role,content,timestamp,active,compacted,_compressed_summary,tool_name) "
            "VALUES (?,?,?,?,?,?,?,?)",
            [(sid, role, content, BASE_TIME + offset, active, compacted, summary, tool)
             for sid, role, content, offset, active, compacted, summary, tool in STATE_MESSAGES],
        )
        connection.commit()
    return path


def _digest(paths) -> dict[str, str]:
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _db_files(home: Path) -> list[Path]:
    directory = home / "neospike"
    return sorted(path for path in directory.iterdir()
                  if path.name.startswith("memory.db") and not path.name.endswith("-shm"))


def _db_state(home: Path) -> tuple[str, bytes]:
    """(main file hash, WAL bytes). A mode=ro reader may leave an empty -wal behind; a missing and
    an empty WAL are the same state."""
    main = home / "neospike" / "memory.db"
    wal = main.with_name(main.name + "-wal")
    return hashlib.sha256(main.read_bytes()).hexdigest(), wal.read_bytes() if wal.exists() else b""


@pytest.fixture
def home(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "home"
    (root / "neospike").mkdir(parents=True)
    (root / "neospike" / "config.json").write_text(json.dumps({"embedding_dimensions": DIMENSIONS}), encoding="utf-8")
    monkeypatch.setattr(cli, "EmbeddingClient", BowEmbedder)
    MemoryStore(root / "neospike" / "memory.db", dimensions=DIMENSIONS).close()
    return root


@pytest.fixture
def state_db(tmp_path: Path) -> Path:
    source = tmp_path / "hermes"
    source.mkdir()
    return _state_db(source / "state.db")


def _run(home: Path, monkeypatch, capsys, *arguments: str, code: int = 0) -> dict:
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(home), "--owner", NAMESPACE, *arguments])
    assert cli.main() == code
    return json.loads(capsys.readouterr().out)


def _rows(home: Path) -> list[tuple]:
    with closing(sqlite3.connect(home / "neospike" / "memory.db")) as connection:
        return [tuple(row) for row in connection.execute(
            "SELECT c.session_id,m.seq,m.role,m.content,m.masked_count,m.created_at FROM conversation_messages m "
            "JOIN conversations c ON c.id=m.conversation_id ORDER BY c.id,m.seq")]


# 1 + 4: only user/assistant text, raw key nowhere in our DB, source unchanged -------------------
def test_import_keeps_only_user_assistant_masks_key_and_leaves_source_untouched(home, state_db, monkeypatch, capsys) -> None:
    source_before = _digest(state_db.parent.iterdir())
    result = _run(home, monkeypatch, capsys, "conv-import", "--state-db", str(state_db))
    assert source_before == _digest(path for path in state_db.parent.iterdir() if path.name in source_before)
    assert {key: result[key] for key in ("sessions_seen", "imported", "skipped_existing", "messages", "masked")} == {
        "sessions_seen": 3, "imported": 3, "skipped_existing": 0, "messages": EXPECTED_ROWS, "masked": 1}
    rows = _rows(home)
    assert [(row[0], row[1], row[2]) for row in rows] == [
        ("noisy", 1, "user"), ("noisy", 2, "assistant"), ("lighthouse", 1, "user"), ("lighthouse", 2, "assistant"),
        ("orphan", 1, "user"), ("orphan", 2, "assistant")]
    assert rows[0][3] == f"My deploy key is {conv.REDACTED} keep it safe." and rows[0][4] == 1
    assert rows[1][3] == "Stored the deploy notes without repeating the key."
    assert rows[1][5] == BASE_TIME + 8  # message time is kept, not the import time
    for path in _db_files(home):
        assert FAKE_KEY.encode() not in path.read_bytes(), path.name


# 2: re-run is a no-op ----------------------------------------------------------------------------
def test_second_run_skips_all_existing_sessions(home, state_db, monkeypatch, capsys) -> None:
    _run(home, monkeypatch, capsys, "conv-import", "--state-db", str(state_db))
    before = _rows(home)
    again = _run(home, monkeypatch, capsys, "conv-import", "--state-db", str(state_db))
    assert again["imported"] == 0 and again["skipped_existing"] == 3 and again["messages"] == 0
    assert _rows(home) == before


# 3: dry run does not change the database ---------------------------------------------------------
def test_dry_run_counts_without_changing_the_db_file(home, state_db, monkeypatch, capsys) -> None:
    _run(home, monkeypatch, capsys, "conv-import", "--state-db", str(state_db), "--limit", "1")
    rows = _rows(home)
    before = _db_state(home)
    result = _run(home, monkeypatch, capsys, "conv-import", "--state-db", str(state_db), "--dry-run", "--index")
    assert result["dry_run"] is True and result["imported"] == 2 and result["skipped_existing"] == 1
    assert result["messages"] == 4 and "index" not in result
    assert _db_state(home) == before
    assert _rows(home) == rows


def test_dry_run_without_a_memory_db_creates_nothing(tmp_path, state_db, monkeypatch, capsys) -> None:
    empty = tmp_path / "empty-home"
    result = _run(empty, monkeypatch, capsys, "conv-import", "--state-db", str(state_db), "--dry-run")
    assert result["imported"] == 3 and result["messages"] == EXPECTED_ROWS
    assert not (empty / "neospike").exists()


# 5: imported conversations are ended ----------------------------------------------------------------
def test_imported_sessions_are_ended(home, state_db, monkeypatch, capsys) -> None:
    _run(home, monkeypatch, capsys, "conv-import", "--state-db", str(state_db))
    with closing(sqlite3.connect(home / "neospike" / "memory.db")) as connection:
        sessions = {entry["session_id"]: entry for entry in conv.list_conversations(connection)}
    assert set(sessions) == {"noisy", "lighthouse", "orphan"}
    assert all(entry["ended_at"] is not None and entry["summary_note_id"] is None for entry in sessions.values())
    assert sessions["noisy"]["ended_at"] == BASE_TIME + 8 and sessions["noisy"]["started_at"] == BASE_TIME - 10
    assert sessions["lighthouse"]["title"] == "Lighthouse height"
    assert sessions["orphan"]["title"] == "Which ferry leaves the northern harbor first?"
    assert sessions["orphan"]["started_at"] == BASE_TIME + 200


# 6: conv-index makes them searchable --------------------------------------------------------------
def test_conv_index_after_import_makes_sessions_searchable(home, state_db, monkeypatch, capsys) -> None:
    imported = _run(home, monkeypatch, capsys, "conv-import", "--state-db", str(state_db))
    assert "index" not in imported  # --index is off by default
    assert _run(home, monkeypatch, capsys, "conv-search", "how tall is the old lighthouse about forty meters")["results"] == []
    indexed = _run(home, monkeypatch, capsys, "conv-index")
    assert indexed["conversations"] == 3 and indexed["pending"] == 0
    found = _run(home, monkeypatch, capsys, "conv-search", "how tall is the old lighthouse about forty meters")["results"]
    assert found and found[0]["session_id"] == "lighthouse" and "forty meters" in found[0]["text"]


def test_index_option_and_limit(home, state_db, monkeypatch, capsys) -> None:
    first = _run(home, monkeypatch, capsys, "conv-import", "--state-db", str(state_db), "--limit", "1", "--index")
    assert first["sessions_seen"] == 1 and first["imported"] == 1 and first["index"]["conversations"] == 1
    rest = _run(home, monkeypatch, capsys, "conv-import", "--state-db", str(state_db))
    assert rest["skipped_existing"] == 1 and rest["imported"] == 2


def test_missing_state_db_is_an_error(home, tmp_path, monkeypatch, capsys) -> None:
    result = _run(home, monkeypatch, capsys, "conv-import", "--state-db", str(tmp_path / "nope.db"), code=1)
    assert result["error"] == "not_found"
    assert not (tmp_path / "nope.db").exists()
