"""Shared provider / database-file helpers .

ThreeTopicEmbedder and KeywordEmbedder map a word to a fixed vector (same rule).
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from neospike import NeospikeProvider
from neospike.store import MemoryStore

OWNER_NS = "default:owner"
UNIT_X = [1.0, 0.0, 0.0]
SUBPROCESS_TIMEOUT_S = 30

_TOPIC_WORDS = (
    ("guitar",),
    ("python",),
    ("coffee",),
)


class ThreeTopicEmbedder:
    """Deterministic 3-d embedder: axis n is 1.0 when a guitar / python / coffee word appears."""

    dimensions = len(_TOPIC_WORDS)

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_document(self, text: str, *, title: str = "") -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.lower()
        return [float(any(word in lowered for word in words)) for words in _TOPIC_WORDS]


def make_provider(
    home: Path,
    session: str = "s-one",
    *,
    cwd: Path | None = None,
    context: str = "primary",
) -> NeospikeProvider:
    """An initialized provider (owner=owner, identity=default); cwd is passed only when given."""
    provider = NeospikeProvider(embedder=ThreeTopicEmbedder())
    options = {"cwd": str(cwd)} if cwd is not None else {}
    provider.initialize(
        session,
        hermes_home=str(home),
        agent_identity="default",
        user_id="owner",
        agent_context=context,
        **options,
    )
    return provider


def boot_provider(home: Path, session_id: str, cwd: Path | None = None) -> NeospikeProvider:
    """make_provider with cwd defaulting to the home itself."""
    return make_provider(home, session_id, cwd=cwd if cwd is not None else home)


def save_item(provider: NeospikeProvider, text: str, scope: str, subject: str, durability: str = "stable", **extra):
    """The parsed JSON reply of neospike_memory_save."""
    payload = {"text": text, "scope": scope, "subject": subject, "durability": durability, "kind": "fact", "basis": "user_said", **extra}
    return json.loads(provider.handle_tool_call("neospike_memory_save", payload))


def write_sessions_db(home: Path, rows: list[tuple[str, str | None]]) -> None:
    """Create <home>/state.db with a sessions table; each row is (id, cwd). git root mirrors cwd."""
    connection = sqlite3.connect(home / "state.db")
    try:
        connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, cwd TEXT, git_repo_root TEXT)")
        connection.executemany(
            "INSERT INTO sessions(id, cwd, git_repo_root) VALUES (?, ?, ?)",
            [(sid, cwd, cwd) for sid, cwd in rows],
        )
        connection.commit()
    finally:
        connection.close()


def make_legacy_db(directory: Path, texts: tuple[str, ...] = ()) -> Path:
    """Create the pre-rename <directory>/memory.sqlite holding one record per text; return its path."""
    path = directory / "memory.sqlite"
    store = MemoryStore(path, dimensions=3)
    try:
        for text in texts:
            store.add(OWNER_NS, text, list(UNIT_X), source="legacy:test")
    finally:
        store.close()
    return path


def run_python(code: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_S,
    )


_SEED_CODE = (
    "import os, sys\n"
    "from pathlib import Path\n"
    "from neospike.store import MemoryStore\n"
    "s = MemoryStore(Path(sys.argv[1]), dimensions=3)\n"
    "s.add(sys.argv[2], sys.argv[3], [1.0, 0.0, 0.0], source='seed:test', kind='memory_item',\n"
    "      metadata={'scope': 'global'})\n"
    "os._exit(0)\n"
)


def seed_wal_db_and_exit(path: Path, text: str) -> None:
    """Write one record in a child process that dies without checkpointing, leaving -wal behind."""
    run_python(_SEED_CODE, str(path), OWNER_NS, text)


def texts_found(path: Path, query: str) -> list[str]:
    store = MemoryStore(path, dimensions=3)
    try:
        return [row.text for row in store.search(OWNER_NS, query, list(UNIT_X), limit=10)]
    finally:
        store.close()


def integrity_of(path: Path) -> str:
    connection = sqlite3.connect(path)
    try:
        return connection.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        connection.close()
