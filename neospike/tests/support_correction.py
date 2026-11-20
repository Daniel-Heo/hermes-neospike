"""Shared helpers for the atomic-correction test suites.

Everything here drives the real provider + SQLite store; no daemon, no mocks of
the store itself.  Failure injection is done with SQLite triggers/authorizers.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager

from neospike.provider import NeospikeProvider

OLD_RULE = "Release gate needs the AMBER-LYNX sign-off stamp."
NEW_RULE = "Release gate now needs INDIGO-CRANE; AMBER-LYNX was revoked and cannot approve anything."

SUBJECT = "release gate"
TOOL_SAVE = "neospike_memory_save"
TOOL_UPDATE = "neospike_memory_update"
TOOL_DELETE = "neospike_memory_delete"

MSG_UPDATE_FAILED = "Atomic memory update failed; no change was saved"
MSG_POLICY_REJECT = "Memory item was rejected by ingestion policy"
MSG_BAD_ID = "Memory ID must be an integer from 1 to 9223372036854775807"
MSG_UNKNOWN_FIELDS = "Unknown atomic-update fields"
MSG_READONLY = "Neospike writes are disabled in this agent context"

CONFLICT = {"updated": False, "reason": "conflict"}
UNAVAILABLE = {"updated": False, "reason": "unavailable"}

NS = "default:local"

# Sessions seeded in the fake Hermes state.db: id -> project folder (None = no cwd).
S_MAIN = "alpha"
S_SIBLING = "alpha2"     # same project as S_MAIN
S_OTHER = "beta"         # different project
S_BARE = "bare"          # session without any project
S_GHOST = "ghost"        # not in state.db at all
S_EMPTY = ""


class KeywordEmbedder:
    """3-d embedder: texts mentioning INDIGO-CRANE land on axis 1, others on axis 0."""

    dimensions = 3

    def embed_document(self, text):
        return [0.0, 1.0, 0.0] if "INDIGO-CRANE" in text else [1.0, 0.0, 0.0]

    def embed_query(self, text):
        return self.embed_document(text)


class FixedVector:
    """Callable stand-in for embed_document that returns one canned vector."""

    def __init__(self, vector):
        self.vector = vector

    def __call__(self, text):
        return self.vector


class RecordingVector:
    """Callable stand-in that records every text it is asked to embed."""

    def __init__(self, vector=(0.0, 1.0, 0.0)):
        self.vector = list(vector)
        self.seen = []

    def __call__(self, text):
        self.seen.append(text)
        return self.vector


class Boom:
    """Callable that always raises (message pretends to hold private data)."""

    def __call__(self, *args, **kwargs):
        raise RuntimeError("injected failure carrying a private payload")


def deny_commit(action, arg1, *rest):
    """sqlite3 authorizer: refuse the COMMIT transaction op, allow all else."""
    if action == sqlite3.SQLITE_TRANSACTION and arg1 == "COMMIT":
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def seed_state_db(root):
    with closing(sqlite3.connect(root / "state.db")) as conn, conn:
        conn.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, cwd TEXT, git_repo_root TEXT)")
        entries = [
            (S_MAIN, str(root / "proj-one"), None),
            (S_SIBLING, str(root / "proj-one"), None),
            (S_OTHER, str(root / "proj-two"), None),
            (S_BARE, None, None),
            (S_EMPTY, None, None),
        ]
        conn.executemany("INSERT INTO sessions VALUES (?,?,?)", entries)


class Harness:
    """Thin wrapper over a live provider with the verbs the tests need."""

    def __init__(self, provider):
        self.p = provider

    # -- raw access ---------------------------------------------------
    @property
    def conn(self):
        return self.p._memory_db._conn

    @property
    def store(self):
        return self.p._memory_db

    def rows(self):
        cur = self.conn.execute("SELECT * FROM items WHERE type='memory' ORDER BY id")
        return [dict(r) for r in cur]

    def fts_ids(self, word):
        cur = self.conn.execute(
            "SELECT rowid FROM items_fts WHERE items_fts MATCH ? ORDER BY rowid", (word,))
        return [r[0] for r in cur]

    def call(self, tool, args):
        return json.loads(self.p.handle_tool_call(tool, args))

    # -- verbs --------------------------------------------------------
    def switch(self, session_id):
        self.p.on_session_switch(session_id)

    def save(self, scope="project", text=OLD_RULE, **extra):
        args = {"text": text, "scope": scope, "subject": SUBJECT, "durability": "ongoing",
                "importance": 0.8, "confidence": 0.9, "tags": ["gate"], "kind": "fact", "basis": "user_said"}
        args.update(extra)
        assert self.call(TOOL_SAVE, args) == {"stored": True}
        return self.rows()[-1]["id"]

    def correct_args(self, memory_id, **overrides):
        args = {"id": memory_id, "expected_text": OLD_RULE, "new_text": NEW_RULE,
                "subject": SUBJECT, "durability": "ongoing"}
        args.update(overrides)
        return args

    def correct(self, memory_id, **overrides):
        return self.call(TOOL_UPDATE, self.correct_args(memory_id, **overrides))

    def forget(self, memory_id):
        return self.call(TOOL_DELETE, {"id": memory_id})

    def mutate(self, tool, memory_id):
        """Run update or delete against one id; same call shape either way."""
        if tool == TOOL_UPDATE:
            return self.correct(memory_id)
        return self.forget(memory_id)

    def seed(self, text=OLD_RULE, *, vec=(1.0, 0.0, 0.0), source="agent:alpha", kind="memory_item", **kw):
        namespace = kw.pop("namespace", NS)
        assert self.store.add(namespace, text, list(vec), source=source, kind=kind, **kw)
        return self.rows()[-1]


@contextmanager
def open_harness(tmp_path, *, with_state_db=True, project_dir="cwd-decoy", session=S_MAIN):
    """Yield a Harness on a fresh provider; always shuts the provider down."""
    if with_state_db:
        seed_state_db(tmp_path)
    provider = NeospikeProvider(embedder=KeywordEmbedder())
    provider.initialize(session, hermes_home=str(tmp_path), cwd=str(tmp_path / project_dir))
    try:
        yield Harness(provider)
    finally:
        provider.shutdown()


def is_denied(result):
    """True when a mutation result reports neither an update nor a removal."""
    return not result.get("updated") and not result.get("removed")
