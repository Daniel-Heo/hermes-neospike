"""Links keep only refers/supersedes; split messages chain their chunks."""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

from neospike import backup as db_backup
from neospike import cli
from neospike import conversations as conv
from neospike.config import EMBEDDING_DIMENSIONS, NeospikeConfig
from neospike.notes import note_related
from neospike.provider import NeospikeProvider
from neospike.service import NeospikeService
from neospike.store import LINK_RELATION_NAMES, SCHEMA_VERSION, MemoryStore
from neospike.tests.support_items import link_rows
from neospike.tests.test_conversations import WordCounter
from neospike.tests.test_notes import DIMENSIONS as BOW_DIMENSIONS
from neospike.tests.test_notes import BowEmbedder
from neospike.tests.test_remember_dedupe import no_remember_tidy  # noqa: F401 (fixture)

# These toy embedders give distinct memories one vector; post-save dedupe is not under test here.
pytestmark = pytest.mark.usefixtures("no_remember_tidy")

DIMENSIONS = EMBEDDING_DIMENSIONS  # the CLI opens the store at the model dimension
NAMESPACE = "owner"
SESSION = "session-live"
CHUNK_TOKENS = 2000  # the 2048 window minus the margin, as in production
SENTENCE_WORDS = 5
MESSAGE_SENTENCES = 1000  # 5000 words = 5000 WordCounter tokens
# v9 links DDL: six relations and the similar-only weight column.
V9_LINKS_SQL = (
    "CREATE TABLE links (id INTEGER PRIMARY KEY, "
    "src_type TEXT NOT NULL CHECK (src_type IN ('note','memory')), src_id INTEGER NOT NULL, "
    "dst_type TEXT NOT NULL CHECK (dst_type IN ('note','memory')), dst_id INTEGER, dst_title TEXT, "
    "relation TEXT NOT NULL DEFAULT 'refers' CHECK (relation IN "
    "('refers','similar','same_context','supersedes','contradicts','part_of')), weight REAL, "
    "created_by TEXT NOT NULL DEFAULT 'code' CHECK (created_by IN ('code','llm','human')), "
    "origin TEXT NOT NULL DEFAULT '' CHECK (origin IN ('','conversation','automation','human','migration')), "
    "session_id TEXT NOT NULL DEFAULT '', evidence TEXT NOT NULL DEFAULT '', "
    "created_at REAL NOT NULL DEFAULT 0, reverted_at REAL)"
)


def _vector(angle: float) -> list[float]:
    return [math.cos(angle), math.sin(angle)] + [0.0] * (DIMENSIONS - 2)


class KeywordEmbedder:
    dimensions = DIMENSIONS
    ANGLES = {"alpha": 0.0, "beta": 0.3, "gamma": 0.9}

    def embed_document(self, text: str, **_kwargs) -> list[float]:
        return _vector(next((angle for word, angle in self.ANGLES.items() if word in text), 1.5))

    embed_query = embed_document


def _relations(store: MemoryStore) -> dict[str, int]:
    return {str(row[0]): int(row[1]) for row in store._conn.execute(
        "SELECT relation,COUNT(*) FROM item_links WHERE relation IN ('refers','supersedes') GROUP BY relation")}


# 1: no save path writes a similar row ------------------------------------------------------------
def test_schema_allows_only_the_v22_relations(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db", dimensions=DIMENSIONS)
    try:
        assert LINK_RELATION_NAMES == ("next", "topic", "refers", "supersedes")
        assert "weight" not in {row[1] for row in store._conn.execute("PRAGMA table_info(item_links)")}
        assert not hasattr(store, "link_similar") and not hasattr(store, "rebuild_similar_links")
        with pytest.raises(sqlite3.IntegrityError):
            store._conn.execute("INSERT INTO item_links(src_id,dst_id,relation) VALUES (1,2,'similar')")
    finally:
        store.close()


def test_note_memory_and_provider_saves_write_no_similar_rows(tmp_path: Path) -> None:
    home = tmp_path / "home"
    NeospikeConfig().save(home)
    provider = NeospikeProvider(embedder=KeywordEmbedder())
    provider.initialize(SESSION, hermes_home=str(home), cwd=str(home))
    try:
        for text in ("The alpha rollout waits for review.", "The beta rollout waits for review."):
            assert json.loads(provider.handle_tool_call("neospike_memory_save", {
                "text": text, "scope": "global", "subject": "rollout", "durability": "ongoing", "kind": "fact", "basis": "user_said"})) == {"stored": True}
        store = provider._memory_db
        service = NeospikeService(store=store, embedder=KeywordEmbedder(), namespace=NAMESPACE)
        hub = service.save_note("Hub", "The alpha hub note body.", author="human")["id"]
        service.save_note("Leaf", "The beta leaf points to [[Hub]] and [[mem:1]].", author="human")
        service.save_note("Lone", "The gamma lone note body.", author="human")
        assert set(_relations(store)) == {"refers"} and _relations(store)["refers"] == 2
        # Similar notes are still computed from vectors, without stored rows.
        assert note_related(store._conn, hub)["similar"]
    finally:
        provider.shutdown()


def test_cli_similar_rebuild_is_gone(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(tmp_path), "similar-rebuild"])
    with pytest.raises(SystemExit):
        cli.main()


# 2: v9 -> v10 migration --------------------------------------------------------------------------
V9_ITEM_TABLES_SQL = (
    "CREATE TABLE memories (id INTEGER PRIMARY KEY, namespace TEXT NOT NULL, content_hash TEXT NOT NULL, "
    "text TEXT NOT NULL, source TEXT NOT NULL, embedding BLOB NOT NULL, created_at REAL NOT NULL, "
    "UNIQUE(namespace, content_hash));"
    "CREATE TABLE notes (id INTEGER PRIMARY KEY, title TEXT NOT NULL UNIQUE, body TEXT NOT NULL, "
    "author TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL, rev INTEGER NOT NULL DEFAULT 1);")
# Old ids of the v9 rows -> the v22 ids: memories keep theirs, notes 1..3 come after the highest memory id (5).
V9_NOTE_IDS = {1: 6, 2: 7, 3: 8}


def _make_v9(path: Path) -> dict[int, tuple]:
    """A v10 file turned back into v9: memories/notes/typed links with weight, chunks without next_chunk_id."""
    store = MemoryStore(path, dimensions=BOW_DIMENSIONS)
    conv.ConversationStore(store).append(SESSION, [("assistant", _long_message())])
    _indexer(store).index_pending()
    store.close()
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executescript(V9_ITEM_TABLES_SQL)
        connection.executemany("INSERT INTO memories VALUES (?,?,?,?,?,X'',1.0)",
                               [(memory_id, NAMESPACE, f"h{memory_id}", f"memory {memory_id}", "agent:s")
                                for memory_id in (3, 4, 5)])
        connection.executemany("INSERT INTO notes(id,title,body,author,created_at,updated_at) VALUES (?,?,?,?,1.0,1.0)",
                               [(1, "A", "a [[B]]", "human"), (2, "B", "b", "human"), (3, "C", "c [[Missing]]", "agent")])
        connection.execute(V9_LINKS_SQL)
        rows = [(1, "note", 1, "note", 2, "B", "refers", None, "code"),
                (2, "memory", 3, "memory", 4, None, "supersedes", None, "human"),
                (3, "note", 1, "note", 3, None, "similar", 0.91, "code"),
                (4, "memory", 3, "memory", 5, None, "similar", 0.88, "code"),
                (5, "note", 2, "note", 3, None, "contradicts", None, "code"),
                (6, "note", 2, "note", 1, None, "part_of", None, "code"),
                (7, "memory", 4, "memory", 5, None, "same_context", None, "code"),
                (8, "note", 3, "note", None, "Missing", "refers", None, "code")]
        connection.executemany("INSERT INTO links(id,src_type,src_id,dst_type,dst_id,dst_title,relation,weight,created_by) "
                               "VALUES (?,?,?,?,?,?,?,?,?)", rows)
        connection.execute("ALTER TABLE conversation_chunks DROP COLUMN next_id")
        connection.execute("UPDATE meta_entries SET value='9' WHERE key='schema_version'")
    return {row[0]: row for row in rows if row[6] in ("refers", "supersedes")}


def test_v9_migration_keeps_refers_supersedes_and_drops_the_rest(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    kept = _make_v9(path)
    store = MemoryStore(path, dimensions=BOW_DIMENSIONS)
    try:
        assert _relations(store) == {"refers": 2, "supersedes": 1}
        rows = {int(row["id"]): (row["id"], row["src_type"], row["src_id"], row["dst_type"], row["dst_id"],
                                 row["dst_title"], row["relation"], row["created_by"]) for row in link_rows(store._conn)}
        assert set(rows) == set(kept)  # the link ids stay (supersede-revert --link)
        for link_id, row in kept.items():
            src = V9_NOTE_IDS.get(row[2], row[2]) if row[1] == "note" else row[2]
            dst = row[4] if row[3] == "memory" or row[4] is None else V9_NOTE_IDS[row[4]]
            title = f"mem:{dst}" if row[3] == "memory" else row[5]
            assert tuple(rows[link_id]) == (row[0], row[1], src, row[3], dst, title, row[6], row[8])
        assert "weight" not in {row[1] for row in store._conn.execute("PRAGMA table_info(item_links)")}
        assert store._conn.execute("SELECT value FROM meta_entries WHERE key=?", ("schema_version",)).fetchone()[0] == str(SCHEMA_VERSION)
        assert (tmp_path / "memory.pre-v10.db").is_file()
        assert _chain_parts(store) == [0, 1, 2]  # filled for the existing chunks
        triggers = {row[0] for row in store._conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        assert "item_links_track_ad" in triggers
    finally:
        store.close()


# split messages: chunk chain, search fields, conv-read -------------------------------------------
def _sentence(index: int) -> str:
    if 200 <= index < 240:
        return "mango kiwi mango kiwi mango."  # part 0 only
    if 500 <= index < 560:
        return "zebra quartz zebra quartz zebra."  # part 1 only
    return "lemon lemon lemon lemon lemon."


def _long_message() -> str:
    return " ".join(_sentence(index) for index in range(MESSAGE_SENTENCES))


def _indexer(store: MemoryStore) -> conv.ConversationIndexer:
    return conv.ConversationIndexer(store, BowEmbedder(), token_counter=WordCounter(), max_tokens=CHUNK_TOKENS,
                                    context_tokens=20)


def _chain_parts(store: MemoryStore, seq: int = 1) -> list[int]:
    """Parts in chain order from the head, asserting the last next is NULL."""
    rows = {int(row[0]): row for row in store._conn.execute(
        "SELECT id,part,next_id FROM conversation_chunks WHERE seq_start=? AND seq_end=?", (seq, seq))}
    targets = {row[2] for row in rows.values() if row[2] is not None}
    [head] = [chunk_id for chunk_id in rows if chunk_id not in targets]
    parts, cursor = [], head
    while cursor is not None:
        parts.append(int(rows[cursor][1]))
        cursor = rows[cursor][2]
    assert len(parts) == len(rows)
    return parts


@pytest.fixture
def chained(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db", dimensions=BOW_DIMENSIONS)
    conv.ConversationStore(store).append(SESSION, [("assistant", _long_message())])
    _indexer(store).index_pending()
    yield store
    store.close()


def test_5000_token_message_is_three_chunks_chained_0_1_2_null(chained: MemoryStore) -> None:
    assert WordCounter.count(_long_message()) == MESSAGE_SENTENCES * SENTENCE_WORDS
    rows = chained._conn.execute("SELECT part,next_id FROM conversation_chunks ORDER BY part").fetchall()
    assert [row[0] for row in rows] == [0, 1, 2] and rows[2][1] is None
    assert _chain_parts(chained) == [0, 1, 2]


def test_read_message_follows_the_chain_and_equals_the_masked_message(chained: MemoryStore) -> None:
    result = conv.read_message(chained._conn, 1, 1)
    assert result["message_parts"] == 3 and result["matches_message"] is True
    assert result["text"] == _long_message()
    secret = "api_key=sk-FAKEKEY1234567890abcdefXYZ"
    conv.ConversationStore(chained).append(SESSION, [("user", _long_message().replace("mango kiwi", secret, 1))])
    _indexer(chained).index_pending()
    masked = conv.read_message(chained._conn, 1, 2)
    stored = chained._conn.execute("SELECT content FROM conversation_messages WHERE seq=2").fetchone()[0]
    assert masked["message_parts"] == 3 and masked["text"] == stored and "sk-FAKEKEY" not in masked["text"]


def test_second_part_hit_carries_message_parts_and_part(chained: MemoryStore) -> None:
    [hit] = conv.search(chained, "zebra quartz", BowEmbedder().embed_query("zebra quartz"), limit=5)
    assert (hit["message_parts"], hit["part"], hit["parts"]) == (3, 1, [1])
    assert hit["read"] == "conv-read --conversation 1 --seq 1" and len(hit["chunk_ids"]) == 1


def test_first_and_second_part_hits_merge_into_one_result(chained: MemoryStore) -> None:
    # Two words per part plus the shared filler "lemon" and part 0's "assistant" label: BoW cos part 0 = 0.82,
    # part 1 = 0.707 clear the 0.7 cosine floor; part 2 (lemon only) = 0.41 stays out.
    query = "mango kiwi zebra quartz lemon assistant"
    results = conv.search(chained, query, BowEmbedder().embed_query(query), limit=5)
    assert len(results) == 1
    [hit] = results
    assert hit["parts"] == [0, 1] and hit["message_parts"] == 3 and len(hit["chunk_ids"]) == 2
    body = hit["text"]
    assert body.count("mango kiwi mango kiwi mango.") == 40 and body.count("zebra quartz zebra quartz zebra.") == 60
    assert body.startswith("assistant: ") and body[len("assistant: "):] in _long_message()  # overlap removed


def test_chain_survives_tail_rechunk_and_full_reindex(chained: MemoryStore) -> None:
    conv.ConversationStore(chained).append(SESSION, [("user", "A short follow-up question.")])
    _indexer(chained).index_pending()
    assert _chain_parts(chained) == [0, 1, 2]
    assert chained._conn.execute("SELECT next_id FROM conversation_chunks WHERE seq_start=2").fetchone()[0] is None
    with chained._conn:
        chained._conn.execute("DELETE FROM conversation_chunks")
        chained._conn.execute("UPDATE conversations SET indexed_seq=0")
    _indexer(chained).index_pending()
    assert _chain_parts(chained) == [0, 1, 2]
    assert conv.read_message(chained._conn, 1, 1)["text"] == _long_message()


def test_cli_conv_read_returns_the_whole_message(tmp_path: Path, monkeypatch, capsys) -> None:
    home = tmp_path / "home"
    (home / "neospike").mkdir(parents=True)
    (home / "neospike" / "config.json").write_text(json.dumps({"embedding_dimensions": BOW_DIMENSIONS}), encoding="utf-8")
    store = MemoryStore(home / "neospike" / "memory.db", dimensions=BOW_DIMENSIONS)
    conv.ConversationStore(store).append(SESSION, [("assistant", _long_message())])
    _indexer(store).index_pending()
    store.close()
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(home), "conv-read", "--conversation", "1", "--seq", "1"])
    assert cli.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["message_parts"] == 3 and output["text"] == _long_message() and output["matches_message"] is True
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(home), "conv-read", "--conversation", "1", "--seq", "9"])
    assert cli.main() == 1
    assert json.loads(capsys.readouterr().out)["error"] == "not_found"


def test_broken_chain_is_reported(chained: MemoryStore) -> None:
    with chained._conn:
        chained._conn.execute("UPDATE conversation_chunks SET next_id=NULL WHERE part=0")
    with pytest.raises(conv.ConversationError) as error:
        conv.read_message(chained._conn, 1, 1)
    assert error.value.code == "broken_chain"


# backup: a migration between backups starts a new chain -------------------------------------------
def test_backup_after_schema_change_is_full(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    MemoryStore(path, dimensions=BOW_DIMENSIONS).close()
    backups = tmp_path / "backups"
    assert db_backup.run_backup(path, backups)["kind"] == "full"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("INSERT INTO item_links(src_id,dst_title,relation) VALUES (1,'X','refers')")
    assert db_backup.run_backup(path, backups)["kind"] == "incremental"
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("UPDATE meta_entries SET value=? WHERE key='schema_version'", (str(SCHEMA_VERSION + 1),))
    result = db_backup.run_backup(path, backups)
    assert result["kind"] == "full" and result["reason"] == "schema_changed"
