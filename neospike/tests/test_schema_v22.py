"""Schema v22 - the v21 -> v22 migration into items / item_links /
item_revisions with its checks and rollback, schema-migrate --dry-run/--apply/--restore, the write guard for a newer
schema, and the old tool names."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
import sys
from array import array
from pathlib import Path

import pytest

from neospike import cli, schema_v22
from neospike.notes import note_body
from neospike.provider import NeospikeProvider
from neospike.store import MemoryStore, next_of
from neospike.tests.support_items import chunk_rows, link_rows, note_rows, topic_labels
from neospike.tests.support_provider import ThreeTopicEmbedder as FakeEmbedder

DIM = 4
OWNER = "default:local"
# The v21 layout, reduced to the tables and columns v22 merges, renames or drops.
V21_DDL = """
CREATE TABLE memories (id INTEGER PRIMARY KEY, namespace TEXT NOT NULL, content_hash TEXT NOT NULL, text TEXT NOT NULL,
    source TEXT NOT NULL, embedding BLOB NOT NULL, created_at REAL NOT NULL, kind TEXT NOT NULL DEFAULT 'episodic',
    confidence REAL NOT NULL DEFAULT 1.0, importance REAL NOT NULL DEFAULT 0.5, expires_at REAL,
    project TEXT NOT NULL DEFAULT '', metadata_json TEXT NOT NULL DEFAULT '{}', accessed_at REAL,
    access_count INTEGER NOT NULL DEFAULT 0, invalid_at REAL, superseded_by INTEGER, archived_at REAL,
    next_piece_id INTEGER, updated_at REAL, UNIQUE(namespace, content_hash));
CREATE VIRTUAL TABLE memories_fts USING fts5(text, content='memories', content_rowid='id', tokenize='unicode61');
CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE INDEX memories_next_piece ON memories(next_piece_id) WHERE next_piece_id IS NOT NULL;
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE notes (id INTEGER PRIMARY KEY, title TEXT NOT NULL UNIQUE, body TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]', project TEXT NOT NULL DEFAULT '', author TEXT NOT NULL,
    created_at REAL NOT NULL, updated_at REAL NOT NULL, rev INTEGER NOT NULL DEFAULT 1, deleted_at REAL,
    accessed_at REAL, access_count INTEGER NOT NULL DEFAULT 0, model TEXT NOT NULL DEFAULT '');
CREATE TABLE note_chunks (id INTEGER PRIMARY KEY, note_id INTEGER NOT NULL, ord INTEGER NOT NULL,
    heading_path TEXT NOT NULL DEFAULT '', text TEXT NOT NULL, text_hash TEXT NOT NULL, embedding BLOB NOT NULL,
    source_json TEXT, UNIQUE(note_id, ord));
CREATE VIRTUAL TABLE note_chunks_fts USING fts5(text, heading_path, content='note_chunks', content_rowid='id');
CREATE TABLE note_revisions (note_id INTEGER NOT NULL, rev INTEGER NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
    edited_by TEXT NOT NULL, edited_at REAL NOT NULL, model TEXT NOT NULL DEFAULT '', PRIMARY KEY(note_id, rev));
CREATE TABLE memory_revisions (memory_id INTEGER NOT NULL, rev INTEGER NOT NULL, text TEXT NOT NULL,
    metadata_json TEXT NOT NULL, edited_at REAL NOT NULL, edited_by TEXT NOT NULL, session_id TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(memory_id, rev));
CREATE TABLE links (id INTEGER PRIMARY KEY, src_type TEXT NOT NULL, src_id INTEGER NOT NULL, dst_type TEXT NOT NULL,
    dst_id INTEGER, dst_title TEXT, relation TEXT NOT NULL DEFAULT 'refers', created_by TEXT NOT NULL DEFAULT 'code',
    origin TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', evidence TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL DEFAULT 0, reverted_at REAL);
CREATE TABLE topics (id INTEGER PRIMARY KEY, note_id INTEGER, name TEXT NOT NULL, centroid BLOB,
    dirty INTEGER NOT NULL DEFAULT 0, dirty_since REAL, created_at REAL NOT NULL, deleted_at REAL, changed_at REAL,
    refreshed_at REAL, fail_count INTEGER NOT NULL DEFAULT 0);
CREATE TABLE topic_members (memory_id INTEGER PRIMARY KEY, topic_id INTEGER NOT NULL, assigned_at REAL NOT NULL);
CREATE TABLE topic_discovery_links (low_id INTEGER NOT NULL, high_id INTEGER NOT NULL, cosine REAL NOT NULL,
    created_at REAL NOT NULL, PRIMARY KEY(low_id, high_id));
CREATE INDEX topic_discovery_links_high ON topic_discovery_links(high_id);
CREATE TABLE topic_discovery_checked (memory_id INTEGER PRIMARY KEY, checked_at REAL NOT NULL, content_hash TEXT);
CREATE TABLE note_absorb_checked (memory_id INTEGER PRIMARY KEY, content_hash TEXT NOT NULL, chunk_id INTEGER NOT NULL);
CREATE TABLE llm_usage (week TEXT NOT NULL, stage TEXT NOT NULL, tokens INTEGER NOT NULL DEFAULT 0,
    calls INTEGER NOT NULL DEFAULT 0, failures INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(week, stage));
CREATE TABLE folders (id INTEGER PRIMARY KEY, name TEXT NOT NULL, color TEXT NOT NULL,
    origin TEXT NOT NULL DEFAULT 'human' CHECK (origin IN ('human','auto')), created_at REAL NOT NULL, updated_at REAL,
    deleted_at REAL);
CREATE TABLE folder_items (folder_id INTEGER NOT NULL REFERENCES folders(id),
    lane TEXT NOT NULL CHECK (lane IN ('memory','note')), item_id INTEGER NOT NULL, PRIMARY KEY (lane, item_id));
CREATE TABLE tombstones (table_name TEXT NOT NULL, row_key TEXT NOT NULL, deleted_at REAL NOT NULL);
"""
# The v21 rows: memories 1..3 (1 -> 2 a split chain), human note 1 (two heading chunks, [[mem:2]]), automatic note 2
# (two chunks made from deleted memories 8 and 9), trashed agent note 3; topics 1 'T', 2 'T' (same name), 3 deleted.
HUMAN_BODY = "# A\nalpha text [[mem:2]]\n\n# B\nbeta text"
HUMAN_CHUNKS = (("A", "# A\nalpha text [[mem:2]]"), ("B", "# B\nbeta text"))
FORMULA_TEXTS = ("Otter rule one.", "Otter rule two.")
# Old note ids -> v22 ids: memories keep 1..3, the three notes take 4..6 in old id order.
NOTE_IDS = {1: 4, 2: 5, 3: 6}


def _vector(*values: float) -> bytes:
    return array("f", values).tobytes()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _v21(path: Path) -> None:
    connection = sqlite3.connect(path)
    with connection:
        connection.executescript(V21_DDL)
        rows = [(1, "Falcon deploys on Monday.", _vector(1, 0, 0, 0), 2), (2, "Heron ships weekly.", _vector(0, 1, 0, 0), None),
                (3, "Otter uses Rust.", _vector(0.6, 0.8, 0, 0), None)]
        for memory_id, text, blob, piece in rows:
            connection.execute("INSERT INTO memories(id,namespace,content_hash,text,source,embedding,created_at,kind,"
                               "updated_at,metadata_json,next_piece_id) VALUES (?,?,?,?,?,?,1.0,'memory_item',1.0,?,?)",
                               (memory_id, OWNER, f"h{memory_id}", text, f"agent:s{memory_id}", blob,
                                json.dumps({"scope": "global", "subject": "s", "session_id": f"s{memory_id}"}), piece))
        connection.executemany("INSERT INTO notes(id,title,body,author,created_at,updated_at,rev,deleted_at,model) "
                               "VALUES (?,?,?,?,2.0,3.0,?,?,?)", [
                                   (1, "Human page", HUMAN_BODY, "human", 2, None, ""),
                                   (2, "otter", "\n\n".join(FORMULA_TEXTS), "formula", 1, None, ""),
                                   (3, "Old agent page", "Agent text that was trashed.", "agent", 1, 4.0, "m1")])
        connection.executemany("INSERT INTO note_chunks(id,note_id,ord,heading_path,text,text_hash,embedding,source_json) "
                               "VALUES (?,?,?,?,?,?,?,?)", [
                                   (1, 1, 0, HUMAN_CHUNKS[0][0], HUMAN_CHUNKS[0][1], _sha(HUMAN_CHUNKS[0][1]), _vector(0, 0, 1, 0), None),
                                   (2, 1, 1, HUMAN_CHUNKS[1][0], HUMAN_CHUNKS[1][1], _sha(HUMAN_CHUNKS[1][1]), _vector(0, 0, 0, 1), None),
                                   (6, 2, 0, "", FORMULA_TEXTS[0], _sha(FORMULA_TEXTS[0]), _vector(0.5, 0.5, 0.5, 0.5),
                                    json.dumps({"memory_id": 8})),
                                   (7, 2, 1, "", FORMULA_TEXTS[1], _sha(FORMULA_TEXTS[1]), _vector(0.5, 0.5, -0.5, 0.5),
                                    json.dumps({"memory_id": 9})),
                                   (9, 3, 0, "", "Agent text that was trashed.", _sha("Agent text that was trashed."),
                                    _vector(0, 0, 0.6, 0.8), None)])
        connection.executemany("INSERT INTO note_revisions VALUES (?,?,?,?,?,?,?)",
                               [(1, 1, "Human page", "# A\nfirst [[mem:2]]", "human", 2.5, "")])
        connection.executemany("INSERT INTO memory_revisions VALUES (?,?,?,?,?,?,?)",
                               [(1, 1, "Falcon deploys on Sunday.", json.dumps({"scope": "global"}), 1.5, "agent", "s1")])
        connection.executemany("INSERT INTO links(id,src_type,src_id,dst_type,dst_id,dst_title,relation,created_by) "
                               "VALUES (?,?,?,?,?,?,?,?)", [
                                   (10, "note", 1, "memory", 2, "mem:2", "refers", "code"),
                                   (11, "memory", 3, "memory", 1, "mem:1", "supersedes", "human"),
                                   (12, "note", 3, "note", 1, "Human page", "refers", "code"),
                                   (13, "note", 99, "memory", 2, "mem:2", "refers", "code")])  # source note gone
        connection.executemany("INSERT INTO meta VALUES (?,?)", [
            ("schema_version", "21"), ("embedding_dimensions", str(DIM)), ("embedding_model", "bge-m3-int8-onnx"),
            ("live_namespace", OWNER), ("review_pending", "0"), ("backup_hash:llm_usage", "x"),
            ("formula_chunk_hwm", "7"), ("topic_manual", json.dumps({"pinned": {"3": 2}, "excluded": {"1": [3]}}))])
        connection.executemany("INSERT INTO topics(id,name,created_at,deleted_at,fail_count) VALUES (?,?,1.0,?,2)",
                               [(1, "T", None), (2, "T", None), (3, "gone", 5.0)])
        connection.executemany("INSERT INTO topic_members VALUES (?,?,1.0)", [(1, 1), (2, 2), (3, 3)])
        connection.execute("INSERT INTO topic_discovery_links VALUES (1,2,0.9,1.0)")
        connection.executemany("INSERT INTO topic_discovery_checked VALUES (?,1.0,?)", [(1, "h1"), (2, "h2")])
        connection.execute("INSERT INTO note_absorb_checked VALUES (1,'h1',7)")
        connection.execute("INSERT INTO llm_usage VALUES ('2026-W46','x',1,1,0)")
        connection.execute("INSERT INTO folders(id,name,color,origin,created_at) VALUES (1,'F','#123456','human',1.0)")
        connection.executemany("INSERT INTO folder_items VALUES (1,?,?)", [("memory", 3), ("note", 1)])
        connection.execute("INSERT INTO tombstones VALUES ('memories','9',1.0)")
    connection.close()


def _names(path: Path) -> set[str]:
    with closing(sqlite3.connect(path)) as connection:
        return {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master")}


def test_open_migrates_v21_in_one_checked_step(tmp_path: Path) -> None:
    path = tmp_path / "neospike" / "memory.db"
    path.parent.mkdir()
    _v21(path)
    store = MemoryStore(path, dimensions=DIM)
    try:
        names = _names(path)
        assert {"items", "items_fts", "item_links", "item_revisions", "meta_entries", "memory_pairs",
                "memory_checks"} <= names
        assert not names & {"memories", "memories_fts", "notes", "note_chunks", "note_chunks_fts", "links", "topics",
                            "topic_members", "memory_revisions", "note_revisions", "meta", "topic_discovery_links",
                            "topic_discovery_checked", "note_absorb_checked", "llm_usage", "memories_ai"}
        conn = store._conn
        columns = {r[1] for r in conn.execute("PRAGMA table_info(items)")}
        assert {"type", "owner", "title", "scope", "author", "kind", "importance", "superseded_by"} <= columns
        assert "expires_at" not in columns
        # memories keep their ids; the scope moved out of metadata_json into its column
        memories = conn.execute("SELECT id,text,scope,metadata_json FROM items WHERE type='memory' ORDER BY id").fetchall()
        assert [(row[0], row[2]) for row in memories] == [(1, "global"), (2, "global"), (3, "global")]
        assert "scope" not in json.loads(memories[0][3])
        assert next_of(conn, 1) == 2  # the piece chain is a next link now
        # notes: heads 4..6 in old id order, the bodies read back unchanged, [[mem:2]] still names memory 2
        heads = {row["id"]: row for row in note_rows(conn)}
        assert sorted(heads) == [4, 5, 6]
        assert (heads[4]["type"], heads[4]["author"], heads[4]["body"]) == ("doc", "human", HUMAN_BODY)
        assert (heads[5]["type"], heads[5]["body"]) == ("note", "\n\n".join(FORMULA_TEXTS))
        assert heads[6]["deleted_at"] == 4.0 and heads[6]["model"] == "m1"
        parts = chunk_rows(conn, 5)
        assert [json.loads(row["source_json"])["memory_id"] for row in parts] == [8, 9]  # automatic parts keep source
        assert [row["embedding"] for row in parts] == [_vector(0.5, 0.5, 0.5, 0.5), _vector(0.5, 0.5, -0.5, 0.5)]
        human = chunk_rows(conn, 4)
        assert [row["heading_path"] for row in human] == ["A", "B"]
        assert [row["embedding"] for row in human] == [_vector(0, 0, 1, 0), _vector(0, 0, 0, 1)]  # vectors reused
        # links: ids kept, ends mapped; the link of a gone source note is dropped
        links = {row["id"]: (row["src_id"], row["dst_id"], row["dst_title"], row["relation"]) for row in link_rows(conn)}
        assert links == {10: (4, 2, "mem:2", "refers"), 11: (3, 1, "mem:1", "supersedes"),
                         12: (6, 4, "Human page", "refers")}
        # topics: links labelled by name, a repeated name gets ' (2)', the deleted topic is gone
        assert topic_labels(conn) == {1: "T", 2: "T (2)"}
        marks = json.loads(conn.execute("SELECT value FROM meta_entries WHERE key='topic_manual'").fetchone()[0])
        assert marks == {"pinned": {"3": "T (2)"}, "excluded": {}}
        revisions = conn.execute("SELECT item_id,rev,title,text FROM item_revisions ORDER BY item_id").fetchall()
        assert [tuple(row) for row in revisions] == [(1, 1, None, "Falcon deploys on Sunday."),
                                                     (4, 1, "Human page", "# A\nfirst [[mem:2]]")]
        assert conn.execute("SELECT created_by FROM folders").fetchone()[0] == "human"
        assert sorted(tuple(r) for r in conn.execute("SELECT item_type,item_id FROM folder_items")) == [
            ("memory", 3), ("note", 4)]
        formula_ids = [row["id"] for row in parts]
        assert [tuple(r) for r in conn.execute("SELECT memory_id,stage,text_hash,chunk_id FROM memory_checks "
                                               "ORDER BY memory_id,stage")] == [
            (1, "absorb", "h1", formula_ids[1]), (1, "topic", "h1", None), (2, "topic", "h2", None)]
        meta = dict(conn.execute("SELECT key,value FROM meta_entries").fetchall())
        assert meta["live_owner"] == OWNER and meta["schema_version"] == "23"  # v21 -> v22 -> v23 in one open
        assert meta["formula_chunk_hwm"] == str(formula_ids[1])
        assert not {"review_pending", "backup_hash:llm_usage", "live_namespace"} & set(meta)
        assert conn.execute("SELECT COUNT(*) FROM tombstones").fetchone()[0] == 0  # old-table tombstones dropped
        assert conn.execute("PRAGMA user_version").fetchone()[0] == schema_v22.SUPPORTED_USER_VERSION
        assert conn.execute("PRAGMA application_id").fetchone()[0] == schema_v22.APPLICATION_ID
        assert [r[0] for r in conn.execute("SELECT rowid FROM items_fts WHERE items_fts MATCH 'Heron'")] == [2]
        assert [hit.id for hit in store.search(OWNER, "x", [1, 0, 0, 0], limit=3)][0] == 1
        assert note_body(conn, 4, "human") == HUMAN_BODY
        assert path.with_name(schema_v22.SNAPSHOT_NAME).is_file()  # the step snapshot of the open path
    finally:
        store.close()


def test_memory_references_follow_an_id_map() -> None:
    """[[mem:N]] (with a heading or alias part too) is rewritten through the map; unmapped ids and titles stay."""
    text = "See [[mem:1]], [[mem:2#part]], [[ mem:3 |alias]], [[mem:9]] and [[mem:1x]] and [[Page]]."
    assert schema_v22.rewrite_memory_refs(text, {1: 11, 2: 12, 3: 13}) == (
        "See [[mem:11]], [[mem:12#part]], [[ mem:13 |alias]], [[mem:9]] and [[mem:1x]] and [[Page]].")


def test_migration_report_counts_by_type_and_link(tmp_path: Path) -> None:
    """(d) row counts per type before -> after, the id map and the [[mem:N]] text of a note body."""
    path = tmp_path / "memory.db"
    _v21(path)
    connection = sqlite3.connect(path)
    try:
        report = schema_v22.migrate_v22(connection)
    finally:
        connection.close()
    assert report["migrated"] and report["problems"] == []
    assert report["items_by_type"] == {"memory": 3, "doc": 1, "note": 2, "formula_parts": 2, "other_parts": 3}
    assert report["items_expected"] == report["items_by_type"]
    assert report["item_links"] == {"refers": 2, "supersedes": 1, "next": 1 + 2 + 2 + 1, "topic": 2}
    assert report["id_map"] == {"notes": {"1": 4, "2": 5, "3": 6}, "memories": "kept"}
    assert report["dropped"]["links"] == 1 and report["dropped"]["deleted_topics"] == ["gone"]
    assert report["renamed_topics"] == {"2": "T (2)"}
    assert report["top5"] == report["top5_before"] != {}
    assert report["parts"]["reused_vector"] == 3 and report["parts"]["embedding_pending"] == 0


def test_failed_check_rolls_everything_back(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "memory.db"
    _v21(path)
    before = _names(path)
    real = schema_v22._verify

    def broken(*args, **kwargs):
        report = real(*args, **kwargs)
        report["problems"].append("forced")
        return report

    monkeypatch.setattr(schema_v22, "_verify", broken)
    connection = sqlite3.connect(path)
    try:
        with pytest.raises(schema_v22.SchemaError, match="forced"):
            schema_v22.migrate_v22(connection)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
    finally:
        connection.close()
    assert _names(path) == before


def test_newer_schema_is_never_opened_for_writing(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    MemoryStore(path, dimensions=DIM).close()
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(f"PRAGMA user_version={schema_v22.SUPPORTED_USER_VERSION + 1}")
    with pytest.raises(schema_v22.SchemaError, match="newer"):
        MemoryStore(path, dimensions=DIM)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA user_version=22")
        connection.execute("PRAGMA application_id=7")
    with pytest.raises(schema_v22.SchemaError, match="not a neospike database"):
        MemoryStore(path, dimensions=DIM)


def _run(home: Path, *arguments: str, capsys) -> tuple[int, dict]:
    code = cli.main(["--home", str(home), *arguments])
    return code, json.loads(capsys.readouterr().out)


def test_schema_migrate_dry_run_apply_and_restore(tmp_path: Path, capsys) -> None:
    home = tmp_path / "home"
    folder = home / "neospike"
    folder.mkdir(parents=True)
    db = folder / "memory.db"
    _v21(db)
    (folder / "config.json").write_text(json.dumps({"debug_enabled": True, "llm_weekly_token_budget": 3}),
                                        encoding="utf-8")
    (folder / "MEMORY_RULE.md").write_text("Use neospike_remember.", encoding="utf-8")
    raw = db.read_bytes()
    code, dry = _run(home, "schema-migrate", "--dry-run", capsys=capsys)
    assert code == 0 and dry["ok"] and dry["problems"] == [] and db.read_bytes() == raw  # read only
    assert dry["rows_after"]["memory_checks"] == 3 and dry["top5"] == dry["top5_before"] != {}
    assert set(dry["fts_integrity"].values()) == {"ok"}
    code, applied = _run(home, "schema-migrate", "--apply", capsys=capsys)
    assert code == 0 and applied["applied"] and applied["backup"] == schema_v22.SNAPSHOT_NAME
    assert applied["config"]["removed"] == ["llm_weekly_token_budget"] and applied["config"]["rule_replacements"] == 1
    assert json.loads((folder / "config.json").read_text(encoding="utf-8")) == {"turn_trace_enabled": True}
    assert (folder / "MEMORY_RULE.md").read_text(encoding="utf-8").strip() == "Use neospike_memory_save."
    assert "items" in _names(db) and _run(home, "schema-migrate", "--apply", capsys=capsys)[1] == {
        "applied": False, "reason": "already v22"}
    code, restored = _run(home, "schema-migrate", "--restore", capsys=capsys)
    assert code == 0 and restored["restored"] and sorted(restored["config_restored"]) == ["MEMORY_RULE.md", "config.json"]
    assert "memories" in _names(db) and "items" not in _names(db)
    assert json.loads((folder / "config.json").read_text(encoding="utf-8"))["debug_enabled"] is True
    assert (folder / restored["replaced_kept_as"]).is_file()


def test_open_in_place_also_moves_config_and_rule(tmp_path: Path) -> None:
    # MemoryStore migrates a v21 database on open without `schema-migrate --apply`; the config move must run too
    # (it was skipped, so three live homes kept the v21 rule text and tool names).
    from neospike.config import DEFAULT_SAVE_CRITERIA, PREVIOUS_SAVE_CRITERIA
    folder = tmp_path / "home" / "neospike"
    folder.mkdir(parents=True)
    db = folder / "memory.db"
    _v21(db)
    (folder / "config.json").write_text(json.dumps({"debug_enabled": True, "llm_weekly_token_budget": 3}),
                                        encoding="utf-8")
    old_rule = PREVIOUS_SAVE_CRITERIA.replace("neospike_memory_save", "neospike_remember")  # the v21 default
    (folder / "MEMORY_RULE.md").write_text(old_rule, encoding="utf-8")
    MemoryStore(db, dimensions=DIM).close()
    assert json.loads((folder / "config.json").read_text(encoding="utf-8")) == {"turn_trace_enabled": True}
    assert (folder / "MEMORY_RULE.md").read_text(encoding="utf-8").strip() == DEFAULT_SAVE_CRITERIA
    assert (folder / "MEMORY_RULE.pre-v22.md").read_text(encoding="utf-8") == old_rule
    assert json.loads((folder / "config.pre-v22.json").read_text(encoding="utf-8"))["debug_enabled"] is True
    edited = folder / "MEMORY_RULE.md"
    edited.write_text("Mine.", encoding="utf-8")
    MemoryStore(db, dimensions=DIM).close()  # already migrated: nothing is moved again
    assert edited.read_text(encoding="utf-8") == "Mine."


def test_config_move_skips_a_database_outside_a_home(tmp_path: Path) -> None:
    db = tmp_path / "copy.db"
    _v21(db)
    MemoryStore(db, dimensions=DIM).close()
    assert schema_v22.home_of_database(db) is None and "items" in _names(db)


def test_only_v22_tool_names_are_listed_and_accepted(tmp_path: Path) -> None:
    # Hermes routes only schema names; the v21 names are no longer registered.
    provider = NeospikeProvider(embedder=FakeEmbedder())
    provider.initialize("s1", hermes_home=str(tmp_path), agent_identity="default", user_id="local",
                        agent_context="primary", cwd=str(tmp_path))
    names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert names == {"neospike_memory_save", "neospike_memory_update", "neospike_memory_delete",
                     "neospike_memory_reconcile", "neospike_memory_search", "neospike_note_read", "neospike_note_write",
                     "neospike_doc_read", "neospike_doc_write"}
    saved = json.loads(provider.handle_tool_call("neospike_memory_save", {
        "text": "Falcon scripts use Python.", "scope": "global", "subject": "Falcon", "durability": "stable", "kind": "fact", "basis": "user_said"}))
    assert saved["stored"] is True
    found = json.loads(provider.handle_tool_call("neospike_memory_search", {"query": "python"}))
    assert found["results"] and found["results"][0]["origin"].startswith("agent:")
    for old in ("neospike_remember", "neospike_search", "neospike_status"):
        assert json.loads(provider.handle_tool_call(old, {})) == {"error": f"Unknown tool: {old}"}


def test_cli_and_mcp_take_the_old_option_with_a_warning(capsys) -> None:
    assert cli.forward_old_command(["--home", "h", "status"]) == ["--home", "h", "memory-count"]
    assert cli.forward_old_command(["forget", "5"]) == ["memory-delete", "--id", "5"]
    assert cli.forward_old_command(["memory-search", "status"]) == ["memory-search", "status"]  # a query word
    from neospike.renames import forward_old_options
    assert forward_old_options(["--namespace=x", "--namespace", "y"]) == ["--owner=x", "--owner", "y"]
    assert capsys.readouterr().err.count("'--namespace' is renamed to '--owner'") == 1  # one warning per option
    from neospike import mcp_server
    assert mcp_server.forward_old_options is forward_old_options
