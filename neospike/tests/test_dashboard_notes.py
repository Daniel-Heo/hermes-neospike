"""Contract tests: dashboard Notes / Memory API over the single-writer CLI path."""
from __future__ import annotations

import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from neospike.config import NeospikeConfig
from neospike.dashboard import plugin_api as api
from neospike.database import canonical_database_path
from neospike.store import MemoryStore
from neospike.tests.test_notes import DIMENSIONS

NAMESPACE = "default:local"
DASHBOARD = Path(api.__file__).resolve().parent
# Child CLI process: stub Hermes core like conftest.py, swap the embedding daemon for the
# deterministic bag-of-words embedder, then run the real CLI entrypoint.
_CHILD_BOOT = (
    "import sys,types;"
    "a=types.ModuleType('agent');m=types.ModuleType('agent.memory_provider');"
    "m.MemoryProvider=type('MemoryProvider',(),{});a.memory_provider=m;"
    "sys.modules.setdefault('agent',a);sys.modules.setdefault('agent.memory_provider',m);"
    "from neospike import cli;from neospike.tests.test_notes import BowEmbedder;"
    "cli.EmbeddingClient=BowEmbedder;sys.exit(cli.main())"
)


class _CliSpy:
    """Records every subprocess argv while running the real subprocess."""

    def __init__(self, run) -> None:
        self.run = run
        self.commands: list[list[str]] = []

    def __call__(self, command, *args, **kwargs):
        self.commands.append(list(command))
        return self.run(command, *args, **kwargs)


class _ReadOnlyConnectGuard:
    """sqlite3.connect for the dashboard process: only mode=ro URIs may be opened."""

    def __init__(self, connect) -> None:
        self.connect = connect
        self.targets: list[str] = []

    def __call__(self, database, *args, **kwargs):
        target = str(database)
        if not (kwargs.get("uri") and target.startswith("file:") and "mode=ro" in target):
            raise AssertionError(f"The dashboard process opened a writable SQLite connection: {target}")
        self.targets.append(target)
        return self.connect(database, *args, **kwargs)


def _child_prefix() -> list[str]:
    return [sys.executable, "-c", _CHILD_BOOT]


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "hermes"
    NeospikeConfig().save(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture()
def client(home: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api, "_cli_argv_base", _child_prefix)
    monkeypatch.setattr(api.subprocess, "run", _CliSpy(subprocess.run))
    # Writes must go through the child process: this process may only open mode=ro connections.
    monkeypatch.setattr(sqlite3, "connect", _ReadOnlyConnectGuard(sqlite3.connect))
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


def _create(client: TestClient, title: str, body: str, **extra) -> dict:
    response = client.post("/notes", json={"title": title, "body": body, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def _db(home: Path) -> Path:
    return canonical_database_path(home / "neospike", "memory")


# 1 ----------------------------------------------------------------------------------------
def test_post_note_goes_through_cli_and_is_searchable_while_dashboard_db_is_read_only(client: TestClient, home: Path) -> None:
    note = _create(client, "Garden plan", "# Garden plan\n\nZebra violinist marmalade 한글 본문.", tags=["home"])
    assert note["author"] == "human" and note["rev"] == 1 and note["created"] is True
    commands = api.subprocess.run.commands
    assert any("note-save" in command and command[:2] == [sys.executable, "-c"] for command in commands)

    found = client.get("/notes", params={"q": "zebra violinist marmalade 한글 본문"}).json()["notes"]
    assert [item["id"] for item in found] == [note["id"]]
    assert any("note-search" in command for command in api.subprocess.run.commands)
    assert client.get("/notes", params={"tag": "home"}).json()["notes"][0]["id"] == note["id"]
    assert client.get("/notes", params={"tag": "other"}).json()["notes"] == []
    assert client.get("/notes/titles", params={"prefix": "gar"}).json()["titles"] == ["Garden plan"]
    assert client.get(f"/notes/{note['id']}").json()["body"].endswith("한글 본문.")

    assert sqlite3.connect.targets and all("mode=ro" in target for target in sqlite3.connect.targets)
    with api._RoConnection() as connection:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("INSERT INTO meta_entries(key,value) VALUES ('probe','1')")

    deleted = client.delete(f"/notes/{note['id']}", params={"expected_rev": 1})
    assert deleted.status_code == 200 and deleted.json()["deleted"] is True
    assert client.get("/notes").json()["notes"] == []
    assert client.get("/notes", params={"q": "zebra violinist marmalade 한글 본문"}).json()["notes"] == []
    assert client.get(f"/notes/{note['id']}").status_code == 404


def test_docs_and_notes_tabs_list_search_and_export_by_author(client: TestClient) -> None:
    """The Docs tab asks for author human, the Notes tab for agent and formula notes."""
    _create(client, "Doc page", "# Doc page\n\nOtter accordion lantern 문서 본문.")
    code, _ = api._cli_call(["note-save"], {"title": "TASK_7", "body": "# TASK_7\n\nOtter accordion lantern 작업 본문.",
                                            "author": "agent"})
    assert code == 0
    docs = client.get("/notes", params={"authors": "human"}).json()
    notes = client.get("/notes", params={"authors": "agent,formula"}).json()
    assert ([item["title"] for item in docs["notes"]], docs["total"]) == (["Doc page"], 1)
    assert ([item["title"] for item in notes["notes"]], notes["total"]) == (["TASK_7"], 1)
    assert client.get("/notes").json()["total"] == 2  # no authors: every note, as before
    every = client.get("/notes", params={"q": "otter accordion lantern 본문"}).json()["notes"]
    found = client.get("/notes", params={"q": "otter accordion lantern 본문", "authors": "human"}).json()["notes"]
    assert sorted(item["title"] for item in every) == ["Doc page", "TASK_7"]  # the search finds both
    assert [item["title"] for item in found] == ["Doc page"]
    assert client.get("/notes", params={"authors": "robot"}).status_code == 422
    exported = client.get("/notes/export", params={"authors": "human"}).json()
    assert [item["path"].rsplit("/", 1)[-1] for item in exported] == ["Doc page.md"]


# 2 ----------------------------------------------------------------------------------------
def test_stale_expected_rev_returns_409_with_current_body(client: TestClient) -> None:
    note = _create(client, "Plan", "First body of the plan.")
    updated = client.put(f"/notes/{note['id']}", json={"title": "Plan", "body": "Second body.", "expected_rev": 1})
    assert updated.status_code == 200 and updated.json()["rev"] == 2
    stale = client.put(f"/notes/{note['id']}", json={"title": "Plan", "body": "Lost update.", "expected_rev": 1})
    assert stale.status_code == 409
    detail = stale.json()["detail"]
    assert detail["error"] == "conflict" and detail["current_rev"] == 2
    assert detail["current"]["body"] == "Second body." and detail["current"]["rev"] == 2


# 3 ----------------------------------------------------------------------------------------
def test_secret_body_returns_422_with_line_numbers_and_db_is_unchanged(client: TestClient, home: Path) -> None:
    note = _create(client, "Setup", "# Setup\n\nConnect to the staging server.")
    body = "# Setup\n\nConnect to the staging server.\napi_key = abcdef123456\nThen restart."
    rejected = client.put(f"/notes/{note['id']}", json={"title": "Setup", "body": body, "expected_rev": 1})
    assert rejected.status_code == 422
    assert rejected.json()["detail"]["error"] == "policy" and rejected.json()["detail"]["lines"] == [4]
    current = client.get(f"/notes/{note['id']}").json()
    assert current["rev"] == 1 and current["body"] == "# Setup\n\nConnect to the staging server."
    with closing(sqlite3.connect(f"file:{_db(home).as_posix()}?mode=ro", uri=True)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM item_revisions").fetchone()[0] == 0


# 4 ----------------------------------------------------------------------------------------
def test_backlink_outgoing_broken_and_similar_notes(client: TestClient) -> None:
    target = _create(client, "B", "# B\n\nCopper lantern harbor notes.")
    source = _create(client, "A", "# A\n\nSee [[B]] and [[Missing page]]. Copper lantern harbor.")
    other = _create(client, "C", "# C\n\nViolet thunder candle orchard.")
    related = client.get(f"/notes/{target['id']}/related").json()
    assert related["backlinks"] == [{"type": "note", "id": source["id"], "title": "A"}]
    assert [item["id"] for item in related["similar"]][:1] == [source["id"]]
    assert len(related["similar"]) <= 5 and target["id"] not in [item["id"] for item in related["similar"]]
    outgoing = client.get(f"/notes/{source['id']}/related").json()
    assert outgoing["outgoing"] == [{"type": "note", "id": target["id"], "title": "B"}]
    assert outgoing["broken"] == [{"type": "note", "id": None, "title": "Missing page"}]
    assert client.get(f"/notes/{other['id']}/related").json()["backlinks"] == []


# 5 ----------------------------------------------------------------------------------------
def test_restore_raises_rev_and_brings_back_the_chosen_body(client: TestClient) -> None:
    note = _create(client, "Recipe", "Original recipe body.")
    assert client.put(f"/notes/{note['id']}", json={"title": "Recipe", "body": "Edited recipe body.", "expected_rev": 1}).status_code == 200
    history = client.get(f"/notes/{note['id']}/revisions").json()
    assert history["current_rev"] == 2 and [item["rev"] for item in history["revisions"]] == [1]
    restored = client.post(f"/notes/{note['id']}/restore", json={"rev": 1})
    assert restored.status_code == 200 and restored.json()["rev"] == 3
    assert client.get(f"/notes/{note['id']}").json()["body"] == "Original recipe body."
    assert [item["rev"] for item in client.get(f"/notes/{note['id']}/revisions").json()["revisions"]] == [2, 1]
    assert client.post(f"/notes/{note['id']}/restore", json={"rev": 1, "expected_rev": 2}).status_code == 409
    assert client.post(f"/notes/{note['id']}/restore", json={"rev": 99}).status_code == 404


def test_memory_item_edit_uses_text_cas_and_delete(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from neospike.tests.test_notes import BowEmbedder
    original = "Project Falcon deploys from the release branch only."
    seed = MemoryStore(_db(home), dimensions=DIMENSIONS)
    try:
        item_id = seed.insert(
            NAMESPACE, original, BowEmbedder().embed_document(original), source="agent:s1", kind="memory_item",
            project="D:/work/falcon", metadata={"scope": "project", "session_id": "s1", "subject": "Falcon deploys", "durability": "stable"},
        )
    finally:
        seed.close()
    client = TestClient(_app(monkeypatch))
    listed = client.get("/memories").json()["items"]
    assert [(item["id"], item["editable"], item["scope"]) for item in listed] == [(item_id, True, "project")]
    edited = "Project Falcon deploys from the release branch after review."
    assert client.put(f"/memories/{item_id}", json={"expected_text": original, "new_text": edited}).json() == {"updated": True, "id": item_id}
    stale = client.put(f"/memories/{item_id}", json={"expected_text": original, "new_text": "Another rewrite of the Falcon rule."})
    assert stale.status_code == 409 and stale.json()["detail"]["current_text"] == edited
    assert client.get("/memories", params={"q": "review"}).json()["items"][0]["text"] == edited
    assert client.delete(f"/memories/{item_id}").status_code == 200
    assert client.delete(f"/memories/{item_id}").status_code == 404


def _app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    monkeypatch.setattr(api, "_cli_argv_base", _child_prefix)
    monkeypatch.setattr(sqlite3, "connect", _ReadOnlyConnectGuard(sqlite3.connect))
    app = FastAPI()
    app.include_router(api.router)
    return app


def test_note_routes_work_before_the_database_exists(client: TestClient) -> None:
    assert client.get("/notes").json() == {"notes": []}
    assert client.get("/notes/titles").json() == {"titles": []}
    assert client.get("/memories").json() == {"items": []}
    assert client.get("/inbox").status_code == 404  # there is no inbox API
    assert client.get("/notes/1").status_code == 404


# 8 ----------------------------------------------------------------------------------------
def test_dashboard_ui_has_four_tabs_and_note_contract() -> None:
    script = (DASHBOARD / "dist" / "index.js").read_text(encoding="utf-8")
    for label in ('"tab.setup": "Setup"', '"tab.notes": "Notes"', '"tab.memory": "Memory"'):
        assert label in script
    # Map is the first (default) tab and Setup the last.
    tab_order = [script.index(f'{{ value: "{value}", label:') for value in ("map", "notes", "docs", "memory", "conversations", "config", "setup")]
    assert '"tab.docs": "Docs"' in script and '"tab.docs": "문서"' in script
    assert '"kind.human": "Docs"' in script and '"kind.human": "문서"' in script and "Human notes" not in script
    assert tab_order == sorted(tab_order) and 'defaultValue: "map"' in script
    for route in ("/notes", "/notes/titles?prefix=", "/related", "/revisions", "/restore", "/memories"):
        assert route in script
    assert '"/inbox' not in script and "tab.inbox" not in script
    for component in ("Tabs", "TabsList", "TabsTrigger", "ConfirmDialog"):
        assert component in script
    assert 'h("textarea"' in script and "expected_rev" in script and "=>" not in script
