"""Memory archive is human only - nothing is archived by age.

An archived memory is out of prefetch, visible to direct search and the dashboard; Unarchive undoes it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neospike import cli
from neospike.config import EMBEDDING_DIMENSIONS, NeospikeConfig
from neospike.provider import NeospikeProvider
from neospike.store import MemoryStore
from neospike.tests.test_dashboard_notes import _db, client, home  # noqa: F401 (fixtures)
from neospike.tests.test_notes import DIMENSIONS, BowEmbedder

SESSION = "session-archive"
ONGOING = "The pilot uses the OTTER approval mark for release."
STABLE = "The pilot always signs with the HERON seal."
KEYWORDS = ("OTTER", "HERON")


class Embedder:
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def embed_document(self, text: str, **_kwargs) -> list[float]:
        axes = [float(word in text) for word in KEYWORDS]
        if not any(axes):
            axes = [0.0] * len(KEYWORDS) + [1.0]
        return axes + [0.0] * (self.dimensions - len(axes))

    embed_query = embed_document

    def take_warnings(self) -> list[str]:
        return []


@pytest.fixture
def provider(tmp_path: Path):
    home_dir = tmp_path / "home"
    NeospikeConfig().save(home_dir)
    instance = NeospikeProvider(embedder=Embedder())
    instance.initialize(SESSION, hermes_home=str(home_dir), cwd=str(home_dir))
    yield instance
    instance.shutdown()


def _call(provider: NeospikeProvider, name: str, args: dict) -> dict:
    return json.loads(provider.handle_tool_call(name, args))


def _remember(provider: NeospikeProvider, text: str, durability: str) -> int:
    assert _call(provider, "neospike_memory_save", {"text": text, "scope": "global", "subject": "pilot",
                                                  "durability": durability, "kind": "fact", "basis": "user_said"}) == {"stored": True}
    return int(provider._memory_db._conn.execute("SELECT MAX(id) FROM items WHERE type='memory'").fetchone()[0])


def _run_cli(provider: NeospikeProvider, monkeypatch, capsys, *arguments: str) -> dict:
    monkeypatch.setattr(cli, "EmbeddingClient", Embedder)
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(provider._home_root),
                                      "--owner", provider._ns_label, *arguments])
    assert cli.main() == 0
    return json.loads(capsys.readouterr().out)


def _archived_at(store: MemoryStore, memory_id: int):
    return store._conn.execute("SELECT archived_at FROM items WHERE type='memory' AND id=?", (memory_id,)).fetchone()[0]


def _search(provider: NeospikeProvider, query: str) -> dict[int, dict]:
    return {item["id"]: item for item in _call(provider, "neospike_memory_search", {"query": query})["results"]
            if item.get("kind") == "memory_item"}


# 2. archive --id: out of prefetch, in direct search; unarchive: back in prefetch -----------------------
def test_archive_cli_and_unarchive(provider: NeospikeProvider, monkeypatch, capsys) -> None:
    ongoing = _remember(provider, ONGOING, "ongoing")
    stable = _remember(provider, STABLE, "stable")
    store = provider._memory_db

    assert _run_cli(provider, monkeypatch, capsys, "archive", "--id", str(ongoing)) == {"archived": True, "id": ongoing}
    assert _run_cli(provider, monkeypatch, capsys, "archive", "--id", str(ongoing))["archived"] is False
    assert store._conn.execute("SELECT COUNT(*) FROM items WHERE type='memory'").fetchone()[0] == 2  # nothing deleted
    assert ONGOING not in provider.prefetch("OTTER approval mark", session_id=SESSION)
    assert STABLE in provider.prefetch("HERON seal", session_id=SESSION)
    found = _search(provider, "OTTER approval mark")
    assert found[ongoing]["archived"] is True and "archived" not in _search(provider, "HERON seal")[stable]
    assert _archived_at(store, ongoing) is not None  # a search hit does not unarchive

    assert _run_cli(provider, monkeypatch, capsys, "unarchive", "--id", str(ongoing))["unarchived"] is True
    assert ONGOING in provider.prefetch("OTTER approval mark", session_id=SESSION)


@pytest.fixture
def item_home(home: Path) -> tuple[Path, int]:  # noqa: F811 (fixture)
    store = MemoryStore(_db(home), dimensions=DIMENSIONS)
    embedder = BowEmbedder()
    memory_id = store.insert("default:local", "Dashboard archive row", embedder.embed_document("Dashboard archive row"),
                             source="t", kind="memory_item", metadata={"scope": "global", "durability": "ongoing"})
    store.close()
    return home, memory_id


# 3. dashboard Memory tab: Archive and Unarchive buttons' endpoints ------------------------------------
def test_dashboard_archives_and_unarchives(item_home: tuple[Path, int], client: TestClient) -> None:  # noqa: F811
    _, memory_id = item_home
    [item] = client.get("/memories").json()["items"]
    assert item["id"] == memory_id and item["archived"] is False
    assert client.post(f"/memories/{memory_id}/unarchive").status_code == 409
    response = client.post(f"/memories/{memory_id}/archive")
    assert response.status_code == 200, response.text
    assert client.get(f"/memories/{memory_id}").json()["archived"] is True
    assert client.post(f"/memories/{memory_id}/archive").status_code == 409
    response = client.post(f"/memories/{memory_id}/unarchive")
    assert response.status_code == 200, response.text
    assert client.get(f"/memories/{memory_id}").json()["archived"] is False
    script = (Path(__file__).resolve().parents[1] / "dashboard" / "dist" / "index.js").read_text(encoding="utf-8")
    assert '"memory.unarchive": "Unarchive"' in script and '"memory.archive": "Archive"' in script and 't("memory.archive")' in script and 'markArchived("archive"' in script
