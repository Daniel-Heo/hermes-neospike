"""GET /map: layout, neighbours, folders, colours, cache; the map runs in the CLI child."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from neospike import folders as folder_family
from neospike import sphere_map
from neospike.config import NeospikeConfig
from neospike.dashboard import plugin_api as api
from neospike.database import canonical_database_path
from neospike.inference import EmbeddingClient
from neospike.notes import NoteService
from neospike.service import NeospikeService
from neospike.store import MemoryStore
from neospike.tests.test_dashboard_notes import _CHILD_BOOT, _ReadOnlyConnectGuard, _child_prefix
from neospike.tests.test_notes import DIMENSIONS, BowEmbedder
from neospike.tests.support_items import chunk_rows

NAMESPACE = "owner"
SYNTHETIC_MEMORIES = 300
SEED = 350
UNIT_TOLERANCE = 1e-6
DAY = 86400.0


def _random_vectors(count: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(count, DIMENSIONS))
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


@pytest.fixture()
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "hermes"
    NeospikeConfig().save(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    sphere_map.clear_cache()
    return home


def _db(home: Path) -> Path:
    return canonical_database_path(home / "neospike", "memory")


def _seed(home: Path, count: int) -> tuple[list[int], list[dict]]:
    """count durable memories with random vectors, three notes (one TASK_, one linking), one folder."""
    store = MemoryStore(_db(home), dimensions=DIMENSIONS)
    try:
        memories = [store.insert(NAMESPACE, f"Synthetic memory number {index}.", vector.tolist(), source="agent:s",
                                 kind="durable") for index, vector in enumerate(_random_vectors(count))]
        notes = NoteService(store, BowEmbedder())
        saved = [notes.save("Garden plan", "# Garden\nTomatoes basil chili compost.", author="human"),
                 notes.save("TASK_900 build", "# Build\nCompile the garden tool [[Garden plan]].", author="agent"),
                 notes.save("Kitchen", "# Kitchen\nBasil pesto and chili oil.", author="agent")]
        folder = folder_family.create_folder(store._conn, "Garden")
        folder_family.move_items(store._conn, folder["id"],
                                 [("memory", memories[0]), ("memory", memories[1]), ("note", saved[0]["id"])])
    finally:
        store.close()
    return memories, saved


def _client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # /map runs `neospike.cli map` in a child (this interpreter, stubbed Hermes core); this process stays read-only.
    monkeypatch.setattr(api, "_cli_argv_base", _child_prefix)
    monkeypatch.setattr(sqlite3, "connect", _ReadOnlyConnectGuard(sqlite3.connect))
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app)


def _full_vectors(home: Path, points: list[dict]) -> np.ndarray:
    """Independent read of the item vectors in the payload's order (note = normalized chunk mean)."""
    with closing(sqlite3.connect(f"file:{_db(home).as_posix()}?mode=ro", uri=True)) as connection:
        rows = []
        for point in points:
            if point["lane"] == "memory":
                blob = connection.execute("SELECT embedding FROM items WHERE type='memory' AND id=?", (point["id"],)).fetchone()[0]
                vector = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
            else:
                connection.row_factory = sqlite3.Row
                chunks = [np.frombuffer(row["embedding"], dtype=np.float32).astype(np.float64)
                          for row in chunk_rows(connection, point["id"])]
                connection.row_factory = None
                vector = np.sum(chunks, axis=0)
            rows.append(vector / np.linalg.norm(vector))
    return np.vstack(rows)


def _reference_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Covariance eigenvectors (not the module's SVD) with the same sign rule."""
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    values, vectors = np.linalg.eigh(centred.T @ centred)
    axes = vectors[:, np.argsort(values)[::-1][:3]].T
    for row in axes:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return mean, axes


# 1 + 3 -------------------------------------------------------------------------------------
def test_map_points_are_unit_deterministic_and_db_file_is_unchanged(home: Path, monkeypatch) -> None:
    _seed(home, 40)
    before = hashlib.sha256(_db(home).read_bytes()).hexdigest()
    client = _client(monkeypatch)
    first = client.get("/map").json()
    sphere_map.clear_cache()
    second = client.get("/map").json()
    third = client.get("/map").json()
    assert hashlib.sha256(_db(home).read_bytes()).hexdigest() == before
    # The dashboard process opens no database for /map (the CLI child does, mode=ro); the guard rejects any write.
    assert all("mode=ro" in target for target in sqlite3.connect.targets)
    assert len(first["points"]) == 43 and first["cached"] is False and third["cached"] is True
    for point in first["points"]:
        assert abs(np.linalg.norm([point["x"], point["y"], point["z"]]) - 1.0) <= UNIT_TOLERANCE
    coords = [[(p["key"], p["x"], p["y"], p["z"]) for p in payload["points"]] for payload in (first, second, third)]
    assert coords[0] == coords[1] == coords[2] and first["hash"] == second["hash"]
    kinds = {p["title"]: p["kind"] for p in first["points"] if p["lane"] == "note"}
    assert kinds == {"Garden plan": "human", "TASK_900 build": "task", "Kitchen": "agent"}
    garden = next(p for p in first["points"] if p["title"] == "Garden plan")
    task = next(p for p in first["points"] if p["kind"] == "task")
    assert garden["backlinks"] == 1 and task["key"] in garden["links"] and garden["key"] in task["links"]
    assert all(p["superseded"] is False for p in first["points"])
    folder = first["folders"][0]
    assert folder["count"] == 3 and folder["color"] == folder_family.MEMBER_PALETTE[0]
    assert {p["key"] for p in first["points"] if p["folder_id"] == folder["id"]} >= {garden["key"]}


# 2 -----------------------------------------------------------------------------------------
def test_neighbors_and_folder_position_match_numpy_reference(home: Path, monkeypatch) -> None:
    memories, saved = _seed(home, 60)
    payload = _client(monkeypatch).get("/map").json()
    points = payload["points"]
    matrix = _full_vectors(home, points)
    keys = [point["key"] for point in points]
    scores = matrix @ matrix.T
    np.fill_diagonal(scores, -np.inf)
    for index, point in enumerate(points):
        expected = [keys[other] for other in np.argsort(-scores[index], kind="stable")[:5]]
        assert point["neighbors"] == expected, point["key"]
    mean, axes = _reference_pca(matrix)
    projected = (matrix - mean) @ axes.T
    projected /= np.linalg.norm(projected, axis=1, keepdims=True)
    got = np.array([[p["x"], p["y"], p["z"]] for p in points])
    assert np.abs(got - projected).max() <= UNIT_TOLERANCE
    member_keys = {f"mem:{memories[0]}", f"mem:{memories[1]}", f"note:{saved[0]['id']}"}
    members = [index for index, key in enumerate(keys) if key in member_keys]
    position = (matrix[members].mean(axis=0) - mean) @ axes.T
    position /= np.linalg.norm(position)
    folder = payload["folders"][0]
    assert np.abs(np.array([folder["x"], folder["y"], folder["z"]]) - position).max() <= UNIT_TOLERANCE


# 6-1 ---------------------------------------------------------------------------------------
def test_searched_note_is_recent_and_eight_day_old_items_are_not(home: Path, monkeypatch) -> None:
    memories, saved = _seed(home, 10)
    old = time.time() - 8 * DAY
    with closing(sqlite3.connect(_db(home))) as connection, connection:
        connection.execute("UPDATE items SET updated_at=?", (old,))
    client = _client(monkeypatch)
    before = {p["key"]: p["recent"] for p in client.get("/map").json()["points"]}
    assert not any(before.values())
    monkeypatch.undo()  # the search below writes through the store (CLI/agent process side)
    store = MemoryStore(_db(home), dimensions=DIMENSIONS)
    try:
        hits = NeospikeService(store=store, embedder=BowEmbedder(), namespace=NAMESPACE).search_notes("basil pesto chili oil")
        assert hits and hits[0]["title"] == "Kitchen"
        row = store._conn.execute("SELECT accessed_at,access_count,updated_at FROM items WHERE id=?",
                                  (saved[2]["id"],)).fetchone()
        assert row["accessed_at"] > old + 7 * DAY and row["access_count"] >= 1 and row["updated_at"] == old
    finally:
        store.close()
    monkeypatch.setenv("HERMES_HOME", str(home))
    after = {p["key"]: p["recent"] for p in _client(monkeypatch).get("/map").json()["points"]}
    assert after[f"note:{saved[2]['id']}"] is True
    assert after[f"mem:{memories[0]}"] is False
    assert {key for key, recent in after.items() if recent} == {f"note:{hit['id']}" for hit in hits}


# Folder recency -----------------------------------------------------------------------------------
def test_folder_recency_three_steps(home: Path) -> None:
    """Member use times 3, 10 and 30 days ago -> recent, mid, old (folder use = latest member max(accessed, updated))."""
    now = 2_000_000_000.0
    store = MemoryStore(_db(home), dimensions=DIMENSIONS)
    try:
        vectors = _random_vectors(5)
        ids = [store.insert(NAMESPACE, f"Recency memory {index}.", vector.tolist(), source="agent:s", kind="durable")
               for index, vector in enumerate(vectors)]
        # (updated_at, accessed_at) in days ago; the later of the two counts.
        ages = {ids[0]: (40, 3), ids[1]: (10, 50), ids[2]: (30, 30), ids[3]: (60, 60), ids[4]: (1, 1)}
        for memory_id, (updated, accessed) in ages.items():
            store._conn.execute("UPDATE items SET updated_at=?, accessed_at=? WHERE id=?",
                                (now - updated * DAY, now - accessed * DAY, memory_id))
        store._conn.commit()
        folder_ids = []
        for name, members in (("Three", [ids[0], ids[3]]), ("Ten", [ids[1]]), ("Thirty", [ids[2]]), ("Empty", [])):
            folder = folder_family.create_folder(store._conn, name)
            folder_ids.append(folder["id"])
            if members:
                folder_family.move_items(store._conn, folder["id"], [("memory", item) for item in members])
    finally:
        store.close()
    with closing(sqlite3.connect(f"file:{_db(home).as_posix()}?mode=ro", uri=True)) as connection:
        payload = sphere_map.build_map(connection, now=now)
    by_name = {folder["name"]: folder for folder in payload["folders"]}
    assert [by_name[name]["recency"] for name in ("Three", "Ten", "Thirty", "Empty")] == ["recent", "mid", "old", "old"]
    assert by_name["Three"]["used_at"] == now - 3 * DAY and by_name["Empty"]["used_at"] is None
    assert payload["recent_days"] == 7.0 and payload["stale_days"] == 14.0
    # The windows are parameters: a 2-day recent window turns the 3-day folder into mid.
    with closing(sqlite3.connect(f"file:{_db(home).as_posix()}?mode=ro", uri=True)) as connection:
        narrow = sphere_map.build_map(connection, recent_window_days=2.0, stale_window_days=5.0, now=now)
    assert {folder["name"]: folder["recency"] for folder in narrow["folders"]}["Three"] == "mid"


def test_recency_boundaries() -> None:
    now = 1_000.0 * DAY
    assert sphere_map.recency(now - 7 * DAY, now, 7.0, 14.0) == "recent"
    assert sphere_map.recency(now - 7 * DAY - 1, now, 7.0, 14.0) == "mid"
    assert sphere_map.recency(now - 14 * DAY, now, 7.0, 14.0) == "mid"
    assert sphere_map.recency(now - 14 * DAY - 1, now, 7.0, 14.0) == "old"
    assert sphere_map.recency(None, now, 7.0, 14.0) == "old"


# Recency days settings -----------------------------------------------------------------------------------
def test_map_days_survive_config_save_and_dashboard_save(home: Path, monkeypatch) -> None:
    NeospikeConfig(map_recent_days=3.0, map_stale_days=9.0).save(home)
    loaded = NeospikeConfig.load(home)
    assert (loaded.map_recent_days, loaded.map_stale_days) == (3.0, 9.0)
    assert NeospikeConfig().map_recent_days == 7.0 and NeospikeConfig().map_stale_days == 14.0
    app = FastAPI()
    app.include_router(api.router)
    saved = TestClient(app).post("/setup/config", json={"embedding_dimensions": DIMENSIONS, "turn_trace_enabled": True})
    assert saved.status_code == 200, saved.text
    reloaded = NeospikeConfig.load(home)
    assert (reloaded.map_recent_days, reloaded.map_stale_days, reloaded.turn_trace_enabled) == (3.0, 9.0, True)
    payload = _client(monkeypatch).get("/map").json()
    assert (payload["recent_days"], payload["stale_days"]) == (3.0, 9.0)
    with pytest.raises(ValueError):
        NeospikeConfig.from_values({"map_recent_days": 0})


# 9 -----------------------------------------------------------------------------------------
def test_map_time_for_300_synthetic_memories(home: Path, monkeypatch, capsys) -> None:
    _seed(home, SYNTHETIC_MEMORIES)
    with closing(sqlite3.connect(f"file:{_db(home).as_posix()}?mode=ro", uri=True)) as connection:
        started = time.perf_counter()
        cold = sphere_map.build_map(connection)
        cold_seconds = time.perf_counter() - started
        started = time.perf_counter()
        warm = sphere_map.build_map(connection)
        warm_seconds = time.perf_counter() - started
    assert len(cold["points"]) == SYNTHETIC_MEMORIES + 3 and warm["cached"] is True
    with capsys.disabled():
        print(f"\n[map timing] items={len(cold['points'])} cold={cold_seconds * 1000:.1f}ms warm={warm_seconds * 1000:.1f}ms")


# 8 -----------------------------------------------------------------------------------------
def test_map_ui_syntax_contract() -> None:
    import shutil
    import subprocess
    dashboard = Path(api.__file__).resolve().parent
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    checked = subprocess.run([node, "--check", str(dashboard / "dist" / "index.js")], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    script = (dashboard / "dist" / "index.js").read_text(encoding="utf-8")
    for needle in ('"tab.map": "Map"', "/map", "/folders/move", '"common.moveToFolder": "Move to folder"', '"map.recentOnly": "Recent only"', '"folder.new": "New folder"',
                   "#F5C518", "#B08E11", "#6E590B", "#E53935", "#FB8C00", "#B9F6CA", "#1E88E5", "noteTitle(title, body)",
                   '"member.expand": "Expand on map"', "backAlpha: 0.25", 'getContext("2d")', "h(MemoEditor", "h(MemoryEntryCard"):
        assert needle in script, needle
    assert "=>" not in script
    assert "#9AA0A6" not in script
    assert "#22C55E" not in script
    assert "#43A047" not in script and "#69F0AE" not in script  # plain notes are #B9F6CA
    assert "neospike-recent-dot" not in script and "MAP_COLORS.recent" not in script  # no ring


# Map view state ---------------------------------------------------------------------------------
def test_map_view_pure_state_with_node() -> None:
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = Path(__file__).resolve().parent / "test_map_view.js"
    checked = subprocess.run([node, str(script)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert "test_map_view: ok" in checked.stdout


# CLI child and cache -------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
VOLATILE_FIELDS = ("cached", "seconds")
MAP_FIELDS = {"points", "folders", "hash", "recent_days", "stale_days", "cached", "seconds"}
POINT_FIELDS = {"key", "id", "lane", "kind", "title", "folder_id", "x", "y", "z", "backlinks", "neighbors",
                "links", "recent", "superseded", "updated_at"}
FOLDER_FIELDS = {"id", "name", "color", "origin", "count", "x", "y", "z", "used_at", "recency"}
CHILD_TIMEOUT_SECONDS = 180
# Runs first in a fresh interpreter: any later `import numpy` raises, so the code under test cannot reach it.
_NUMPY_BLOCKER = (
    "import sys\n"
    "class _NoNumpy:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name == 'numpy' or name.startswith('numpy.'):\n"
    "            raise ImportError('numpy is blocked in this test')\n"
    "        return None\n"
    "sys.meta_path.insert(0, _NoNumpy())\n"
    "assert 'numpy' not in sys.modules\n"
)
_CACHE_HIT_SCRIPT = _NUMPY_BLOCKER + (
    "import json, sqlite3\n"
    "from pathlib import Path\n"
    "from neospike import sphere_map\n"
    "home = Path(sys.argv[1])\n"
    "db = (home / 'neospike' / 'memory.db').as_posix()\n"
    "connection = sqlite3.connect(f'file:{db}?mode=ro', uri=True)\n"
    "try:\n"
    "    payload = sphere_map.build_map(connection, cache_dir=sphere_map.cache_directory(home))\n"
    "except ImportError as exc:\n"
    "    payload = {'error': str(exc)}\n"
    "payload['numpy_loaded'] = 'numpy' in sys.modules\n"
    "sys.stdout.write(json.dumps(payload))\n"
)
_ROUTE_SCRIPT = _NUMPY_BLOCKER + (
    "import json\n"
    "from fastapi import FastAPI\n"
    "from fastapi.testclient import TestClient\n"
    "from neospike.dashboard import plugin_api as api\n"
    "imported_numpy = 'numpy' in sys.modules\n"
    "def _prefix():\n"
    "    return [sys.executable, '-c', sys.argv[1]]\n"  # the CLI child is a separate, unblocked interpreter
    "api._cli_argv_base = _prefix\n"
    "app = FastAPI()\n"
    "app.include_router(api.router)\n"
    "response = TestClient(app).get('/map')\n"
    "sys.stdout.write(json.dumps({'status': response.status_code, 'body': response.json(),\n"
    "                             'imported_numpy': imported_numpy, 'numpy_loaded': 'numpy' in sys.modules}))\n"
)


def _child_env(home: Path) -> dict[str, str]:
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(REPO_ROOT), "PYTHONIOENCODING": "utf-8"}
    env.pop("HERMES_PYTHON", None)
    return env


def _run_cli_map(home: Path) -> dict:
    result = subprocess.run([*_child_prefix(), "--home", str(home), "map"], capture_output=True,
                            timeout=CHILD_TIMEOUT_SECONDS, env=_child_env(home), cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    return json.loads(result.stdout.decode("utf-8"))


def _run_script(script: str, home: Path, *arguments: str) -> dict:
    result = subprocess.run([sys.executable, "-c", script, *arguments], capture_output=True,
                            timeout=CHILD_TIMEOUT_SECONDS, env=_child_env(home), cwd=REPO_ROOT)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    return json.loads(result.stdout.decode("utf-8"))


def _stable(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key not in VOLATILE_FIELDS}


# Route shape and cache file ------------------------------------------------------------------------------------
def test_cli_map_has_the_route_shape_and_keeps_one_cache_file(home: Path) -> None:
    _seed(home, 40)
    cache_dir = sphere_map.cache_directory(home)
    cache_dir.mkdir(parents=True)
    stale = cache_dir / "map-0000.json"
    stale.write_text("{}", encoding="utf-8")
    payload = _run_cli_map(home)
    assert set(payload) == MAP_FIELDS and payload["cached"] is False and len(payload["points"]) == 43
    assert all(set(point) == POINT_FIELDS for point in payload["points"])
    assert payload["folders"] and all(set(folder) == FOLDER_FIELDS for folder in payload["folders"])
    assert [path.name for path in cache_dir.glob("map-*.json")] == [f"map-{payload['hash']}.json"]
    assert not stale.exists()
    # Same payload as the in-process build (no file cache) that the dashboard route serves.
    with closing(sqlite3.connect(f"file:{_db(home).as_posix()}?mode=ro", uri=True)) as connection:
        direct = sphere_map.build_map(connection)
    assert _stable(payload) == _stable(direct)


def test_map_cache_never_creates_the_embedding_daemon_folder(home: Path) -> None:
    """A plain mkdir of the daemon's .cache (inherited ACL) made ensure_daemon refuse it forever."""
    _seed(home, 40)
    payload = _run_cli_map(home)
    daemon_dir = EmbeddingClient(home).directory
    assert sphere_map.cache_file(sphere_map.cache_directory(home), payload["hash"]).is_file()
    assert sphere_map.cache_directory(home) != daemon_dir
    assert not (home / "neospike" / ".cache").exists() and not daemon_dir.exists()


# Cache hit without numpy ------------------------------------------------------------------------------------
def test_second_call_hits_the_cache_file_without_importing_numpy(home: Path) -> None:
    _seed(home, 40)
    first = _run_cli_map(home)
    second = _run_cli_map(home)
    assert first["cached"] is False and second["cached"] is True and _stable(first) == _stable(second)
    hit = _run_script(_CACHE_HIT_SCRIPT, home, str(home))
    assert hit.pop("numpy_loaded") is False and hit["cached"] is True
    assert _stable(hit) == _stable(first)
    # Control: without the file the same blocked process must reach numpy (the blocker is effective).
    for path in sphere_map.cache_directory(home).glob("map-*.json"):
        path.unlink()
    miss = _run_script(_CACHE_HIT_SCRIPT, home, str(home))
    assert "numpy is blocked" in miss["error"]


# Dashboard route without numpy ------------------------------------------------------------------------------------
def test_dashboard_route_imports_and_serves_map_without_numpy(home: Path) -> None:
    _seed(home, 40)
    expected = _run_cli_map(home)
    routed = _run_script(_ROUTE_SCRIPT, home, _CHILD_BOOT)
    assert routed["imported_numpy"] is False and routed["numpy_loaded"] is False
    assert routed["status"] == 200, routed["body"]
    # The route adds nothing to the CLI's points (topic points come from the CLI; none seeded here).
    assert _stable(routed["body"]) == _stable(expected)


class _FailingCli:
    def __init__(self, message: str) -> None:
        self.message = message

    def __call__(self, arguments, payload=None, *, namespace=None):
        assert arguments == ["map"]
        return 1, {"error": "unavailable", "message": self.message}


def test_map_route_passes_the_cli_error_message(home: Path, monkeypatch) -> None:
    message = "The map needs numpy in the Hermes plugin Python: No module named 'numpy'"
    monkeypatch.setattr(api, "_cli_call", _FailingCli(message))
    app = FastAPI()
    app.include_router(api.router)
    response = TestClient(app).get("/map")
    assert response.status_code == 503 and response.json() == {"detail": message}


# Map status ------------------------------------------------------------------------------------
_MAP_STATUS_SCRIPT = (
    "const assert = require('node:assert/strict');"
    "const view = require(process.argv[1]);"
    "assert.equal(view.mapStatusText(false, null, 0), null);"
    "assert.equal(view.mapStatusText(true, null, 0), 'No memories or notes with embeddings yet.');"
    "assert.equal(view.mapStatusText(true, null, 3), null);"
    "assert.equal(view.mapStatusText(true, 'The map needs numpy', 0), 'Could not load the map: The map needs numpy');"
    "assert.equal(view.mapStatusText(true, 'boom', 5), 'Could not load the map: boom');"
    "console.log('map status: ok');"
)


def test_map_error_text_replaces_the_empty_notice_with_node() -> None:
    import shutil
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = Path(api.__file__).resolve().parent / "dist" / "index.js"
    checked = subprocess.run([node, "-e", _MAP_STATUS_SCRIPT, str(script)], capture_output=True, text=True)
    assert checked.returncode == 0, checked.stderr
    assert "map status: ok" in checked.stdout


# Timing ------------------------------------------------------------------------------------
def test_map_route_time_for_300_items_including_the_cli_child(home: Path, monkeypatch, capsys) -> None:
    _seed(home, SYNTHETIC_MEMORIES)
    client = _client(monkeypatch)
    started = time.perf_counter()
    cold = client.get("/map").json()
    cold_seconds = time.perf_counter() - started
    started = time.perf_counter()
    warm = client.get("/map").json()
    warm_seconds = time.perf_counter() - started
    assert len(cold["points"]) == SYNTHETIC_MEMORIES + 3 and cold["cached"] is False and warm["cached"] is True
    with capsys.disabled():
        print(f"\n[map route timing] items={len(cold['points'])} cold={cold_seconds * 1000:.1f}ms "
              f"(build {cold['seconds'] * 1000:.1f}ms) warm={warm_seconds * 1000:.1f}ms "
              f"(build {warm['seconds'] * 1000:.1f}ms)")


# Point kinds --------------------------------------------------------------------------------------
def _map_point_kinds() -> set[str]:
    """Every point kind the map backend can emit: memory, each note author, task, topic."""
    from neospike.store import NOTE_AUTHOR_NAMES
    return {"memory", "task", sphere_map.TOPIC_LANE, *NOTE_AUTHOR_NAMES}


def _object_keys(script: str, prefix: str) -> set[str]:
    import re
    start = script.index(prefix) + len(prefix)
    body = script[start:script.index("}", start)]
    return set(re.findall(r"(\w+)\s*:", body))


def _missing_map_kinds(script: str) -> list[str]:
    """Kinds the frontend would hide (kinds default), colour wrongly (KIND_CLASSES) or leave unnamed."""
    import re
    shown = _object_keys(script, "const [kinds, setKinds] = useState({")
    classed = _object_keys(script, "const KIND_CLASSES = {")
    boxes = set(re.findall(r'\{ value: "(\w+)", label: "kind\.\1" \}', script))
    missing = []
    for kind in sorted(_map_point_kinds()):
        if kind not in shown:
            missing.append(f"kinds default: {kind}")
        if kind == sphere_map.TOPIC_LANE:
            continue  # topic colour comes from the lane (isTopic) and has no checkbox
        if kind not in classed:
            missing.append(f"KIND_CLASSES: {kind}")
        if kind not in boxes:
            missing.append(f"kind checkbox: {kind}")
        if script.count(f'"kind.{kind}":') != 2:
            missing.append(f"en/ko label: kind.{kind}")
    return missing


def test_frontend_knows_every_map_point_kind_including_formula() -> None:
    dashboard = Path(api.__file__).resolve().parent
    script = (dashboard / "dist" / "index.js").read_text(encoding="utf-8")
    assert "formula" in _map_point_kinds()
    assert _missing_map_kinds(script) == []
    assert 'const KIND_CLASSES = { memory: "other", human: "doc", agent: "note", formula: "note"' in script
    # The check itself must catch a frontend without formula.
    without = script.replace("formula: true, ", "").replace('formula: "note", ', "")
    assert "kinds default: formula" in _missing_map_kinds(without)
    assert "KIND_CLASSES: formula" in _missing_map_kinds(without)


# Panel hooks order -----------------------------------------------------------------------
_HOOK_CALL = r"(?<![\w.])use[A-Z]\w*\("


def _component_body(lines: list[str], name: str) -> tuple[list[str], str]:
    start = next(index for index, line in enumerate(lines) if line.lstrip().startswith(f"function {name}("))
    indent = len(lines[start]) - len(lines[start].lstrip())
    closing = " " * indent + "}"
    end = next(index for index in range(start + 1, len(lines)) if lines[index].rstrip() == closing)
    return lines[start + 1:end], " " * (indent + 2)


def _hooks_after_return(script: str, names: tuple[str, ...]) -> list[str]:
    """Hook calls at a component's top level after its first top-level return (or `if` block that returns)."""
    import re
    lines = script.splitlines()
    found = []
    for name in names:
        body, pad = _component_body(lines, name)
        returned = False
        for index, line in enumerate(body):
            if not (line.startswith(pad) and not line.startswith(pad + " ")):
                continue
            text = line.strip()
            if returned and re.search(_HOOK_CALL, text):
                found.append(f"{name}: {text}")
            elif text.startswith("return") or (text.startswith("if (") and "return" in text):
                returned = True
            elif text.startswith("if (") and text.endswith("{"):
                block_end = next(k for k in range(index + 1, len(body)) if body[k].startswith(pad + "}"))
                returned = returned or any("return" in body[k] for k in range(index + 1, block_end))
    return found


def test_map_panel_hooks_run_in_the_same_order_for_topics_and_items() -> None:
    dashboard = Path(api.__file__).resolve().parent
    script = (dashboard / "dist" / "index.js").read_text(encoding="utf-8")
    assert _hooks_after_return(script, ("ViewEntry",)) == []  # a hook after an early return blanks the tab
    assert _hooks_after_return(script, ("ThemeView", "RecordView", "ThemeEntry", "LinkedRoll")) == []
    # A PanelItem with a hook after an early return must be caught.
    old = ("  function ViewEntry(props) {\n    const panel = props.panel;\n"
           "    const connections = useConnectionList(panel.type, panel.id, panel.seq, props.version);\n"
           "    if (panel.type === MAP_VIEW.TOPIC_LANE) {\n      return h(ThemeEntry, { id: panel.id });\n    }\n"
           "    const [reloads, setReloads] = useState(0);\n    return null;\n  }\n")
    assert _hooks_after_return(old, ("ViewEntry",)) == ["ViewEntry: const [reloads, setReloads] = useState(0);"]
