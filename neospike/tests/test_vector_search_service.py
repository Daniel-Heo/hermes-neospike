"""Vector search in the embedding service: the wire answer equals the full comparison, a search does not wait behind
an embedding, and a service started from other code is replaced (real model; test homes only)."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from neospike import inference, vector_index
from neospike.inference import EmbeddingClient, EmbeddingDaemon, EmbeddingUnavailable
from neospike.store import MemoryStore, dot_similarities
from neospike.tests.support_embed_service import (
    ON_WINDOWS,
    kill_and_wait,
    poll_until,
    profile_root,
    spawn_python,
    spawn_service,
    stop_daemon_of,
    wait_model_loaded,
)
from neospike.tests.test_wincompat import read_line, remove_tree

DIMENSIONS = 1024
ROWS = 1000
SEED = 528
NAMESPACE = "owner"
TOP = 10
THRESHOLD = 0.2
CLUSTERS = 40
NOISE = 0.6
RESCORE_TOLERANCE = 1e-12
ROUNDS = 20
LONG_TEXT_WORDS = 400  # about 800 tokens: a document embedding of a noticeable length (DirectML hung on 1,800)
QUEUED_DOCUMENTS = 12
SEARCH_WHILE_EMBEDDING_SECONDS = 1.0
START_SECONDS = 180
CLIENTS = 2
PACKAGE_COPY_DIR = "older-code"
_PACKAGE = Path(inference.__file__).resolve().parent
_SKIPPED_COPY = shutil.ignore_patterns("tests", "__pycache__", "dist", "node_modules")

CLIENT_CODE = """
import json, sys
from neospike.inference import EmbeddingClient
client = EmbeddingClient(sys.argv[1], startup_timeout=float(sys.argv[2]))
client.ensure_daemon()
print(json.dumps(client.request('ping')), flush=True)
"""


@pytest.fixture
def svc_home(tmp_path, monkeypatch):
    with profile_root(tmp_path, monkeypatch) as root:
        home = root / "profile"
        home.mkdir()
        yield home
        stop_daemon_of(home)
        if ON_WINDOWS:
            remove_tree(home)


def _vectors(count: int, seed: int = SEED) -> np.ndarray:
    generator = np.random.default_rng(seed)
    centers = generator.normal(size=(CLUSTERS, DIMENSIONS))
    rows = centers[generator.integers(0, CLUSTERS, count)] + NOISE * generator.normal(size=(count, DIMENSIONS))
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    return rows.astype(np.float32)


def _memories(store: MemoryStore, vectors: np.ndarray) -> list[int]:
    old = time.time() - 60  # an existing database: outside the recheck window of the next sync
    rows = [(NAMESPACE, hashlib.sha256(f"m{index}".encode()).hexdigest(), f"m {index}", "agent:s", vector.tobytes(),
             old, old, "memory_item", 1.0, 0.5, "", json.dumps({})) for index, vector in enumerate(vectors)]
    with store._write_scope():
        store._conn.executemany(
            "INSERT INTO items(type,owner,text_hash,text,origin,embedding,created_at,updated_at,kind,confidence,"
            "importance,project,metadata_json) VALUES ('memory',?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    return [int(row[0]) for row in store._conn.execute("SELECT id FROM items WHERE type='memory' ORDER BY id")]


class ClientFor:
    """vector_index._search_client for one service home."""

    def __init__(self, home: Path) -> None:
        self.home = home

    def __call__(self) -> EmbeddingClient:
        return EmbeddingClient(self.home, startup_timeout=START_SECONDS)


def _search_ms(store: MemoryStore, ids: list[int], query: np.ndarray) -> float:
    started = time.perf_counter()
    for _ in range(ROUNDS):
        vector_index.similarities(store._conn, "items", ids, query, threshold=THRESHOLD)
    return (time.perf_counter() - started) / ROUNDS * 1000


def test_spawned_service_search_matches_the_full_comparison(svc_home, tmp_path, monkeypatch) -> None:
    """Another process's answer: the top N and every row over the floor with their exact cosines, NaN elsewhere
    (also for an allowed id whose row is gone)."""
    service = spawn_service(svc_home)
    try:
        assert '"ready"' in read_line(service.stdout, START_SECONDS)
        monkeypatch.setattr(vector_index, "_search_client", ClientFor(svc_home))
        store = MemoryStore(tmp_path / "memory.db", dimensions=DIMENSIONS)
        try:
            vectors = _vectors(ROWS + 1)
            ids = _memories(store, vectors[:ROWS])
            allowed = ids[::3] + [max(ids) + 1000]  # a part of the rows, plus one id with no row
            query = vectors[ROWS]
            exact = np.asarray(dot_similarities(query, [vectors[ids.index(rowid)].tobytes()
                                                        for rowid in allowed[:-1]]) + [math.nan])
            top = vector_index.similarities(store._conn, "items", allowed, query, limit=TOP)
            got = {index for index, score in enumerate(top) if math.isfinite(score)}
            assert got == set(np.argsort(-np.nan_to_num(exact, nan=-9))[:TOP].tolist())
            assert all(abs(top[index] - exact[index]) < RESCORE_TOLERANCE for index in got)
            ranged = vector_index.similarities(store._conn, "items", allowed, query, threshold=THRESHOLD)
            above = set(np.flatnonzero(np.nan_to_num(exact, nan=-9) >= THRESHOLD).tolist())
            assert above and {index for index, score in enumerate(ranged) if math.isfinite(score)} == above
            assert all(abs(ranged[index] - exact[index]) < RESCORE_TOLERANCE for index in above)
            assert math.isnan(top[-1]) and math.isnan(ranged[-1])
            many = vector_index.similarities_many(store._conn, "items", allowed, [query, vectors[0]], limit=TOP)
            assert many[0] == top and len(many[1]) == len(allowed)
        finally:
            store.close()
    finally:
        EmbeddingClient(svc_home).request_shutdown()
        service.wait(timeout=START_SECONDS)


def test_bad_search_requests_are_refused(svc_home, tmp_path) -> None:
    daemon = EmbeddingDaemon(svc_home)
    daemon.start()
    try:
        client = EmbeddingClient(svc_home, startup_timeout=START_SECONDS)
        good = {"path": str(tmp_path / "missing.db"), "table": "items", "ids": [1], "vectors": [[1.0]], "limit": 1}
        bad = [good, {**good, "path": "relative.db"}]
        database = tmp_path / "memory.db"
        MemoryStore(database, dimensions=DIMENSIONS).close()
        real = {**good, "path": str(database)}
        bad += [{**real, "table": "sqlite_master"}, {**real, "column": "text"}, {**real, "ids": [True]},
                {**real, "ids": list(range(vector_index_max_ids() + 1))}, {**real, "vectors": [[1.0], [1.0, 2.0]]},
                {**real, "vectors": []}, {**real, "limit": 0}, {**real, "limit": None}]
        for fields in bad:
            with pytest.raises(EmbeddingUnavailable):
                client.request(vector_index.SEARCH_OP, **fields)
        assert client.request(vector_index.SEARCH_OP, **real) == {"results": [{"ids": [], "scores": []}]}
    finally:
        daemon.close()


def vector_index_max_ids() -> int:
    from neospike.vector_search import MAX_IDS
    return MAX_IDS


class EmbedJob:
    """Runs one document embedding and records when it ended."""

    def __init__(self, client: EmbeddingClient, text: str) -> None:
        self.client, self.text = client, text
        self.ended = None
        self.error = None

    def __call__(self) -> None:
        try:
            self.client.embed_document(self.text)
        except Exception as error:  # reported by the test
            self.error = error
        self.ended = time.perf_counter()


class WorkerBusy:
    """Predicate: the model thread has taken a job from its queue."""

    def __init__(self, daemon: EmbeddingDaemon, queued: int) -> None:
        self.daemon, self.queued = daemon, queued

    def __call__(self) -> bool:
        return self.daemon.jobs.qsize() < self.queued


def test_search_does_not_wait_behind_embeddings(svc_home, tmp_path, monkeypatch) -> None:
    """Real model: while document embeddings run on the model thread, a search answers from its own thread.
    Prints the search time of 1,000 stored 1,024-wide vectors, idle and during the embeddings."""
    daemon = EmbeddingDaemon(svc_home)
    daemon.start()
    store = MemoryStore(tmp_path / "memory.db", dimensions=DIMENSIONS)
    try:
        client = EmbeddingClient(svc_home, timeout=inference._MAX_REQUEST_SECONDS, startup_timeout=START_SECONDS)
        client.embed_query("warm")  # waits for the model load
        monkeypatch.setattr(vector_index, "_search_client", ClientFor(svc_home))
        vectors = _vectors(ROWS + 1)
        ids = _memories(store, vectors[:ROWS])
        query = vectors[ROWS]
        vector_index.similarities(store._conn, "items", ids, query, threshold=THRESHOLD)  # build
        idle_ms = _search_ms(store, ids, query)
        text = " ".join(f"word{index % 97}" for index in range(LONG_TEXT_WORDS))
        jobs = [EmbedJob(client, f"{index} {text}") for index in range(QUEUED_DOCUMENTS)]
        threads = [threading.Thread(target=job) for job in jobs]
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        assert poll_until(WorkerBusy(daemon, QUEUED_DOCUMENTS), START_SECONDS)
        searched_at = time.perf_counter()
        busy = vector_index.similarities(store._conn, "items", ids, query, threshold=THRESHOLD)
        search_seconds = time.perf_counter() - searched_at
        for thread in threads:
            thread.join(START_SECONDS)
        ends = [job.ended for job in jobs]
        assert all(ends) and not [job.error for job in jobs if job.error], [job.error for job in jobs]
        embed_seconds = max(ends) - started
        print(f"measure search {ROWS}x{DIMENSIONS}: idle {idle_ms:.1f}ms, during embedding "
              f"{search_seconds * 1000:.1f}ms ({QUEUED_DOCUMENTS} documents took {embed_seconds:.2f}s)")
        assert any(math.isfinite(score) for score in busy)
        assert searched_at + search_seconds < max(ends)  # answered while the embeddings still ran
        assert search_seconds < SEARCH_WHILE_EMBEDDING_SECONDS
    finally:
        store.close()
        daemon.close()


def _older_code_service(home: Path, root: Path):
    """A service started from a copy of the package (other file stamps: another fingerprint)."""
    copy_root = root / PACKAGE_COPY_DIR
    shutil.copytree(_PACKAGE, copy_root / _PACKAGE.name, ignore=_SKIPPED_COPY, copy_function=shutil.copyfile)
    service = spawn_service(home, cwd=str(copy_root))
    assert '"ready"' in read_line(service.stdout, START_SECONDS)
    wait_model_loaded(service)
    return service


def _ping(home: Path) -> dict:
    return EmbeddingClient(home).request("ping")


def test_same_code_keeps_the_service(svc_home) -> None:
    service = spawn_service(svc_home)
    try:
        assert '"ready"' in read_line(service.stdout, START_SECONDS)
        before = _ping(svc_home)
        assert before["fingerprint"] == inference.code_fingerprint()
        EmbeddingClient(svc_home, startup_timeout=START_SECONDS).ensure_daemon()
        assert _ping(svc_home)["pid"] == before["pid"]
    finally:
        EmbeddingClient(svc_home).request_shutdown()
        service.wait(timeout=START_SECONDS)


def test_other_code_is_replaced_once_by_two_clients_at_once(svc_home) -> None:
    """Two client processes find a service of other code: it stops, one new service starts and both use it."""
    old = _older_code_service(svc_home, svc_home.parent)
    try:
        old_ping = _ping(svc_home)
        assert old_ping["fingerprint"] != inference.code_fingerprint()
        clients = [spawn_python(CLIENT_CODE, str(svc_home), str(START_SECONDS)) for _ in range(CLIENTS)]
        answers = []
        for client in clients:
            out, err = client.communicate(timeout=START_SECONDS * 2)
            assert client.returncode == 0, err
            answers.append(json.loads(out.strip().splitlines()[-1]))
        assert old.wait(timeout=START_SECONDS) == 0
        assert len({answer["pid"] for answer in answers}) == 1
        assert answers[0]["pid"] != old_ping["pid"]
        assert answers[0]["fingerprint"] == inference.code_fingerprint()
        assert _ping(svc_home)["pid"] == answers[0]["pid"]
    finally:
        kill_and_wait(old)


def test_search_on_a_service_without_search_replaces_it(svc_home, tmp_path, monkeypatch) -> None:
    """A search reaching a service that does not know the operation (older code) replaces it, then answers."""
    old = _older_code_service(svc_home, tmp_path)
    try:
        old_pid = _ping(svc_home)["pid"]
        store = MemoryStore(tmp_path / "memory.db", dimensions=DIMENSIONS)
        try:
            vectors = _vectors(TOP + 1)
            ids = _memories(store, vectors[:TOP])
            refusing = RefusingClientFor(svc_home)
            monkeypatch.setattr(vector_index, "_search_client", refusing)
            scores = vector_index.similarities(store._conn, "items", ids, vectors[TOP], limit=TOP)
        finally:
            store.close()
        assert refusing.refused and all(math.isfinite(score) for score in scores)
        assert old.wait(timeout=START_SECONDS) == 0
        assert _ping(svc_home)["pid"] != old_pid
    finally:
        kill_and_wait(old)


def test_a_service_left_in_a_test_home_is_stopped_by_the_cleanup(svc_home) -> None:
    """What a test (or a child process of it) started under its tmp_path does not outlive it (root conftest)."""
    import conftest
    service = spawn_service(svc_home)
    try:
        assert '"ready"' in read_line(service.stdout, START_SECONDS)
        assert conftest.stop_services_under(svc_home.parent) == [svc_home]
        assert service.wait(timeout=START_SECONDS) == 0
        assert conftest.stop_services_under(svc_home.parent) == []
    finally:
        kill_and_wait(service)


class RefusingClientFor(ClientFor):
    """Clients whose first search gets the answer a service of older code gives (it has no such operation):
    stands in for an old service without keeping old code around."""

    def __init__(self, home: Path) -> None:
        super().__init__(home)
        self.refused = False

    def __call__(self) -> EmbeddingClient:
        return _RefusingClient(self, self.home, startup_timeout=START_SECONDS)


class _RefusingClient(EmbeddingClient):
    def __init__(self, owner: RefusingClientFor, home: Path, **options) -> None:
        super().__init__(home, **options)
        self.owner = owner

    def _exchange(self, op, *args, **kwargs):
        if op == vector_index.SEARCH_OP and not self.owner.refused:
            self.owner.refused = True
            raise EmbeddingUnavailable(inference._ERR["bad_op"])
        return super()._exchange(op, *args, **kwargs)
