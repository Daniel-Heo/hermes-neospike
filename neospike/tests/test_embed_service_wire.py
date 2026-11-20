"""Embedding service over the wire: framing, singleton, queue priority, auth, shutdown (fake backends only)."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from neospike.tests.support_embed_service import (
    ON_WINDOWS,
    GatedBackend,
    RecordingBackend,
    cache_dir,
    kill_and_wait,
    ping_pid,
    poll_until,
    profile_root,
    spawn_service,
    stop_daemon_of,
)

from neospike._wincompat import connect_socket
from neospike import inference
from neospike.inference import (
    INPUT_CHAR_CAP,
    EmbeddingClient,
    EmbeddingDaemon,
    _read_frame,
    _write_frame,
)
from neospike.tests.test_wincompat import (
    graceful_popen_kwargs,
    read_line,
    remove_tree,
    service_pid,
    stop_gracefully,
)

# Singleton-service load: 8 clients at once.
PARALLEL_CLIENTS = 8


@pytest.fixture
def svc_home(tmp_path, monkeypatch):
    with profile_root(tmp_path, monkeypatch) as root:
        home = root / 'profile'
        home.mkdir()
        yield home
        stop_daemon_of(home)
        if ON_WINDOWS:
            remove_tree(home)


class QueueDepth:
    """Predicate: the daemon queue holds exactly n jobs."""

    def __init__(self, daemon, n):
        self.daemon, self.n = daemon, n

    def __call__(self):
        return self.daemon.jobs.qsize() == self.n


class QueueDepthOrDone(QueueDepth):
    def __init__(self, daemon, n, future):
        super().__init__(daemon, n)
        self.future = future

    def __call__(self):
        return super().__call__() or self.future.done()


@pytest.mark.parametrize('glyph', ['\u00e9', '\U0001F600'], ids=['two-byte', 'four-byte'])
def test_input_of_exactly_the_character_cap_round_trips(svc_home, glyph):
    backend = RecordingBackend([1.0])
    daemon = EmbeddingDaemon(svc_home, backend=backend)
    daemon.start()
    try:
        payload = glyph * INPUT_CHAR_CAP
        assert EmbeddingClient(svc_home).embed_document(payload) == [1.0]
        assert backend.values() == [payload]
    finally:
        daemon.close()


def test_one_daemon_per_profile_serves_parallel_queries_and_cleans_up(svc_home):
    backend = RecordingBackend([1.0, 0.0])
    daemon = EmbeddingDaemon(svc_home, backend=backend)
    daemon.start()
    try:
        endpoint = daemon.socket_path
        if ON_WINDOWS:
            # loopback TCP: the endpoint file only publishes the port, the token still guards it
            assert int(endpoint.read_text(encoding='ascii')) > 0
        else:
            assert endpoint.stat().st_mode & 0o777 == 0o600
        with pytest.raises(RuntimeError, match='already running'):
            EmbeddingDaemon(svc_home, backend=backend).start()
        client = EmbeddingClient(svc_home)
        texts = [f'q{n}' for n in range(12)]
        with ThreadPoolExecutor(max_workers=PARALLEL_CLIENTS) as pool:
            answers = list(pool.map(client.embed_query, texts))
        assert answers == [[1.0, 0.0]] * len(texts)
        assert sorted(backend.values()) == sorted(texts)
    finally:
        daemon.close()
    assert not daemon.socket_path.exists()


# A ping reply this late misses one poll slice (inference._POLL_SLICE) but fits the startup budget.
SLOW_PING_SECONDS = 0.6


class _SlowPingDaemon(EmbeddingDaemon):
    """Answers every connection only after SLOW_PING_SECONDS: alive, but busy."""

    def _serve_connection(self, conn):
        time.sleep(SLOW_PING_SECONDS)
        super()._serve_connection(conn)


def test_busy_daemon_that_answers_after_a_poll_slice_is_not_a_failure(svc_home):
    """A live daemon slower than one slice does not make ensure_daemon raise TimeoutError."""
    assert inference._POLL_SLICE < SLOW_PING_SECONDS
    daemon = _SlowPingDaemon(svc_home, backend=RecordingBackend([1.0, 0.0]))
    daemon.start()
    try:
        assert EmbeddingClient(svc_home, startup_timeout=5).embed_query('busy') == [1.0, 0.0]
        with pytest.raises(TimeoutError, match='startup deadline'):
            EmbeddingClient(svc_home, startup_timeout=SLOW_PING_SECONDS / 2).ensure_daemon()
    finally:
        daemon.close()

def test_queue_reserves_room_for_interactive_and_serves_it_before_indexing(svc_home):
    backend = GatedBackend('first', [1.0, 0.0])
    daemon = EmbeddingDaemon(svc_home, backend=backend, queue_size=1)
    daemon.start()
    client = EmbeddingClient(svc_home, timeout=3)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            head = pool.submit(client.embed_document, 'first')
            assert backend.entered.wait(2)
            indexing = pool.submit(client.embed_document, 'second')
            poll_until(QueueDepth(daemon, 1), 2)
            interactive = pool.submit(client.embed_query, 'urgent')
            poll_until(QueueDepthOrDone(daemon, 2, interactive), 1)
            try:
                assert not interactive.done(), 'a full indexing queue turned away an interactive query'
                with pytest.raises(RuntimeError, match='queue full'):
                    client.embed_document('overflow')
            finally:
                backend.release.set()
            assert head.result() == indexing.result() == interactive.result() == [1.0, 0.0]
        assert backend.values() == ['first', 'urgent', 'second']
    finally:
        backend.release.set()
        daemon.close()


def test_bad_token_and_expired_deadline_leave_the_single_worker_intact(svc_home):
    backend = GatedBackend('stall', [1.0])
    daemon = EmbeddingDaemon(svc_home, backend=backend)
    daemon.start()
    try:
        with connect_socket(daemon.socket_path, 5) as conn:
            _write_frame(conn, {'token': 'not-it', 'op': 'query', 'value': 'classified'})
            assert 'Unauthorized' in _read_frame(conn)['error']
        with ThreadPoolExecutor(max_workers=2) as pool:
            stalled = pool.submit(EmbeddingClient(svc_home, timeout=.2).embed_query, 'stall')
            assert backend.entered.wait(2)
            with pytest.raises(RuntimeError, match='deadline'):
                EmbeddingClient(svc_home, timeout=.05).embed_query('late')
            with pytest.raises(RuntimeError, match='deadline'):
                stalled.result(timeout=2)
            assert ping_pid(EmbeddingClient(svc_home)) > 0
            backend.release.set()
        assert EmbeddingClient(svc_home).embed_query('next') == [1.0]
        assert backend.values() == ['stall', 'next']  # 'classified' and 'late' never reached the model
    finally:
        backend.release.set()
        daemon.close()


def test_local_model_runner_is_shared_between_clients_and_freed_on_close(svc_home):
    from neospike.embedder import ONNX_MODEL_FILE, resolve_model_dir
    if not (resolve_model_dir() / ONNX_MODEL_FILE).is_file():
        pytest.skip('Local model weights not installed')
    daemon = EmbeddingDaemon(svc_home)
    daemon.start()
    try:
        vector = EmbeddingClient(svc_home).embed_query('approved red company logo')
        model = daemon.backend.text
        assert len(vector) == 1024
        assert abs(sum(x * x for x in vector) - 1) < .001
        assert len(EmbeddingClient(svc_home).embed_document('ACME NORTH')) == 1024
        assert daemon.backend.text is model
    finally:
        daemon.close()
    assert daemon.backend.text is None


def test_provider_wires_the_profile_client_and_loads_no_model(svc_home):
    from neospike import NeospikeProvider
    provider = NeospikeProvider()
    provider.initialize('desktop', hermes_home=str(svc_home))
    try:
        assert isinstance(provider._vectorizer, EmbeddingClient)
        assert provider._vectorizer.directory == svc_home / 'neospike' / '.cache'
    finally:
        provider.shutdown()


def test_restarted_service_issues_a_new_token_and_unknown_profile_has_no_service(svc_home):
    first = spawn_service(svc_home)
    try:
        assert 'ready' in read_line(first.stdout, 10)
        old_token = (cache_dir(svc_home) / 'token').read_text()
    finally:
        kill_and_wait(first, 10)
    second = spawn_service(svc_home)
    try:
        ready = read_line(second.stdout, 10)
        assert 'ready' in ready, second.stderr.read()
        assert (cache_dir(svc_home) / 'token').read_text() != old_token
        with connect_socket(cache_dir(svc_home) / 'i.sock', 5) as conn:
            _write_frame(conn, {'token': old_token, 'op': 'ping'})
            assert 'Unauthorized' in _read_frame(conn)['error']
        with pytest.raises(FileNotFoundError):
            EmbeddingClient(svc_home / 'unrelated-profile').request('ping')
        assert ping_pid(EmbeddingClient(svc_home)) == service_pid(second, ready)
    finally:
        second.terminate()
        second.wait(timeout=10)


def test_service_process_is_a_singleton_and_exits_cleanly_on_stop(svc_home):
    process = spawn_service(svc_home, **graceful_popen_kwargs())
    try:
        ready = read_line(process.stdout, 10)
        assert 'ready' in ready, process.stderr.read()
        pid = service_pid(process, ready)
        client = EmbeddingClient(svc_home)
        seen = set()
        guard = threading.Lock()

        def record_ping():
            found = ping_pid(client)
            with guard:
                seen.add(found)

        workers = [threading.Thread(target=record_ping) for _ in range(16)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(10)
        assert seen == {pid}
        second = spawn_service(svc_home)
        _, err = second.communicate(timeout=10)
        assert second.returncode != 0
        assert 'already running' in err
        assert ping_pid(client) == pid
    finally:
        stop_gracefully(process)
        process.wait(timeout=10)
    assert process.returncode == 0
    assert not (cache_dir(svc_home) / 'i.sock').exists()


def test_profile_homes_resolve_to_the_shared_root_unless_overridden(tmp_path, monkeypatch):
    root = tmp_path / 'hermes'
    expected = root.resolve() / 'neospike' / '.cache'
    monkeypatch.delenv('NEOSPIKE_INFERENCE_HOME', raising=False)
    assert EmbeddingClient(root / 'profiles' / 'lotto').directory == expected
    assert EmbeddingClient(root).directory == expected
    override = tmp_path / 'elsewhere'
    monkeypatch.setenv('NEOSPIKE_INFERENCE_HOME', str(override))
    assert EmbeddingClient(root / 'profiles' / 'lotto').directory == override.resolve() / 'neospike' / '.cache'


@pytest.mark.parametrize('bad', [0, -1, float('inf'), float('nan')])
def test_client_rejects_unusable_startup_timeout(tmp_path, bad):
    with pytest.raises(ValueError, match='Startup timeout'):
        EmbeddingClient(tmp_path, startup_timeout=bad)
