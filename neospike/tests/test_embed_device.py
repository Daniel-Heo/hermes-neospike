"""Embedding device: DirectML when it works, otherwise the CPU; the daemon loads the model at start and
reports the device (fake onnxruntime and fake backends only)."""
from __future__ import annotations

import re

import pytest
from packaging.requirements import Requirement

from neospike import embedder
from neospike.inference import EmbeddingClient, EmbeddingDaemon
from neospike.tests.support_embed_service import (
    ON_WINDOWS,
    RecordingBackend,
    poll_until,
    profile_root,
    stop_daemon_of,
)
from neospike.tests.test_install_check import REPO_ROOT
from neospike.tests.test_wincompat import remove_tree

CPU = "CPUExecutionProvider"
DML = "DmlExecutionProvider"
MODEL_PATH = "model.onnx"
THREADS = 4
PROBE_FEED = {"input_ids": [[0, 2]]}
WARM_SECONDS = 5


class _FakeOptions:
    def __init__(self) -> None:
        self.enable_mem_pattern = True
        self.execution_mode = "parallel"
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0


class _FakeExecutionMode:
    ORT_SEQUENTIAL = "sequential"


class _FakeSession:
    def __init__(self, runtime: "_FakeRuntime", options: _FakeOptions, providers: list) -> None:
        self.runtime = runtime
        self.options = options
        self.wanted = [entry if isinstance(entry, str) else entry[0] for entry in providers]
        self.provider_options = dict(entry for entry in providers if not isinstance(entry, str))
        self.runs = 0

    def get_providers(self) -> list[str]:
        if DML in self.wanted and self.runtime.session_drops_directml:
            return [CPU]
        return list(self.wanted)

    def run(self, _outputs, _feed):
        if DML in self.wanted and self.runtime.directml_run_fails:
            raise RuntimeError("device removed")
        self.runs += 1
        return [[[1.0]]]


class _FakeRuntime:
    """Stand-in for the onnxruntime module: which providers exist and how a DirectML session behaves."""

    ExecutionMode = _FakeExecutionMode
    SessionOptions = _FakeOptions

    def __init__(self, *, directml: bool, create_fails: bool = False, run_fails: bool = False,
                 session_drops_directml: bool = False) -> None:
        self.available = [DML, CPU] if directml else [CPU]
        self.directml_create_fails = create_fails
        self.directml_run_fails = run_fails
        self.session_drops_directml = session_drops_directml
        self.sessions: list[_FakeSession] = []

    def get_available_providers(self) -> list[str]:
        return list(self.available)

    def InferenceSession(self, _path, *, sess_options, providers):  # noqa: N802 (onnxruntime's name)
        session = _FakeSession(self, sess_options, providers)
        if DML in session.wanted and self.directml_create_fails:
            raise RuntimeError("no DirectX 12 adapter")
        self.sessions.append(session)
        return session


def _open(monkeypatch, runtime: _FakeRuntime, device: str = embedder.DEVICE_AUTO):
    monkeypatch.setattr(embedder, "ort", runtime)
    return embedder.open_session(MODEL_PATH, THREADS, device, PROBE_FEED)


def test_directml_is_taken_when_its_session_runs(monkeypatch) -> None:
    runtime = _FakeRuntime(directml=True)
    session, device, note = _open(monkeypatch, runtime)
    assert (device, note) == (embedder.DEVICE_DIRECTML, "")
    assert session.wanted == [DML, CPU] and session.provider_options == {DML: {"device_id": 0}}
    # DirectML requires both settings; the probe ran once on the session that is kept.
    assert session.options.enable_mem_pattern is False and session.options.execution_mode == "sequential"
    assert session.runs == 1 and runtime.sessions == [session]


def test_build_without_directml_runs_on_the_cpu_with_the_old_settings(monkeypatch) -> None:
    runtime = _FakeRuntime(directml=False)
    session, device, note = _open(monkeypatch, runtime)
    assert device == embedder.DEVICE_CPU and "not in this onnxruntime build" in note
    assert session.wanted == [CPU] and session.options.enable_mem_pattern is True
    assert (session.options.intra_op_num_threads, session.options.inter_op_num_threads) == (THREADS, 1)
    assert session.runs == 1


@pytest.mark.parametrize("failure", ["create_fails", "run_fails", "session_drops_directml"])
def test_a_directml_session_that_fails_falls_back_to_the_cpu(monkeypatch, failure) -> None:
    runtime = _FakeRuntime(directml=True, **{failure: True})
    session, device, note = _open(monkeypatch, runtime)
    assert device == embedder.DEVICE_CPU and note.startswith("DirectML failed")
    assert session.wanted == [CPU] and session.runs == 1


def test_cpu_request_never_opens_a_directml_session(monkeypatch) -> None:
    runtime = _FakeRuntime(directml=True)
    session, device, note = _open(monkeypatch, runtime, embedder.DEVICE_CPU)
    assert (device, note) == (embedder.DEVICE_CPU, "")
    assert [made.wanted for made in runtime.sessions] == [[CPU]] and session.wanted == [CPU]


@pytest.mark.parametrize("runtime", [_FakeRuntime(directml=False), _FakeRuntime(directml=True, create_fails=True)])
def test_directml_request_is_an_error_when_it_cannot_run(monkeypatch, runtime) -> None:
    with pytest.raises(embedder.DeviceUnavailable):
        _open(monkeypatch, runtime, embedder.DEVICE_DIRECTML)
    assert runtime.sessions == []


def test_unknown_device_is_refused(monkeypatch) -> None:
    with pytest.raises(ValueError, match="Unknown embedding device"):
        _open(monkeypatch, _FakeRuntime(directml=True), "cuda")


class _WarmBackend(RecordingBackend):
    """Backend with the model runner's warm()/device: the device is known only after warm()."""

    def __init__(self, *, warm_error: Exception | None = None) -> None:
        super().__init__()
        self.device = None
        self.device_note = None
        self.warm_error = warm_error
        self.warm_calls = 0

    def warm(self) -> None:
        self.warm_calls += 1
        if self.warm_error is not None:
            raise self.warm_error
        self.device, self.device_note = embedder.DEVICE_DIRECTML, ""


class _Warmed:
    def __init__(self, backend: _WarmBackend) -> None:
        self.backend = backend

    def __call__(self) -> bool:
        return self.backend.warm_calls == 1


@pytest.fixture
def svc_home(tmp_path, monkeypatch):
    with profile_root(tmp_path, monkeypatch) as root:
        home = root / "profile"
        home.mkdir()
        yield home
        stop_daemon_of(home)
        if ON_WINDOWS:
            remove_tree(home)


def test_daemon_loads_the_model_at_start_and_ping_reports_the_device(svc_home) -> None:
    backend = _WarmBackend()
    daemon = EmbeddingDaemon(svc_home, backend=backend)
    daemon.start()
    try:
        assert poll_until(_Warmed(backend), WARM_SECONDS)  # no request was sent
        client = EmbeddingClient(svc_home)
        assert client.device() == embedder.DEVICE_DIRECTML
        assert client.request("ping")["device"] == embedder.DEVICE_DIRECTML
        assert client.embed_query("hello") == [1.0] and backend.warm_calls == 1
    finally:
        daemon.close()


def test_failed_warm_up_leaves_the_daemon_serving(svc_home) -> None:
    backend = _WarmBackend(warm_error=RuntimeError("model load failed"))
    daemon = EmbeddingDaemon(svc_home, backend=backend)
    daemon.start()
    try:
        assert poll_until(_Warmed(backend), WARM_SECONDS)
        client = EmbeddingClient(svc_home)
        assert client.device() is None
        assert client.embed_query("hello") == [1.0]
    finally:
        daemon.close()


class _Platform:
    """Marker environment of one system."""

    def __init__(self, sys_platform: str) -> None:
        self.environment = {"sys_platform": sys_platform}

    def applies(self, requirement: Requirement) -> bool:
        return requirement.marker is None or requirement.marker.evaluate(self.environment)


def _onnxruntime_requirements() -> list[Requirement]:
    text = (REPO_ROOT / "neospike" / "plugin.yaml").read_text(encoding="utf-8-sig")
    return [Requirement(spec) for spec in re.findall(r'- "(onnxruntime[^"]*)"', text)]


@pytest.mark.parametrize("sys_platform, package",
                         [("win32", "onnxruntime-directml"), ("linux", "onnxruntime"), ("darwin", "onnxruntime")])
def test_each_system_installs_exactly_one_onnxruntime_package(sys_platform, package) -> None:
    # Both packages write the same onnxruntime module folder; two in one environment overwrite each other.
    platform = _Platform(sys_platform)
    assert [item.name for item in _onnxruntime_requirements() if platform.applies(item)] == [package]


def test_both_onnxruntime_packages_share_one_version_range() -> None:
    assert {str(item.specifier) for item in _onnxruntime_requirements()} == {"<2,>=1.20"}


def test_backend_without_warm_up_reports_no_device(svc_home) -> None:
    daemon = EmbeddingDaemon(svc_home, backend=RecordingBackend())
    daemon.start()
    try:
        assert EmbeddingClient(svc_home).device() is None
    finally:
        daemon.close()
