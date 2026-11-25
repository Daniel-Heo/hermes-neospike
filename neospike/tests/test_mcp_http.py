"""The shared neospike-mcp HTTP server: token gate, Host check, one server at a time, the stop of the previous
server (stop request and forced stop), the token file, the start from register().

Server processes run the real entry point (python -c boot -> mcp_server.main) on a free test port with the
bag-of-words embedder; no Hermes, no real HERMES_HOME. The helpers here are shared with test_mcp_server.py.
"""
from __future__ import annotations

import http.client
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from neospike import mcp_http
from neospike.config import NeospikeConfig
from neospike.mcp_headers import TOKEN_VARIABLE, authorization_headers, read_env_token

HAS_MCP = importlib.util.find_spec("mcp") is not None
needs_mcp = pytest.mark.skipif(not HAS_MCP, reason="optional extra [mcp] is not installed")
REPO = Path(__file__).resolve().parents[2]
TOKEN = "a" * 64
INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                         "clientInfo": {"name": "test", "version": "0"}}}
MCP_ACCEPT = "application/json, text/event-stream"
# Server process: repo on sys.path (the starter drops PYTHONPATH), stub Hermes core, BowEmbedder, real main().
SERVER_BOOT = (
    "import sys,types;sys.path.insert(0,sys.argv[1]);"
    "a=types.ModuleType('agent');m=types.ModuleType('agent.memory_provider');"
    "m.MemoryProvider=type('MemoryProvider',(),{});a.memory_provider=m;"
    "sys.modules.setdefault('agent',a);sys.modules.setdefault('agent.memory_provider',m);"
    "from neospike import mcp_server as s;from neospike.tests.test_notes import BowEmbedder;"
    "s.EmbeddingClient=BowEmbedder;"
    "sys.exit(s.main(sys.argv[2:]))"
)
# A process that only sleeps: stands in for a server that ignores the stop request, or a foreign program.
SLEEPER = "import time;time.sleep(120)"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((mcp_http.HOST, 0))
        return probe.getsockname()[1]


def prepare_home(home: Path, token: str | None = TOKEN, port: int | None = None) -> int:
    """A test home: config.json with a free mcp_port and (unless token is None) the token in .env."""
    port = port or free_port()
    NeospikeConfig(mcp_port=port).save(home)
    if token is not None:
        (home / ".env").write_text(f"OTHER=1\n{TOKEN_VARIABLE}={token}\n", encoding="utf-8")
    return port


def server_command(home: Path) -> list[str]:
    return [sys.executable, "-c", SERVER_BOOT, str(REPO), "--home", str(home)]


def stop_all(home: Path) -> None:
    """Teardown: the recorded server goes, whatever state the test left."""
    record = mcp_http.read_record(home)
    mcp_http.stop_server(home, read_env_token(home / ".env"))
    if record is not None and record.names_live_server():
        mcp_http.kill_process(record.pid)


class RunningServer:
    """`with RunningServer(home) as record:` a started server of *home* (prepare_home first)."""

    def __init__(self, home: Path) -> None:
        self.home = home

    def __enter__(self) -> mcp_http.ServerRecord:
        state = mcp_http.ensure_running(self.home, server_command(self.home))
        assert state == mcp_http.STARTED, (state, (self.home / "neospike" / ".cache" / "mcp_http.log").read_text(
            encoding="utf-8", errors="replace") if (self.home / "neospike" / ".cache" / "mcp_http.log").is_file()
            else "")
        return mcp_http.read_record(self.home)

    def __exit__(self, *exc_info) -> None:
        stop_all(self.home)


def raw_post(port: int, path: str, payload: dict, headers: dict) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection(mcp_http.HOST, port, timeout=10)
    try:
        connection.request("POST", path, body=json.dumps(payload).encode("utf-8"),
                           headers={"Content-Type": "application/json", "Accept": MCP_ACCEPT, **headers})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


@pytest.fixture()
def home(tmp_path: Path):
    home = tmp_path / "hermes"
    home.mkdir()
    yield home
    stop_all(home)


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", SLEEPER])


def _fake_record(home: Path, process: subprocess.Popen, port: int, start_shift: int = 0) -> mcp_http.ServerRecord:
    """A record naming *process* (its real start time and executable, unless start_shift moves the time)."""
    deadline = time.monotonic() + 10
    while mcp_http.process_start_time(process.pid) is None and time.monotonic() < deadline:
        time.sleep(0.05)
    record = mcp_http.ServerRecord(pid=process.pid, parent_pid=os.getpid(), port=port, started_at=time.time(),
                                   start_time=mcp_http.process_start_time(process.pid) + start_shift,
                                   executable=mcp_http.process_executable(process.pid))
    mcp_http.write_record(home, record)
    return record


# -- the gate and the Host check -----------------------------------------------------------------------------

@needs_mcp
def test_missing_or_wrong_token_gets_401_and_the_right_one_reaches_the_tools(home: Path) -> None:
    port = prepare_home(home)
    with RunningServer(home):
        for headers in ({}, authorization_headers("b" * 64), {"Authorization": TOKEN}):
            status, body = raw_post(port, mcp_http.MCP_PATH, INITIALIZE, headers)
            assert status == 401 and json.loads(body) == {"error": "unauthorized"}, headers
        status, body = raw_post(port, mcp_http.MCP_PATH, INITIALIZE, authorization_headers(TOKEN))
        assert status == 200 and json.loads(body)["result"]["serverInfo"]["name"] == "neospike-mcp"
        listed = raw_post(port, mcp_http.MCP_PATH, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                          authorization_headers(TOKEN))
        assert listed[0] == 200 and len(json.loads(listed[1])["result"]["tools"]) == 6  # stateless: no session
        assert raw_post(port, mcp_http.SHUTDOWN_PATH, {}, {})[0] == 401  # the stop path is gated too


@needs_mcp
def test_a_foreign_host_header_is_refused_even_with_the_token(home: Path) -> None:
    port = prepare_home(home)
    with RunningServer(home):
        status, _body = raw_post(port, mcp_http.MCP_PATH, INITIALIZE,
                                 {**authorization_headers(TOKEN), "Host": "evil.example:80"})
        assert status in (400, 421)


def test_without_a_token_the_server_does_not_start_and_says_why(home: Path) -> None:
    prepare_home(home, token=None)
    assert mcp_http.serve(object(), home) == mcp_http.EXIT_NO_TOKEN
    log = (home / "neospike" / ".cache" / "mcp_http.log").read_text(encoding="utf-8")
    assert f"not started: {TOKEN_VARIABLE} is missing" in log and TOKEN not in log


def test_a_port_taken_by_another_program_stops_the_start(home: Path, monkeypatch) -> None:
    port = prepare_home(home)
    monkeypatch.setattr(mcp_http.HttpTuning, "BIND_RETRY_SECONDS", 0.3)
    with socket.socket() as other:
        other.bind((mcp_http.HOST, port))
        other.listen(1)
        assert mcp_http.serve(object(), home) == mcp_http.EXIT_PORT_BUSY
    log = (home / "neospike" / ".cache" / "mcp_http.log").read_text(encoding="utf-8")
    assert f"not started: port {port} is in use by another program" in log
    assert mcp_http.read_record(home) is None


# -- one server: the new one stops the old one ---------------------------------------------------------------

@needs_mcp
def test_a_second_start_stops_the_first_server_and_one_remains(home: Path) -> None:
    port = prepare_home(home)
    with RunningServer(home) as first:
        second_state = mcp_http.ensure_running(home, server_command(home))
        second = mcp_http.read_record(home)
        assert second_state == mcp_http.STARTED and second.pid != first.pid
        assert mcp_http.wait_gone(first.pid, 10)
        assert mcp_http.server_status(port, TOKEN)["pid"] == second.pid
        log = (home / "neospike" / ".cache" / "mcp_http.log").read_text(encoding="utf-8")
        assert f"previous server pid={first.pid} stopped on request" in log and TOKEN not in log


@needs_mcp
def test_two_starts_at_once_leave_one_server(home: Path) -> None:
    port = prepare_home(home)
    states: list[str] = []
    starters = [threading.Thread(target=_Starter(home, states)) for _ in range(2)]
    for starter in starters:
        starter.start()
    for starter in starters:
        starter.join(120)
    assert states == [mcp_http.STARTED, mcp_http.STARTED]
    record = mcp_http.read_record(home)
    assert mcp_http.server_status(port, TOKEN)["pid"] == record.pid
    with socket.socket() as probe:  # nobody else listens: the port is held by the one server only
        with pytest.raises(OSError):
            probe.bind((mcp_http.HOST, port))


class _Starter:
    def __init__(self, home: Path, states: list[str]) -> None:
        self.home, self.states = home, states

    def __call__(self) -> None:
        self.states.append(mcp_http.ensure_running(self.home, server_command(self.home)))


def test_a_server_that_ignores_the_stop_request_is_stopped_by_force(home: Path, monkeypatch) -> None:
    prepare_home(home)
    monkeypatch.setattr(mcp_http.HttpTuning, "STOP_WAIT_SECONDS", 0.5)
    stuck = _sleeper()
    try:
        _fake_record(home, stuck, free_port())
        assert mcp_http.stop_server(home, TOKEN) == mcp_http.KILLED
        assert stuck.wait(10) is not None
    finally:
        stuck.kill()


def test_a_record_of_another_process_never_kills_it(home: Path, monkeypatch) -> None:
    """The PID is alive but its start time is not the recorded one (a reused PID): left alone."""
    prepare_home(home)
    monkeypatch.setattr(mcp_http.HttpTuning, "STOP_WAIT_SECONDS", 0.5)
    other = _sleeper()
    try:
        _fake_record(home, other, free_port(), start_shift=1)
        assert mcp_http.stop_server(home, TOKEN) == mcp_http.NOT_RUNNING
        time.sleep(0.3)
        assert other.poll() is None
    finally:
        other.kill()


def test_switched_off_home_starts_nothing(home: Path) -> None:
    prepare_home(home)
    config = json.loads((home / "neospike" / "config.json").read_text(encoding="utf-8"))
    (home / "neospike" / "config.json").write_text(json.dumps({**config, "enabled": False}), encoding="utf-8")
    assert mcp_http.ensure_running(home, [sys.executable, "-c", "raise SystemExit(9)"]) == mcp_http.DISABLED
    assert (home / ".env").read_text(encoding="utf-8") == f"OTHER=1\n{TOKEN_VARIABLE}={TOKEN}\n"
    assert mcp_http.read_record(home) is None and not (home / "neospike" / ".cache" / "mcp_http.lock").exists()


# -- the token -----------------------------------------------------------------------------------------------

def test_token_is_made_once_and_kept_with_the_other_lines(home: Path) -> None:
    env = home / ".env"
    env.write_bytes(b"FIRST=1\r\n# comment\r\nLAST=2")
    with mcp_http.start_lock(home):
        made = mcp_http.ensure_token(home)
    with mcp_http.start_lock(home):
        again = mcp_http.ensure_token(home)
    assert made == again and len(made) == 64 and int(made, 16) >= 0
    assert env.read_bytes() == b"FIRST=1\r\n# comment\r\nLAST=2\r\n" + f"{TOKEN_VARIABLE}={made}\r\n".encode()
    assert mcp_http.remove_token(home) is True
    assert env.read_bytes() == b"FIRST=1\r\n# comment\r\nLAST=2\r\n"


def test_token_made_at_once_by_many_starters_is_one_line(home: Path) -> None:
    tokens: list[str] = []
    makers = [threading.Thread(target=_TokenMaker(home, tokens)) for _ in range(8)]
    for maker in makers:
        maker.start()
    for maker in makers:
        maker.join(60)
    text = (home / ".env").read_text(encoding="utf-8")
    assert text.count(TOKEN_VARIABLE) == 1 and len(set(tokens)) == 1 and len(tokens) == 8


class _TokenMaker:
    def __init__(self, home: Path, tokens: list[str]) -> None:
        self.home, self.tokens = home, tokens

    def __call__(self) -> None:
        with mcp_http.start_lock(self.home):
            self.tokens.append(mcp_http.ensure_token(self.home))


def test_start_makes_the_missing_token_before_the_server(home: Path) -> None:
    prepare_home(home, token=None)
    state = mcp_http.ensure_running(home, [sys.executable, "-c", "raise SystemExit(5)"])
    assert state == mcp_http.START_FAILED
    first = read_env_token(home / ".env")
    mcp_http.ensure_running(home, [sys.executable, "-c", "raise SystemExit(5)"])
    assert len(first) == 64 and read_env_token(home / ".env") == first


def test_headers_helper_prints_only_the_header(home: Path) -> None:
    prepare_home(home)
    helper = REPO / "neospike" / "mcp_headers.py"
    shown = subprocess.run([sys.executable, "-I", str(helper), str(home)], capture_output=True, text=True, timeout=30)
    assert shown.returncode == 0 and json.loads(shown.stdout) == {"Authorization": "Bearer " + TOKEN}
    (home / ".env").write_text("OTHER=1\n", encoding="utf-8")
    missing = subprocess.run([sys.executable, "-I", str(helper), str(home)], capture_output=True, text=True, timeout=30)
    assert missing.returncode == 1 and missing.stdout == "" and TOKEN_VARIABLE in missing.stderr


# -- the start from register() -------------------------------------------------------------------------------

class _Recorder:
    def __init__(self) -> None:
        self.roots: list[Path] = []
        self.done = threading.Event()

    def __call__(self, root, command=None) -> str:
        self.roots.append(Path(root))
        self.done.set()
        return mcp_http.STARTED


def test_register_outside_hermes_starts_nothing(monkeypatch) -> None:
    from neospike import provider
    monkeypatch.setitem(sys.modules, "hermes_constants", None)  # import fails as outside Hermes
    assert provider.start_mcp_server() is False


def test_every_profile_load_in_one_process_starts_one_root_server(tmp_path: Path, monkeypatch) -> None:
    from neospike import provider
    root = (tmp_path / "root").resolve()
    homes = [root, root / "profiles" / "design", root / "profiles" / "lotto"]
    current = types.SimpleNamespace(home=homes[0])
    constants = types.ModuleType("hermes_constants")
    constants.get_hermes_home = _CurrentHome(current)
    monkeypatch.setitem(sys.modules, "hermes_constants", constants)
    monkeypatch.delenv("NEOSPIKE_INFERENCE_HOME", raising=False)
    monkeypatch.setattr(mcp_http, "_started_roots", set())
    recorder = _Recorder()
    monkeypatch.setattr(mcp_http, "ensure_running", recorder)
    started = []
    for profile_home in homes + homes:  # two plugin loads per profile
        current.home = profile_home
        started.append(provider.start_mcp_server())
    assert recorder.done.wait(10)
    time.sleep(0.2)
    assert started == [True] + [False] * 5 and recorder.roots == [root]


class _CurrentHome:
    def __init__(self, current) -> None:
        self.current = current

    def __call__(self) -> str:
        return str(self.current.home)


def test_switching_a_profile_off_leaves_the_root_server(tmp_path: Path) -> None:
    from neospike.dashboard import plugin_api
    profile = (tmp_path / "profiles" / "design").resolve()
    profile.mkdir(parents=True)
    assert plugin_api._stop_mcp_server(profile) == "not_root"
    assert plugin_api._stop_mcp_server(tmp_path.resolve()) == mcp_http.NOT_RUNNING
