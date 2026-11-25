"""neospike-mcp as one shared local HTTP server (MCP streamable-http) for Claude Code and Hermes.

The server listens on 127.0.0.1:<mcp_port of config.json>, path /mcp, stateless (a client that loses the
server just sends its next request to the new one). Every request must carry
``Authorization: Bearer <NEOSPIKE_MCP_TOKEN of HERMES_HOME/.env>``; the SDK's Host check (DNS rebinding
protection for 127.0.0.1) stays on.

Lifecycle: the plugin's register() calls start_in_background() once per process; ensure_running() then starts a
new detached server under a lock file, and the new server stops the one before it (an authenticated stop
request first, then a forced stop only for the process its record names, with the same start time and
executable). So the last started server is the only one. It is not tied to the process that started it.

Files in <home>/neospike/.cache: mcp_http.log (start, stop reason, port, parent PID; never queries or the
token), mcp_http.pid (the running server's record) and mcp_http.lock (one starter at a time).
"""
from __future__ import annotations

import errno
import hmac
import http.client
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import _wincompat as _wc
from ._wincompat import fcntl
from .config import NeospikeConfig, read_enabled
from .hermes_runtime import launch_command
from .mcp_headers import BEARER_PREFIX, TOKEN_VARIABLE, authorization_headers, env_file_path, parse_env_value, \
    read_env_token

SERVER_NAME = "neospike-mcp"
SERVER_MODULE = "neospike.mcp_server"
HOST = "127.0.0.1"
MCP_PATH = "/mcp"
STATUS_PATH = "/neospike/status"
SHUTDOWN_PATH = "/neospike/shutdown"
CACHE_PARTS = ("neospike", ".cache")
LOG_NAME, RECORD_NAME, LOCK_NAME = "mcp_http.log", "mcp_http.pid", "mcp_http.lock"
EXIT_NO_TOKEN, EXIT_PORT_BUSY = 3, 4
# ensure_running() outcomes.
STARTED, START_FAILED, START_TIMEOUT, DISABLED = "started", "failed", "timeout", "disabled"
# stop_server() outcomes.
NOT_RUNNING, STOPPED, KILLED, LEFT_RUNNING = "not_running", "stopped", "killed", "left_running"
# Windows socket errors of a taken port: WSAEADDRINUSE, and WSAEACCES under SO_EXCLUSIVEADDRUSE.
_PORT_TAKEN_WINERRORS = (10048, 10013)
_PORT_TAKEN_ERRNOS = (errno.EADDRINUSE, errno.EACCES)


class HttpTuning:
    """Waits of the server lifecycle (starting values, no measurement behind them)."""
    STOP_WAIT_SECONDS = 5.0  # after the stop request, before a forced stop
    KILL_WAIT_SECONDS = 5.0
    BIND_RETRY_SECONDS = 10.0  # the old server may still hold the port for a moment
    READY_WAIT_SECONDS = 60.0  # the Hermes bootstrap and the SDK import come first
    LOCK_WAIT_SECONDS = 120.0
    POLL_SECONDS = 0.1
    REQUEST_TIMEOUT_SECONDS = 3.0
    GRACEFUL_SHUTDOWN_SECONDS = 3
    LISTEN_BACKLOG = 128
    LOG_MAX_BYTES = 1_000_000  # one older copy (.1) is kept


def cache_dir(home: str | Path) -> Path:
    return Path(home).expanduser().joinpath(*CACHE_PARTS)


def server_url(port: int) -> str:
    return f"http://{HOST}:{port}{MCP_PATH}"


def configured_port(home: str | Path) -> int:
    return NeospikeConfig.load(home).mcp_port


class ServerLog:
    """Appends one stamped line per event to <home>/neospike/.cache/mcp_http.log."""

    def __init__(self, home: str | Path) -> None:
        self.path = cache_dir(home) / LOG_NAME

    def write(self, text: str) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.is_file() and self.path.stat().st_size > HttpTuning.LOG_MAX_BYTES:
                os.replace(self.path, self.path.with_name(LOG_NAME + ".1"))
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.path, "a", encoding="utf-8") as log_file:
                log_file.write(f"{stamp} pid={os.getpid()} {text}\n")
        except OSError:
            pass


# -- token ----------------------------------------------------------------------------------------------------

def ensure_token(home: str | Path) -> str:
    """NEOSPIKE_MCP_TOKEN of <home>/.env; made once with secrets.token_hex(32) and appended when missing (an
    existing value is never changed, other lines keep their bytes and order). Call it under start_lock()."""
    path = Path(env_file_path(Path(home).expanduser()))
    token = read_env_token(path)
    if token:
        return token
    token = secrets.token_hex(32)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raw = None
    newline = b"\r\n" if raw and b"\r\n" in raw else b"\n"
    lead = newline if raw and not raw.endswith((b"\n", b"\r")) else b""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as env_file:
        env_file.write(lead + f"{TOKEN_VARIABLE}={token}".encode("ascii") + newline)
    if raw is None:
        _wc.chmod(path, 0o600)
    return token


def remove_token(home: str | Path, dry_run: bool = False) -> bool:
    """Delete every NEOSPIKE_MCP_TOKEN line of <home>/.env (True = there was one); other lines stay."""
    path = Path(env_file_path(Path(home).expanduser()))
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return False
    bom = raw.startswith(b"\xef\xbb\xbf")
    lines = raw.decode("utf-8-sig").splitlines(keepends=True)
    kept = [line for line in lines if parse_env_value(line, TOKEN_VARIABLE) is None]
    if len(kept) == len(lines):
        return False
    if not dry_run:
        path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + "".join(kept).encode("utf-8"))
    return True


class start_lock:
    """`with start_lock(home):` one starter (and one token writer) at a time, across processes and threads."""

    def __init__(self, home: str | Path, wait_seconds: float = HttpTuning.LOCK_WAIT_SECONDS) -> None:
        self.path = cache_dir(home) / LOCK_NAME
        self.wait_seconds = wait_seconds
        self.handle: Any = None

    def __enter__(self) -> "start_lock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+")
        deadline = time.monotonic() + self.wait_seconds
        while True:
            try:
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    self.handle.close()
                    raise TimeoutError(f"{self.path} is still locked after {self.wait_seconds:g} s") from None
                time.sleep(HttpTuning.POLL_SECONDS)

    def __exit__(self, *exc_info: Any) -> None:
        try:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
        finally:
            self.handle.close()


# -- processes ------------------------------------------------------------------------------------------------

class _Win32:
    """The few kernel32 calls the process checks need (Windows only)."""
    QUERY_LIMITED_INFORMATION = 0x1000
    TERMINATE = 0x0001
    STILL_ACTIVE = 259
    ACCESS_DENIED = 5
    PATH_CHARS = 32768
    _kernel32: Any = None

    @classmethod
    def kernel32(cls) -> Any:
        if cls._kernel32 is None:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            k32.CloseHandle.argtypes = (wintypes.HANDLE,)
            k32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            k32.GetProcessTimes.argtypes = (wintypes.HANDLE, *(ctypes.POINTER(wintypes.FILETIME),) * 4)
            k32.QueryFullProcessImageNameW.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                       ctypes.POINTER(wintypes.DWORD))
            k32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
            cls._kernel32 = k32
        return cls._kernel32

    @classmethod
    def open(cls, pid: int, access: int) -> Any:
        return cls.kernel32().OpenProcess(access, False, pid)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if _wc.IS_WINDOWS:
        import ctypes
        from ctypes import wintypes
        handle = _Win32.open(pid, _Win32.QUERY_LIMITED_INFORMATION)
        if not handle:
            return ctypes.get_last_error() == _Win32.ACCESS_DENIED
        try:
            code = wintypes.DWORD()
            return bool(_Win32.kernel32().GetExitCodeProcess(handle, ctypes.byref(code))) \
                and code.value == _Win32.STILL_ACTIVE
        finally:
            _Win32.kernel32().CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return _linux_state(pid) != "Z"


def _linux_stat_fields(pid: int) -> list[str]:
    try:
        text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii", errors="replace")
    except OSError:
        return []
    return text.rpartition(")")[2].split()


def _linux_state(pid: int) -> str:
    fields = _linux_stat_fields(pid)
    return fields[0] if fields else ""


def process_start_time(pid: int) -> int | None:
    """Creation time of *pid* (Windows FILETIME, Linux clock ticks since boot); None when unknown."""
    if _wc.IS_WINDOWS:
        import ctypes
        from ctypes import wintypes
        handle = _Win32.open(pid, _Win32.QUERY_LIMITED_INFORMATION)
        if not handle:
            return None
        try:
            times = [wintypes.FILETIME() for _ in range(4)]
            if not _Win32.kernel32().GetProcessTimes(handle, *(ctypes.byref(value) for value in times)):
                return None
            return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        finally:
            _Win32.kernel32().CloseHandle(handle)
    fields = _linux_stat_fields(pid)
    start_field = 19  # field 22 of /proc/<pid>/stat, counted after the command name
    return int(fields[start_field]) if len(fields) > start_field else None


def process_executable(pid: int) -> str:
    """Full path of the program *pid* runs; "" when unknown."""
    if _wc.IS_WINDOWS:
        import ctypes
        from ctypes import wintypes
        handle = _Win32.open(pid, _Win32.QUERY_LIMITED_INFORMATION)
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(_Win32.PATH_CHARS)
            buffer = ctypes.create_unicode_buffer(_Win32.PATH_CHARS)
            if not _Win32.kernel32().QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return ""
            return buffer.value
        finally:
            _Win32.kernel32().CloseHandle(handle)
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return ""


def kill_process(pid: int) -> None:
    if _wc.IS_WINDOWS:
        handle = _Win32.open(pid, _Win32.TERMINATE)
        if handle:
            try:
                _Win32.kernel32().TerminateProcess(handle, 1)
            finally:
                _Win32.kernel32().CloseHandle(handle)
        return
    import signal
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def wait_gone(pid: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while pid_alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(HttpTuning.POLL_SECONDS)
    return True


# -- the server record ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerRecord:
    """What a running server wrote about itself at start (<home>/neospike/.cache/mcp_http.pid)."""
    pid: int
    parent_pid: int
    port: int
    started_at: float
    start_time: int | None
    executable: str

    @classmethod
    def of_this_process(cls, port: int) -> "ServerRecord":
        pid = os.getpid()
        return cls(pid=pid, parent_pid=os.getppid(), port=port, started_at=time.time(),
                   start_time=process_start_time(pid), executable=process_executable(pid) or sys.executable)

    def names_live_server(self) -> bool:
        """The recorded PID is alive and is the very process that wrote the record (same start time and
        executable); a reused PID or an unknown start time is not."""
        if self.start_time is None or not pid_alive(self.pid):
            return False
        return (process_start_time(self.pid) == self.start_time
                and os.path.normcase(process_executable(self.pid)) == os.path.normcase(self.executable))


def record_path(home: str | Path) -> Path:
    return cache_dir(home) / RECORD_NAME


def read_record(home: str | Path) -> ServerRecord | None:
    try:
        values = json.loads(record_path(home).read_text(encoding="utf-8"))
        return ServerRecord(pid=int(values["pid"]), parent_pid=int(values["parent_pid"]), port=int(values["port"]),
                            started_at=float(values["started_at"]),
                            start_time=None if values.get("start_time") is None else int(values["start_time"]),
                            executable=str(values["executable"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def write_record(home: str | Path, record: ServerRecord) -> None:
    target = record_path(home)
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_name(RECORD_NAME + ".tmp")
    scratch.write_text(json.dumps(asdict(record)), encoding="utf-8")
    os.replace(scratch, target)


def remove_own_record(home: str | Path, pid: int) -> None:
    record = read_record(home)
    if record is not None and record.pid == pid:
        record_path(home).unlink(missing_ok=True)


# -- client side ----------------------------------------------------------------------------------------------

def request(port: int, method: str, path: str, token: str,
            timeout: float = HttpTuning.REQUEST_TIMEOUT_SECONDS) -> tuple[int, bytes]:
    """(status, body) of one authenticated request to the local server; OSError when nothing answers."""
    connection = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        connection.request(method, path, body=b"" if method == "POST" else None, headers=authorization_headers(token))
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def server_status(port: int, token: str) -> dict | None:
    """The status answer of a neospike-mcp server on *port* (it accepted *token*); None otherwise."""
    try:
        status, body = request(port, "GET", STATUS_PATH, token)
        values = json.loads(body) if status == 200 else None
    except (OSError, ValueError):
        return None
    return values if isinstance(values, dict) and values.get("server") == SERVER_NAME else None


def stop_server(home: str | Path, token: str, log: ServerLog | None = None) -> str:
    """Stop the server <home>'s record names: the stop request first; after STOP_WAIT_SECONDS a forced stop,
    only when the record still names that very process. A record of a dead or reused PID is ignored."""
    log = log or ServerLog(home)
    record = read_record(home)
    if record is None or record.pid == os.getpid() or not record.names_live_server():
        return NOT_RUNNING
    try:
        request(record.port, "POST", SHUTDOWN_PATH, token)
    except OSError:
        pass
    if wait_gone(record.pid, HttpTuning.STOP_WAIT_SECONDS):
        log.write(f"previous server pid={record.pid} stopped on request")
        return STOPPED
    if not record.names_live_server():
        return STOPPED
    kill_process(record.pid)
    if wait_gone(record.pid, HttpTuning.KILL_WAIT_SECONDS):
        log.write(f"previous server pid={record.pid} did not stop on request; forced stop")
        return KILLED
    log.write(f"previous server pid={record.pid} is still running after a forced stop")
    return LEFT_RUNNING


def stop_running(home: str | Path) -> str:
    """Stop the shared server of *home*'s root home (switch turned off, uninstall)."""
    root = _wc.shared_inference_home(home)
    return stop_server(root, read_env_token(env_file_path(root)))


# -- server side ----------------------------------------------------------------------------------------------

def _port_taken(exc: OSError) -> bool:
    return getattr(exc, "winerror", None) in _PORT_TAKEN_WINERRORS or exc.errno in _PORT_TAKEN_ERRNOS


def bind_listener(port: int, retry_seconds: float = HttpTuning.BIND_RETRY_SECONDS) -> socket.socket:
    """A listening socket on 127.0.0.1:*port* that no other process can share (SO_EXCLUSIVEADDRUSE on
    Windows). A taken port is retried for *retry_seconds*, then its OSError is raised."""
    deadline = time.monotonic() + retry_seconds
    while True:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            else:  # POSIX: two listeners still cannot share the port; only TIME_WAIT is skipped
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((HOST, port))
            listener.listen(HttpTuning.LISTEN_BACKLOG)
            return listener
        except OSError as exc:
            listener.close()
            if not _port_taken(exc) or time.monotonic() >= deadline:
                raise
        time.sleep(HttpTuning.POLL_SECONDS)


class ServerControl:
    """The running server's status answer and its stop switch (set by a stop request)."""

    def __init__(self, record: ServerRecord) -> None:
        self.record = record
        self.uvicorn_server: Any = None
        self.reason = ""

    def status(self) -> dict:
        return {"server": SERVER_NAME, "pid": self.record.pid, "port": self.record.port,
                "started_at": self.record.started_at}

    def request_stop(self, reason: str) -> None:
        self.reason = self.reason or reason
        if self.uvicorn_server is not None:
            self.uvicorn_server.should_exit = True


async def _send_json(send: Any, status: int, payload: dict, extra_headers: tuple = ()) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                            *extra_headers]})
    await send({"type": "http.response.body", "body": body})


class BearerTokenGate:
    """ASGI middleware: an HTTP request without `Authorization: Bearer <token>` gets 401 and reaches nothing
    behind it. The status and stop paths are answered here; everything else goes to the MCP app."""

    def __init__(self, app: Any, token: str, control: ServerControl) -> None:
        self.app = app
        self.expected = (BEARER_PREFIX + token).encode("utf-8")
        self.control = control

    def authorized(self, scope: dict) -> bool:
        given = b""
        for name, value in scope.get("headers") or ():
            if name.lower() == b"authorization":
                given = value
        return hmac.compare_digest(given, self.expected)

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if not self.authorized(scope):
            await _send_json(send, 401, {"error": "unauthorized"}, ((b"www-authenticate", b"Bearer"),))
            return
        path, method = scope.get("path", ""), scope.get("method", "")
        if path == STATUS_PATH and method == "GET":
            await _send_json(send, 200, self.control.status())
        elif path == SHUTDOWN_PATH and method == "POST":
            self.control.request_stop("stop request")
            await _send_json(send, 202, {"stopping": True})
        else:
            await self.app(scope, receive, send)


def serve(mcp_server: Any, home: str | Path) -> int:
    """Run *mcp_server* (an SDK MCPServer) as the one shared HTTP server of *home*; returns the exit code."""
    home = Path(home).expanduser()
    log = ServerLog(home)
    token = read_env_token(env_file_path(home))
    if not token:
        log.write(f"not started: {TOKEN_VARIABLE} is missing in {env_file_path(home)}")
        return EXIT_NO_TOKEN
    port = configured_port(home)
    previous = stop_server(home, token, log)
    try:
        listener = bind_listener(port)
    except OSError as exc:
        log.write(f"not started: port {port} is in use by another program ({exc.strerror or exc})")
        return EXIT_PORT_BUSY
    record = ServerRecord.of_this_process(port)
    write_record(home, record)
    log.write(f"started: port={port} parent_pid={record.parent_pid} previous={previous}")
    control = ServerControl(record)
    app = BearerTokenGate(mcp_server.streamable_http_app(streamable_http_path=MCP_PATH, stateless_http=True,
                                                         json_response=True, host=HOST), token, control)
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, log_config=None, access_log=False, lifespan="on",
                                           timeout_graceful_shutdown=HttpTuning.GRACEFUL_SHUTDOWN_SECONDS))
    control.uvicorn_server = server
    try:
        server.run(sockets=[listener])
    except BaseException as exc:
        control.request_stop(f"{type(exc).__name__}")
        raise
    finally:
        listener.close()
        remove_own_record(home, record.pid)
        log.write(f"stopped: {control.reason or 'signal'}")
    return 0


# -- starting -------------------------------------------------------------------------------------------------

def _detached_kwargs() -> dict:
    if _wc.IS_WINDOWS:
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return {"close_fds": True, "creationflags": flags}
    return {"close_fds": True, "start_new_session": True}


def server_command(root: Path, env: dict) -> list[str]:
    """neospike.mcp_server through hermes_launch.py on the Hermes runtime python (env updated in place)."""
    command = launch_command(root, [SERVER_MODULE, "--home", str(root)])
    command[0] = _wc.detached_interpreter(command[0], env)
    return command


def spawn_server(root: Path, command: list[str] | None = None) -> subprocess.Popen:
    env = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    env["HERMES_HOME"], env["PYTHONIOENCODING"] = str(root), "utf-8"
    command = command or server_command(root, env)
    log_path = cache_dir(root) / LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log_file:  # the server's own stderr (SDK warnings, a failed import)
        return subprocess.Popen(command, env=env, cwd=str(Path(__file__).resolve().parent.parent),
                                stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file, **_detached_kwargs())


def wait_ready(root: Path, process: subprocess.Popen, token: str, since: float,
               seconds: float = HttpTuning.READY_WAIT_SECONDS) -> str:
    """STARTED once a record written after *since* names a server that answers; START_FAILED when the started
    process ended with an error code first. (A bootstrap may hand over to another interpreter and end with 0.)"""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        record = read_record(root)
        if record is not None and record.started_at >= since:
            status = server_status(record.port, token)
            if status is not None and status.get("pid") == record.pid:
                return STARTED
        code = process.poll()
        if code not in (None, 0):
            return START_FAILED
        time.sleep(HttpTuning.POLL_SECONDS)
    return START_TIMEOUT


def ensure_running(home: str | Path, command: list[str] | None = None) -> str:
    """Start a new shared server for *home*'s root home (it stops the one before it); the token is made first
    when missing. Nothing happens while Neospike is switched off there. One starter at a time (start_lock)."""
    root = _wc.shared_inference_home(home)
    if not read_enabled(root):
        return DISABLED
    with start_lock(root):
        token = ensure_token(root)
        since = time.time()
        process = spawn_server(root, command)
        state = wait_ready(root, process, token, since)
    ServerLog(root).write(f"start request: {state}")
    return state


class _BackgroundStart:
    """Thread body of start_in_background: ensure_running, with any failure written to the log."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def __call__(self) -> None:
        try:
            ensure_running(self.root)
        except Exception as exc:  # a missing Hermes runtime, a lock timeout: logged, never raised into Hermes
            ServerLog(self.root).write(f"start failed: {type(exc).__name__}: {exc}")


_started_roots: set[str] = set()
_started_lock = threading.Lock()


def start_in_background(home: str | Path) -> bool:
    """Run ensure_running on a daemon thread, once per process and root home (register() runs for every profile
    and for each plugin load). False when this process already did."""
    root = _wc.shared_inference_home(home)
    with _started_lock:
        if str(root) in _started_roots:
            return False
        _started_roots.add(str(root))
    threading.Thread(target=_BackgroundStart(root), name="neospike-mcp-start", daemon=True).start()
    return True
