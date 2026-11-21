# neospike windows patch v1
"""Profile-private embedding service (runs in the foreground; it never starts or stops Hermes).

One worker thread owns the model and loads it the first time a request needs it. Clients have no
local fallback. A native call cannot be interrupted once it is running, so the bounded queues and
the per-request deadlines are what stop a stuck call from accumulating more work behind it.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hmac
import itertools
import json
import math
import os
import pathlib
import queue
import secrets
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

from . import _wincompat as _wc
from ._wincompat import fcntl
from .config import EMBEDDING_DIMENSIONS, PAUSED_MESSAGE, read_enabled

# One input may fill the full token window. The bge-m3 tokenizer emits at most 16
# characters (plus a space) per piece, so 17 chars/token is the ceiling. A frame holds that input
# fully JSON-escaped (up to 6 bytes per character) plus a fixed allowance for the protocol fields.
# Kept at the old 8192-token size after the window became 2048: a longer query is cut to the window
# by the embedder, so the character cap only bounds the frame and must not refuse queries it used to take.
INPUT_CHAR_CAP = 8192 * 17
_FRAME_OVERHEAD = 65_536
FRAME_BYTE_CAP = INPUT_CHAR_CAP * 6 + _FRAME_OVERHEAD
_OPS = frozenset(('query', 'document'))
SHUTDOWN_OP = 'shutdown'
# release_daemon() results.
DAEMON_STOPPED, DAEMON_NOT_RUNNING, DAEMON_IN_USE = 'stopped', 'not_running', 'in_use'
_SHUTDOWN_REPLY_SECONDS = 5
_PROVIDER_ID = 'neospike'
_PRIORITY_RANK = {'interactive': 0, 'indexing': 1}
_UNIX_SOCKET_LIMIT = 104
_MAX_REQUEST_SECONDS = 120
_FAILURE_BACKOFF_SECONDS = 5
_POLL_SLICE = .25
_RECV_CHUNK = 4096
_ACCEPT_TICK = .2
_JOB_TICK = .1
_CONN_IO_TIMEOUT = 2
_CONN_JOIN_SECONDS = 3
_LISTEN_BACKLOG = 32
_DEFAULT_REQUEST_SECONDS = 30
_SPIN_SLEEP = .025
_LAUNCH_GUARD_NAME = 'launch.lock'
_MAIN_WAIT_SLICE = .5
_TOKEN_READ_LIMIT = 256
# Mirror embedder.TOKENIZER_JSON / ONNX_MODEL_FILE / MODEL_FOLDER: importing embedder loads the model packages.
MODEL_FILE_NAMES = ('tokenizer.json', 'sentence_transformers_int8.onnx')
MODEL_FOLDER = 'bge-m3-onnx'
_MODEL_DIR_ENV = 'HERMES_BGE_M3_DIR'
_MODEL_MISSING_EXIT = 1

_ERR = {
    'dir_owner': 'Inference directory must be owned by this user, not a symlink',
    'dir_insecure': 'Insecure inference directory',
    'file_insecure': 'Insecure inference lifecycle file',
    'token_insecure': 'Insecure inference token',
    'frame_short': 'Incomplete inference frame',
    'frame_big': 'Inference frame too large',
    'bad_op': 'Unknown inference operation',
    'bad_input': 'Invalid inference input',
    'bad_timeout': 'Invalid timeout',
    'bad_priority': 'Invalid priority',
    'bad_bounds': 'Queue and connection bounds must be positive',
    'bad_startup': 'Startup timeout must be positive and finite',
    'long_path': 'Profile path too long for a Unix socket (104-byte limit)',
    'already_up': 'Inference service already running',
    'denied': 'Unauthorized inference request',
    'queue_full': 'Inference queue full',
    'deadline': 'Inference deadline exceeded',
    'start_deadline': 'Inference startup deadline exceeded',
    'no_credentials': 'Live inference socket has no usable credentials',
    'recent_failure': 'Inference startup recently failed; retry on later demand',
    'owner_busy': 'Inference service owns profile but is not ready',
    'launching': 'Inference launch in progress; service is not ready',
    'spawn_failed': 'Inference service failed to start; see private inference.log',
    'model_missing': 'Embedding model file not found: {path} (put bge-m3-onnx there or set HERMES_BGE_M3_DIR)',
}


class EmbeddingUnavailable(RuntimeError):
    """The embedding service cannot answer: an error reply of the daemon, or a daemon that cannot be started.
    A RuntimeError, so callers that catch RuntimeError keep working."""


def missing_model_message(home=None):
    """The model-missing error for the first model file the daemon would fail to load, or None when all exist.
    Resolves the folder as embedder.resolve_model_dir does in a daemon whose HERMES_HOME is *home*
    (None: this process's HERMES_HOME)."""
    configured = os.environ.get(_MODEL_DIR_ENV)
    if configured:
        model_dir = pathlib.Path(configured).expanduser()
    else:
        base = _wc.default_hermes_home() if home is None else home
        model_dir = _wc.shared_inference_home(base) / 'models' / MODEL_FOLDER
    for name in MODEL_FILE_NAMES:
        if not (model_dir / name).is_file():
            return _ERR['model_missing'].format(path=model_dir / name)
    return None


def _not_mine(info):
    return info.st_uid != _wc.getuid()


def _ensure_owner_dir(path):
    path = pathlib.Path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode)) or _wc.is_link_stat(info) or _not_mine(info) or _wc.foreign_owner(path):
        raise PermissionError(_ERR['dir_owner'])
    _wc.chmod(path, 0o700)
    return path


def _open_owner_file(path, *, unshared=False):
    if unshared:
        fd = _wc.open_unshared(path)
    else:
        fd = _wc.open_nofollow(path, os.O_CREAT | os.O_RDWR | _wc.O_NONBLOCK, 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or _not_mine(info) or info.st_nlink != 1 or _wc.insecure_mode(info.st_mode, fd):
        os.close(fd)
        raise PermissionError(_ERR['file_insecure'])
    return os.fdopen(fd, mode='r+')


def _read_frame(conn):
    buffered = b''
    while True:
        newline_at = buffered.find(b'\n')
        if newline_at != -1:
            return json.loads(buffered[:newline_at])
        chunk = conn.recv(min(_RECV_CHUNK, FRAME_BYTE_CAP + 1 - len(buffered)))
        if not chunk:
            raise ValueError(_ERR['frame_short'])
        buffered += chunk
        if FRAME_BYTE_CAP < len(buffered):
            raise ValueError(_ERR['frame_big'])


def _write_frame(conn, payload):
    wire = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode() + b'\n'
    if FRAME_BYTE_CAP < len(wire):
        raise ValueError(_ERR['frame_big'])
    conn.sendall(wire)


class LocalModelRunner:
    def __init__(self, profile_home):
        self.home = pathlib.Path(profile_home)
        self.text = None
        self.warnings = []

    @property
    def device(self):
        """Device the loaded model runs on ('cpu' or 'directml'); None before the model is loaded."""
        return getattr(self.text, 'device', None)

    @property
    def device_note(self):
        """Why DirectML was not taken ('' when it was); None before the model is loaded."""
        return getattr(self.text, 'device_note', None)

    def warm(self):
        """Load the model (its load runs one short input, so the first real request pays no setup)."""
        if self.text is None:
            missing = missing_model_message()
            if missing:
                raise FileNotFoundError(missing)
            from .embedder import OnnxDenseEmbedder
            self.text = OnnxDenseEmbedder()

    def take_warnings(self):
        """Degradations of the most recent request (only the single worker thread calls this)."""
        drained, self.warnings = self.warnings, []
        return drained

    def __call__(self, op, value, *, title='none'):
        if op not in _OPS:
            raise ValueError(_ERR['bad_op'])
        self.warm()
        if op == 'query':
            result = self.text.embed_query(value)
        else:
            result = self.text.embed_document(value, title=title)
        self.warnings.extend(self.text.take_warnings())
        return result


class EmbeddingDaemon:
    def __init__(self, profile_home, *, backend=None, queue_size=32, connections=48):
        self.home = pathlib.Path(profile_home)
        self.directory = self.home / 'neospike' / '.cache'
        self.socket_path = self.directory / 'i.sock'
        self.backend = LocalModelRunner(profile_home) if backend is None else backend
        if min(queue_size, connections) <= 0:
            raise ValueError(_ERR['bad_bounds'])
        self.jobs = queue.PriorityQueue(maxsize=queue_size * 2)
        self.lane_slots = {lane: threading.BoundedSemaphore(queue_size) for lane in _PRIORITY_RANK}
        self.conn_slots = threading.BoundedSemaphore(connections)
        self.ticket_seq = itertools.count()
        self.stop_event = threading.Event()
        # Set by an authenticated shutdown request or a stop signal; main() waits on it.
        self.shutdown_requested = threading.Event()
        self.lifetime_lock = None
        self.server_sock = None
        self.secret = None
        self.owns_token = False
        self.loop_threads = []
        self.live_conns = set()
        self.live_conns_guard = threading.Lock()

    def start(self):
        _ensure_owner_dir(self.directory)
        # In TCP mode the address lives in a port file, so the AF_UNIX path-length limit does not apply.
        if _wc.USE_AF_UNIX and len(os.fsencode(self.socket_path)) >= _UNIX_SOCKET_LIMIT:
            raise ValueError(_ERR['long_path'])
        lock_fd = _wc.open_nofollow(self.directory / 'owner.lock', os.O_RDWR | os.O_CREAT, 0o600)
        self.lifetime_lock = os.fdopen(lock_fd, 'w')
        try:
            fcntl.flock(self.lifetime_lock, fcntl.LOCK_NB | fcntl.LOCK_EX)
        except BlockingIOError:
            self.lifetime_lock.close()
            self.lifetime_lock = None
            raise RuntimeError(_ERR['already_up']) from None
        try:
            self._publish_token()
            self.socket_path.unlink(missing_ok=True)
            self.server_sock = _wc.listen_socket(self.socket_path, backlog=_LISTEN_BACKLOG)
            self.server_sock.settimeout(_ACCEPT_TICK)
            for loop in (self._worker_loop, self._accept_loop):
                thread = threading.Thread(target=loop, daemon=True)
                self.loop_threads.append(thread)
                thread.start()
        except BaseException:
            self.close()
            raise

    def _publish_token(self):
        self.secret = secrets.token_hex(32)
        with _open_owner_file(self.directory / 'token') as token_file:
            token_file.truncate()
            token_file.write(self.secret)
        self.owns_token = True

    def _accept_loop(self):
        while not self.stop_event.is_set():
            try:
                conn, _address = self.server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            if not self.conn_slots.acquire(blocking=False):
                conn.close()
                continue
            thread = threading.Thread(target=self._serve_connection, args=(conn,), daemon=True)
            with self.live_conns_guard:
                self.live_conns.add(thread)
            thread.start()

    def _authenticated_request(self, conn):
        request = _read_frame(conn)
        if not isinstance(request, dict) or not hmac.compare_digest(str(request.get('token', '')), self.secret):
            raise PermissionError(_ERR['denied'])
        return request

    def _enqueue(self, request):
        """Check a query/document request and put it on the queue; returns (future, timeout)."""
        if request.get('op') not in _OPS:
            raise ValueError(_ERR['bad_op'])
        text = request.get('value')
        if not isinstance(text, str) or len(text) > INPUT_CHAR_CAP:
            raise ValueError(_ERR['bad_input'])
        seconds = float(request.get('timeout', _DEFAULT_REQUEST_SECONDS))
        if not (math.isfinite(seconds) and 0 < seconds <= _MAX_REQUEST_SECONDS):
            raise ValueError(_ERR['bad_timeout'])
        lane = request.get('priority', 'interactive')
        if lane not in _PRIORITY_RANK:
            raise ValueError(_ERR['bad_priority'])
        future = concurrent.futures.Future()
        deadline = time.monotonic() + seconds
        if not self.lane_slots[lane].acquire(blocking=False):
            raise queue.Full
        try:
            self.jobs.put_nowait((_PRIORITY_RANK[lane], next(self.ticket_seq), deadline, future, request))
        except BaseException:
            self.lane_slots[lane].release()
            raise
        return future, seconds

    def _serve_connection(self, conn):
        try:
            conn.settimeout(_CONN_IO_TIMEOUT)
            request = self._authenticated_request(conn)
            if request.get('op') == 'ping':
                # device: where the model runs; None until the model is loaded (the start-up warm-up).
                _write_frame(conn, {'result': {'pid': os.getpid(), 'device': getattr(self.backend, 'device', None)}})
                return
            if request.get('op') == SHUTDOWN_OP:
                # Set before the answer, so a caller that got it knows the stop is under way; main() then
                # leaves its wait, and close() waits for this connection and the in-flight model call.
                self.shutdown_requested.set()
                _write_frame(conn, {'result': {'pid': os.getpid(), 'stopping': True}})
                return
            future, seconds = self._enqueue(request)
            try:
                result, warnings = future.result(timeout=seconds)
            except TimeoutError:
                future.cancel()
                raise TimeoutError(_ERR['deadline']) from None
            response = {'result': result}
            if warnings:
                response['warnings'] = warnings
            _write_frame(conn, response)
        except Exception as error:
            message = _ERR['queue_full'] if isinstance(error, queue.Full) else str(error)
            with contextlib.suppress(OSError, ValueError):
                _write_frame(conn, {'error': message})
        finally:
            conn.close()
            self.conn_slots.release()
            with self.live_conns_guard:
                self.live_conns.discard(threading.current_thread())

    def _run_job(self, deadline, future, request):
        if deadline <= time.monotonic():
            future.cancel()
        if not future.set_running_or_notify_cancel():
            return
        take_warnings = getattr(self.backend, 'take_warnings', None)
        try:
            result = self.backend(request['op'], request['value'], title=request.get('title', 'none'))
        except Exception as error:
            if callable(take_warnings):
                take_warnings()
            future.set_exception(error)
        else:
            future.set_result((result, take_warnings() if callable(take_warnings) else []))

    def _warm_backend(self):
        """Load the model before the first request, on the worker thread: the daemon answers pings meanwhile
        and requests queue behind it. A failure is left to the first request, which reports it to its caller."""
        warm = getattr(self.backend, 'warm', None)
        if not callable(warm):
            return
        try:
            warm()
        except Exception as error:
            print(json.dumps({'warm_up_failed': f'{type(error).__name__}: {error}'}), flush=True)
            return
        print(json.dumps({'device': getattr(self.backend, 'device', None),
                          'device_note': getattr(self.backend, 'device_note', None)}), flush=True)

    def _worker_loop(self):
        self._warm_backend()
        while not self.stop_event.is_set():
            try:
                _rank, _ticket, deadline, future, request = self.jobs.get(timeout=_JOB_TICK)
                self.lane_slots[request.get('priority', 'interactive')].release()
            except queue.Empty:
                continue
            try:
                self._run_job(deadline, future, request)
            finally:
                self.jobs.task_done()

    def close(self):
        self.stop_event.set()
        if self.server_sock:
            self.server_sock.close()
        # The singleton lock stays held until an in-flight native call has returned.
        for thread in self.loop_threads:
            thread.join()
        while not self.jobs.empty():
            self.jobs.get_nowait()[3].cancel()
            self.jobs.task_done()
        with self.live_conns_guard:
            conn_threads = list(self.live_conns)
        for thread in conn_threads:
            thread.join(timeout=_CONN_JOIN_SECONDS)
        if isinstance(self.backend, LocalModelRunner):
            self.backend.text = None
        if self.lifetime_lock:
            self.socket_path.unlink(missing_ok=True)
            if self.owns_token:
                (self.directory / 'token').unlink(missing_ok=True)
            self.lifetime_lock.close()
            self.lifetime_lock = None


class _Countdown:
    """Startup budget shared by every wait: rest() is what remains, left() caps it to one poll slice."""

    def __init__(self, seconds):
        self._deadline = time.monotonic() + seconds

    def rest(self):
        """Everything still left in the budget; raises once none is."""
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(_ERR['start_deadline'])
        return remaining

    def left(self):
        return min(self.rest(), _POLL_SLICE)


def _try_flock(handle):
    """Take the exclusive non-blocking lock; True if we got it."""
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _stamp(handle, text=''):
    handle.seek(0)
    handle.truncate()
    if text:
        handle.write(text)
    handle.flush()


class EmbeddingClient:
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, profile_home, *, timeout: float = 30, startup_timeout: float = 10, paused=None):
        # A single shared daemon serves every profile (_wincompat.shared_inference_home); memory DBs stay per profile.
        self.home = _wc.shared_inference_home(profile_home)
        # paused(): True refuses every embedding request before a daemon is reached or started (the on/off switch).
        self.paused = paused
        self.directory = self.home / 'neospike' / '.cache'
        self.timeout = timeout
        if startup_timeout <= 0 or not math.isfinite(startup_timeout):
            raise ValueError(_ERR['bad_startup'])
        self.startup_timeout = startup_timeout
        self._reported_warnings = []
        self._reported_warnings_lock = threading.Lock()

    def take_warnings(self):
        """Degradations the daemon reported since the previous call (e.g. an over-long input was truncated)."""
        with self._reported_warnings_lock:
            drained, self._reported_warnings = self._reported_warnings, []
        return drained

    def _ping(self, countdown):
        """One authenticated ping. Connecting gets a single slice; the reply gets the entire remaining budget.
        A daemon that took the connection but answers slowly is busy rather than dead, and reconnecting
        on every slice would only stack abandoned connections in its backlog."""
        try:
            self._exchange('ping', socket_timeout=countdown.left(), reply_timeout=countdown.rest())
        except TimeoutError:
            raise TimeoutError(_ERR['start_deadline']) from None

    def _daemon_responds(self, countdown):
        try:
            self._ping(countdown)
        except ConnectionRefusedError:
            return False
        except FileNotFoundError:
            pass
        else:
            return True
        # No token file: probe only the transport (files are never repaired here), so a live daemon
        # that lacks credentials cannot be mistaken for a dead one.
        try:
            _wc.connect_socket(self.directory / 'i.sock', countdown.left()).close()
        except (FileNotFoundError, ConnectionRefusedError):
            return False
        # The daemon may have appeared between the two probes, so authenticate one more time.
        try:
            self._ping(countdown)
        except FileNotFoundError:
            raise PermissionError(_ERR['no_credentials']) from None
        except ConnectionRefusedError:
            return False
        return True

    def _spawn_daemon(self, guard):
        # Detached: no shell, no console window, no inherited pipes, and no tie to this app's lifetime.
        with _open_owner_file(self.directory / 'inference.log') as log_file:
            _stamp(log_file)
            child_env = {name: setting for name, setting in os.environ.items() if name != 'PYTHONPATH'}
            child_env['HERMES_HOME'] = str(self.home)  # the shared daemon finds models/cache under its own home
            command = _wc.inference_command(self.home, ['--home', str(self.home),
                                                         *_wc.launch_fd_args(guard.fileno())], child_env)
            return subprocess.Popen(
                command, env=child_env, cwd=str(pathlib.Path(__file__).resolve().parent.parent),
                stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file,
                **_wc.detached_popen_kwargs(guard.fileno()))

    def _open_launch_guard(self):
        guard_path = self.directory / _LAUNCH_GUARD_NAME
        if not _wc.IS_WINDOWS:
            return _open_owner_file(guard_path)
        # Windows cannot hand a flock to a new process; the inherited no-sharing handle serves as the guard.
        try:
            return _open_owner_file(guard_path, unshared=True)
        except BlockingIOError:
            raise EmbeddingUnavailable(_ERR['launching']) from None

    def ensure_daemon(self):
        countdown = _Countdown(self.startup_timeout)
        if self._daemon_responds(countdown):
            return
        # A directory that already exists with unsafe ownership is rejected here.
        _ensure_owner_dir(self.directory)
        with _open_owner_file(self.directory / 'startup.lock') as gate:
            while not _try_flock(gate):
                time.sleep(min(_SPIN_SLEEP, countdown.left()))
            if self._daemon_responds(countdown):
                return
            # A daemon without its model files could only answer errors while staying resident.
            missing = missing_model_message(self.home)
            if missing:
                raise EmbeddingUnavailable(missing)
            # Failure backoff shared by every starter, including one that was killed mid-launch.
            gate.seek(0)
            stamped = gate.read().strip()
            if stamped and 0 <= time.monotonic() - float(stamped) < _FAILURE_BACKOFF_SECONDS:
                raise EmbeddingUnavailable(_ERR['recent_failure'])
            # The lifetime lock is authoritative, even if its owner is unable to answer a ping.
            with _open_owner_file(self.directory / 'owner.lock') as lifetime:
                if not _try_flock(lifetime):
                    raise EmbeddingUnavailable(_ERR['owner_busy'])
            # The second flock survives exec, so the child holds the guard from its first instruction
            # even when this starter exits; a timestamp or a PID could not pass that along.
            with self._open_launch_guard() as guard:
                if not _try_flock(guard):
                    raise EmbeddingUnavailable(_ERR['launching'])
                _stamp(gate, str(time.monotonic()))
                try:
                    child = self._spawn_daemon(guard)
                    # Reap the child for as long as we live; never kill it on timeout or exit (the lifetime lock owns it).
                    threading.Thread(target=child.wait, daemon=True).start()
                    while not self._daemon_responds(countdown):
                        if child.poll() is not None:
                            raise EmbeddingUnavailable(_ERR['spawn_failed'])
                        time.sleep(min(_SPIN_SLEEP, countdown.left()))
                    _stamp(gate)
                except BaseException:
                    _stamp(gate, str(time.monotonic()))
                    raise

    def request(self, op, value='', *, priority='interactive', **extra):
        if op in _OPS:
            if self.paused is not None and self.paused():
                raise EmbeddingUnavailable(PAUSED_MESSAGE)
            self.ensure_daemon()
        return self._exchange(op, value, priority=priority, **extra)

    def device(self):
        """Device the running daemon's model is on ('cpu' or 'directml'); None while it is still loading.
        Never starts a daemon (FileNotFoundError / ConnectionRefusedError when none runs)."""
        return self._exchange('ping', socket_timeout=self.startup_timeout).get('device')

    def request_shutdown(self):
        """Ask the running daemon to stop; True when it accepted, False when none runs. Never starts one."""
        try:
            self._exchange(SHUTDOWN_OP, socket_timeout=_SHUTDOWN_REPLY_SECONDS)
        except (FileNotFoundError, ConnectionRefusedError, ConnectionResetError):
            return False
        return True

    def _read_token(self):
        folder = self.directory.lstat()
        if (not stat.S_ISDIR(folder.st_mode)) or _wc.is_link_stat(folder) or _not_mine(folder) \
                or _wc.insecure_mode(folder.st_mode, self.directory):
            raise PermissionError(_ERR['dir_insecure'])
        fd = _wc.open_nofollow(self.directory / 'token', os.O_RDONLY | _wc.O_NONBLOCK)
        with os.fdopen(fd) as token_file:
            info = os.fstat(token_file.fileno())
            if not stat.S_ISREG(info.st_mode) or _not_mine(info) or info.st_nlink != 1 \
                    or _wc.insecure_mode(info.st_mode, token_file.fileno()):
                raise PermissionError(_ERR['token_insecure'])
            return token_file.read(_TOKEN_READ_LIMIT)

    def _exchange(self, op, value='', *, priority='interactive', socket_timeout=None, reply_timeout=None,
                  **extra):
        """socket_timeout limits the connect (and also the reply, unless reply_timeout is given)."""
        token = self._read_token()
        connect_wait = self.timeout + 1 if socket_timeout is None else socket_timeout
        with _wc.connect_socket(self.directory / 'i.sock', connect_wait) as conn:
            if reply_timeout is not None:
                conn.settimeout(reply_timeout)
            _write_frame(conn, dict(token=token, op=op, value=str(value),
                                    timeout=self.timeout, priority=priority, **extra))
            response = _read_frame(conn)
        if 'error' in response:
            raise EmbeddingUnavailable(response['error'])
        warnings = response.get('warnings')
        if warnings and isinstance(warnings, list):
            with self._reported_warnings_lock:
                self._reported_warnings.extend(str(warning) for warning in warnings)
        return response['result']

    def embed_query(self, passage):
        return self.request('query', passage)

    def embed_document(self, passage, *, title='none'):
        return self.request('document', passage, title=title, priority='indexing')


def daemon_users(profile_home):
    """The other homes served by the daemon of *profile_home* that still use Neospike: memory.provider is
    neospike and the on/off switch is on (the root home, then its Neospike profiles)."""
    from .cron.neospike_cron_common import configured_memory_provider, target_homes
    own = pathlib.Path(profile_home).expanduser().resolve()
    homes = target_homes(_wc.shared_inference_home(profile_home))
    return [home for home in homes if home.resolve() != own
            and configured_memory_provider(home) == _PROVIDER_ID and read_enabled(home)]


def release_daemon(profile_home):
    """(DAEMON_STOPPED | DAEMON_NOT_RUNNING | DAEMON_IN_USE, the homes still using it). *profile_home* stops
    using the shared daemon (uninstall, or its switch turned off); the daemon is asked to stop only when no
    other home uses it. OSError when a running daemon could not be reached."""
    users = daemon_users(profile_home)
    if users:
        return DAEMON_IN_USE, users
    stopped = EmbeddingClient(profile_home).request_shutdown()
    return (DAEMON_STOPPED if stopped else DAEMON_NOT_RUNNING), []


class _StopFlagSetter:
    """Signal handler object that sets an event (used instead of a lambda)."""

    def __init__(self, event):
        self._event = event

    def __call__(self, *_signal_args):
        self._event.set()


def main():
    parser = argparse.ArgumentParser(description='Foreground private Neospike inference; Ctrl-C/SIGTERM to stop')
    parser.add_argument('--home', default=_wc.default_hermes_home())
    parser.add_argument('--launch-fd', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    home = pathlib.Path(args.home).expanduser()
    os.environ.setdefault("HERMES_HOME", str(home))
    daemon = EmbeddingDaemon(home)
    stop_requested = daemon.shutdown_requested
    # On Windows a graceful stop is delivered as CTRL_BREAK_EVENT (SIGBREAK).
    for signal_number in (signal.SIGINT, signal.SIGTERM, *_wc.stop_signals()):
        signal.signal(signal_number, _StopFlagSetter(stop_requested))
    try:
        # However it was started, a daemon without its model files never reports ready.
        missing = missing_model_message()
        if missing:
            print(json.dumps({'error': missing}), file=sys.stderr, flush=True)
            raise SystemExit(_MODEL_MISSING_EXIT)
        daemon.start()
    finally:
        # Drop the inherited guard only after start() holds owner.lock for itself.
        if args.launch_fd is not None:
            _wc.close_launch_fd(args.launch_fd, daemon.directory / _LAUNCH_GUARD_NAME)
    print(json.dumps({'ready': str(daemon.socket_path), 'pid': os.getpid()}), flush=True)
    try:
        # Windows cannot break an unbounded wait with a signal, hence the short slice on that platform.
        while not stop_requested.wait(_MAIN_WAIT_SLICE if _wc.IS_WINDOWS else None):
            pass
    finally:
        daemon.close()


if __name__ == '__main__':
    main()
