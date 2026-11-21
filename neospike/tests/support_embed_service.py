"""Shared helpers for the embedding-service and setup-route tests."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

from neospike import _wincompat as wc
from neospike import hermes_runtime
from neospike.inference import EmbeddingClient
from neospike.tests.test_wincompat import HARD_KILL, wait_pid_exit

ON_WINDOWS = os.name == 'nt'
SLEEPER_CODE = 'import time; time.sleep(60)'
WAIT_SLICE = .02
# Seconds the client keeps refusing a new launch after a failed one (inference._FAILURE_BACKOFF_SECONDS).
BACKOFF_WINDOW = 5


def cache_dir(home: Path) -> Path:
    return Path(home) / 'neospike' / '.cache'


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    return env


def poll_until(check, seconds: float) -> bool:
    """Call check() until it returns truthy or the time is up."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if check():
            return True
        time.sleep(WAIT_SLICE)
    return bool(check())


FAKE_CHECKOUT_DIR = 'hermes-checkout'
FAKE_BOOTSTRAP = '"""Test stand-in for the Hermes checkout\'s hermes_bootstrap: selects nothing."""\n'


def fake_hermes_checkout(directory: Path) -> Path:
    """A folder with an empty hermes_bootstrap.py, for hermes_launch.py to import instead of Hermes's."""
    checkout = Path(directory) / FAKE_CHECKOUT_DIR
    checkout.mkdir(parents=True, exist_ok=True)
    (checkout / (hermes_runtime.BOOTSTRAP_MODULE + '.py')).write_text(FAKE_BOOTSTRAP, encoding='utf-8')
    return checkout


def use_test_runtime(tmp_path: Path, monkeypatch) -> Path:
    """Child processes start through hermes_launch.py on this interpreter and the fake checkout, never the
    installed Hermes (HERMES_PYTHON is the runtime python fallback when there is no runtime.json)."""
    checkout = fake_hermes_checkout(tmp_path)
    monkeypatch.setenv(hermes_runtime.ENV_PYTHON, sys.executable)
    monkeypatch.setenv(hermes_runtime.ENV_CHECKOUT, str(checkout))
    return checkout


@contextmanager
def profile_root(tmp_path: Path, monkeypatch):
    """Yield a directory that holds throw-away profile homes.

    POSIX uses a short /tmp path so the AF_UNIX path limit is never hit; Windows uses pytest's temp
    root (it purges itself, and a just-killed venv launcher may still hold a log file there).
    """
    use_test_runtime(tmp_path, monkeypatch)
    if ON_WINDOWS:
        wc.chmod(tmp_path, 0o700)  # drop any foreign ACE inherited from the temp root
        holder = nullcontext(str(tmp_path))
    else:
        holder = tempfile.TemporaryDirectory(prefix='neo5-', dir='/tmp')
    with holder as name:
        yield Path(name)


def stop_daemon_of(home: Path, seconds: float = 5) -> None:
    """Terminate whatever daemon still answers for this profile (best effort)."""
    try:
        pid = EmbeddingClient(home, timeout=.2).request('ping')['pid']
    except (OSError, RuntimeError, ValueError):
        return
    os.kill(pid, signal.SIGTERM)
    if ON_WINDOWS:  # TerminateProcess skips cleanup: wait until the handles are released
        wait_pid_exit(pid, seconds)
    else:
        sock = cache_dir(home) / 'i.sock'
        poll_until(path_gone(sock), seconds)


class _Gone:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self) -> bool:
        return not self.path.exists()


def path_gone(path: Path) -> _Gone:
    """Predicate object: true once path no longer exists."""
    return _Gone(path)


def share_with_others(path: Path) -> None:
    """Make path readable by other accounts (0o644/0o755, or an Everyone read ACE on Windows)."""
    if ON_WINDOWS:
        subprocess.run(['icacls', str(path), '/grant', '*S-1-1-0:(R)'], check=True, capture_output=True)
    else:
        path.chmod(0o755 if path.is_dir() else 0o644)


def lock_down(path: Path) -> None:
    if ON_WINDOWS:
        subprocess.run(['icacls', str(path), '/remove:g', '*S-1-1-0'], check=True, capture_output=True)
    else:
        path.chmod(0o700 if path.is_dir() else 0o600)


def permission_snapshot(path: Path):
    """Mode bits on POSIX; owner and DACL entries on Windows."""
    return wc._read_security(path) if ON_WINDOWS else path.stat().st_mode & 0o777


def make_private_dir(home: Path) -> Path:
    target = cache_dir(home)
    target.mkdir(parents=True, mode=0o700)
    return target


class SleepingPopen:
    """Stands in for subprocess.Popen: the daemon command is replaced by a process that never gets
    ready, while real exec and inherited descriptors still happen."""

    def __init__(self, real_popen, launch_log: Path | None = None):
        self.real_popen = real_popen
        self.launch_log = launch_log
        self.children: list = []

    def __call__(self, argv, **kwargs):
        child = self.real_popen([sys.executable, '-c', SLEEPER_CODE], **kwargs)
        self.children.append(child)
        if self.launch_log is not None:
            with self.launch_log.open('a') as out:
                out.write(f'{child.pid}\n')
        return child

    def reap(self) -> None:
        for child in self.children:
            child.kill()
            child.wait(timeout=5)


def spawn_python(source: str, *args: str, **popen_kwargs) -> subprocess.Popen:
    popen_kwargs.setdefault('stdout', subprocess.PIPE)
    popen_kwargs.setdefault('stderr', subprocess.PIPE)
    return subprocess.Popen([sys.executable, '-c', source, *args], text=True, env=clean_env(), **popen_kwargs)


def spawn_service(home: Path, *extra: str, **popen_kwargs) -> subprocess.Popen:
    argv = [sys.executable, '-m', 'neospike.inference', '--home', str(home), *extra]
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=clean_env(), **popen_kwargs)


def kill_and_wait(process: subprocess.Popen, seconds: float = 5) -> None:
    if process.poll() is None:
        process.kill()
    process.wait(timeout=seconds)


def ping_pid(client: EmbeddingClient) -> int:
    return client.request('ping')['pid']


def hard_kill(pid: int) -> None:
    os.kill(pid, HARD_KILL)


class RecordingBackend:
    """Daemon backend that logs each call and answers a fixed vector."""

    def __init__(self, vector=None):
        self.vector = [1.0] if vector is None else vector
        self.calls: list = []
        self.guard = threading.Lock()

    def __call__(self, op, value, **kwargs):
        with self.guard:
            self.calls.append((op, value))
        return list(self.vector)

    def values(self) -> list:
        with self.guard:
            return [value for _, value in self.calls]


class GatedBackend(RecordingBackend):
    """Blocks inside the call whose value equals `hold` until release is set."""

    def __init__(self, hold: str, vector=None):
        super().__init__(vector)
        self.hold = hold
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, op, value, **kwargs):
        out = super().__call__(op, value, **kwargs)
        if value == self.hold:
            self.entered.set()
            self.release.wait(5)
        return out
