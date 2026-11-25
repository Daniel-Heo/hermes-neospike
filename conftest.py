from __future__ import annotations

import contextlib
import gc
import re
import shutil
import sqlite3
import sys
import threading
import types
from pathlib import Path

import pytest

try:
    import agent.memory_provider  # noqa: F401
except ImportError:
    agent = types.ModuleType("agent")
    memory_provider = types.ModuleType("agent.memory_provider")

    class MemoryProvider:
        pass

    memory_provider.MemoryProvider = MemoryProvider
    agent.memory_provider = memory_provider
    sys.modules["agent"] = agent
    sys.modules["agent.memory_provider"] = memory_provider


# Tests must never launch the real hermes/claude/uv/hf (PEP 578 audit hook; no off switch).
_BLOCKED_PROGRAMS = frozenset({"hermes", "claude", "uv", "hf"})
_STRIPPED_EXTENSIONS = (".exe", ".cmd", ".bat")
_POPEN_EVENT = "subprocess.Popen"


class RealCommandBlocked(RuntimeError):
    """Raised by the audit hook before a blocked program's process is created."""


def _first_token(command) -> str:
    """First program token of *command*: list/tuple element 0, or the quote-aware first word of a string
    (Windows passes the already-joined command line)."""
    if isinstance(command, (list, tuple)):
        return str(command[0]) if command else ""
    text = str(command or "").lstrip()
    if text.startswith('"'):
        end = text.find('"', 1)
        return text[1:end] if end >= 0 else text[1:]
    return text.split(None, 1)[0] if text else ""


def _program_name(path: str) -> str:
    """Lower-case base name without .exe/.cmd/.bat."""
    name = re.split(r"[\\/]",str(path).strip())[-1].lower()
    for extension in _STRIPPED_EXTENSIONS:
        if name.endswith(extension):
            return name[: -len(extension)]
    return name


def blocked_program(executable, command) -> str | None:
    """The blocked program name *executable* (else the command's first token) launches, or None."""
    name = _program_name(executable if executable else _first_token(command))
    return name if name in _BLOCKED_PROGRAMS else None


def _guard_real_commands(event: str, args: tuple) -> None:
    if event != _POPEN_EVENT:
        return
    executable, command = args[0], args[1]
    name = blocked_program(executable, command)
    if name:
        raise RealCommandBlocked(f"Blocked a real `{name}` command in a test: "
                                 f"executable={executable!r}, args={command!r}. Pass a fake runner or fake subprocess.run.")


sys.addaudithook(_guard_real_commands)


_REPO_ROOT = Path(__file__).resolve().parent
_LEFTOVER_DIRS = (".pytest_cache",)


def _inside_repo(path: Path) -> bool:
    return path != _REPO_ROOT and _REPO_ROOT in path.parents


def _close_leaked_databases() -> None:
    """Tests that never call provider.shutdown() leave memory.db open (the conversation indexer's daemon thread
    keeps the provider alive), and Windows refuses to delete an open file. Shut those providers down, then close
    any sqlite connection still open."""
    gc.collect()
    provider_type = getattr(sys.modules.get("neospike.provider"), "NeospikeProvider", None)
    objects = gc.get_objects()
    for obj in objects:
        if provider_type is not None and isinstance(obj, provider_type):
            with contextlib.suppress(Exception):
                obj.shutdown()
    for obj in objects:
        if isinstance(obj, sqlite3.Connection):
            with contextlib.suppress(Exception):
                obj.close()


def pytest_sessionfinish(session, exitstatus):
    for name in _LEFTOVER_DIRS:
        shutil.rmtree(name, ignore_errors=True)
    # pytest clears --basetemp when a session starts but never when it ends, so a basetemp inside the repo
    # (a run with --basetemp=.test-tmp) kept a copy of every test home. Remove it once the session is done.
    basetemp = session.config.option.basetemp
    if basetemp and _inside_repo(Path(basetemp).resolve()):
        _close_leaked_databases()
        shutil.rmtree(Path(basetemp).resolve(), ignore_errors=True)


# Vector search goes through the embedding service: one in-process service in a throw-away home serves every test
# that searches in the pytest process (started on the first search, closed when the session ends). A child process a
# test starts (a dashboard CLI, an MCP server) searches through a service of the child's own home; such a service is
# stopped when its test ends (_no_service_outlives_its_test).
SEARCH_HOME_DIR = "search-service"


def _search_daemon_type(base):
    """The service without its start-up model load: these tests only search, and the load's report line would
    land in whichever test's captured stdout is running then. The model still loads on a first embedding."""

    class SearchFirstDaemon(base):
        def _warm_backend(self):
            return None

    return SearchFirstDaemon


class SessionSearchService:
    """Stands in for vector_index._search_client: starts the session's service on the first call."""

    def __init__(self, factory) -> None:
        self._factory = factory
        self._guard = threading.Lock()
        self.home = None
        self.daemon = None

    def __call__(self):
        from neospike import _wincompat as wc
        from neospike.inference import EmbeddingClient, EmbeddingDaemon
        with self._guard:
            if self.daemon is None:
                home = self._factory.mktemp(SEARCH_HOME_DIR)
                if wc.IS_WINDOWS:
                    wc.chmod(home, 0o700)  # drop any foreign ACE inherited from the temp root
                daemon = _search_daemon_type(EmbeddingDaemon)(home)
                daemon.start()
                self.home, self.daemon = home, daemon
        return EmbeddingClient(self.home)

    def state(self, path: str, table: str = "items", column: str = "embedding"):
        """The service's index of one database file and vector column (None before its first search)."""
        return self.daemon.searcher.state(path, table, column) if self.daemon is not None else None

    def close(self) -> None:
        with self._guard:
            if self.daemon is not None:
                self.daemon.close()
                self.daemon = None


@pytest.fixture(scope="session")
def search_service(tmp_path_factory):
    service = SessionSearchService(tmp_path_factory)
    yield service
    service.close()


@pytest.fixture(autouse=True)
def _vector_search_through_session_service(search_service, monkeypatch):
    from neospike import vector_index
    monkeypatch.setattr(vector_index, "_search_client", search_service)


SERVICE_ADDRESS_GLOB = "neospike/.cache/i.sock"  # the socket (or, in TCP mode, its port file) of a running service


def stop_services_under(root: Path) -> list[Path]:
    """Ask every embedding service whose files sit under root (a test's own homes) to stop; the homes asked."""
    from neospike.inference import EmbeddingClient
    asked = []
    for address in sorted(Path(root).rglob(SERVICE_ADDRESS_GLOB)):
        client = EmbeddingClient(address.parent.parent.parent)
        client.directory = address.parent  # this home's own service, whatever NEOSPIKE_INFERENCE_HOME says
        with contextlib.suppress(OSError, RuntimeError, ValueError):
            if client.request_shutdown():
                asked.append(address.parent.parent.parent)
    return asked


@pytest.fixture(autouse=True)
def _no_service_outlives_its_test(request):
    """A service started in a test's tmp_path (by the test or a child process of it) never outlives the test."""
    yield
    root = request.node.funcargs.get("tmp_path")
    if root is not None:
        stop_services_under(root)
