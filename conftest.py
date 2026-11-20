from __future__ import annotations

import contextlib
import gc
import re
import shutil
import sqlite3
import sys
import types
from pathlib import Path

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
