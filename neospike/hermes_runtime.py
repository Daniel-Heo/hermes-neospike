"""Where the Hermes launcher and its pythons are. Stdlib only, no package-relative imports:
the cron wrapper loads this file by path from <home>/plugins/neospike/.

Two different interpreters (measured on native Windows):
- runtime python: command[0] of `hermes --print-runtime-command`, a bare interpreter that becomes Hermes only
  through the hermes_bootstrap -c prologue. install.py runs its probes there.
- CLI python: a venv that has the Neospike dependencies (tokenizers, onnxruntime, ...). The Dashboard,
  the cron scripts and the MCP server run `-m neospike.*` with it. The runtime python lacks them.

Both are cached in <home>/neospike/runtime.json; a cached path is used only while the file exists.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CACHE_PARTS = ("neospike", "runtime.json")
CACHE_RUNTIME, CACHE_CLI = "runtime_python", "cli_python"
ENV_PYTHON = "HERMES_PYTHON"
LAUNCHER_TIMEOUT_SECONDS = 120
PROFILES_DIR, AGENT_DIR = "profiles", "hermes-agent"
WINDOWS_LAUNCHERS = ("hermes.exe", "hermes.cmd")
POSIX_LAUNCHERS = ("hermes",)


class HermesRuntimeNotFound(RuntimeError):
    """No candidate exists; the message lists every place tried and how to set HERMES_PYTHON."""


def _is_windows(windows: bool | None) -> bool:
    return os.name == "nt" if windows is None else windows


def venv_python(venv: Path, *, windows: bool | None = None) -> Path:
    return venv / ("Scripts/python.exe" if _is_windows(windows) else "bin/python")


def hermes_root_home(home) -> Path:
    """The home holding the Hermes code (hermes-agent); a profile home (<root>/profiles/<name>) maps to
    <root>, any other home to itself. Per-profile paths (plugins, neospike data) stay under *home*."""
    path = Path(home).expanduser()
    return path.parent.parent if path.parent.name == PROFILES_DIR else path


def hermes_agent_dir(home) -> Path:
    """<root home>/hermes-agent: profiles have no checkout of their own."""
    return hermes_root_home(home) / AGENT_DIR


def checkout_python(home: Path, *, windows: bool | None = None) -> Path:
    """<root home>/hermes-agent/venv python (backward compatible last resort)."""
    return venv_python(hermes_agent_dir(home) / "venv", windows=windows)


def cache_file(home: Path) -> Path:
    return Path(home).joinpath(*CACHE_PARTS)


def read_cache(home: Path) -> dict:
    try:
        data = json.loads(cache_file(home).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_cache(home: Path, values: dict) -> None:
    """Merge *values* into runtime.json atomically (temp file in the same folder + os.replace)."""
    target = cache_file(home)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {**read_cache(home), **values}
    descriptor, temporary = tempfile.mkstemp(prefix=".runtime-", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=1)
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _cached(home: Path, key: str) -> str | None:
    value = read_cache(home).get(key)
    return value if isinstance(value, str) and value and Path(value).is_file() else None


def launcher_candidates(home: Path, *, windows: bool | None = None, user_home: Path | None = None,
                        which=None) -> list[str]:
    """<root home>/bin/hermes(.exe) -> `hermes` on PATH -> ~/.local/bin/hermes (Desktop's link on macOS).
    A profile has no bin of its own; it runs the root launcher."""
    names = WINDOWS_LAUNCHERS if _is_windows(windows) else POSIX_LAUNCHERS
    found = (which or shutil.which)("hermes")
    local_bin = Path(user_home) if user_home is not None else Path.home()
    return ([str(hermes_root_home(home) / "bin" / name) for name in names] + ([found] if found else [])
            + [str(local_bin / ".local" / "bin" / name) for name in names])


def find_launcher(home: Path, *, windows: bool | None = None, user_home: Path | None = None,
                  which=None) -> str | None:
    found = shutil.which("hermes") if which is None else which("hermes")
    for candidate in launcher_candidates(home, windows=windows, user_home=user_home, which=which):
        if candidate == found or Path(candidate).is_file():
            return candidate
    return None


def launcher_python(launcher: str, runner=None) -> str | None:
    """command[0] of `<launcher> --print-runtime-command` (a JSON array), else None."""
    try:
        done = (runner or subprocess.run)([launcher, "--print-runtime-command"], capture_output=True, text=True,
                                          encoding="utf-8", errors="replace", timeout=LAUNCHER_TIMEOUT_SECONDS)
        command = json.loads(done.stdout) if done.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return command[0] if isinstance(command, list) and command and isinstance(command[0], str) else None


def _not_found(what: str, tried: list[str]) -> HermesRuntimeNotFound:
    return HermesRuntimeNotFound(
        f"Hermes {what} not found. Tried:\n  " + "\n  ".join(tried)
        + f"\nSet {ENV_PYTHON} to the Hermes venv python (e.g. `export {ENV_PYTHON}=~/.hermes/hermes-agent/venv/bin/python`"
        + f", Windows `setx {ENV_PYTHON} %LOCALAPPDATA%\\hermes\\hermes-agent\\venv\\Scripts\\python.exe`).")


def runtime_python(home: Path, *, launcher: str | None = None, environ=None, windows: bool | None = None,
                   user_home: Path | None = None, which=None, runner=None, cache: bool = True) -> tuple[str, str]:
    """(python Hermes really starts, where it came from).

    runtime.json -> launcher --print-runtime-command (cached on success) -> HERMES_PYTHON -> checkout venv.
    HERMES_PYTHON names the CLI venv, a different interpreter from the gateway's, so it follows the launcher
    cache=False neither reads nor writes runtime.json (--check, --dry-run).
    """
    home = Path(home)
    environ = os.environ if environ is None else environ
    tried = []
    if cache:
        cached = _cached(home, CACHE_RUNTIME)
        if cached:
            return cached, f"{cache_file(home)}"
        tried.append(f"{cache_file(home)} ({CACHE_RUNTIME})")
    launcher = launcher or find_launcher(home, windows=windows, user_home=user_home, which=which)
    if launcher:
        python = launcher_python(launcher, runner)
        if python:
            if cache:
                write_cache(home, {CACHE_RUNTIME: python})
            return python, f"{launcher} --print-runtime-command"
        tried.append(f"{launcher} --print-runtime-command (no command)")
    else:
        tried += [f"launcher {path}" for path in launcher_candidates(home, windows=windows, user_home=user_home,
                                                                     which=which)]
    configured = environ.get(ENV_PYTHON)
    if configured:
        return configured, ENV_PYTHON
    tried.append(f"{ENV_PYTHON} (unset)")
    venv = checkout_python(home, windows=windows)
    if venv.is_file():
        return str(venv), "checkout venv"
    tried.append(str(venv))
    raise _not_found("runtime python", tried)


def cli_python(home: Path, *, environ=None, windows: bool | None = None) -> str:
    """The venv python with the Neospike dependencies: HERMES_PYTHON -> runtime.json -> checkout venv.

    install.py records its choice in runtime.json (on macOS the PM environment venv when there is no checkout
    venv), so later callers find it without running the launcher.
    """
    home = Path(home)
    environ = os.environ if environ is None else environ
    configured = environ.get(ENV_PYTHON)
    if configured:
        return configured
    cached = _cached(home, CACHE_CLI)
    if cached:
        return cached
    venv = checkout_python(home, windows=windows)
    if venv.is_file():
        return str(venv)
    raise _not_found("CLI python", [f"{ENV_PYTHON} (unset)", f"{cache_file(home)} ({CACHE_CLI})", str(venv)])
