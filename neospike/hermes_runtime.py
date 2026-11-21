"""Where the Hermes launcher and its runtime python are, and the command that runs a Neospike module there.
Stdlib only, no package-relative imports: the cron wrapper loads this file by path from <home>/plugins/neospike/.

The runtime python is command[0] of `hermes --print-runtime-command`, a bare interpreter that becomes Hermes
only through the hermes_bootstrap prologue, which selects the dependency environment the Hermes package
manager prepared (plugin.yaml python_dependencies). Every process outside the gateway (MCP server, cron
scripts, embedding daemon, Dashboard CLI calls, model download) runs through hermes_launch.py with it
(launch_command), so no package is installed into any Hermes environment and no environment folder is
written into a command. The runtime python is cached in <home>/neospike/runtime.json; a cached path is used
only while the file exists.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CACHE_PARTS = ("neospike", "runtime.json")
CACHE_RUNTIME = "runtime_python"
LAUNCH_SCRIPT = "hermes_launch.py"
ISOLATED_FLAG = "-I"
BOOTSTRAP_MODULE = "hermes_bootstrap"
ENV_PYTHON = "HERMES_PYTHON"
ENV_CHECKOUT = "NEOSPIKE_HERMES_CHECKOUT"
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


def write_cache(home: Path, values: dict, drop: tuple[str, ...] = ()) -> None:
    """Merge *values* into runtime.json and remove the *drop* keys, atomically (temp file in the same
    folder + os.replace). Every other key stays as it is."""
    target = cache_file(home)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = {key: value for key, value in read_cache(home).items() if key not in drop}
    data.update(values)
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
                   user_home: Path | None = None, which=None, runner=None, cache: bool = True,
                   run_launcher: bool = True) -> tuple[str, str]:
    """(python Hermes really starts, where it came from).

    runtime.json -> launcher --print-runtime-command (cached on success) -> HERMES_PYTHON -> checkout venv.
    HERMES_PYTHON names the CLI venv, a different interpreter from the gateway's, so it follows the launcher
    cache=False neither reads nor writes runtime.json (--check, --dry-run). run_launcher=False skips the
    launcher (it may finish a pending Hermes update when it starts), for every caller but install.py.
    """
    home = Path(home)
    environ = os.environ if environ is None else environ
    tried = []
    if cache:
        cached = _cached(home, CACHE_RUNTIME)
        if cached:
            return cached, f"{cache_file(home)}"
        tried.append(f"{cache_file(home)} ({CACHE_RUNTIME})")
    launcher = launcher or (find_launcher(home, windows=windows, user_home=user_home, which=which)
                            if run_launcher else None)
    if not run_launcher:
        tried.append("launcher (not run outside install.py)")
    elif launcher:
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


def current_runtime_python(home: Path, **runtime_options) -> str:
    """This interpreter when the Hermes bootstrap already runs in this process (the gateway, or a module
    started by hermes_launch.py), else runtime_python(home) without running the launcher (runtime.json
    that install.py wrote, a profile's own and then its root home's -> HERMES_PYTHON -> checkout venv)."""
    if BOOTSTRAP_MODULE in sys.modules:
        return sys.executable
    home, root = Path(home), hermes_root_home(home)
    if root != home.expanduser() and not _cached(home, CACHE_RUNTIME):
        shared = _cached(root, CACHE_RUNTIME)
        if shared:
            return shared
    return runtime_python(home, **{"run_launcher": False, **runtime_options})[0]


def launch_script() -> Path:
    """hermes_launch.py next to this file: the copy of the plugin that is running."""
    return Path(__file__).resolve().with_name(LAUNCH_SCRIPT)


def launch_command(home: Path, target, *, python: str | None = None, script: Path | None = None,
                   **runtime_options) -> list[str]:
    """[runtime python, -I, hermes_launch.py, <hermes checkout>, *target]; *target* is `module, args...`.
    *python* defaults to current_runtime_python(home, **runtime_options)
    (HermesRuntimeNotFound when there is none). The command names no dependency environment folder: the
    bootstrap picks the current one at every start."""
    python = python or current_runtime_python(home, **runtime_options)
    return [str(python), ISOLATED_FLAG, str(script or launch_script()), str(hermes_checkout(home)), *target]


def hermes_checkout(home: Path) -> Path:
    """The Hermes checkout hermes_bootstrap is imported from: NEOSPIKE_HERMES_CHECKOUT, else hermes_agent_dir."""
    configured = os.environ.get(ENV_CHECKOUT)
    return Path(configured).expanduser() if configured else hermes_agent_dir(home)
