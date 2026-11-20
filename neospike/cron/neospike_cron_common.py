"""Shared helpers of the Neospike cron scripts. Stdlib only.

install.py copies this file next to the scripts into <HERMES_HOME>/scripts/ (the only folder Hermes cron
runs scripts from), so a script's home is the parent of its folder. The CLI runs the way the Dashboard
runs it (plugin_api.py): hermes_runtime.cli_python (HERMES_PYTHON, runtime.json, checkout venv), PYTHONPATH=<home>/plugins,
HERMES_HOME, HERMES_BGE_M3_DIR=<home>/models/bge-m3-onnx.

Run directly, it is a CLI wrapper `<python> neospike_cron_common.py [--home <home>] <cli args>`
(without --home: the root home), and `--list-homes` prints target_homes() one per line.

One cron in the root home serves every profile (<root>/profiles/<name>) whose config.yaml sets
memory.provider to neospike and that has a <profile>/neospike folder.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

CLI_MODULE = "neospike.cli"
# Variables the Hermes cron overlay may set for its own interpreter; they would redirect the CLI's venv python.
_FOREIGN_ENV = ("PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP")
PROFILES_DIR, DATA_DIR, PROVIDER_NAME = "profiles", "neospike", "neospike"
HOME_OPTION, LIST_HOMES_OPTION = "--home", "--list-homes"


def home_dir() -> Path:
    """<home>/scripts/<this file> -> <home>."""
    return Path(__file__).resolve().parent.parent


def yaml_scalar(value: str) -> str:
    return value.split(" #", 1)[0].strip().strip("'\"")


def _yaml_data(text: str):
    """PyYAML result, or None when PyYAML is absent (the cron python) or the file does not parse."""
    try:
        import yaml
    except ImportError:
        return None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None


def configured_memory_provider(home: Path) -> str | None:
    """memory.provider of <home>/config.yaml (PyYAML when importable, else the top-level `memory:` block)."""
    try:
        text = (home / "config.yaml").read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    data = _yaml_data(text)
    if data is not None:
        memory = data.get("memory") if isinstance(data, dict) else None
        value = memory.get("provider") if isinstance(memory, dict) else None
        return None if value is None else str(value)
    inside, indent = False, None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            inside, indent = re.match(r"memory:\s*(#.*)?$", line) is not None, None
            continue
        if inside:
            width = len(line) - len(line.lstrip())
            indent = width if indent is None else indent
            match = re.match(r"\s*provider:(.*)$", line)
            if width == indent and match:
                return yaml_scalar(match.group(1)) or None
    return None


def target_homes(root: Path) -> list[Path]:
    """*root*, then (sorted) every <root>/profiles/<name> that uses Neospike: memory.provider is neospike and
    <profile>/neospike exists. Read at every run, so a profile added later is picked up."""
    profiles = root / PROFILES_DIR
    found = sorted(path for path in profiles.iterdir() if path.is_dir()) if profiles.is_dir() else []
    return [root] + [path for path in found
                     if (path / DATA_DIR).is_dir() and configured_memory_provider(path) == PROVIDER_NAME]


def _hermes_runtime(home: Path):
    """neospike/hermes_runtime.py loaded by path: the cron python cannot import the package (no PYTHONPATH)."""
    path = home / "plugins" / "neospike" / "hermes_runtime.py"
    spec = importlib.util.spec_from_file_location("neospike_hermes_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli_python(home: Path) -> str:
    """Same choice as the Dashboard (hermes_runtime.cli_python)."""
    return _hermes_runtime(home).cli_python(home)


def cli_env(home: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in _FOREIGN_ENV}
    env.update({
        "HERMES_HOME": str(home),
        "PYTHONPATH": str(home / "plugins"),
        "HERMES_BGE_M3_DIR": str(home / "models" / "bge-m3-onnx"),
        "PYTHONIOENCODING": "utf-8",
    })
    return env


def cli_command(home: Path, args: list[str]) -> list[str]:
    return [cli_python(home), "-m", CLI_MODULE, "--home", str(home), *args]


def run_cli(home: Path, args: list[str], timeout: float) -> subprocess.CompletedProcess:
    """Run one CLI command with captured UTF-8 output; cwd=home so the result does not depend on the cron cwd."""
    return subprocess.run(cli_command(home, args), capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=cli_env(home), cwd=str(home))


def last_line(*texts: str) -> str:
    """Last non-empty line of the first text that has one (stderr before stdout)."""
    for text in texts:
        lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return "(no output)"


def main(argv: list[str]) -> int:
    home = home_dir()
    if argv[:1] == [LIST_HOMES_OPTION]:
        print("\n".join(str(path) for path in target_homes(home)))
        return 0
    if argv[:1] == [HOME_OPTION] and len(argv) >= 2:
        home, argv = Path(argv[1]).expanduser().resolve(), argv[2:]
    return subprocess.run(cli_command(home, argv), env=cli_env(home), cwd=str(home)).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
