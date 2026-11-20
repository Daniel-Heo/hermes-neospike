"""Tests cannot launch the real hermes/claude/uv/hf; install.py looks subprocess.run up at call time."""
from __future__ import annotations

import importlib.util
import inspect
import subprocess
from pathlib import Path

import pytest

import conftest

REPO_ROOT = Path(__file__).resolve().parents[2]
MISSING_HERMES = "C:/nonexistent_dir/hermes.exe"
FAKE_CLAUDE = "C:/nonexistent_dir/claude.cmd"
MIN_RUNNER_FUNCTIONS = 6


def _install_module():
    spec = importlib.util.spec_from_file_location("install_no_real_commands", REPO_ROOT / "install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Recorder:
    """Fake subprocess.run: records every command, never launches anything."""

    def __init__(self):
        self.commands: list[list[str]] = []

    def __call__(self, command, *args, **kwargs):
        self.commands.append(list(command))
        return subprocess.CompletedProcess(command, 1, "", "")


class _WhichClaudeOnly:
    """Fake shutil.which: only `claude` exists (at a path that does not exist on disk)."""

    def __call__(self, name, *args, **kwargs):
        return FAKE_CLAUDE if name == "claude" else None


@pytest.mark.parametrize("executable, command", [
    (None, ["hermes"]),
    (None, ["HERMES.EXE", "--version"]),
    (None, r'"C:\x y\claude.cmd" mcp list'),
    (None, "uv pip install"),
    (None, ["hf", "download"]),
    (None, "hf download"),
    (r"C:\bin\Claude.BAT", "anything"),
])
def test_guard_blocks_real_programs(executable, command):
    assert conftest.blocked_program(executable, command) is not None


@pytest.mark.parametrize("executable, command", [
    (None, r'C:\Python\python.exe -c "print(1)"'),
    (None, ["python.exe", "-c", "print(1)"]),
    (None, r'"C:\hermes dir\python.exe" -c "x"'),
    (None, ""),
])
def test_guard_passes_other_programs(executable, command):
    assert conftest.blocked_program(executable, command) is None


def test_subprocess_run_is_blocked_before_process_creation():
    with pytest.raises(conftest.RealCommandBlocked):
        subprocess.run([MISSING_HERMES, "--version"], capture_output=True, timeout=5)


def test_install_runner_defaults_are_none():
    install = _install_module()
    defaults = {}
    for name, function in inspect.getmembers(install, inspect.isfunction):
        parameter = inspect.signature(function).parameters.get("runner")
        if function.__module__ == install.__name__ and parameter is not None:
            defaults[name] = parameter.default
    assert len(defaults) >= MIN_RUNNER_FUNCTIONS, defaults
    assert {name: value for name, value in defaults.items() if value is not None} == {}


def test_check_installation_without_runner_uses_patched_run(monkeypatch, tmp_path):
    install = _install_module()
    recorder = _Recorder()
    monkeypatch.setattr(install.subprocess, "run", recorder)
    install.audit_installation(tmp_path, "hermes", [])
    assert recorder.commands and recorder.commands[0] == ["hermes", "memory", "status"]


def test_register_mcp_without_runner_uses_patched_run(monkeypatch, tmp_path):
    install = _install_module()
    recorder = _Recorder()
    monkeypatch.setattr(install.subprocess, "run", recorder)
    monkeypatch.setattr(install.shutil, "which", _WhichClaudeOnly())
    install.apply_mcp_registration("python", tmp_path)
    assert [command[:3] for command in recorder.commands][-1] == [FAKE_CLAUDE, "mcp", "add-json"]
