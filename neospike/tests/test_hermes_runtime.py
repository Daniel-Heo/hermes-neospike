"""One place finds the Hermes launcher and its runtime python, and builds the command that runs a Neospike module
there through hermes_launch.py (tmp homes only)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from neospike import hermes_runtime as hr

LAYOUTS = [pytest.param(True, "hermes.exe", "Scripts/python.exe", id="windows"),
           pytest.param(False, "hermes", "bin/python", id="posix")]


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def _no_path(_name):
    return None


class _PrintRuntime:
    """Fake subprocess.run for `<launcher> --print-runtime-command`."""

    def __init__(self, python: str) -> None:
        self.python, self.commands = python, []

    def __call__(self, command, **_kwargs):
        self.commands.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps([self.python, "-I", "-c", "..."]), "")


class _NoLauncherRun:
    """Runner that fails the test if a library caller runs the Hermes launcher."""

    def __init__(self) -> None:
        self.commands = []

    def __call__(self, command, **_kwargs):
        self.commands.append(command)
        raise AssertionError(f"launcher run: {command}")


@pytest.fixture
def outside_hermes(monkeypatch):
    """This test process is not a bootstrapped Hermes process, and no checkout override is set."""
    monkeypatch.delitem(sys.modules, hr.BOOTSTRAP_MODULE, raising=False)
    monkeypatch.delenv(hr.ENV_CHECKOUT, raising=False)


def test_launch_command_is_runtime_python_launch_script_checkout_and_target(tmp_path, outside_hermes):
    home = tmp_path / "home"
    runtime = str(_touch(tmp_path / "runtime" / "python.exe"))
    hr.write_cache(home, {hr.CACHE_RUNTIME: runtime})
    command = hr.launch_command(home, ["neospike.cli", "--help"], environ={})
    assert command == [runtime, "-I", str(hr.launch_script()), str(home / "hermes-agent"), "neospike.cli", "--help"]
    assert hr.launch_script() == Path(hr.__file__).resolve().with_name("hermes_launch.py")
    assert hr.launch_script().is_file()


def test_launch_command_never_runs_the_launcher(tmp_path, outside_hermes):
    """Outside install.py no launcher runs (it may finish a pending Hermes update): HERMES_PYTHON follows."""
    home, runner = tmp_path / "home", _NoLauncherRun()
    launcher = str(_touch(home / "bin" / "hermes.exe"))
    command = hr.launch_command(home, ["neospike.cli"], environ={"HERMES_PYTHON": "/chosen/python"},
                                which={"hermes": launcher}.get, runner=runner)
    assert command[0] == "/chosen/python" and runner.commands == []
    assert not hr.cache_file(home).exists()


def test_bootstrapped_process_launches_with_its_own_interpreter(tmp_path, outside_hermes, monkeypatch):
    monkeypatch.setitem(sys.modules, hr.BOOTSTRAP_MODULE, object())
    command = hr.launch_command(tmp_path / "home", ["neospike.inference"], runner=_NoLauncherRun())
    assert command[0] == sys.executable


def test_profile_without_its_own_cache_uses_the_root_runtime_python(tmp_path, outside_hermes):
    root = tmp_path / "home"
    profile = root / "profiles" / "design"
    runtime = str(_touch(tmp_path / "runtime" / "python.exe"))
    hr.write_cache(root, {hr.CACHE_RUNTIME: runtime})
    command = hr.launch_command(profile, ["neospike.cli"], environ={}, runner=_NoLauncherRun())
    assert command[0] == runtime and command[3] == str(root / "hermes-agent")


def test_checkout_override_names_the_bootstrap_folder(tmp_path, outside_hermes, monkeypatch):
    monkeypatch.setenv(hr.ENV_CHECKOUT, str(tmp_path / "checkout"))
    command = hr.launch_command(tmp_path / "home", ["neospike.cli"], python="/runtime/python")
    assert command[:4] == ["/runtime/python", "-I", str(hr.launch_script()), str(tmp_path / "checkout")]


@pytest.mark.parametrize(("windows", "launcher", "venv"), LAYOUTS)
def test_vanished_cache_entries_are_ignored(tmp_path, windows, launcher, venv):
    home = tmp_path / "home"
    checkout = _touch(home / "hermes-agent" / "venv" / venv)
    hr.write_cache(home, {hr.CACHE_RUNTIME: str(tmp_path / "gone")})
    python, source = hr.runtime_python(home, environ={}, windows=windows, user_home=tmp_path / "user",
                                       which=_no_path, runner=_PrintRuntime("unused"))
    assert (python, source) == (str(checkout), "checkout venv")


@pytest.mark.parametrize(("windows", "launcher", "venv"), LAYOUTS)
def test_local_bin_launcher_is_found_and_cached(tmp_path, windows, launcher, venv):
    """macOS Desktop links `hermes` into ~/.local/bin; that alone is enough (no checkout venv)."""
    home, user = tmp_path / "home", tmp_path / "user"
    fake = _touch(user / ".local" / "bin" / launcher)
    assert hr.find_launcher(home, windows=windows, user_home=user, which=_no_path) == str(fake)
    runner = _PrintRuntime(str(tmp_path / "runtime" / "python"))
    _touch(Path(runner.python))
    python, source = hr.runtime_python(home, environ={"HERMES_PYTHON": "/cli/venv/python"}, windows=windows,
                                       user_home=user, which=_no_path, runner=runner)
    assert python == runner.python and source == f"{fake} --print-runtime-command"  # launcher beats HERMES_PYTHON
    assert hr.read_cache(home) == {hr.CACHE_RUNTIME: runner.python}
    again = hr.runtime_python(home, environ={}, windows=windows, user_home=user, which=_no_path, runner=runner)
    assert again[0] == runner.python and len(runner.commands) == 1  # second call reads runtime.json


@pytest.mark.parametrize(("windows", "launcher", "venv"), LAYOUTS)
def test_launcher_order_home_bin_then_path_then_local_bin(tmp_path, windows, launcher, venv):
    home, user = tmp_path / "home", tmp_path / "user"
    _touch(user / ".local" / "bin" / launcher)
    on_path = str(_touch(tmp_path / "path" / launcher))
    assert hr.find_launcher(home, windows=windows, user_home=user, which={"hermes": on_path}.get) == on_path
    in_home = _touch(home / "bin" / launcher)
    assert hr.find_launcher(home, windows=windows, user_home=user, which={"hermes": on_path}.get) == str(in_home)


@pytest.mark.parametrize(("windows", "launcher", "venv"), LAYOUTS)
def test_profile_uses_root_bin_launcher_before_path(tmp_path, windows, launcher, venv):
    """A profile has no bin; a stale `hermes` on PATH must not beat <root>/bin."""
    root, user = tmp_path / "home", tmp_path / "user"
    profile = root / "profiles" / "design"
    profile.mkdir(parents=True)
    on_path = str(_touch(tmp_path / "path" / launcher))
    in_root = _touch(root / "bin" / launcher)
    assert hr.find_launcher(profile, windows=windows, user_home=user, which={"hermes": on_path}.get) == str(in_root)


def test_live_launcher_script_runs(tmp_path):
    """A real fake launcher process (python script) answers --print-runtime-command."""
    script = tmp_path / "fake_hermes.py"
    script.write_text("import json, sys\nprint(json.dumps([sys.executable, '-I']))\n", encoding="utf-8")
    def runner(command, **kwargs):  # the runner gets [launcher, flag]; prefix the interpreter
        return subprocess.run([sys.executable, str(script), *command[1:]], **kwargs)
    assert hr.launcher_python(str(script), runner) == sys.executable


@pytest.mark.parametrize(("windows", "launcher", "venv"), LAYOUTS)
def test_nothing_found_names_every_place_and_hermes_python(tmp_path, windows, launcher, venv):
    home, user = tmp_path / "home", tmp_path / "user"
    with pytest.raises(hr.HermesRuntimeNotFound) as runtime_error:
        hr.runtime_python(home, environ={}, windows=windows, user_home=user, which=_no_path)
    message = str(runtime_error.value)
    for place in (hr.cache_file(home), home / "bin" / launcher, user / ".local" / "bin" / launcher,
                  home / "hermes-agent" / "venv" / venv):
        assert str(place) in message, message
    assert "Set HERMES_PYTHON" in message
    assert not hr.cache_file(home).exists()  # a failed lookup writes nothing


def test_cache_false_neither_reads_nor_writes(tmp_path):
    home = tmp_path / "home"
    hr.write_cache(home, {hr.CACHE_RUNTIME: str(_touch(tmp_path / "stale"))})
    before = hr.cache_file(home).read_bytes()
    runner = _PrintRuntime("/fresh/python")
    launcher = str(_touch(home / "bin" / "hermes"))
    assert hr.runtime_python(home, launcher=launcher, runner=runner, cache=False)[0] == "/fresh/python"
    assert hr.cache_file(home).read_bytes() == before
