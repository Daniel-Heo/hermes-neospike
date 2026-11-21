"""hermes_launch.py follows the Hermes launcher prologue, in its order, with a stand-in hermes_bootstrap.

Order of the prologue (the command `hermes --print-runtime-command` prints): drop PYTHONHOME, PYTHONPATH and
VIRTUAL_ENV -> checkout first on sys.path -> sys._hermes_pin_default_home -> import hermes_bootstrap ->
HERMES_HOME from hermes_constants when unset. Neospike adds the plugins folder at the end, after the bootstrap.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neospike import hermes_launch as launch

REPO_PACKAGE = Path(launch.__file__).resolve().parent
DEFAULT_ROOT = "default-hermes-root"
# The stand-in bootstrap records what the real one would see when it is imported.
RECORDING_BOOTSTRAP = """import os, sys
SEEN = {
    "path": list(sys.path),
    "env": {name: os.environ.get(name) for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "HERMES_HOME")},
    "pin": getattr(sys, "_hermes_pin_default_home", None),
    "argv": list(sys.argv),
}
"""
CONSTANTS = "from pathlib import Path\n\ndef get_default_hermes_root():\n    return Path({root!r})\n"
TARGET_MODULE = "neospike_launch_target"
TARGET_CODE = "import sys\nsys.modules['neospike_launch_target_ran'] = {'name': __name__, 'argv': list(sys.argv)}\n"
HERMES_MODULES = ("hermes_bootstrap", "hermes_constants", TARGET_MODULE, TARGET_MODULE + "_ran")
PROCESS_TIMEOUT_SECONDS = 60
CODE_RUNNERS = {"exec", "eval", "compile"}
PROBE_MODULE = "neospike_launch_probe"
PROBE_CODE = ("import json, os, sys, neospike\nprint(json.dumps({'path': sys.path, 'file': neospike.__file__, "
              "'pythonpath': os.environ.get('PYTHONPATH'), 'argv': sys.argv}))\n")
# The launcher's dropped variables, written out here (not taken from the module under test).
LAUNCHER_DROPPED = ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")


def _checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "hermes_bootstrap.py").write_text(RECORDING_BOOTSTRAP, encoding="utf-8")
    (checkout / "hermes_constants.py").write_text(CONSTANTS.format(root=str(tmp_path / DEFAULT_ROOT)),
                                                  encoding="utf-8")
    return checkout


@pytest.fixture
def clean_process(monkeypatch):
    """sys.path (without this repository, which pytest and an editable install put there), sys.argv, the pin
    flag, the launcher's variables and the stand-in modules are restored."""
    foreign = {os.path.normcase(str(REPO_PACKAGE)), os.path.normcase(str(REPO_PACKAGE.parent))}
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path
                                      if os.path.normcase(os.path.abspath(entry or os.curdir)) not in foreign])
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(sys, launch.PIN_DEFAULT_HOME_FLAG, None, raising=False)
    for name in (*LAUNCHER_DROPPED, "HERMES_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name in HERMES_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield monkeypatch
    for name in HERMES_MODULES:
        sys.modules.pop(name, None)


def test_prepare_follows_the_launcher_prologue_order(tmp_path, clean_process):
    checkout = _checkout(tmp_path)
    for name in LAUNCHER_DROPPED:
        clean_process.setenv(name, str(tmp_path / "foreign"))
    sys.path.insert(0, str(REPO_PACKAGE))  # what a launch without -I would put first
    launch.prepare(str(checkout))
    seen = sys.modules["hermes_bootstrap"].SEEN
    # At the bootstrap import: variables dropped, checkout first, pin set, home not settled yet,
    # no plugins folder and no package folder on sys.path.
    assert seen["env"] == {"PYTHONHOME": None, "PYTHONPATH": None, "VIRTUAL_ENV": None, "HERMES_HOME": None}
    assert seen["path"][0] == str(checkout)
    assert seen["pin"] is True
    assert launch.plugins_dir() not in seen["path"] and str(REPO_PACKAGE) not in seen["path"]
    # After it: HERMES_HOME from hermes_constants, the plugins folder last.
    assert os.environ["HERMES_HOME"] == str(tmp_path / DEFAULT_ROOT)
    assert sys.path[0] == str(checkout) and sys.path[-1] == launch.plugins_dir()
    assert launch.plugins_dir() == str(REPO_PACKAGE.parent)


def test_prepare_keeps_a_set_home_and_the_arguments(tmp_path, clean_process):
    """HERMES_HOME already set wins (the launcher only fills it in); sys.argv is untouched during the import,
    because a bootstrap restart re-runs the launch file with the same arguments."""
    checkout = _checkout(tmp_path)
    clean_process.setenv("HERMES_HOME", str(tmp_path / "profile"))
    sys.argv[:] = ["hermes_launch.py", str(checkout), "neospike.cli", "--home", "x"]
    launch.prepare(str(checkout))
    seen = sys.modules["hermes_bootstrap"].SEEN
    assert seen["env"]["HERMES_HOME"] == str(tmp_path / "profile") == os.environ["HERMES_HOME"]
    assert seen["argv"] == ["hermes_launch.py", str(checkout), "neospike.cli", "--home", "x"]


def test_run_starts_a_module_as_main_with_its_arguments(tmp_path, clean_process):
    (tmp_path / (TARGET_MODULE + ".py")).write_text(TARGET_CODE, encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    sys.argv[:] = ["hermes_launch.py", "checkout", TARGET_MODULE, "--flag", "value"]
    launch.run([TARGET_MODULE, "--flag", "value"])
    ran = sys.modules[TARGET_MODULE + "_ran"]
    assert ran["name"] == "__main__"
    assert ran["argv"][1:] == ["--flag", "value"] and ran["argv"][0].endswith(TARGET_MODULE + ".py")


@pytest.mark.parametrize("argv", [[], ["checkout"]])
def test_missing_arguments_print_the_usage(argv, clean_process, tmp_path):
    with pytest.raises(SystemExit) as stopped:
        launch.main(argv)
    assert "usage:" in str(stopped.value.code)


def test_launch_script_runs_no_code_string():
    """Only modules start: the launch file calls no exec, eval or compile (a code-string runner)."""
    tree = ast.parse((REPO_PACKAGE / "hermes_launch.py").read_text(encoding="utf-8-sig"))
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not called & CODE_RUNNERS


def _launch(checkout: Path, target: list[str], tmp_path: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "foreign"), "HERMES_HOME": str(tmp_path / "home")}
    return subprocess.run([sys.executable, "-I", str(REPO_PACKAGE / "hermes_launch.py"), str(checkout), *target],
                          capture_output=True, text=True, encoding="utf-8", timeout=PROCESS_TIMEOUT_SECONDS,
                          env=env, cwd=str(tmp_path))


def test_isolated_process_imports_this_package_through_the_plugins_folder(tmp_path):
    """A real `python -I hermes_launch.py` with the stand-in bootstrap (no Hermes runs): PYTHONPATH is
    ignored and dropped, the package folder is not on sys.path, and neospike imports from the plugins folder
    (an editable install of this venv may already list it, so its position is checked in-process above)."""
    checkout = _checkout(tmp_path)
    (checkout / (PROBE_MODULE + ".py")).write_text(PROBE_CODE, encoding="utf-8")
    done = _launch(checkout, [PROBE_MODULE, "tail"], tmp_path)
    assert done.returncode == 0, done.stderr
    seen = json.loads(done.stdout.strip().splitlines()[-1])
    assert seen["path"][0] == str(checkout) and str(REPO_PACKAGE.parent) in seen["path"]
    assert str(REPO_PACKAGE) not in seen["path"] and seen["pythonpath"] is None
    assert Path(seen["file"]).resolve().parent == REPO_PACKAGE
    assert seen["argv"][1:] == ["tail"]


@pytest.mark.parametrize(("target", "code"), [
    (["neospike.runtime_check", "imports", "json"], 0),
    (["neospike.runtime_check", "present", "json", "neospike_missing_dep"], 1),
    (["neospike.runtime_check"], 2),
])
def test_runtime_check_runs_as_a_module_through_the_launch_script(tmp_path, target, code):
    done = _launch(_checkout(tmp_path), target, tmp_path)
    assert done.returncode == code, done.stderr
