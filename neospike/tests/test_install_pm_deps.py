"""numpy is declared in plugin.yaml and installed by the Hermes package manager (fake subprocesses only)."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from neospike.tests.test_install_check import (HEALTHY_STATUS, REPO_ROOT, _FakeRunner, _install_module,
                                               _runtime_check_of, _runtime_venv)

NUMPY = "numpy>=2.0,<3"
FAISS = "faiss-cpu>=1.16,<2"
TOKENIZERS = "tokenizers>=0.22,<1"
ONNXRUNTIME = "onnxruntime>=1.20,<2"
HUGGINGFACE_HUB = "huggingface-hub>=1.0,<2"
DECLARATION = (b'python_dependencies:\n  - "numpy>=2.0,<3"\n  - "faiss-cpu>=1.16,<2"\n'
               b'  - "tokenizers>=0.22,<1"\n  - "onnxruntime>=1.20,<2"\n'
               b'  - "huggingface-hub>=1.0,<2"\n')  # LF form


def _plugin_yaml_lf() -> bytes:
    """The repository plugin.yaml with CRLF folded to LF: a checkout may give either line ending."""
    return (REPO_ROOT / "neospike" / "plugin.yaml").read_bytes().replace(b"\r\n", b"\n")


class _PmRunner(_FakeRunner):
    """_FakeRunner plus `hermes plugins enable` (exit 0); keeps each call's env to check HERMES_HOME."""

    def __init__(self, import_code: int, sites: list[str]) -> None:
        super().__init__(HEALTHY_STATUS, import_code, sites)
        self.envs: list[dict] = []

    def __call__(self, command: list[str], **kwargs) -> subprocess.CompletedProcess:
        self.envs.append(kwargs.get("env") or {})
        if command[1:3] == ["plugins", "enable"]:
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 0, "Plugin 'neospike' is already enabled.", "")
        return super().__call__(command, **kwargs)


def _installer(tmp_path: Path, monkeypatch, import_code: int, *argv: str):
    install = _install_module()
    home = tmp_path / "home"
    checkout = home / "hermes-agent" / "venv" / "Scripts" / "python.exe"
    checkout.parent.mkdir(parents=True)
    checkout.write_text("", encoding="utf-8")
    site, runtime_venv_python = _runtime_venv(home)
    runner = _PmRunner(import_code, [str(site)])
    monkeypatch.setenv("HERMES_PYTHON", str(checkout))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setattr(install.subprocess, "run", runner)
    monkeypatch.setattr(install.shutil, "which", {"hermes": "hermes", "uv": "uv"}.get)
    monkeypatch.setattr(install.sys, "argv", ["install.py", "--home", str(home), *argv])
    return install, home, runner, runtime_venv_python


def _is_provider_check(command: list[str]) -> bool:
    """The post-sync import check or the --check provider check (not the environment probe)."""
    return (_runtime_check_of(command) or ["sites"])[0] in ("imports", "provider")


def test_install_enables_the_plugin_through_hermes_pm(tmp_path, monkeypatch):
    install, home, runner, runtime_venv_python = _installer(tmp_path, monkeypatch, 0, "--skip-models")
    assert install.main() == 0
    enable = ["hermes", "plugins", "enable", "neospike", "--no-allow-tool-override"]
    assert enable in runner.commands
    at = runner.commands.index(enable)
    env = runner.envs[at]
    assert env["HERMES_HOME"] == str(home) and env["PYTHONIOENCODING"] == "utf-8"
    assert (home / "plugins" / "neospike" / "plugin.yaml").is_file()  # copied before the PM sync
    assert any(_is_provider_check(command) for command in runner.commands[at + 1:])  # load check after enable
    assert not any(command[1:3] == ["pip", "install"] for command in runner.commands)  # the PM installs


def test_install_stops_when_the_provider_does_not_load_after_pm_sync(tmp_path, monkeypatch):
    install, home, runner, runtime_venv_python = _installer(tmp_path, monkeypatch, 1, "--skip-models")
    with pytest.raises(SystemExit) as raised:
        install.main()
    message = str(raised.value)
    assert "hermes pm status" in message and str(runtime_venv_python) in message and "numpy" in message
    assert not any(command[1:3] == ["config", "set"] for command in runner.commands)  # provider stays unset
    assert not any(command[1:3] == ["pip", "install"] for command in runner.commands)


class _NoModelRunner(_PmRunner):
    """A new home without models: the imports work, is_available() is False."""

    def __call__(self, command: list[str], **kwargs) -> subprocess.CompletedProcess:
        if _runtime_check_of(command) == ["provider"]:
            self.envs.append(kwargs.get("env") or {})
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 1, "", "NeospikeProvider().is_available() is False")
        return super().__call__(command, **kwargs)


def test_skip_models_new_home_installs_to_the_end(tmp_path, monkeypatch):
    install, home, _, _ = _installer(tmp_path, monkeypatch, 0, "--skip-models")
    runner = _NoModelRunner(0, install.subprocess.run.sites)
    monkeypatch.setattr(install.subprocess, "run", runner)
    assert install.main() == 0
    checks = [_runtime_check_of(command) for command in runner.commands if _is_provider_check(command)]
    assert checks and all(check[0] == "imports" and "numpy" in check for check in checks)
    assert ["hermes", "config", "set", "memory.provider", "neospike"] in runner.commands


def test_dry_run_prints_the_enable_command(tmp_path, monkeypatch, capsys):
    install, _, runner, _ = _installer(tmp_path, monkeypatch, 0, "--dry-run")
    assert install.main() == 0
    out = capsys.readouterr().out
    assert "hermes plugins enable neospike --no-allow-tool-override" in out
    launch = str(install.installed_launch_script(install.Path(install.sys.argv[2])))
    assert f"{launch} " in out and " neospike.runtime_check imports numpy faiss neospike neospike.provider" in out
    assert not any(command[1:3] == ["plugins", "enable"] for command in runner.commands)


def test_repository_plugin_yaml_declares_the_runtime_packages():
    assert _plugin_yaml_lf().endswith(DECLARATION)
    install = _install_module()
    assert install.PLUGIN_PIP_REQUIREMENTS == (NUMPY, FAISS, TOKENIZERS, ONNXRUNTIME, HUGGINGFACE_HUB)
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8-sig")
    assert re.findall(r'"(numpy[^"]*)"', pyproject) == [NUMPY]
    assert re.findall(r'"(faiss[^"]*)"', pyproject) == [FAISS]
    assert re.findall(r'"(tokenizers[^"]*)"', pyproject) == [TOKENIZERS]
    assert re.findall(r'"(onnxruntime[^"]*)"', pyproject) == [ONNXRUNTIME]
    assert re.findall(r'"(huggingface-hub[^"]*)"', pyproject) == [HUGGINGFACE_HUB]
    assert "faiss" in install.IMPORT_CHECK[1:]  # the post-sync check fails when PM left faiss out


def _no_installation_failures(home, hermes, required, runner=None) -> list[str]:
    """audit_installation's runner default is bound to the real subprocess.run at import; never let it run."""
    return []


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize(("declared", "reported"), [(False, True), (True, False)])
def test_check_reports_plugin_yaml_without_the_numpy_declaration(tmp_path, monkeypatch, declared, reported, newline):
    install, home, _, _ = _installer(tmp_path, monkeypatch, 0, "--check")
    monkeypatch.setattr(install, "audit_installation", _no_installation_failures)
    plugin_yaml = home / "plugins" / "neospike" / "plugin.yaml"
    plugin_yaml.parent.mkdir(parents=True)
    text = _plugin_yaml_lf()
    assert DECLARATION in text
    text = text if declared else text.replace(DECLARATION, b"")
    plugin_yaml.write_bytes(text.replace(b"\n", newline))
    with pytest.raises(SystemExit) as raised:  # other artifacts (models, crons) are missing in this fake home
        install.main()
    assert ("python_dependencies" in str(raised.value)) is reported
