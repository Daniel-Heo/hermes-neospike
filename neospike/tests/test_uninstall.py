"""install.py --uninstall against scratch homes with fake subprocesses: each step, --dry-run, a second run, the
data that stays, and a profile home whose shared embedding service another home still uses."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from neospike.tests.test_install_check import _add_job, _install_module, _remove_job

HERMES = "hermes"
CLAUDE = r"C:\fake\claude.cmd"
CONFIG_YAML = "memory:\n  provider: neospike\nplugins:\n  enabled:\n    - other\n    - neospike\n"


@pytest.fixture(scope="module")
def install():
    return _install_module()


class _UninstallRunner:
    """subprocess.run stand-in: `claude mcp get neospike-mcp` shows *mcp_home* and `claude mcp get neospike` shows
    *old_home* (None: not registered); the removals change the scratch home the way Hermes and Claude Code would."""

    def __init__(self, mcp_home: Path | None, old_home: Path | None = None) -> None:
        self.mcp_homes: dict[str, Path | None] = {"neospike-mcp": mcp_home, "neospike": old_home}
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs) -> subprocess.CompletedProcess:
        self.commands.append(command)
        home = Path(kwargs.get("env", {}).get("HERMES_HOME", "."))
        if command[1:3] == ["mcp", "get"]:
            shown = self.mcp_homes.get(command[3])
            if shown is None:
                return subprocess.CompletedProcess(command, 1, "", "No MCP server found")
            return subprocess.CompletedProcess(command, 0, f"{command[3]}:\n  Args: --home {shown}\n", "")
        if command[1:3] == ["mcp", "remove"]:
            self.mcp_homes[command[-1]] = None
        elif command[1:3] == ["cron", "remove"]:
            _remove_job(home, command[3])
        elif command[1:3] == ["plugins", "remove"]:  # hermes_cli _remove_user_plugin + _forget_plugin_config
            shutil.rmtree(home / "plugins" / "neospike")
            (home / "config.yaml").write_text("memory:\n  provider: ''\nplugins:\n  enabled:\n    - other\n",
                                              encoding="utf-8")
        elif command[1:3] == ["memory", "off"]:
            (home / "config.yaml").write_text(
                (home / "config.yaml").read_text(encoding="utf-8").replace("provider: neospike", "provider: ''"),
                encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")


class _Release:
    """inference.release_daemon stand-in: records the homes it was called for."""

    def __init__(self) -> None:
        self.homes: list[Path] = []

    def __call__(self, home: Path):
        self.homes.append(home)
        return "stopped", []


def _installed_root(tmp_path: Path, install) -> Path:
    """A root home as install.py leaves it: plugin + previous copy, cron job and scripts, data, model, config."""
    home = tmp_path / "hermes"
    (home / "plugins" / "neospike").mkdir(parents=True)
    (home / "plugins" / "neospike" / "plugin.yaml").write_text("name: neospike\n", encoding="utf-8")
    (home / "plugins" / "neospike.previous").mkdir()
    install.copy_cron_scripts(home)
    (home / "scripts" / "user_script.py").write_text("# not ours\n", encoding="utf-8")
    _add_job(home, "neospike-daily")
    _add_job(home, "user-job")
    (home / "neospike").mkdir()
    (home / "neospike" / "memory.db").write_bytes(b"data")
    (home / "neospike" / ".install" / "previous").mkdir(parents=True)
    (home / "models" / "bge-m3-onnx").mkdir(parents=True)
    (home / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
    return home


def _wire(monkeypatch, install, runner, release: _Release) -> None:
    monkeypatch.setattr(install.subprocess, "run", runner)
    monkeypatch.setattr(install.shutil, "which", {"claude": CLAUDE}.get)
    monkeypatch.setattr(install.inference, "release_daemon", release)


def _job_names(home: Path) -> list[str]:
    return [job["name"] for job in json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8"))["jobs"]]


def test_dry_run_prints_every_step_and_changes_nothing(tmp_path, monkeypatch, capsys, install):
    home = _installed_root(tmp_path, install)
    runner, release = _UninstallRunner(home, home), _Release()
    _wire(monkeypatch, install, runner, release)
    before = sorted(str(path) for path in home.rglob("*"))
    assert install.run_uninstall(HERMES, home, dry_run=True) == 0
    out = capsys.readouterr().out
    assert sorted(str(path) for path in home.rglob("*")) == before
    assert runner.commands == [[CLAUDE, "mcp", "get", "neospike-mcp"], [CLAUDE, "mcp", "get", "neospike"]]
    assert release.homes == []  # read only
    assert "mcp remove -s user neospike-mcp\n" in out and "mcp remove -s user neospike\n" in out
    for line in ("+ hermes cron remove job0", f"+ remove {home / 'scripts' / 'neospike_daily.py'}",
                 "+ ask the embedding service of", "+ hermes plugins remove neospike",
                 f"+ remove {home / 'plugins' / 'neospike.previous'}",
                 f"+ remove {home / 'neospike' / '.install'}", f"Kept your Neospike data (memories, notes, "
                 f"conversations, backups): {home / 'neospike'}", f"Kept the model: {home / 'models' / 'bge-m3-onnx'}"):
        assert line in out
    assert "Uninstalled." not in out


def test_uninstall_removes_each_step_and_keeps_the_data(tmp_path, monkeypatch, capsys, install):
    home = _installed_root(tmp_path, install)
    runner, release = _UninstallRunner(home), _Release()
    _wire(monkeypatch, install, runner, release)
    assert install.run_uninstall(HERMES, home, dry_run=False) == 0
    assert [command[1:3] for command in runner.commands] == [["cron", "remove"], ["mcp", "get"], ["mcp", "remove"],
                                                             ["mcp", "get"], ["plugins", "remove"]]
    assert _job_names(home) == ["user-job"]
    assert not any((home / "scripts" / name).exists() for name in install.CRON_FILES)
    assert (home / "scripts" / "user_script.py").is_file()
    assert release.homes == [home]
    assert not (home / "plugins" / "neospike").exists() and not (home / "plugins" / "neospike.previous").exists()
    assert not (home / "neospike" / ".install").exists()
    assert (home / "neospike" / "memory.db").read_bytes() == b"data" and (home / "models" / "bge-m3-onnx").is_dir()
    assert "Restart the Hermes gateway" in capsys.readouterr().out


def test_second_run_skips_what_is_gone(tmp_path, monkeypatch, capsys, install):
    home = _installed_root(tmp_path, install)
    runner, release = _UninstallRunner(home), _Release()
    _wire(monkeypatch, install, runner, release)
    install.run_uninstall(HERMES, home, dry_run=False)
    capsys.readouterr()
    runner.commands.clear()
    assert install.run_uninstall(HERMES, home, dry_run=False) == 0
    out = capsys.readouterr().out
    assert runner.commands == [[CLAUDE, "mcp", "get", "neospike-mcp"], [CLAUDE, "mcp", "get", "neospike"]]
    for line in ("Cron job neospike-daily: not registered (skipped)", "MCP server neospike-mcp: not registered",
                 "MCP server neospike: not registered",
                 "Plugin folder", "not there (skipped)"):
        assert line in out
    assert (home / "neospike" / "memory.db").is_file()


def test_uninstall_removes_the_old_name_only_when_it_points_here(tmp_path, monkeypatch, capsys, install):
    home = _installed_root(tmp_path, install)
    for old_home, removed in ((home, True), (tmp_path / "other", False)):
        runner, release = _UninstallRunner(home, old_home), _Release()
        _wire(monkeypatch, install, runner, release)
        install.uninstall_mcp(home, dry_run=False, runner=runner)
        removals = [command[-1] for command in runner.commands if command[1:3] == ["mcp", "remove"]]
        assert removals == (["neospike-mcp", "neospike"] if removed else ["neospike-mcp"])
        assert runner.mcp_homes["neospike"] == (None if removed else old_home)


def test_folder_already_gone_turns_memory_off_and_unlists_the_plugin(tmp_path, monkeypatch, capsys, install):
    home = _installed_root(tmp_path, install)
    shutil.rmtree(home / "plugins" / "neospike")
    runner, release = _UninstallRunner(None), _Release()
    _wire(monkeypatch, install, runner, release)
    install.run_uninstall(HERMES, home, dry_run=False)
    assert [HERMES, "memory", "off"] in runner.commands
    assert (home / "config.yaml").read_text(encoding="utf-8") == (
        "memory:\n  provider: ''\nplugins:\n  enabled:\n    - other\n")


@pytest.mark.parametrize(("text", "after"), [
    ("plugins:\n  enabled: [other, neospike]\n", "plugins:\n  enabled: [other]\n"),
    ("\ufeffplugins:\r\n  enabled:\r\n  - neospike\r\n  - other\r\n", "\ufeffplugins:\r\n  enabled:\r\n  - other\r\n"),
])
def test_unlist_plugin_enabled_keeps_the_other_lines(tmp_path, install, text, after):
    (tmp_path / "config.yaml").write_bytes(text.encode("utf-8"))
    assert install.unlist_plugin_enabled(tmp_path) is True
    assert (tmp_path / "config.yaml").read_bytes() == after.encode("utf-8")
    assert install.unlist_plugin_enabled(tmp_path) is False


def test_profile_home_leaves_cron_mcp_and_the_shared_service_to_the_root(tmp_path, monkeypatch, capsys, install):
    root = _installed_root(tmp_path, install)
    profile = root / "profiles" / "bot"
    (profile / "plugins" / "neospike").mkdir(parents=True)
    (profile / "neospike").mkdir()
    (profile / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
    runner, release = _UninstallRunner(root), _Release()
    _wire(monkeypatch, install, runner, release)
    install.run_uninstall(HERMES, profile, dry_run=False)
    out = capsys.readouterr().out
    assert [command[1:3] for command in runner.commands] == [["plugins", "remove"]]
    assert release.homes == []  # the root home still uses Neospike
    assert f"left running: still used by {root}" in out
    assert _job_names(root) == ["neospike-daily", "user-job"]
    assert f"Kept the model (shared with the root home): {root / 'models' / 'bge-m3-onnx'}" in out


def test_uninstall_option_is_parsed(monkeypatch, install):
    monkeypatch.setattr(install.sys, "argv", ["install.py", "--uninstall", "--dry-run"])
    parsed = install._parse_args()
    assert parsed.uninstall and parsed.dry_run
