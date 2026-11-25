"""install.py and the shared MCP HTTP server: the token in .env, the Hermes config.yaml entry, --check,
--dry-run and --uninstall (fake subprocesses, temporary homes; no real hermes or claude)."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

from neospike.mcp_headers import TOKEN_VARIABLE, read_env_token

REPO_ROOT = Path(__file__).resolve().parents[2]
URL = "http://127.0.0.1:47831/mcp"
ENTRY = ("  neospike-mcp:\n    url: http://127.0.0.1:47831/mcp\n    headers:\n"
         "      Authorization: \"Bearer ${NEOSPIKE_MCP_TOKEN}\"\n")
# What another conversation put there by hand: kept by install and uninstall.
OTHER_SERVERS = ("model:\n  default: x\nmcp_servers:\n  neospike_mcp:\n    command: python\n    args:\n"
                 "      - -m\n      - neospike.mcp_server\n  other:\n    url: http://127.0.0.1:9/mcp\n"
                 "# trailing comment\nplugins:\n  enabled:\n    - neospike\n")


def _install_module():
    spec = importlib.util.spec_from_file_location("install_under_test", REPO_ROOT / "install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def install(monkeypatch):
    module = _install_module()
    monkeypatch.setattr(module.shutil, "which", {}.get)  # no claude: Claude Code steps are skipped
    monkeypatch.delenv("NEOSPIKE_INFERENCE_HOME", raising=False)
    return module


def _config(home: Path) -> str:
    return (home / "config.yaml").read_text(encoding="utf-8")


def test_token_is_made_once_and_a_rerun_keeps_it(install, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=x\n", encoding="utf-8")
    assert install.ensure_mcp_token(tmp_path) is True
    token = read_env_token(tmp_path / ".env")
    assert re.fullmatch(r"[0-9a-f]{64}", token)
    assert install.ensure_mcp_token(tmp_path) is False
    assert (tmp_path / ".env").read_text(encoding="utf-8") == f"ANTHROPIC_API_KEY=x\n{TOKEN_VARIABLE}={token}\n"


def test_hermes_entry_is_appended_and_a_rerun_changes_nothing(install, tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_bytes(b"\xef\xbb\xbfmodel:\n  default: x\n")
    assert install.apply_hermes_mcp_registration(tmp_path) is True
    assert (tmp_path / "config.yaml").read_bytes().startswith(b"\xef\xbb\xbf")
    assert _config(tmp_path) == "﻿model:\n  default: x\nmcp_servers:\n" + ENTRY
    assert install.hermes_mcp_registered(tmp_path)
    before = (tmp_path / "config.yaml").read_bytes()
    assert install.apply_hermes_mcp_registration(tmp_path) is False
    assert (tmp_path / "config.yaml").read_bytes() == before


def test_hermes_entry_joins_other_servers_and_keeps_them(install, tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(OTHER_SERVERS, encoding="utf-8")
    install.apply_hermes_mcp_registration(tmp_path)
    text = _config(tmp_path)
    assert text == OTHER_SERVERS.replace("    url: http://127.0.0.1:9/mcp\n", "    url: http://127.0.0.1:9/mcp\n" + ENTRY)
    assert install.hermes_mcp_registered(tmp_path)
    assert install.remove_hermes_mcp_registration(tmp_path) is True
    assert _config(tmp_path) == OTHER_SERVERS  # the stdio neospike_mcp entry and the other server stay


def test_hermes_entry_with_an_old_url_is_rewritten(install, tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("mcp_servers:\n" + ENTRY.replace("47831", "40000") + "other: 1\n",
                                          encoding="utf-8")
    assert not install.hermes_mcp_registered(tmp_path)
    install.apply_hermes_mcp_registration(tmp_path)
    assert _config(tmp_path) == "mcp_servers:\n" + ENTRY + "other: 1\n"


def test_empty_section_becomes_a_block_and_removal_empties_it(install, tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("mcp_servers: {}\r\nmodel: x\r\n", encoding="utf-8", newline="")
    install.apply_hermes_mcp_registration(tmp_path)
    assert (tmp_path / "config.yaml").read_bytes() == ("mcp_servers:\n" + ENTRY + "model: x\n").replace(
        "\n", "\r\n").encode()
    install.remove_hermes_mcp_registration(tmp_path)
    assert (tmp_path / "config.yaml").read_bytes() == b"mcp_servers: {}\r\nmodel: x\r\n"


def test_a_flow_mapping_with_servers_is_not_rewritten(install, tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text("mcp_servers: {a: {url: x}}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not a block mapping"):
        install.apply_hermes_mcp_registration(tmp_path)


def test_registration_writes_token_and_hermes_entry_but_not_for_a_profile(install, tmp_path: Path) -> None:
    profile = tmp_path / "profiles" / "design"
    profile.mkdir(parents=True)
    install.apply_mcp_registration("/runtime/python", profile, None)
    assert not (profile / ".env").exists() and not (profile / "config.yaml").exists()
    install.apply_mcp_registration("/runtime/python", tmp_path, None)
    assert read_env_token(tmp_path / ".env") and install.hermes_mcp_registered(tmp_path)


def test_check_reports_each_missing_mcp_part(install, tmp_path: Path, monkeypatch) -> None:
    problems = install.mcp_problems(tmp_path)
    assert len(problems) == 2 and TOKEN_VARIABLE in problems[0] and "mcp_servers.neospike-mcp" in problems[1]
    install.apply_mcp_registration("/runtime/python", tmp_path, None)
    assert install.mcp_problems(tmp_path) == []
    monkeypatch.setattr(install.shutil, "which", {"claude": "claude"}.get)
    absent = _ClaudeGet("")
    assert install.mcp_problems(tmp_path, absent) == [
        f"Claude Code MCP server neospike-mcp is not registered for {tmp_path} (run install.py again)"]
    assert install.mcp_problems(tmp_path, _ClaudeGet(f"neospike-mcp:\n  Type: http\n  URL: {URL}\n")) == []


class _ClaudeGet:
    """`claude mcp get` stand-in: exit 0 with *shown*, or exit 1 when it is empty; records the commands."""

    def __init__(self, shown: str) -> None:
        self.shown = shown
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        return subprocess.CompletedProcess(command, 0 if self.shown else 1, self.shown, "")


def test_uninstall_removes_both_registrations_and_the_token(install, tmp_path: Path, monkeypatch, capsys) -> None:
    (tmp_path / ".env").write_text("KEEP=1\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(OTHER_SERVERS, encoding="utf-8")
    install.apply_mcp_registration("/runtime/python", tmp_path, None)
    monkeypatch.setattr(install.shutil, "which", {"claude": "claude"}.get)
    claude = _ClaudeGet(f"neospike-mcp:\n  URL: {URL}\n")
    removed = []
    monkeypatch.setattr(install, "_plan_or_run", _Recorded(removed))
    install.uninstall_mcp(tmp_path, False, claude)
    assert ["claude", "mcp", "remove", "-s", "user", "neospike-mcp"] in removed
    assert _config(tmp_path) == OTHER_SERVERS
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "KEEP=1\n"
    assert "MCP server: not_running" in capsys.readouterr().out


class _Recorded:
    def __init__(self, commands: list) -> None:
        self.commands = commands

    def __call__(self, command, dry_run, env=None) -> None:
        self.commands.append(command)


def test_uninstall_dry_run_changes_nothing(install, tmp_path: Path, capsys) -> None:
    install.apply_mcp_registration("/runtime/python", tmp_path, None)
    env_before, config_before = (tmp_path / ".env").read_bytes(), (tmp_path / "config.yaml").read_bytes()
    install.uninstall_mcp(tmp_path, True)
    out = capsys.readouterr().out
    assert "+ stop the MCP server" in out and "+ remove mcp_servers.neospike-mcp" in out
    assert f"+ remove {TOKEN_VARIABLE}" in out
    assert (tmp_path / ".env").read_bytes() == env_before and (tmp_path / "config.yaml").read_bytes() == config_before


def test_dry_run_plans_the_token_and_both_registrations(install, tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(install, "migration_preview", _NoSteps())
    monkeypatch.setattr(install, "cron_argv_list", _NoSteps())
    install._print_dry_run("hermes", tmp_path)
    out = capsys.readouterr().out
    assert f"+ add {TOKEN_VARIABLE} to" in out and "+ write mcp_servers.neospike-mcp in" in out
    add = next(line for line in out.splitlines() if "add-json" in line)
    assert '"type":"http"' in add.replace('\\"', '"') and URL in add
    assert not (tmp_path / ".env").exists() and not (tmp_path / "config.yaml").exists()


class _NoSteps:
    def __call__(self, *args, **kwargs) -> list:
        return []


def test_registered_entry_is_json_without_the_token(install, tmp_path: Path) -> None:
    install.ensure_mcp_token(tmp_path)
    payload = json.dumps(install.mcp_server_entry("/runtime/python", tmp_path))
    assert read_env_token(tmp_path / ".env") not in payload and "headersHelper" in payload
