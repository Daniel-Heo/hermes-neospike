"""install.py --check fails on a NOT installed status and on a provider import failure (fake subprocesses)."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PYTHON = r"C:\fake\python-3.14.7\python.exe"
HEALTHY_STATUS = "  Provider:  neospike\n\n  Plugin:    installed ✓\n  Status:    available ✓\n"
BROKEN_STATUS = "  Provider:  neospike\n\n  Plugin:    NOT installed ✗\n"
IMPORT_ERROR = ("ImportError: neospike needs numpy. Environment without numpy: x\\Lib\\site-packages. "
                "Run `python install.py` again from the neospike repository.")


def _install_module():
    spec = importlib.util.spec_from_file_location("install_under_test", REPO_ROOT / "install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _add_job(home: Path, name: str) -> str:
    """Append one job to <home>/cron/jobs.json the way `hermes cron create` would; returns its id."""
    jobs = home / "cron" / "jobs.json"
    data = json.loads(jobs.read_text(encoding="utf-8")) if jobs.is_file() else {"jobs": []}
    job_id = f"job{len(data['jobs'])}"
    data["jobs"].append({"id": job_id, "name": name})
    jobs.parent.mkdir(parents=True, exist_ok=True)
    jobs.write_text(json.dumps(data), encoding="utf-8")
    return job_id


def _remove_job(home: Path, job_id: str) -> None:
    jobs = home / "cron" / "jobs.json"
    data = json.loads(jobs.read_text(encoding="utf-8"))
    data["jobs"] = [job for job in data["jobs"] if job["id"] != job_id]
    jobs.write_text(json.dumps(data), encoding="utf-8")


class _FakeRunner:
    """Stands in for subprocess.run: memory status, --print-runtime-command, and the provider import."""

    def __init__(self, status: str, import_code: int, sites: list[str] | None = None) -> None:
        self.status = status
        self.import_code = import_code
        self.sites = sites or []
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs) -> subprocess.CompletedProcess:
        self.commands.append(command)
        if command[1:] == ["memory", "status"]:
            return subprocess.CompletedProcess(command, 0, self.status, "")
        if command[1:] == ["--print-runtime-command"]:
            return subprocess.CompletedProcess(command, 0, json.dumps([RUNTIME_PYTHON, "-I", "-c", "..."]), "")
        if len(command) > 3 and "json.dumps" in command[3]:  # the environment probe
            return subprocess.CompletedProcess(command, 0, "bootstrap noise\n" + json.dumps(self.sites), "")
        if command[1:3] == ["config", "set"]:  # what `hermes config set memory.provider X` leaves in config.yaml
            config = Path(kwargs.get("env", {}).get("HERMES_HOME", "")) / "config.yaml"
            if config.is_file():
                text = config.read_text(encoding="utf-8")  # the comment-preserving writer keeps the other lines
                config.write_text(re.sub(r"(?m)^  provider: .*$", f"  provider: {command[4]}", text, count=1),
                                  encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[1:4] == ["pip", "install", "--python"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[1:3] == ["cron", "remove"]:  # removal of the dropped weekly job
            _remove_job(Path(kwargs["env"]["HERMES_HOME"]), command[3])
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[1:3] in (["cron", "create"], ["cron", "edit"]):
            if command[2] == "create":
                _add_job(Path(kwargs["env"]["HERMES_HOME"]), command[command.index("--name") + 1])
            return subprocess.CompletedProcess(command, 0, "", "")
        stderr = IMPORT_ERROR if self.import_code else ""
        return subprocess.CompletedProcess(command, self.import_code, "", stderr)


@pytest.mark.parametrize(("status", "import_code", "expected"), [
    (HEALTHY_STATUS, 0, None),
    (BROKEN_STATUS, 0, "Plugin:    installed ✓"),
    (HEALTHY_STATUS, 1, "install.py"),
])
def test_check_fails_on_status_or_import(tmp_path, monkeypatch, status, import_code, expected):
    monkeypatch.delenv("HERMES_PYTHON", raising=False)
    install = _install_module()
    runner = _FakeRunner(status, import_code)
    failures = install.audit_installation(tmp_path, "hermes", [], runner)
    assert runner.commands[-1][0] == RUNTIME_PYTHON  # the import ran in the launcher's python, not the venv
    assert "is_available()" in runner.commands[-1][3]  # --check keeps the is_available assertion
    if expected is None:
        assert failures == []
    else:
        assert len(failures) == 1 and expected in failures[0], failures


def test_runtime_command_beats_hermes_python(tmp_path, monkeypatch):
    """HERMES_PYTHON may name another venv; the launcher's runtime python is checked instead."""
    monkeypatch.setenv("HERMES_PYTHON", r"C:\fake\hermes-agent\venv\Scripts\python.exe")
    install = _install_module()
    runner = _FakeRunner(HEALTHY_STATUS, 0)
    assert install.audit_installation(tmp_path, "hermes", [], runner) == []
    assert runner.commands[-1][0] == RUNTIME_PYTHON


def _runtime_venv(root: Path) -> tuple[Path, Path]:
    """A fake PM environment venv: (its site-packages, its python)."""
    venv = root / "installs" / "id" / "environments" / "hash" / "venv"
    (venv / "Lib" / "site-packages").mkdir(parents=True)
    (venv / "Scripts").mkdir()
    (venv / "pyvenv.cfg").write_text("include-system-site-packages = false\n", encoding="utf-8")
    return venv / "Lib" / "site-packages", venv / "Scripts" / "python.exe"


def test_checks_run_through_hermes_bootstrap(tmp_path, monkeypatch):
    """The provider import and the environment probe see the bootstrapped sys.path."""
    monkeypatch.delenv("HERMES_PYTHON", raising=False)
    install = _install_module()
    site, _ = _runtime_venv(tmp_path)
    runner = _FakeRunner(HEALTHY_STATUS, 0, [str(site)])
    assert install.audit_installation(tmp_path, "hermes", [], runner) == []
    assert "import hermes_bootstrap" in runner.commands[-1][3]
    assert runner.commands[-1][4:] == [str(tmp_path / "hermes-agent"), str(tmp_path / "plugins")]


def test_runtime_environment_is_the_venv_owning_site_packages(tmp_path):
    install = _install_module()
    site, python = _runtime_venv(tmp_path)
    runner = _FakeRunner(HEALTHY_STATUS, 0, [str(site)])
    environment, seen = install.probe_runtime_env(RUNTIME_PYTHON, tmp_path, runner)
    assert environment == python and str(site) in seen
    assert runner.commands[0][0] == RUNTIME_PYTHON and "import hermes_bootstrap" in runner.commands[0][3]
    environment, _ = install.probe_runtime_env(RUNTIME_PYTHON, tmp_path, _FakeRunner(HEALTHY_STATUS, 0, []))
    assert environment is None


def test_profile_home_probes_the_root_hermes_agent(tmp_path):
    """hermes-agent lives only in the root home; plugins stay the profile's own."""
    install = _install_module()
    profile = tmp_path / "profiles" / "design"
    runner = _FakeRunner(HEALTHY_STATUS, 0, [])
    install.probe_runtime_env(RUNTIME_PYTHON, profile, runner)
    assert runner.commands[0][4:] == [str(tmp_path / "hermes-agent"), str(profile / "plugins")]


def test_install_leaves_numpy_to_the_hermes_pm(tmp_path, monkeypatch):
    """install.py keeps the checkout-venv install; numpy reaches the gateway's PM environment through
    plugin.yaml python_dependencies, never a direct pip install (test_install_pm_deps.py)."""
    install = _install_module()
    home = tmp_path / "home"
    checkout = home / "hermes-agent" / "venv" / "Scripts" / "python.exe"
    checkout.parent.mkdir(parents=True)
    checkout.write_text("", encoding="utf-8")
    site, runtime_venv_python = _runtime_venv(home)
    runner = _FakeRunner(HEALTHY_STATUS, 0, [str(site)])
    monkeypatch.setenv("HERMES_PYTHON", str(checkout))
    monkeypatch.setattr(install.subprocess, "run", runner)
    monkeypatch.setattr(install.shutil, "which", {"hermes": "hermes", "uv": "uv"}.get)
    monkeypatch.setattr(install.sys, "argv", ["install.py", "--home", str(home), "--skip-models"])
    assert install.main() == 0
    installs = [command for command in runner.commands if command[1:3] == ["pip", "install"]]
    assert len(installs) == 1 and installs[0][3:5] == ["--python", str(checkout)]
    assert not any(str(runtime_venv_python) in command for command in installs)
    assert set(install.cron_name_set(home)) == {install.DAILY_JOB_NAME}  # a new install has no weekly job
    assert (home / "neospike" / "MEMORY_RULE.md").read_text(encoding="utf-8").startswith("Store with high recall")
    assert not any("update_job" in " ".join(command) for command in runner.commands)


# The weekly LLM job is gone; an existing install loses its job and files, daily stays ------------
def test_install_removes_the_weekly_job_and_scripts(tmp_path, monkeypatch):
    install = _install_module()
    home = tmp_path / "home"
    checkout = home / "hermes-agent" / "venv" / "Scripts" / "python.exe"
    checkout.parent.mkdir(parents=True)
    checkout.write_text("", encoding="utf-8")
    site, _ = _runtime_venv(home)
    daily_id = _add_job(home, install.DAILY_JOB_NAME)
    _add_job(home, "neospike-weekly-review")
    scripts = home / "scripts"
    scripts.mkdir()
    for name in ("neospike_weekly_monitor.py", "weekly_review_prompt.md", "weekly_review_prompt_kor.md"):
        (scripts / name).write_text("old\n", encoding="utf-8")
    runner = _FakeRunner(HEALTHY_STATUS, 0, [str(site)])
    monkeypatch.setenv("HERMES_PYTHON", str(checkout))
    monkeypatch.setattr(install.subprocess, "run", runner)
    monkeypatch.setattr(install.shutil, "which", {"hermes": "hermes", "uv": "uv"}.get)
    monkeypatch.setattr(install.sys, "argv", ["install.py", "--home", str(home), "--skip-models"])
    assert install.main() == 0
    records = install.read_cron_jobs(home)
    assert set(records) == {install.DAILY_JOB_NAME} and records[install.DAILY_JOB_NAME]["id"] == daily_id
    assert sorted(path.name for path in scripts.iterdir()) == sorted(install.CRON_FILES)
    assert install.cron_problems(home) == [] and install.old_install_traces(home) == []
    edits = [command for command in runner.commands if command[1:3] == ["cron", "edit"]]
    assert [command[3] for command in edits] == [install.DAILY_JOB_NAME]


def test_check_reports_a_left_over_weekly_job(tmp_path):
    install = _install_module()
    _add_job(tmp_path, "neospike-weekly-review")
    assert "Old cron job remains: neospike-weekly-review" in install.old_install_traces(tmp_path)


# macOS layout (no checkout venv) and the Claude Code MCP registration ------------------------------
def test_install_without_checkout_venv_uses_the_runtime_environment(tmp_path, monkeypatch):
    install = _install_module()
    home = tmp_path / "home"
    site, runtime_venv_python = _runtime_venv(home)
    runtime_venv_python.write_text("", encoding="utf-8")
    runner = _FakeRunner(HEALTHY_STATUS, 0, [str(site)])
    monkeypatch.delenv("HERMES_PYTHON", raising=False)
    monkeypatch.setattr(install.subprocess, "run", runner)
    monkeypatch.setattr(install.shutil, "which", {"hermes": "hermes", "uv": "uv"}.get)
    monkeypatch.setattr(install.sys, "argv", ["install.py", "--home", str(home), "--skip-models"])
    assert install.main() == 0
    installs = [command for command in runner.commands if command[1:3] == ["pip", "install"]]
    assert len(installs) == 1 and installs[0][3:5] == ["--python", str(runtime_venv_python)]
    cache = json.loads((home / "neospike" / "runtime.json").read_text(encoding="utf-8"))
    assert cache == {"runtime_python": RUNTIME_PYTHON, "cli_python": str(runtime_venv_python)}


def test_mcp_registration_overwrites_with_the_hand_made_shape(tmp_path, monkeypatch):
    install = _install_module()
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs.get("check", False)))
        return subprocess.CompletedProcess(command, 1, "", "No MCP server named neospike")

    monkeypatch.setattr(install.shutil, "which", {"claude": "claude"}.get)
    install.apply_mcp_registration("/venv/python", tmp_path, runner)
    (remove, remove_check), (add, add_check) = calls
    assert remove == ["claude", "mcp", "remove", "-s", "user", "neospike"] and remove_check is False
    assert add[:6] == ["claude", "mcp", "add-json", "-s", "user", "neospike"] and add_check is True
    assert json.loads(add[6]) == {
        "type": "stdio", "command": "/venv/python",
        "args": ["-P", "-m", "neospike.mcp_server", "--home", str(tmp_path), "--owner", "default:local"],
        "env": {"HERMES_HOME": str(tmp_path),
                "HERMES_BGE_M3_DIR": str(tmp_path / "models" / "bge-m3-onnx"),
                "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(tmp_path / "plugins")}}
    assert " " not in add[6]  # compact: survives claude.cmd / cmd.exe


def test_profile_home_models_are_the_root_models(tmp_path, monkeypatch):
    """Models are shared with the root home (shared_inference_home), never per profile."""
    install = _install_module()
    monkeypatch.delenv("NEOSPIKE_INFERENCE_HOME", raising=False)
    root = tmp_path.resolve()
    profile = root / "profiles" / "design"
    env = install.mcp_server_entry("/venv/python", profile)["env"]
    assert env["HERMES_BGE_M3_DIR"] == str(root / "models" / "bge-m3-onnx")
    assert env["PYTHONPATH"] == str(profile / "plugins")


def test_mcp_server_entry_starts_with_safe_path_flag(tmp_path):
    assert _install_module().mcp_server_entry("/venv/python", tmp_path)["args"][0] == "-P"


def test_mcp_registration_is_skipped_without_claude(tmp_path, monkeypatch, capsys):
    install = _install_module()
    monkeypatch.setattr(install.shutil, "which", {}.get)
    install.apply_mcp_registration("/venv/python", tmp_path, None)
    assert capsys.readouterr().out.strip() == "Claude Code (claude) not found; MCP server neospike not registered."


def test_mcp_registration_is_skipped_for_a_profile_home(tmp_path, monkeypatch, capsys):
    """The single user-scope MCP entry keeps pointing at the root home."""
    install = _install_module()
    monkeypatch.delenv("NEOSPIKE_INFERENCE_HOME", raising=False)
    monkeypatch.setattr(install.shutil, "which", {"claude": "claude"}.get)
    profile = tmp_path / "profiles" / "design"
    profile.mkdir(parents=True)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)

    install.apply_mcp_registration("/venv/python", profile, runner)
    assert calls == []
    assert capsys.readouterr().out.strip() == (
        f"Profile home {profile}: Claude Code MCP server neospike left pointing at {tmp_path.resolve()}.")
    install.apply_mcp_registration("/venv/python", tmp_path, runner)
    assert [command[2] for command in calls] == ["remove", "add-json"]  # the root home still registers


# A config.yaml whose `plugins:` block holds a column-0 comment block
REAL_PLUGINS_BLOCK = ("plugins:\n  # Deadline\n  clone_timeout_seconds: 300\n\n# ====\n# Model Configuration\n# ====\n"
                      "  enabled:\n    - alpha\n    - beta\n  disabled: []\nmodel:\n  default: x\n")


def test_memory_provider_is_read_from_the_memory_block(tmp_path):
    from neospike.cron.neospike_cron_common import configured_memory_provider
    (tmp_path / "config.yaml").write_text(
        'model:\n  provider: "anthropic"\nmemory:\n  # note\n  provider: local_x  # comment\n'
        '  nested:\n    provider: other\n', encoding="utf-8")
    assert configured_memory_provider(tmp_path) == "local_x"


# The dashboard shows a user plugin only when plugins.enabled lists it -----------------------------
@pytest.mark.parametrize("before, after", [
    (REAL_PLUGINS_BLOCK, REAL_PLUGINS_BLOCK.replace("    - beta\n", "    - beta\n    - neospike\n")),
    ("﻿plugins:\r\n  enabled: [alpha]  # c\r\n", "﻿plugins:\r\n  enabled: [alpha, neospike]  # c\r\n"),
    ("plugins:\n  enabled: []\n", "plugins:\n  enabled: [neospike]\n"),
    ("plugins:\n  disabled: []\nmodel: {}\n", "plugins:\n  enabled:\n    - neospike\n  disabled: []\nmodel: {}\n"),
    ("model: {}", "model: {}\nplugins:\n  enabled:\n    - neospike\n"),
])
def test_neospike_is_added_to_plugins_enabled(tmp_path, before, after):
    install = _install_module()
    (tmp_path / "config.yaml").write_text(before, encoding="utf-8", newline="")
    assert install.list_plugin_enabled(tmp_path, dry_run=True) is True
    assert install.list_plugin_enabled(tmp_path) is True
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8", newline="") == after
    assert install.list_plugin_enabled(tmp_path) is False  # second run: no duplicate
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8", newline="") == after


def test_listed_neospike_is_left_alone(tmp_path):
    install = _install_module()
    text = "plugins:\n  enabled:\n  - alpha\n  - neospike  # dashboard\n"
    (tmp_path / "config.yaml").write_text(text, encoding="utf-8", newline="")
    assert install.list_plugin_enabled(tmp_path) is False
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8", newline="") == text


# A profile home leaves the cron to the root home; skills lose the renamed tool names ------------------------
class _Launcher:
    """Function object standing in for install.locate_launcher (the project uses no lambdas)."""

    def __call__(self, home: Path) -> str:
        return "hermes"


def test_profile_dry_run_leaves_the_cron_to_the_root(tmp_path, monkeypatch, capsys):
    install = _install_module()
    monkeypatch.delenv("NEOSPIKE_INFERENCE_HOME", raising=False)
    monkeypatch.setattr(install, "locate_launcher", _Launcher())
    monkeypatch.setattr(install.shutil, "which", {"claude": "claude"}.get)
    profile = tmp_path / "profiles" / "design"
    profile.mkdir(parents=True)
    _add_job(profile, install.DROPPED_JOBS[0])
    monkeypatch.setattr(install.sys, "argv", ["install.py", "--home", str(profile), "--dry-run"])
    assert install.main() == 0
    out = capsys.readouterr().out
    assert f"Profile home {profile}: cron job neospike-daily is run by the root home's cron " \
           f"({tmp_path.resolve()})." in out.splitlines()
    assert "+ hermes cron remove job0" in out  # the dropped weekly job of the profile is still removed
    assert "cron create" not in out and "+ copy" not in out and "update_job" not in out
    assert install.cron_problems(profile) != [] and install.skip_cron_for_profile(tmp_path) is False


def test_skill_old_names_are_backed_up_and_renamed(tmp_path):
    import zipfile
    install = _install_module()
    skill = tmp_path / "skills" / "notes" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    before = ("Call neospike_memory_remember, then neospike_memory_forget.\r\n"
              "Keep neospike_memory_search and neospike_memory_remembered as they are.\r\n")
    skill.write_bytes(b"\xef\xbb\xbf" + before.encode("utf-8"))
    (tmp_path / "skills" / "clean.md").write_text("nothing old\n", encoding="utf-8")
    assert install.old_install_traces(tmp_path) == [f"Old names remain in skill file: {skill}"]
    install.migrate_skill_names(tmp_path)
    (backup,) = tmp_path.glob("neospike_migration_backup_*.zip")
    with zipfile.ZipFile(backup) as archive:
        assert archive.namelist() == ["skills/notes/SKILL.md"]
        assert archive.read("skills/notes/SKILL.md").decode("utf-8-sig") == before
    assert skill.read_bytes().decode("utf-8-sig") == (
        "Call neospike_memory_save, then neospike_memory_delete.\r\n"
        "Keep neospike_memory_search and neospike_memory_remembered as they are.\r\n")
    assert skill.read_bytes().startswith(b"\xef\xbb\xbf")
    assert install.old_install_traces(tmp_path) == []
    assert install.migrate_skill_names(tmp_path) is None  # second run: nothing left, no new backup


def test_skill_mcp_search_names_become_mem_search():
    install = _install_module()
    text = ("MCP: `memory_search`·`note_search` first.\r\n"
            "Keep neospike_memory_search, the CLI note-search and note_search_p as they are.\r\n")
    assert install.apply_skill_renames(text) == (
        "MCP: `mem_search`·`mem_search` first.\r\n"
        "Keep neospike_memory_search, the CLI note-search and note_search_p as they are.\r\n")


def test_ensure_memory_rule_keeps_an_edited_file(tmp_path):
    install = _install_module()
    rule = install.seed_memory_rule(tmp_path)
    assert rule.read_text(encoding="utf-8").startswith("Store with high recall")
    rule.write_text("My own rule.", encoding="utf-8")
    install.seed_memory_rule(tmp_path)  # a reinstall never overwrites the user's rules
    assert rule.read_text(encoding="utf-8") == "My own rule."


def test_install_writes_memory_rule_into_every_neospike_profile(tmp_path, monkeypatch):
    install = _install_module()
    bot, other = tmp_path / "profiles" / "bot", tmp_path / "profiles" / "other"
    (bot / "neospike").mkdir(parents=True)
    (bot / "config.yaml").write_text("memory:\n  provider: neospike\n", encoding="utf-8")
    other.mkdir(parents=True)  # no neospike folder: left alone
    for target in install.target_homes(tmp_path):
        install.seed_memory_rule(target)
    assert (tmp_path / "neospike" / "MEMORY_RULE.md").is_file() and (bot / "neospike" / "MEMORY_RULE.md").is_file()
    assert not (other / "neospike").exists()
