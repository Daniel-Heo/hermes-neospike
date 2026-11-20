"""The cron scripts run against a scratch HERMES_HOME, and install.py never registers a cron twice."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from neospike.config import EMBEDDING_DIMENSIONS, NeospikeConfig
from neospike.store import DEFAULT_NAMESPACE, MemoryStore

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_TIMEOUT_SECONDS = 600
# Minimal Hermes state.db (tables conv-import reads; see test_conv_import.py), no sessions.
STATE_SCHEMA = """
CREATE TABLE sessions (id TEXT PRIMARY KEY, origin TEXT NOT NULL, started_at REAL NOT NULL, ended_at REAL, title TEXT);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
    tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL NOT NULL,
    _compressed_summary INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL DEFAULT 1, compacted INTEGER NOT NULL DEFAULT 0
);
"""


def _install_module():
    spec = importlib.util.spec_from_file_location("install_under_test", REPO_ROOT / "install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def install():
    return _install_module()


@pytest.fixture
def home(tmp_path, install) -> Path:
    """A scratch Hermes home: the plugin under plugins/, the cron scripts under scripts/ (as install.py puts them)."""
    root = tmp_path / "hermes"
    shutil.copytree(REPO_ROOT / "neospike", root / "plugins" / "neospike",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "dashboard"))
    install.copy_cron_scripts(root)
    return root


def _state_db(home: Path) -> None:
    with closing(sqlite3.connect(home / "state.db")) as connection:
        connection.executescript(STATE_SCHEMA)


def _run_script(home: Path, name: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "HERMES_PYTHON": sys.executable, "PYTHONIOENCODING": "utf-8"}
    env.pop("HERMES_HOME", None)  # the script must take its home from its own location
    return subprocess.run([sys.executable, str(home / "scripts" / name)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=SCRIPT_TIMEOUT_SECONDS, env=env,
                          cwd=str(home / "scripts"))


def test_daily_success_prints_nothing(home):
    _state_db(home)
    done = _run_script(home, "neospike_daily.py")
    assert (done.returncode, done.stdout) == (0, ""), (done.stdout, done.stderr[-2000:])
    assert list((home / "neospike" / "backups").glob("full-*.db"))  # the backup stage really ran


def test_daily_failure_names_the_stage_and_keeps_going(home):
    """No state.db: conv-import fails; the stages before and after it still run."""
    done = _run_script(home, "neospike_daily.py")
    lines = done.stdout.strip().splitlines()
    assert done.returncode == 1 and len(lines) == 1, (done.stdout, done.stderr[-2000:])
    assert lines[0].startswith(f"{home.resolve()}: FAILED conv-import (exit 1):") and "state.db not found" in lines[0], lines


def test_daily_backup_failure_is_listed_first(home, monkeypatch, capsys):
    sys.path.insert(0, str(home / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location("daily_under_test", home / "scripts" / "neospike_daily.py")
        daily = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(daily)
    finally:
        sys.path.remove(str(home / "scripts"))
    stages = _FailingStages({"topic-to-note", "backup"})
    monkeypatch.setattr(daily, "run_stage", stages)
    monkeypatch.setattr(daily.common, "home_dir", _HomeOf(home))
    assert daily.main() == 1
    # topic-to-note after topic-discover; embed-pending first (it runs the save follow-up, so
    # note-absorb and topic-assign are no stages). Inbox-maintain, lint and inbox-review-code are gone.
    assert stages.order == ["backup", "link-prune", "conv-import", "conv-index", "embed-pending", "pref-reconcile",
                            "topic-discover", "topic-to-note"]
    assert capsys.readouterr().out.splitlines() == [f"{home}: FAILED backup | FAILED topic-to-note"]


class _FailingStages:
    def __init__(self, names: set[str]) -> None:
        self.names = names
        self.order: list[str] = []

    def __call__(self, home: Path, name: str, args: list[str]) -> str | None:
        self.order.append(name)
        return f"FAILED {name}" if name in self.names else None


class _HomeOf:
    """Function object standing in for common.home_dir (the project uses no lambdas)."""

    def __init__(self, home: Path) -> None:
        self.home = home

    def __call__(self) -> Path:
        return self.home


def _vector(axis: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMENSIONS
    vector[axis] = 1.0
    return vector


TOPIC_CLUSTER_SIZE = 6  # topics._Tuning.GROUP_MAX_WITHOUT_TOPIC + 1: the smallest linked group discover turns into a topic


def test_daily_discovers_topics(home):
    """The daily run creates a topic from a cluster of unassigned memories. A topic is a
    tag and 6 members (not more than 10) stay a topic."""
    _state_db(home)
    store = MemoryStore(home / "neospike" / "memory.db", dimensions=EMBEDDING_DIMENSIONS)
    try:
        for index in range(TOPIC_CLUSTER_SIZE):
            store.insert(DEFAULT_NAMESPACE, f"Deploy note {index}: the build server uses port 8080.", _vector(0),
                         source="agent:s", kind="memory_item",
                         metadata={"scope": "global", "durability": "ongoing", "session_id": "s"})
    finally:
        store.close()
    done = _run_script(home, "neospike_daily.py")
    assert (done.returncode, done.stdout) == (0, ""), (done.stdout, done.stderr[-2000:])
    with closing(sqlite3.connect(home / "neospike" / "memory.db")) as connection:
        topics = connection.execute("SELECT DISTINCT label FROM item_links WHERE relation='topic'").fetchall()
        members = connection.execute("SELECT COUNT(*) FROM item_links WHERE relation='topic'").fetchone()[0]
    assert len(topics) == 1, topics
    assert members == TOPIC_CLUSTER_SIZE


class _FakeHermes:
    """Stands in for subprocess.run of `hermes cron create/edit`: create adds the job to jobs.json."""

    def __init__(self, home: Path) -> None:
        self.jobs = home / "cron" / "jobs.json"
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs) -> subprocess.CompletedProcess:
        self.commands.append(command)
        if command[1:3] == ["cron", "create"]:
            data = json.loads(self.jobs.read_text(encoding="utf-8")) if self.jobs.is_file() else {"jobs": []}
            data["jobs"].append({"id": str(len(data["jobs"])), "name": command[command.index("--name") + 1]})
            self.jobs.parent.mkdir(parents=True, exist_ok=True)
            self.jobs.write_text(json.dumps(data), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")


def test_register_twice_creates_each_job_once(home, install, monkeypatch):
    """Only the daily job is registered (no weekly LLM job)."""
    fake = _FakeHermes(home)
    monkeypatch.setattr(install.subprocess, "run", fake)
    install.apply_cron_jobs("hermes", home)
    install.apply_cron_jobs("hermes", home)
    verbs = [(command[2], command[command.index("--name") + 1] if "--name" in command else command[3])
             for command in fake.commands]
    assert verbs == [("create", install.DAILY_JOB_NAME), ("edit", install.DAILY_JOB_NAME)]
    names = [job["name"] for job in json.loads(fake.jobs.read_text(encoding="utf-8"))["jobs"]]
    assert names == [install.DAILY_JOB_NAME]
    daily = fake.commands[0]
    assert daily[daily.index("--failure-deliver") + 1] == "bot-chat"
    assert not (home / "scripts" / "neospike_weekly_monitor.py").exists()


def test_check_crons_sees_missing_jobs_and_changed_scripts(home, install, monkeypatch):
    assert install.cron_problems(home) == [f"Cron job not registered: {install.DAILY_JOB_NAME}"]
    monkeypatch.setattr(install.subprocess, "run", _FakeHermes(home))
    install.apply_cron_jobs("hermes", home)
    assert install.cron_problems(home) == []
    (home / "scripts" / "neospike_daily.py").write_text("# edited\n", encoding="utf-8")
    assert install.cron_problems(home) == [f"Cron script differs from the repository: {home / 'scripts' / 'neospike_daily.py'}"]


# One root cron serves every Neospike profile ------------------------------------------------------
def _load_script(home: Path, name: str, alias: str):
    sys.path.insert(0, str(home / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(alias, home / "scripts" / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(home / "scripts"))
    return module


def _profile(root: Path, name: str, provider: str, data: bool = True) -> Path:
    profile = root / "profiles" / name
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(f"model:\n  default: x\nmemory:\n  memory_enabled: true\n  provider: {provider}  # c\n",
                                         encoding="utf-8")
    if data:
        (profile / "neospike").mkdir()
    return profile


def test_target_homes_are_the_root_and_its_neospike_profiles(home):
    common = _load_script(home, "neospike_cron_common.py", "common_under_test")
    design = _profile(home, "design", "neospike")
    _profile(home, "lotto", "honcho")
    _profile(home, "empty", "neospike", data=False)  # provider set but never installed
    assert common.target_homes(home) == [home, design]


class _StagesPerHome:
    """run_stage stand-in: every stage of *failing* fails; records (home, stage)."""

    def __init__(self, failing: Path) -> None:
        self.failing = failing
        self.calls: list[tuple[Path, str]] = []

    def __call__(self, home: Path, name: str, args: list[str]) -> str | None:
        self.calls.append((home, name))
        return f"FAILED {name}" if home == self.failing and name in ("backup", "topic-to-note") else None


def test_daily_keeps_going_after_a_failed_home(home, monkeypatch, capsys):
    daily = _load_script(home, "neospike_daily.py", "daily_homes_under_test")
    design = _profile(home, "design", "neospike")
    stages = _StagesPerHome(home)
    monkeypatch.setattr(daily, "run_stage", stages)
    monkeypatch.setattr(daily.common, "home_dir", _HomeOf(home))
    assert daily.main() == 1
    assert [name for target, name in stages.calls if target == design] == [name for name, _ in daily.stages(design)]
    assert capsys.readouterr().out.splitlines() == [f"{home}: FAILED backup | FAILED topic-to-note", f"{design}: ok"]
