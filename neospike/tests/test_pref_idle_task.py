"""The preference tidy-up runs as a Hermes plugin idle task with ctx.llm, not as a daily cron stage."""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from neospike import cli
from neospike import pref_reconcile as pr
from neospike import preferences as prefs
from neospike import provider as provider_module
from neospike.config import EMBEDDING_DIMENSIONS, NeospikeConfig
from neospike.database import canonical_database_path
from neospike.store import MemoryStore

PACKAGE = Path(__file__).resolve().parents[1]
DAILY_SCRIPT = PACKAGE / "cron" / "neospike_daily.py"
OWNER = "alice"
PAIR_COSINE = 0.85


class FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text


class FakePluginLlm:
    """Stands in for Hermes's PluginLlm: records complete() calls and returns the prepared answers."""

    def __init__(self, *answers: str, error: Exception | None = None) -> None:
        self.answers = list(answers)
        self.error = error
        self.calls: list[tuple[list[dict], dict]] = []

    def complete(self, messages: list[dict], **options) -> FakeResult:
        self.calls.append((messages, options))
        if self.error is not None:
            raise self.error
        return FakeResult(self.answers.pop(0) if self.answers else "[]")


class FakeRun:
    name = pr.IDLE_TASK_NAME
    idle_for_seconds = 7200.0
    last_run_at = None
    run_count = 0


class GeneralContext:
    """A real PluginContext's surface used by register(): llm, hooks, provider, auxiliary and idle tasks."""

    def __init__(self, plugin_llm) -> None:
        self.llm = plugin_llm
        self.hooks: list[str] = []
        self.providers: list = []
        self.aux_tasks: dict[str, dict] = {}
        self.idle_tasks: dict[str, tuple] = {}

    def register_hook(self, name, callback) -> None:
        self.hooks.append(name)

    def register_memory_provider(self, provider) -> None:
        self.providers.append(provider)

    def register_auxiliary_task(self, key, **options) -> None:
        self.aux_tasks[key] = options

    def register_idle_task(self, name, fn, **options) -> None:
        self.idle_tasks[name] = (fn, options)  # keyed like Hermes: a second load replaces the first


class CollectorContext(GeneralContext):
    """The memory loader's collector: register_* only, no llm (getattr(ctx, "llm") raises AttributeError)."""

    def __init__(self) -> None:
        super().__init__(None)
        del self.llm

    def __getattr__(self, attr):
        raise AttributeError(attr)


def pair_vectors() -> tuple[list[float], list[float]]:
    left = [0.0] * EMBEDDING_DIMENSIONS
    right = [0.0] * EMBEDDING_DIMENSIONS
    left[0] = 1.0
    right[0], right[1] = PAIR_COSINE, math.sqrt(1.0 - PAIR_COSINE * PAIR_COSINE)
    return left, right


def seeded_home(tmp_path: Path) -> tuple[Path, list[int]]:
    store = MemoryStore(canonical_database_path(tmp_path / "neospike", "memory"), dimensions=EMBEDDING_DIMENSIONS)
    try:
        ids = []
        for text, vector in zip(("Short answers at work", "Short answers at work in the office"), pair_vectors()):
            fields = prefs.clean_fields({"type": "preference", "text": text, "situation": "", "basis": "user_said"})
            ids.append(store.add_preference(OWNER, fields, embedding=vector, situation_embedding=None))
    finally:
        store.close()
    return tmp_path, ids


def registered_task(tmp_path: Path, plugin_llm) -> tuple[GeneralContext, object]:
    ctx = GeneralContext(plugin_llm)
    assert pr.register_idle_task(ctx, home=tmp_path) is True
    return ctx, ctx.idle_tasks[pr.IDLE_TASK_NAME][0]


# (a) idle task registration and its switches ------------------------------------------------------------------
def test_idle_task_is_daily_and_uses_the_registered_auxiliary_slot(tmp_path: Path) -> None:
    home, ids = seeded_home(tmp_path)
    plugin_llm = FakePluginLlm(json.dumps([{"pair": ids, "action": "keep", "keep_id": None, "text": None}]))
    ctx, task = registered_task(home, plugin_llm)
    assert ctx.idle_tasks[pr.IDLE_TASK_NAME][1] == {"interval_hours": 24.0}
    assert list(ctx.aux_tasks) == ["neospike_pref_reconcile"]
    summary = task(FakeRun())
    assert summary.startswith("asked 1 of 1 pairs in 1 calls: keep 1")
    options = plugin_llm.calls[0][1]
    assert options["task"] == "neospike_pref_reconcile" and options["temperature"] == 0
    assert task(FakeRun()) == ("asked 0 of 1 pairs in 0 calls: keep 0, update 0, delete 0, skipped 0, deferred 0; "
                               "memory pairs: judged 0 of 0 in 0 calls, applied 0, keep 0, skipped 0, dropped 0, "
                               "deferred 0")


def test_paused_neospike_runs_nothing(tmp_path: Path) -> None:
    home, _ids = seeded_home(tmp_path)
    (home / "neospike" / "config.json").write_text(json.dumps({"enabled": False}), encoding="utf-8")
    plugin_llm = FakePluginLlm()
    _ctx, task = registered_task(home, plugin_llm)
    assert task(FakeRun()) == "paused" and plugin_llm.calls == []


def test_switched_off_tidy_up_runs_nothing(tmp_path: Path) -> None:
    home, _ids = seeded_home(tmp_path)
    NeospikeConfig(pref_reconcile_enabled=False).save(home)
    plugin_llm = FakePluginLlm()
    _ctx, task = registered_task(home, plugin_llm)
    assert task(FakeRun()) == "disabled" and plugin_llm.calls == []


def test_failed_llm_call_fails_the_run(tmp_path: Path) -> None:
    home, _ids = seeded_home(tmp_path)
    _ctx, task = registered_task(home, FakePluginLlm(error=RuntimeError("provider down")))
    with pytest.raises(RuntimeError, match="LLM call failed: RuntimeError: provider down"):
        task(FakeRun())


def test_context_without_llm_schedules_nothing(tmp_path: Path) -> None:
    ctx = CollectorContext()
    assert pr.register_idle_task(ctx, home=tmp_path) is False
    assert ctx.idle_tasks == {} and ctx.aux_tasks == {}


def scheduled_lines(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records
            if record.name == pr.logger.name and record.levelname == "INFO" and " scheduled as idle task " in record.getMessage()]


def test_successful_registration_logs_one_info_line(tmp_path: Path, caplog) -> None:
    caplog.set_level("INFO", logger=pr.logger.name)
    registered_task(tmp_path, FakePluginLlm())
    assert scheduled_lines(caplog) == [
        "Neospike pref-reconcile scheduled as idle task 'neospike-pref-reconcile' (every 24 hours)"]


def test_unscheduled_loads_log_no_scheduled_line(tmp_path: Path, caplog) -> None:
    caplog.set_level("INFO", logger=pr.logger.name)
    failing = GeneralContext(FakePluginLlm())
    failing.register_idle_task = None  # calling it raises TypeError
    assert pr.register_idle_task(failing, home=tmp_path) is False
    assert pr.register_idle_task(CollectorContext(), home=tmp_path) is False
    assert scheduled_lines(caplog) == []


def test_command_line_run_explains_it_needs_hermes(tmp_path: Path, capsys) -> None:
    assert cli.main(["--home", str(tmp_path), "pref-reconcile"]) == 1
    captured = capsys.readouterr()
    assert "runs inside Hermes as the idle task 'neospike-pref-reconcile'" in captured.err
    assert json.loads(captured.out) == {"error": pr.NO_LLM_MESSAGE}


# (b) the daily cron job has no LLM stage ------------------------------------------------------------------------
def test_daily_job_has_no_pref_reconcile_stage() -> None:
    sys.path.insert(0, str(DAILY_SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("daily_without_pref_reconcile", DAILY_SCRIPT)
        daily = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(daily)
    finally:
        sys.path.remove(str(DAILY_SCRIPT.parent))
    assert "pref-reconcile" not in [name for name, _args in daily.stages(Path("home"))]


# (c) no Hermes internals ----------------------------------------------------------------------------------------
def imported_modules(source: str) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def test_pref_reconcile_imports_no_hermes_agent_module() -> None:
    names = imported_modules((PACKAGE / "pref_reconcile.py").read_text(encoding="utf-8-sig"))
    assert not [name for name in names if name == "agent" or name.startswith("agent.")]


# (d) both register() loads ------------------------------------------------------------------------------------
def test_both_loads_register_hooks_once_each_and_one_idle_task(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(pr, "_hermes_home", HomeOf(tmp_path))
    collector, general = CollectorContext(), GeneralContext(FakePluginLlm())
    provider_module.register(collector)
    provider_module.register(general)
    provider_module.register(general)  # a forced reload of the general plugin
    assert collector.hooks == ["transform_llm_output", "on_session_end"]
    assert general.hooks == ["transform_llm_output", "on_session_end"] * 2
    assert len(collector.providers) == 1 and collector.idle_tasks == {}
    assert list(general.idle_tasks) == [pr.IDLE_TASK_NAME]


class HomeOf:
    """Stands in for pref_reconcile._hermes_home."""

    def __init__(self, home: Path) -> None:
        self.home = home

    def __call__(self) -> Path:
        return self.home
