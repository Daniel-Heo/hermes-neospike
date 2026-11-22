"""The on/off switch ("enabled" in config.json): while it is off the provider recalls, stores and follows up
nothing, its tools, the MCP searches and embedding requests are refused, the daily job skips the home, and the
shared embedding service is asked to stop unless another home uses it. Turned back on, the next call works."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from neospike import inference, mcp_server
from neospike.config import PAUSED_MESSAGE, NeospikeConfig, read_enabled
from neospike.cron import neospike_cron_common as common
from neospike.dashboard import plugin_api as api
from neospike.inference import EmbeddingClient, EmbeddingDaemon
from neospike.tests.support_embed_service import RecordingBackend, cache_dir, spawn_service
from neospike.tests.support_provider import make_provider, save_item
from neospike.tests.test_cron_scripts import _HomeOf, _load_script, _profile
from neospike.tests.test_embed_service_wire import svc_home  # noqa: F401  (fixture)
from neospike.tests.test_install_check import _install_module
from neospike.tests.test_setup_routes import FakeCompleted, FakeRun, hermes_home, http  # noqa: F401  (fixture)
from neospike.tests.test_wincompat import read_line

SERVICE_EXIT_SECONDS = 10


def _switch(home: Path, on: bool) -> None:
    """The dashboard save: only "enabled" changes, the other keys stay."""
    path = home / "neospike" / "config.json"
    values = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**values, "enabled": on}), encoding="utf-8")


class _Conversations:
    """ConversationStore stand-in: records what sync_turn appends."""

    def __init__(self) -> None:
        self.appended: list[str] = []

    def append(self, session_id: str, turns) -> bool:
        self.appended.append(session_id)
        return False


class _Always:
    def __call__(self) -> bool:
        return True


# ---- config -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(("text", "on"), [(None, True), ('{"enabled": false}', False), ('{"enabled": "false"}', False),
                                          ('{"enabled": true}', True), ("not json", True), ('{"enabled": 3}', True)])
def test_only_an_explicit_false_pauses(tmp_path, text, on):
    if text is not None:
        (tmp_path / "neospike").mkdir()
        (tmp_path / "neospike" / "config.json").write_text(text, encoding="utf-8")
    assert read_enabled(tmp_path) is on
    assert common.home_enabled(tmp_path) is on  # the cron copy of the same rule


def test_config_round_trips_the_switch(tmp_path):
    assert NeospikeConfig.load(tmp_path).enabled is True
    NeospikeConfig(enabled=False).save(tmp_path)
    assert NeospikeConfig.load(tmp_path).enabled is False and read_enabled(tmp_path) is False


# ---- provider -----------------------------------------------------------------------------------------------
def test_paused_provider_recalls_stores_and_answers_nothing_until_switched_back_on(tmp_path):
    home = tmp_path / "hermes"
    provider = make_provider(home)
    try:
        assert save_item(provider, "The user plays guitar every evening", "global", "hobby").get("stored", True)
        conversations = provider._talk_store = _Conversations()
        _switch(home, False)
        assert provider.prefetch("guitar") == ""
        reply = json.loads(provider.handle_tool_call("neospike_memory_search", {"query": "guitar"}))
        assert reply == {"error": PAUSED_MESSAGE}
        provider._mark_saved("s-one")
        provider.sync_turn("guitar?", "yes", session_id="s-one")
        assert conversations.appended == [] and provider._saved_in_turn == set()
        _switch(home, True)  # no restart: the next call reads the switch again
        assert "guitar" in provider.prefetch("guitar")
        assert json.loads(provider.handle_tool_call("neospike_memory_search", {"query": "guitar"}))["results"]
        provider.sync_turn("guitar?", "yes", session_id="s-one")
        assert conversations.appended == ["s-one"]
    finally:
        provider.shutdown()


def test_paused_client_refuses_before_any_daemon_is_started(tmp_path):
    client = EmbeddingClient(tmp_path, paused=_Always())
    with pytest.raises(RuntimeError, match="Neospike is paused"):
        client.embed_document("text")
    assert not (tmp_path / "neospike" / ".cache").exists()


# ---- embedding service --------------------------------------------------------------------------------------
def test_shutdown_request_is_answered_then_sets_the_stop_event(svc_home):  # noqa: F811
    daemon = EmbeddingDaemon(svc_home, backend=RecordingBackend())
    daemon.start()
    try:
        assert EmbeddingClient(svc_home).request_shutdown() is True
        assert daemon.shutdown_requested.is_set()
    finally:
        daemon.close()
    assert EmbeddingClient(svc_home).request_shutdown() is False  # nothing runs, nothing is started
    assert not (cache_dir(svc_home) / "i.sock").exists()


def test_release_keeps_the_service_while_another_home_uses_it(svc_home):  # noqa: F811
    daemon = EmbeddingDaemon(svc_home, backend=RecordingBackend())
    daemon.start()
    bot = _profile(svc_home, "bot", "neospike")
    try:
        assert inference.release_daemon(svc_home) == (inference.DAEMON_IN_USE, [bot])
        assert not daemon.shutdown_requested.is_set()
        _switch(bot, False)  # a paused home no longer counts as a user
        assert inference.release_daemon(svc_home) == (inference.DAEMON_STOPPED, [])
        assert daemon.shutdown_requested.is_set()
    finally:
        daemon.close()
    assert inference.release_daemon(svc_home) == (inference.DAEMON_NOT_RUNNING, [])


def test_service_process_exits_cleanly_on_a_shutdown_request(svc_home):  # noqa: F811
    process = spawn_service(svc_home)
    try:
        assert "ready" in read_line(process.stdout, SERVICE_EXIT_SECONDS), process.stderr.read()
        assert EmbeddingClient(svc_home).request_shutdown() is True
        assert process.wait(timeout=SERVICE_EXIT_SECONDS) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=SERVICE_EXIT_SECONDS)
    assert not (cache_dir(svc_home) / "i.sock").exists()


# ---- daily cron ---------------------------------------------------------------------------------------------
class _StageHomes:
    def __init__(self) -> None:
        self.homes: list[Path] = []

    def __call__(self, home: Path, until=None) -> list[str]:
        self.homes.append(home)
        return []


def test_daily_job_skips_a_paused_home(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    _install_module().copy_cron_scripts(root)
    daily = _load_script(root, "neospike_daily.py", "daily_pause_under_test")
    design, lotto = _profile(root, "design", "neospike"), _profile(root, "lotto", "neospike")
    _switch(lotto, False)
    ran = _StageHomes()
    monkeypatch.setattr(daily, "run_home", ran)
    monkeypatch.setattr(daily.common, "home_dir", _HomeOf(root))
    assert daily.main() == 0
    assert ran.homes == [root, design]
    _switch(lotto, True)
    ran.homes.clear()
    assert daily.main() == 0 and ran.homes == [root, design, lotto]


# ---- MCP server ---------------------------------------------------------------------------------------------
def test_mcp_search_is_refused_while_paused(tmp_path):
    _switch(tmp_path, False)
    tools = mcp_server.NeospikeTools(tmp_path)
    with pytest.raises(mcp_server.ToolError) as refused:
        tools.mem_search("guitar")
    assert json.loads(str(refused.value)) == {"error": "paused", "message": PAUSED_MESSAGE}
    _switch(tmp_path, True)
    with pytest.raises(mcp_server.ToolError) as missing:
        tools.mem_search("guitar")
    assert json.loads(str(missing.value))["error"] == "not_found"  # past the switch: no database yet


# ---- dashboard ----------------------------------------------------------------------------------------------
class _ReleaseRecorder:
    def __init__(self) -> None:
        self.homes: list[Path] = []

    def __call__(self, home: Path):
        self.homes.append(home)
        return inference.DAEMON_IN_USE, [home / "profiles" / "bot"]


def test_dashboard_switch_saves_reports_the_service_and_shows_the_state(http, monkeypatch):  # noqa: F811
    release = _ReleaseRecorder()
    monkeypatch.setattr(api.inference, "release_daemon", release)
    monkeypatch.setattr(api.shutil, "which", {}.get)
    monkeypatch.setattr(api.subprocess, "run", FakeRun(FakeCompleted()))
    home = hermes_home()
    off = http.post("/setup/config", json={"enabled": False})
    assert off.status_code == 200
    assert off.json()["daemon"] == {"state": "in_use", "users": [str(home / "profiles" / "bot")]}
    assert release.homes == [home] and read_enabled(home) is False
    assert http.get("/setup/status").json()["config"]["enabled"] is False
    on = http.post("/setup/config", json={"enabled": True})
    assert on.status_code == 200 and "daemon" not in on.json() and release.homes == [home]
    assert http.get("/setup/status").json()["config"]["enabled"] is True


def test_dashboard_tables_carry_the_switch_in_both_languages():
    source = Path(api.__file__).resolve().parent / "dist" / "index.js"
    text = source.read_text(encoding="utf-8-sig")
    assert (source.parent.parent / "dist" / "index.js").read_bytes() == source.read_bytes()
    for key in ("config.enabled.title", "config.enabled.label", "config.enabled.copy", "config.enabled.stopped",
                "config.enabled.not_running", "config.enabled.in_use", "config.enabled.error"):
        assert text.count(f'"{key}":') == 2, key
