"""Preference/constraint cleanup model: install default (auto only when missing), the " API" mark rule, Config tab
strings and the shipped bundle. Nothing here runs hermes, claude, uv or hf."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from neospike.dashboard import plugin_api as api

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "neospike" / "dashboard"
FAKE_HERMES = "hermes-fake"
PICKER_KEYS = ("config.prefModel.title", "config.prefModel.auto", "config.prefModel.copy",
               "config.prefModel.current", "config.prefModel.listFailed", "config.prefModel.unchanged")


def _install_module():
    spec = importlib.util.spec_from_file_location("install_pref_model", REPO_ROOT / "install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_home(home: Path, config_text: str | None, profile: bool = False) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    if profile:
        (home / "neospike").mkdir(exist_ok=True)
    if config_text is not None:
        (home / "config.yaml").write_text(config_text, encoding="utf-8")
    return home


def no_pyyaml(text: str) -> None:
    """_yaml_data without PyYAML (the cron python): the text fallback is used."""
    return None


class RecordRun:
    """Stand-in for subprocess.run: records (argv, HERMES_HOME) and runs nothing."""

    def __init__(self):
        self.calls: list[tuple[list[str], str]] = []

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), kwargs.get("env", {}).get("HERMES_HOME", "")))


PROFILE_CONFIG = "memory:\n  provider: neospike\n"
PINNED_CONFIG = ("memory:\n  provider: neospike\nauxiliary:\n  neospike_pref_reconcile:\n"
                 "    provider: openrouter\n    model: some/model\n")


@pytest.fixture
def homes(tmp_path):
    root = _write_home(tmp_path / "root", "memory:\n  provider: neospike\nauxiliary:\n  vision:\n    provider: auto\n")
    fresh = _write_home(root / "profiles" / "fresh", PROFILE_CONFIG, profile=True)
    pinned = _write_home(root / "profiles" / "pinned", PINNED_CONFIG, profile=True)
    return root, fresh, pinned


@pytest.mark.parametrize("without_pyyaml", [False, True])
def test_install_sets_auto_only_where_the_slot_is_missing(homes, monkeypatch, without_pyyaml):
    install = _install_module()
    if without_pyyaml:
        monkeypatch.setattr(install, "_yaml_data", no_pyyaml)
    root, fresh, pinned = homes
    record = RecordRun()
    monkeypatch.setattr(install.subprocess, "run", record)
    before = (pinned / "config.yaml").read_bytes()
    install.apply_pref_model_default(FAKE_HERMES, root)
    argv = [FAKE_HERMES, "config", "set", "auxiliary.neospike_pref_reconcile.provider", "auto"]
    assert record.calls == [(argv, str(root)), (argv, str(fresh))]
    assert (pinned / "config.yaml").read_bytes() == before


def test_slot_detection_needs_the_key_under_auxiliary(tmp_path, monkeypatch):
    install = _install_module()
    elsewhere = _write_home(tmp_path / "a", "model:\n  neospike_pref_reconcile: x\nauxiliary:\n  vision: {}\n")
    missing = tmp_path / "missing"
    for yaml_reader in (install._yaml_data, no_pyyaml):
        monkeypatch.setattr(install, "_yaml_data", yaml_reader)
        assert not install.pref_model_configured(elsewhere)
        assert not install.pref_model_configured(missing)
        assert install.pref_model_configured(_write_home(tmp_path / "b", PINNED_CONFIG))


def test_task_name_is_the_one_pref_reconcile_uses():
    from neospike import pref_reconcile
    install = _install_module()
    script = (DASHBOARD / "dist" / "index.js").read_text(encoding="utf-8-sig")
    assert install.PREF_MODEL_TASK == pref_reconcile.LLM_TASK
    assert f'const PREF_MODEL_TASK = "{pref_reconcile.LLM_TASK}";' in script


@pytest.mark.parametrize(("provider", "auth_types", "marked"), [
    ("anthropic", ["api_key"], True),            # ANTHROPIC_API_KEY
    ("anthropic", ["oauth"], False),             # Claude subscription token (sk-ant-oat...) or Hermes PKCE login
    ("anthropic", ["api_key", "oauth"], False),  # an OAuth token is used first
    ("openrouter", ["api_key", "api_key"], True),
    ("openai-codex", ["oauth"], False),          # ChatGPT subscription
    ("openai-codex", ["api_key"], False),
    ("nous", [], False),                         # no credential known
])
def test_api_mark_only_for_api_key_credentials(provider, auth_types, marked):
    assert api.api_key_authenticated(provider, auth_types) is marked


class FakePools:
    """Stand-in for agent.credential_pool: auth types per provider; unknown providers raise like a missing pool."""

    def __init__(self, table):
        self.table = table

    def __call__(self, provider):
        if provider not in self.table:
            raise KeyError(provider)
        return self.table[provider]


def test_route_returns_the_api_key_providers(monkeypatch):
    monkeypatch.setattr(api, "_pool_auth_types", FakePools({
        "anthropic": ["oauth"], "openrouter": ["api_key"], "openai-codex": ["oauth"], "deepseek": ["api_key"]}))
    application = FastAPI()
    application.include_router(api.router)
    reply = TestClient(application).get("/config/api-key-providers",
                                        params={"providers": "anthropic,openrouter,openai-codex,ghost,deepseek,openrouter"})
    assert reply.status_code == 200
    assert reply.json() == {"providers": ["openrouter", "deepseek"]}


def test_bundle_label_rule_appends_api_suffix():
    script = (DASHBOARD / "dist" / "index.js").read_text(encoding="utf-8-sig")
    assert 'const API_KEY_SUFFIX = " API";' in script
    assert "function prefModelLabel(model, apiKey) { return apiKey ? model + API_KEY_SUFFIX : model; }" in script
    panel = script[script.index("function SettingsView("):script.index("// ---- Notes / Memory.")]
    assert "h(PrefModelView, null)" in panel
    assert "pref_reconcile_enabled" not in script[script.index("function PrefModelView("):script.index("const CRITERIA_EDITOR_ROWS")]


def test_picker_strings_exist_in_both_languages():
    script = (DASHBOARD / "dist" / "index.js").read_text(encoding="utf-8-sig")
    start = script.index("const STRINGS = {")
    en = script[script.index("en: {", start):script.index("ko: {", start)]
    ko = script[script.index("ko: {", start):script.index("let language", start)]
    keys = re.compile(r'"([a-z]+\.[\w.]+)":')
    assert set(keys.findall(en)) - set(keys.findall(ko)) == set()
    for key in PICKER_KEYS:
        assert f'"{key}":' in en and f'"{key}":' in ko
