"""The memory creation rules of system_prompt_block are MEMORY_RULE.md of the Hermes home (the only
source; DEFAULT_SAVE_CRITERIA without the file), edited in the dashboard Config tab; the save tool
description does not repeat them."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from neospike.config import (DEFAULT_SAVE_CRITERIA, NeospikeConfig, read_memory_rule, memory_rule_file_path,
                             migrate_config_files, save_language_rule, write_memory_rule)
from neospike.provider import NeospikeProvider

LANG_EN = save_language_rule("en")  # follows the save criteria
from neospike.tests.support_provider import ThreeTopicEmbedder as FakeEmbedder
from neospike.tests.test_sphere_map import _client, home  # noqa: F401 (fixture)

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
DEFAULT_SENTENCES = ("Store with high recall", "Short facts, decisions", "Save one fact per memory",
                     "Do not save raw transcript")
CUSTOM = "Keep every build decision of the Falcon project as its own memory item."


def _prompt(home_dir: Path, rule: str | None) -> str:
    NeospikeConfig().save(home_dir)
    if rule is not None:
        memory_rule_file_path(home_dir).write_text(rule, encoding="utf-8")
    provider = NeospikeProvider(embedder=FakeEmbedder())
    provider._home_root = home_dir  # the home initialize() records; no model or database is needed here
    return provider.system_prompt_block()


def test_default_prompt_holds_the_default_text_once():
    prompt = NeospikeProvider(embedder=FakeEmbedder()).system_prompt_block()
    assert prompt.count(DEFAULT_SAVE_CRITERIA) == 1
    assert "Put text over 2000 characters" in prompt and "Markdown memory" in prompt


def test_rule_file_replaces_the_default_sentences(tmp_path):
    prompt = _prompt(tmp_path, f"{CUSTOM} Prefer one fact per item.")
    assert f"{CUSTOM} Prefer one fact per item. {LANG_EN} Put text over 2000 characters" in prompt
    assert not [s for s in DEFAULT_SENTENCES if s in prompt]


def test_default_text_in_the_file_gives_the_default_prompt(tmp_path):
    default = NeospikeProvider(embedder=FakeEmbedder()).system_prompt_block()
    assert _prompt(tmp_path, DEFAULT_SAVE_CRITERIA) == default


def test_empty_rule_file_leaves_no_criteria(tmp_path):
    prompt = _prompt(tmp_path, "")
    assert not [s for s in DEFAULT_SENTENCES if s in prompt]
    assert f"cite the source when material. {LANG_EN} Put text over 2000 characters" in prompt


def test_missing_file_uses_the_default_even_with_an_old_list(tmp_path):
    # A v21 config.json save_criteria list is not read; the code constant applies.
    NeospikeConfig().save(tmp_path)
    config = tmp_path / "neospike" / "config.json"
    config.write_text(json.dumps({"save_criteria": [{"text": CUSTOM, "enabled": True}]}), encoding="utf-8")
    assert read_memory_rule(tmp_path) == DEFAULT_SAVE_CRITERIA


def test_remember_description_has_no_repeated_criteria():
    description = next(item["description"] for item in NeospikeProvider(embedder=FakeEmbedder()).get_tool_schemas()
                       if item["name"] == "neospike_memory_save")
    for phrase in ("atomic, reusable", "one fact per call", "never save raw", "transient commands", "tool output"):
        assert phrase not in description, phrase
    assert "Text over 2000 characters" in description and "neospike_memory_reconcile" in description


def test_memory_rule_file_wins(tmp_path):
    memory_rule_file_path(tmp_path).parent.mkdir(parents=True)
    memory_rule_file_path(tmp_path).write_text("\ufeff  Hand edited rule.\n", encoding="utf-8")  # BOM and blanks ignored
    assert read_memory_rule(tmp_path) == "Hand edited rule."
    assert f"material. Hand edited rule. {LANG_EN} Put text over 2000 characters" in _prompt(tmp_path, None)
    assert write_memory_rule(tmp_path, "   ") == "" and read_memory_rule(tmp_path) == ""  # empty file = no rules
    assert not [s for s in DEFAULT_SENTENCES if s in _prompt(tmp_path, None)]
    memory_rule_file_path(tmp_path).write_text("x" * 2001, encoding="utf-8")  # broken file -> default
    assert read_memory_rule(tmp_path) == DEFAULT_SAVE_CRITERIA


def test_config_move_renames_drops_and_moves_the_old_list(tmp_path):
    # Config move (only in tests; Hermes runs it on the installs).
    config = tmp_path / "neospike" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"turn_trace_enabled": True, "episodic_retention_days": 30, "llm_weekly_token_budget": 5,
                                  "visual_enabled": False, "recall_log_months": 6,
                                  "save_criteria": [{"text": "Use neospike_remember for facts.", "enabled": True},
                                                    {"text": "Off.", "enabled": False}]}), encoding="utf-8")
    report = migrate_config_files(tmp_path)
    saved = json.loads(config.read_text(encoding="utf-8"))
    assert saved == {"turn_trace_enabled": True, "recall_log_months": 6}
    assert read_memory_rule(tmp_path) == "Use neospike_memory_save for facts."
    assert report["rule_created"] and sorted(report["removed"]) == ["episodic_retention_days", "llm_weekly_token_budget",
                                                                    "save_criteria", "visual_enabled"]
    loaded = NeospikeConfig.load(tmp_path)
    assert loaded.turn_trace_enabled


def test_config_move_replaces_old_tool_names_in_the_rule_file(tmp_path):
    write_memory_rule(tmp_path, "Save with neospike_remember; fix with neospike_update, not neospike_remember_x.")
    report = migrate_config_files(tmp_path)
    assert read_memory_rule(tmp_path) == "Save with neospike_memory_save; fix with neospike_memory_update, " \
                                         "not neospike_remember_x."
    assert report["rule_replacements"] == 2 and migrate_config_files(tmp_path)["rule_replacements"] == 0


def test_old_config_keys_are_read_until_moved(tmp_path):
    config = tmp_path / "neospike" / "config.json"
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"debug_enabled": True}), encoding="utf-8")
    assert NeospikeConfig.load(tmp_path).turn_trace_enabled is True  # the old key is read
    config.write_text(json.dumps({"debug_enabled": True, "turn_trace_enabled": False}), encoding="utf-8")
    assert NeospikeConfig.load(tmp_path).turn_trace_enabled is False  # the new key wins


def test_api_reads_saves_and_refuses(home: Path, monkeypatch) -> None:
    client = _client(monkeypatch)
    first = client.get("/config/memory-rule").json()
    assert first == {"text": DEFAULT_SAVE_CRITERIA, "default": DEFAULT_SAVE_CRITERIA, "max_chars": 2000}
    saved = client.put("/config/memory-rule", json={"text": f"  {CUSTOM} "}).json()
    assert saved["saved"] is True and saved["text"] == CUSTOM
    assert memory_rule_file_path(home).read_text(encoding="utf-8") == CUSTOM + "\n"
    too_long = client.put("/config/memory-rule", json={"text": "w " * 1001})
    secret = client.put("/config/memory-rule", json={"text": "token: abcdefghijklmnop"})
    assert too_long.status_code == 422 and secret.status_code == 422
    assert client.get("/config/memory-rule").json()["text"] == CUSTOM  # refused saves changed nothing


def _script() -> str:
    source = (DASHBOARD / "src" / "index.js").read_text(encoding="utf-8")
    assert source == (DASHBOARD / "dist" / "index.js").read_text(encoding="utf-8")
    return source


def test_panel_is_one_editor_with_save_and_default():
    script = _script()
    panel = script[script.index("function CommitCriteriaView()"):script.index("// ---- Notes / Memory.")]
    persist = panel[panel.index("function save("):panel.index("function resetToDefault(")]
    assert "apiFault(e).text" in persist and persist.count("setDraft") == 0  # a refusal shows why, the draft stays
    # Default is always enabled and needs no server answer; its text equals config.DEFAULT_SAVE_CRITERIA.
    assert 'outlined: true, onClick: resetToDefault }' in panel
    assert 'disabled: !draft.length, onClick: save }' in panel  # Save is enabled whenever the editor holds text
    assert f"const DEFAULT_MEMORY_RULE = {json.dumps(DEFAULT_SAVE_CRITERIA)};" in script
    for needed in ('onClick: save', 'onClick: resetToDefault', 'setDraft(DEFAULT_MEMORY_RULE)', 't("config.criteria.chars"',
                   "neospike-criteria-editor", '"/config/memory-rule"'):
        assert needed in panel, needed
    for gone in ('type: "checkbox"', 't("common.delete")', 't("config.criteria.add")', "criteriaPreview"):
        assert gone not in panel, gone
    assert panel.index("onClick: save") < panel.index('h("textarea"') and panel.index("onClick: resetToDefault") < panel.index('h("textarea"')


def test_en_keys_all_in_ko_and_css_uses_theme_variables():
    script = _script()
    start = script.index("const STRINGS = {")
    en = script[script.index("en: {", start):script.index("ko: {", start)]
    ko = script[script.index("ko: {", start):script.index("let language", start)]
    keys = re.compile(r'"([a-z]+\.[\w.]+)":')
    assert set(keys.findall(en)) - set(keys.findall(ko)) == set()
    for key in ("config.criteria.title", "connected.rel.nextPiece", "connected.rel.previousPiece"):
        assert f'"{key}":' in en and f'"{key}":' in ko
    css = (DASHBOARD / "src" / "style.css").read_text(encoding="utf-8")
    assert css == (DASHBOARD / "dist" / "style.css").read_text(encoding="utf-8")
    criteria = "\n".join(line for line in css.splitlines() if "neospike-criteria" in line)
    assert criteria and not re.search(r"#[0-9a-fA-F]{3,8}\b|rgb\(", criteria)


def test_config_sections_use_numbered_step_cards():
    # Config sections look like Setup's step cards; the
    # rules card is step "1".
    script = _script()
    panel = script[script.index("function SettingsView("):script.index("// ---- Notes / Memory.")]
    rules = 'h("div", { className: "neospike-step-head" }, h(Badge, null, "1"), h(CardTitle, null, t("config.criteria.title")))'
    assert rules in panel and "neospike-hero" not in panel and "TokenQuotaView" not in panel
    config = panel[:panel.index("const CRITERIA_EDITOR_ROWS")]
    assert 'className: "neospike-steps" }, h(CommitCriteriaView, null)' in config


def test_connected_relations_are_translated():
    script = _script()
    assert "item.relations.map(edgeRelationText)" in script
    assert '"next piece": "connected.rel.nextPiece", "previous piece": "connected.rel.previousPiece"' in script
