"""A change of the memory creation rules (dashboard Config tab) reaches an open conversation once, in front
of its next recall block, instead of waiting for a new conversation; the system prompt is not rebuilt.
The Config tab screen (title, no Edit button, Add saves a blank item at once)."""
from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from neospike.config import DEFAULT_SAVE_CRITERIA, SAVE_LANGUAGE_RULE, NeospikeConfig, write_memory_rule
from neospike.provider import (
    CRITERIA_CHANGED_NOTICE,
    CRITERIA_CLEARED_NOTICE,
    NeospikeProvider,
)
from neospike.store import SCHEMA_VERSION, MemoryStore
from neospike.tests.support_provider import ThreeTopicEmbedder as FakeEmbedder
from neospike.tests.test_sphere_map import _client, home  # noqa: F401 (fixture)

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
CUSTOM = "Keep every build decision of the Falcon project as its own memory item."
LATER = "Keep every release date of the Falcon project as its own memory item."
QUERY = "Which language does the indexing pipeline use, python?"
MEMORY = "The project uses Python for its indexing pipeline."


def _set_rules(home_dir: Path, items: list[dict]) -> None:
    """The checked texts in order (the v21 list) as MEMORY_RULE.md, the only rules source."""
    write_memory_rule(home_dir, " ".join(item["text"] for item in items if item["enabled"]))


def _state(home_dir: Path) -> sqlite3.Connection:
    """A minimal Hermes state.db: the sessions columns the provider reads and the messages columns it checks."""
    connection = sqlite3.connect(home_dir / "state.db")
    try:
        connection.executescript(
            "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, git_repo_root TEXT, cwd TEXT);"
            "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
            "role TEXT NOT NULL, content TEXT, api_content TEXT);")
    except BaseException:
        connection.close()
        raise
    return connection


def _provider(home_dir: Path, session_id: str, *, build_prompt: bool = True) -> NeospikeProvider:
    with closing(_state(home_dir)) as connection, connection:
        connection.execute("INSERT OR IGNORE INTO sessions VALUES (?, NULL, ?)", (session_id, str(home_dir / "project")))
    provider = NeospikeProvider(embedder=FakeEmbedder())
    provider.initialize(session_id, hermes_home=str(home_dir), agent_identity="default", user_id="owner",
                        agent_context="primary", cwd=str(home_dir / "project"))
    if build_prompt:  # a new conversation; a restored one gets its stored prompt and Hermes skips this call
        provider.system_prompt_block()
    return provider


def _turn(home_dir: Path, session_id: str, recall: str | None) -> None:
    """Hermes stores the user row; api_content holds the composed recall when the prefetch arrived in time."""
    with closing(_state(home_dir)) as connection, connection:
        connection.execute(
            "INSERT INTO messages(session_id, role, content, api_content) VALUES (?, 'user', ?, ?)",
            (session_id, "question", None if recall is None else "question\n\n<memory-context>\n" + recall
             + "\n</memory-context>"))


def _notice(rules: str | None, language: str = "English") -> str:
    head = re.escape((CRITERIA_CHANGED_NOTICE + rules if rules else CRITERIA_CLEARED_NOTICE) + " "
                     + SAVE_LANGUAGE_RULE.format(language=language))
    return head + r" \[neospike-rules [0-9a-f]{12}\.[0-9a-f]{8}\]"


def test_system_block_names_the_save_language(tmp_path):
    english = SAVE_LANGUAGE_RULE.format(language="English")
    assert english in _provider(tmp_path, "s0").system_prompt_block()  # nothing saved: English
    NeospikeConfig(extraction_language="ko").save(tmp_path)
    block = _provider(tmp_path, "s1").system_prompt_block()
    assert SAVE_LANGUAGE_RULE.format(language="Korean") in block and english not in block
    assert block.index(DEFAULT_SAVE_CRITERIA) < block.index(SAVE_LANGUAGE_RULE.format(language="Korean"))


def test_language_change_mid_conversation_notifies_once(tmp_path):
    provider = _with_memory(tmp_path, "s1")
    before = provider.prefetch(QUERY, session_id="s1")
    assert "Memory creation rules" not in before
    NeospikeConfig(extraction_language="ko").save(tmp_path)
    first = provider.prefetch(QUERY, session_id="s1")
    assert re.fullmatch(_notice(DEFAULT_SAVE_CRITERIA, "Korean") + "\n" + re.escape(before), first)
    _turn(tmp_path, "s1", first)
    assert provider.prefetch(QUERY, session_id="s1") == before  # delivered: once only


def _with_memory(home_dir: Path, session_id: str) -> NeospikeProvider:
    provider = _provider(home_dir, session_id)
    reply = json.loads(provider.handle_tool_call("neospike_memory_save", {
        "text": MEMORY, "scope": "project", "subject": "indexing pipeline", "durability": "stable",
        "importance": 0.8, "confidence": 0.95, "tags": ["python"], "kind": "fact", "basis": "user_said"}))
    assert reply.get("stored") is True, reply
    return provider


def test_change_mid_conversation_notifies_once_before_the_recall(tmp_path):
    provider = _with_memory(tmp_path, "s1")
    before = provider.prefetch(QUERY, session_id="s1")
    assert MEMORY in before and "Memory creation rules" not in before
    _set_rules(tmp_path, [{"text": CUSTOM, "enabled": True}])
    first = provider.prefetch(QUERY, session_id="s1")
    assert re.fullmatch(_notice(CUSTOM) + "\n" + re.escape(before), first)  # notice first; recall unchanged
    _turn(tmp_path, "s1", first)
    assert provider.prefetch(QUERY, session_id="s1") == before  # delivered: once only


def test_no_change_leaves_the_recall_byte_identical(tmp_path):
    provider = _with_memory(tmp_path, "s1")
    _set_rules(tmp_path, [{"text": DEFAULT_SAVE_CRITERIA, "enabled": True}])  # saved but the same text
    first = provider.prefetch(QUERY, session_id="s1")
    assert first == provider._build_recall(QUERY, "s1") and first.startswith("<neospike_recall>")
    assert provider.prefetch("nothing matches this", session_id="s1") == ""


def test_latest_change_only_and_none_checked(tmp_path):
    provider = _provider(tmp_path, "s1")
    _set_rules(tmp_path, [{"text": CUSTOM, "enabled": True}])
    _set_rules(tmp_path, [{"text": CUSTOM, "enabled": False}, {"text": LATER, "enabled": True}])
    sent = provider.prefetch(QUERY, session_id="s1")
    assert re.fullmatch(_notice(LATER), sent)  # empty recall: notice only
    _turn(tmp_path, "s1", sent)
    _set_rules(tmp_path, [{"text": LATER, "enabled": False}])
    sent = provider.prefetch(QUERY, session_id="s1")
    assert re.fullmatch(_notice(None), sent)
    _turn(tmp_path, "s1", sent)
    assert provider.prefetch(QUERY, session_id="s1") == ""


def test_timed_out_notice_is_sent_again(tmp_path):
    provider = _provider(tmp_path, "s1")
    _set_rules(tmp_path, [{"text": CUSTOM, "enabled": True}])
    first = provider.prefetch(QUERY, session_id="s1")
    _turn(tmp_path, "s1", None)  # Hermes dropped the prefetch after 8 s: the row has no marker
    second = provider.prefetch(QUERY, session_id="s1")
    assert re.fullmatch(_notice(CUSTOM), first) and re.fullmatch(_notice(CUSTOM), second) and first != second
    _turn(tmp_path, "s1", second)
    assert provider.prefetch(QUERY, session_id="s1") == ""


def test_delivered_notice_is_not_sent_again_and_state_is_read_only_while_pending(tmp_path, monkeypatch):
    provider = _provider(tmp_path, "s1")
    _set_rules(tmp_path, [{"text": CUSTOM, "enabled": True}])
    _turn(tmp_path, "s1", provider.prefetch(QUERY, session_id="s1"))
    _turn(tmp_path, "s1", None)  # the next turn's row may already be stored when its prefetch runs
    checks = []
    original = provider._notice_seen_in_state

    def counted(session_id, marker):
        checks.append(marker)
        return original(session_id, marker)

    monkeypatch.setattr(provider, "_notice_seen_in_state", counted)
    assert provider.prefetch(QUERY, session_id="s1") == ""
    assert provider.prefetch(QUERY, session_id="s1") == ""
    assert len(checks) == 1  # confirmed once; an ordinary turn does not read state.db


def test_unreadable_state_sends_the_notice_again(tmp_path, monkeypatch):
    provider = _provider(tmp_path, "s1")
    _set_rules(tmp_path, [{"text": CUSTOM, "enabled": True}])
    _turn(tmp_path, "s1", provider.prefetch(QUERY, session_id="s1"))
    real_connect = sqlite3.connect

    def refuse_state(target, *args, **kwargs):
        if "state.db" in str(target) and "mode=ro" in str(target):
            raise sqlite3.OperationalError("unable to open database file")
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr("neospike.provider.sqlite3.connect", refuse_state)
    assert re.fullmatch(_notice(CUSTOM), provider.prefetch(QUERY, session_id="s1"))
    monkeypatch.setattr("neospike.provider.sqlite3.connect", real_connect)
    other = _provider(tmp_path, "s2", build_prompt=False)
    assert re.fullmatch(_notice(CUSTOM), other.prefetch(QUERY, session_id="s2"))
    assert re.fullmatch(_notice(CUSTOM), other.prefetch(QUERY, session_id="s2"))  # no user row: sent again


def test_restart_follows_the_record_and_unrecorded_means_default(tmp_path):
    _provider(tmp_path, "known")  # prompt built with the default rules
    _set_rules(tmp_path, [{"text": CUSTOM, "enabled": True}])
    _provider(tmp_path, "custom")  # prompt built with CUSTOM
    restarted = _provider(tmp_path, "custom", build_prompt=False)  # new provider, same session, stored prompt
    assert restarted.prefetch(QUERY, session_id="custom") == ""
    assert re.fullmatch(_notice(CUSTOM), restarted.prefetch(QUERY, session_id="known"))
    assert re.fullmatch(_notice(CUSTOM), restarted.prefetch(QUERY, session_id="older"))  # no record: default
    _set_rules(tmp_path, [{"text": DEFAULT_SAVE_CRITERIA, "enabled": True}])
    again = _provider(tmp_path, "older2", build_prompt=False)
    assert again.prefetch(QUERY, session_id="older2") == ""  # no record and default rules: nothing to say


def test_new_conversation_after_a_change_gets_no_notice(tmp_path):
    _set_rules(tmp_path, [{"text": CUSTOM, "enabled": True}])
    provider = _provider(tmp_path, "fresh")
    assert CUSTOM in provider.system_prompt_block()
    assert provider.prefetch(QUERY, session_id="fresh") == ""


def test_failed_recall_still_delivers_the_notice(tmp_path, monkeypatch):
    provider = _provider(tmp_path, "s1")
    _set_rules(tmp_path, [{"text": CUSTOM, "enabled": True}])

    def broken(*_args, **_kwargs):
        raise RuntimeError("recall down")

    monkeypatch.setattr(provider, "_fused_search", broken)
    sent = provider.prefetch(QUERY, session_id="s1")
    assert re.fullmatch(_notice(CUSTOM), sent)
    _turn(tmp_path, "s1", sent)
    with pytest.raises(RuntimeError):  # without a notice a failure is Hermes's to handle, as before
        provider.prefetch(QUERY, session_id="s1")


def test_compression_carries_what_the_conversation_knows(tmp_path):
    _set_rules(tmp_path, [{"text": CUSTOM, "enabled": True}])
    provider = _provider(tmp_path, "parent")
    provider.system_prompt_block()  # Hermes rebuilds the prompt before it announces the new session id
    provider.on_session_switch("child", parent_session_id="parent", reset=False, reason="compression")
    assert provider.prefetch(QUERY, session_id="child") == ""
    provider.on_session_switch("other", parent_session_id="child", reset=False, reason="resume")
    assert re.fullmatch(_notice(CUSTOM), provider.prefetch(QUERY, session_id="other"))  # unrecorded: default


def test_schema_v20_table(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite", dimensions=3)
    assert SCHEMA_VERSION >= 20
    store.store_session_criteria("s", "a")
    store.store_session_criteria("s", "b")
    assert store._conn.execute("SELECT criteria_hash FROM session_save_criteria WHERE session_id=?",
                               ("s",)).fetchone()[0] == "b"


def test_saved_rule_file_reaches_the_prompt(home: Path, monkeypatch) -> None:
    client = _client(monkeypatch)
    assert client.put("/config/memory-rule", json={"text": CUSTOM}).status_code == 200
    provider = NeospikeProvider(embedder=FakeEmbedder())
    provider._home_root = home
    assert (f"material. {CUSTOM} {SAVE_LANGUAGE_RULE.format(language='English')} Put text over 2000 characters"
            in provider.system_prompt_block())


def _script() -> str:
    source = (DASHBOARD / "src" / "index.js").read_text(encoding="utf-8")
    assert source == (DASHBOARD / "dist" / "index.js").read_text(encoding="utf-8")
    return source


def _panel(script: str) -> str:
    return script[script.index("function SettingsView("):script.index("// ---- Notes / Memory.")]  # takes props


def test_screen_has_no_edit_button():
    panel = _panel(_script())
    assert 't("common.edit")' not in panel and "rows: CRITERIA_EDITOR_ROWS" in panel


def test_title_in_both_languages_and_removed_texts():
    script = _script()
    assert '"config.criteria.title": "Memory creation rules"' in script
    assert '"config.criteria.title": "기억 생성 규칙"' in script
    assert "config.criteria.newItem" not in script and "config.criteria.add" not in script
    assert "config.none" not in script and "config.criteria.when" not in script
    panel = _panel(script)
    assert 'h(Badge, null, "1"), h(CardTitle, null, t("config.criteria.title"))' in panel  # the only card
    css = (DASHBOARD / "src" / "style.css").read_text(encoding="utf-8")
    assert css == (DASHBOARD / "dist" / "style.css").read_text(encoding="utf-8")
    assert ".neospike-criteria-editor {" in css and ".neospike-criteria-list" not in css
