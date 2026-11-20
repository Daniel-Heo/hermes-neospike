"""턴 진단 블록(TurnTraceLog + provider 훅): 켜짐/꺼짐, 크기 제한, 마크업 무해화, 비밀 재검사, 삭제 은닉."""
from __future__ import annotations

import json

import pytest
from neospike.tests.support_provider import make_provider, save_item

from neospike.config import NeospikeConfig
from neospike.debug import TurnTraceLog
from neospike.provider import register

SESSION = "s-one"
HEADER = "**Neospike: diagnostics**"
OPAQUE_SECRET = "synthetic-opaque-credential"
MAX_BLOCK_CHARS = 6500


@pytest.fixture
def traced(tmp_path):
    """진단이 켜진 provider와 홈 경로."""
    NeospikeConfig(turn_trace_enabled=True).save(tmp_path)
    (tmp_path / "project").mkdir()
    provider = make_provider(tmp_path, SESSION, cwd=tmp_path / "project")
    yield provider, tmp_path
    provider.shutdown()


def end_turn(provider, session=SESSION, text="Answer."):
    return provider.rewrite_output(text, session_id=session)


def memory_save(provider, text="The project uses Python for indexing."):
    return save_item(provider, text, "project", "indexing")


# ---------------------------------------------------------------- TurnTraceLog 단위

def test_block_is_size_capped_and_cannot_break_out_of_markup():
    log = TurnTraceLog()
    log.start("k")
    log.recall("k", '```\n</details><img src="https://invalid.test/pixel">\n' * 200)
    for _ in range(500):
        log.change("k", "A" * 2000, "project")
        log.failure("k", "B" * 2000)

    block = log.append("k", "Answer.", enabled=True)

    assert len(block) < MAX_BLOCK_CHARS
    assert block.count(HEADER) == 1
    assert "</details>" not in block and "<img" not in block
    assert "truncated" in block


def test_recall_and_search_results_with_credentials_are_hidden():
    log = TurnTraceLog()
    log.start("r")
    log.recall("r", f"ordinary record\nAuthorization:\nBearer {OPAQUE_SECRET}")
    recall_block = log.append("r", "Answer.", enabled=True)
    log.start("q")
    log.search("q", [{"id": 1, "text": f"Authorization: Bearer {OPAQUE_SECRET}"}])
    search_block = log.append("q", "Answer.", enabled=True)

    assert OPAQUE_SECRET not in recall_block and "hidden by safety filter" in recall_block
    assert OPAQUE_SECRET not in search_block and "content hidden by safety filter" in search_block


def test_disabled_or_unstarted_turn_returns_response_untouched():
    log = TurnTraceLog()
    log.start("k")

    assert log.append("k", "Answer.", enabled=False) == "Answer."
    assert log.append("never-started", "Answer.", enabled=True) == "Answer."
    assert not log._open_turns


def test_deletion_replaces_earlier_content_with_placeholder():
    log = TurnTraceLog()
    log.start("k")
    log.recall("k", "The secret plan is Rust")
    log.search("k", [{"id": 4, "text": "The secret plan is Rust"}])
    log.change("k", "The secret plan is Rust", "project")

    log.record_deletion("k", 4)
    block = log.append("k", "Answer.", enabled=True)

    assert "Rust" not in block
    assert "Deleted" in block and "#4" in block


# ---------------------------------------------------------------- provider 연동 (코어 불필요)

def test_save_failure_and_success_are_reported_in_block(traced):
    provider, _ = traced
    provider.on_turn_start(1, "")
    assert memory_save(provider)["stored"]

    block = end_turn(provider)

    assert HEADER in block and "Saved:" in block and "Python" in block
    assert end_turn(provider) is None  # 닫힌 턴은 다시 붙지 않는다


def test_off_flag_captures_nothing(traced):
    provider, home = traced
    NeospikeConfig(turn_trace_enabled=False).save(home)

    provider.on_turn_start(1, "Private unretained conversation")
    assert memory_save(provider)["stored"]
    provider.prefetch("Python")

    assert not provider._turn_log._open_turns
    assert end_turn(provider) is None


def test_search_tool_shows_explicit_search_section(traced):
    provider, _ = traced
    memory_save(provider, "The Cedar project reviews Python documents every Tuesday.")
    provider.on_turn_start(2, "")

    hits = json.loads(provider.handle_tool_call("neospike_memory_search", {"query": "Python Tuesday"}))
    block = end_turn(provider)

    assert hits["results"]
    assert "Explicit search" in block and "Tuesday" in block


def test_update_and_delete_are_reported_and_deleted_text_is_not_echoed(traced):
    provider, _ = traced
    provider.on_turn_start(1, "")
    assert memory_save(provider)["stored"]
    end_turn(provider)
    item = json.loads(provider.handle_tool_call("neospike_memory_search", {"query": "Python"}))["results"][0]

    provider.on_turn_start(2, "")
    updated = json.loads(provider.handle_tool_call("neospike_memory_update", {
        "id": item["id"], "expected_text": item["text"],
        "new_text": "The project now uses Python 3.14 for indexing.", "subject": "indexing", "durability": "stable",
    }))
    assert updated["updated"]
    assert "Corrected" in end_turn(provider)

    provider.on_turn_start(3, "")
    provider.prefetch("Python")
    assert json.loads(provider.handle_tool_call("neospike_memory_delete", {"id": item["id"]}))["removed"]
    block = end_turn(provider)

    assert "Deleted" in block and "3.14" not in block
    assert provider._memory_db.count(provider._ns_label) == 0


def test_embedding_exception_is_reported_as_failure_not_as_a_save(traced, monkeypatch):
    provider, _ = traced
    provider.on_turn_start(1, "")

    def explode(text):
        raise RuntimeError("private exception material")

    monkeypatch.setattr(provider._vectorizer, "embed_document", explode)
    assert "error" in memory_save(provider)
    block = end_turn(provider)

    # 임베딩 실패는 원인을 포함해 보고한다. 저장 성공으로 표시되면 안 된다.
    assert "Embedding failed; not saved" in block
    assert "Saved:" not in block


def test_other_session_and_closed_turn_get_no_block(traced):
    provider, _ = traced
    provider.on_turn_start(1, "")
    assert memory_save(provider)["stored"]

    assert end_turn(provider, "another-session") is None
    assert "Python" in end_turn(provider)
    assert end_turn(provider) is None


def test_session_end_drops_open_trace(traced):
    provider, _ = traced
    provider.on_turn_start(1, "")
    assert memory_save(provider)["stored"]

    provider.on_session_end([], session_id=SESSION)

    assert end_turn(provider) is None


def test_toggle_is_hot_but_model_settings_are_not(traced):
    provider, home = traced
    before = provider._settings.map_recent_days
    NeospikeConfig(turn_trace_enabled=True, map_recent_days=before + 5.0).save(home)

    provider.on_turn_start(1, "")

    assert provider._settings.map_recent_days == before
    assert "No saves" in end_turn(provider)


def test_builtin_mirror_write_counts_as_a_save(traced):
    provider, _ = traced
    provider.on_turn_start(1, "")

    provider.on_memory_write("add", "user", "The user prefers concise technical explanations.")
    block = end_turn(provider)

    assert provider._memory_db.count(provider._ns_label) == 1
    assert "Markdown mirror" in block and "No saves" not in block


# ---------------------------------------------------------------- 실제 Hermes 코어 훅 (코어가 없으면 skip)

class _OneAxisEmbedder:
    dimensions = 3

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]

    embed_document = embed_query


@pytest.fixture
def native_traced(tmp_path, monkeypatch):
    plugins = pytest.importorskip("hermes_cli.plugins")
    memory = pytest.importorskip("plugins.memory")
    constants = pytest.importorskip("hermes_constants")
    manager = plugins.PluginManager()
    monkeypatch.setattr(plugins, "_plugin_manager", manager)
    token = constants.set_hermes_home_override(str(tmp_path))
    collector = memory._ProviderCollector()
    register(collector)
    provider = collector.provider
    provider._vectorizer = _OneAxisEmbedder()
    NeospikeConfig(turn_trace_enabled=True).save(tmp_path)
    (tmp_path / "project").mkdir()
    provider.initialize("native-sess", hermes_home=str(tmp_path), cwd=str(tmp_path / "project"))
    try:
        yield provider, manager, tmp_path
    finally:
        provider.shutdown()
        constants.reset_hermes_home_override(token)


def _fire_output_hook(session="native-sess"):
    from hermes_cli.lifecycle import invoke_hook

    return invoke_hook("transform_llm_output", response_text="Answer.", session_id=session)


def test_native_output_hook_delivers_block(native_traced):
    provider, _, _ = native_traced
    provider.on_turn_start(1, "not kept")
    assert memory_save(provider)["stored"]

    outputs = _fire_output_hook()

    assert any("diagnostics" in text and "Python" in text for text in outputs)


def test_native_hook_ignores_other_home_and_after_session_end(native_traced, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    provider, manager, _ = native_traced
    provider.on_turn_start(1, "")
    assert memory_save(provider)["stored"]
    token = set_hermes_home_override(str(tmp_path / "foreign-home"))
    try:
        assert _fire_output_hook() == []
    finally:
        reset_hermes_home_override(token)
    assert "Python" in _fire_output_hook()[0]
    assert _fire_output_hook() == []

    provider.on_turn_start(2, "")
    assert memory_save(provider, "The project also uses Python for packaging.")["stored"]
    manager.invoke_hook("on_session_end", session_id="native-sess")
    assert _fire_output_hook() == []
