"""Agent guidance: the per-turn block and tool descriptions carry the storage rules."""
from __future__ import annotations

import pytest

from neospike.mcp_server import TOOL_DESCRIPTIONS
from neospike.provider import NeospikeProvider
from neospike.tests.support_provider import ThreeTopicEmbedder as FakeEmbedder

# One keyword group per guidance item 1-7.
PROMPT_KEYWORDS = {
    "storage choice": ("2000 characters", "two or more # headings", "neospike_note_write", "doc (a note the user wrote)"),
    "one fact per memory": ("one fact per memory",),
    "correct vs supersede": ("mode=correct", "mode=supersede", "only part", "one-off exception", "session id"),
    "docs = human notes": ("Edit a doc (a note the user wrote) only if the user asks",),
    "save-time older items": ("older list", "neospike_memory_reconcile"),
    "result reading": ("replaced by mem:N", "archived"),
}


@pytest.fixture(scope="module")
def provider_texts():
    provider = NeospikeProvider(embedder=FakeEmbedder())
    schemas = {item["name"]: item["description"] for item in provider.get_tool_schemas()}
    return provider.system_prompt_block(), schemas


@pytest.mark.parametrize("item", list(PROMPT_KEYWORDS))
def test_prompt_block_has_guidance_item(provider_texts, item):
    prompt, _ = provider_texts
    missing = [word for word in PROMPT_KEYWORDS[item] if word not in prompt]
    assert not missing, (item, missing)


# Features that do not exist (weekly review, automatic inbox review, automatic note_proposal, topic pages) are not
# described, and the prompt plus the tool descriptions stay within the budget.
REMOVED_WORDS = ("weekly", "Sonnet", "note_proposal", "approval review", "topic page", "TOPIC",
                 "inbox", "neospike_review", "neospike_approve", "neospike_reject")  # there is no inbox
TEXT_BUDGET_CHARS = 7271


def test_removed_features_are_not_described_and_the_text_fits(provider_texts):
    prompt, schemas = provider_texts
    texts = [prompt, *schemas.values()]
    assert [word for word in REMOVED_WORDS if any(word in text for text in texts)] == []
    assert len(prompt) + sum(len(text) for text in schemas.values()) <= TEXT_BUDGET_CHARS


def test_tool_descriptions_repeat_the_rules(provider_texts):
    _, schemas = provider_texts
    # What to save lives only in the prompt's save criteria; the remember description keeps size/behaviour.
    assert "one fact per call" not in schemas["neospike_memory_save"] and "never save raw" not in schemas["neospike_memory_save"]
    assert "neospike_memory_reconcile" in schemas["neospike_memory_save"]
    assert "newest wins in one call" in schemas["neospike_memory_reconcile"]
    assert "only when the user asks" in schemas["neospike_note_write"]
    assert "only when the user asks" in schemas["neospike_doc_write"]
    assert "one-off exception" in schemas["neospike_memory_update"]
    assert "mem:N is the current decision" in schemas["neospike_memory_search"]
    assert "mem:N is the current decision" in TOOL_DESCRIPTIONS["mem_search"]
