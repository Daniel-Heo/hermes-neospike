"""Daily preference tidy-up: candidate pairs, LLM decisions (a fake LLM), keep memory, settings and the stage."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from neospike import pref_reconcile as pr
from neospike import preferences as prefs
from neospike.config import NeospikeConfig
from neospike.store import MemoryStore
from neospike.dashboard import plugin_api as api
from neospike.tests.test_setup_routes import FakeCompleted, FakeRun, FixedAnswer, hermes_home, http  # noqa: F401  (fixture)

DIMENSIONS = 8
OWNER, OTHER_OWNER, THIRD_OWNER = "alice", "bob", "carol"
DAILY_SCRIPT = Path(__file__).resolve().parents[1] / "cron" / "neospike_daily.py"


def unit(*parts: float) -> list[float]:
    values = list(parts) + [0.0] * (DIMENSIONS - len(parts))
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def at_cos(cos: float, axis: int = 1, sign: float = 1.0) -> list[float]:
    """A unit vector with the given cosine to e0, the rest on *axis*."""
    values = [0.0] * DIMENSIONS
    values[0], values[axis] = cos, sign * math.sqrt(1.0 - cos * cos)
    return values


def plane(first: int, cos: float) -> tuple[list[float], list[float]]:
    """Two unit vectors on axes (first, first+1) with the given cosine, orthogonal to other planes."""
    left = [0.0] * DIMENSIONS
    right = [0.0] * DIMENSIONS
    left[first] = 1.0
    right[first], right[first + 1] = cos, math.sqrt(1.0 - cos * cos)
    return left, right


@pytest.fixture()
def store(tmp_path: Path):
    opened = MemoryStore(tmp_path / "memory.db", dimensions=DIMENSIONS)
    yield opened
    opened.close()


def add(store: MemoryStore, owner: str, text: str, vector: list[float] | None, *, basis: str = "user_said",
        situation: str = "", pref_type: str = "preference") -> int:
    fields = prefs.clean_fields({"type": pref_type, "text": text, "situation": situation, "basis": basis})
    pref_id = store.add_preference(owner, fields, embedding=vector, situation_embedding=None)
    assert pref_id is not None
    return pref_id


class FakeLlm:
    """Returns the prepared answers in order and records every message list."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.calls: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.answers.pop(0) if self.answers else "[]"


class FailingLlm:
    def __call__(self, messages: list[dict]) -> str:
        raise RuntimeError("provider down")


def answer(*decisions: dict) -> str:
    return "```json\n" + json.dumps(list(decisions)) + "\n```"


def ids_of(pairs) -> list[tuple[int, int]]:
    return [(pair[1]["id"], pair[2]["id"]) for pair in pairs]


# candidate pairs ----------------------------------------------------------------------------------------------
def test_candidates_keep_similar_pairs_and_drop_low_duplicate_other_owner_and_pending(store) -> None:
    base = add(store, OWNER, "I like dark roast coffee in the morning", unit(1.0))
    similar = add(store, OWNER, "I prefer dark coffee early", at_cos(0.85, axis=1))
    add(store, OWNER, "I like dark roast coffee at morning", at_cos(0.95, axis=1, sign=-1.0))  # save duplicate
    add(store, OWNER, "Tea is fine too", at_cos(0.70, axis=2))  # below 0.80
    add(store, OWNER, "Coffee without sugar", None)  # no vector yet
    add(store, OTHER_OWNER, "I like dark roast coffee in the morning, always", unit(1.0))  # other owner
    far_words = add(store, THIRD_OWNER, "Short answers", unit(1.0))
    long_text = add(store, THIRD_OWNER, "Give short answers with no preamble and no closing summary", at_cos(0.95))
    assert sorted(ids_of(pr.candidate_pairs(store))) == sorted([(base, similar), (far_words, long_text)])


def test_candidates_are_ordered_by_cosine(store) -> None:
    low = plane(0, 0.82)
    high = plane(2, 0.88)
    a, b = add(store, OWNER, "alpha one", low[0]), add(store, OWNER, "beta two three four five six", low[1])
    c, d = add(store, OWNER, "gamma one", high[0]), add(store, OWNER, "delta two three four five six", high[1])
    assert ids_of(pr.candidate_pairs(store)) == [(c, d), (a, b)]


# decisions ----------------------------------------------------------------------------------------------------
def three_pairs(store) -> dict[str, tuple[int, int]]:
    rows = {}
    for name, first in (("keep", 0), ("update", 2), ("delete", 4)):
        left, right = plane(first, 0.85)
        rows[name] = (add(store, OWNER, f"{name} rule for work", left, situation=""),
                      add(store, OWNER, f"{name} rule for the work at the office today", right))
    return rows


def test_three_decisions_apply_and_keep_pair_is_not_asked_again(store) -> None:
    rows = three_pairs(store)
    llm = FakeLlm(answer(
        {"pair": list(rows["keep"]), "action": "keep", "keep_id": None, "text": None},
        {"pair": list(rows["update"]), "action": "update", "keep_id": rows["update"][1], "text": "merged rule"},
        {"pair": list(rows["delete"]), "action": "delete", "keep_id": rows["delete"][0], "text": None}))
    result = pr.reconcile(store, llm, language="ko")
    assert (result["candidates"], result["asked"], result["calls"]) == (3, 3, 1)
    assert (result["keep"], result["update"], result["delete"], result["skipped"]) == (1, 1, 1, 0)
    assert store.preference(OWNER, rows["keep"][0]) and store.preference(OWNER, rows["keep"][1])
    merged = store.preference(OWNER, rows["update"][1])
    assert merged["text"] == "merged rule" and not merged["has_vector"]  # embed-pending fills it
    assert store.preference(OWNER, rows["update"][0]) is None
    assert store.preference(OWNER, rows["delete"][0]) and store.preference(OWNER, rows["delete"][1]) is None
    sent = json.loads(llm.calls[0][1]["content"].split("\n", 1)[1])
    assert set(sent[0]["rows"][0]) == {"id", "type", "text", "situation", "basis", "updated_at"}

    again = FakeLlm()
    second = pr.reconcile(store, again, language="ko")
    assert (second["candidates"], second["already_kept"], second["asked"], second["calls"]) == (1, 1, 0, 0)
    assert again.calls == []


def test_changed_text_makes_a_kept_pair_a_new_question(store) -> None:
    left, right = plane(0, 0.85)
    a, b = add(store, OWNER, "rule a", left), add(store, OWNER, "rule b for the long office day", right)
    pr.reconcile(store, FakeLlm(answer({"pair": [a, b], "action": "keep"})), language="ko")
    assert store.change_preference(OWNER, a, {"text": "rule a2"}, embedding=left) == "updated"
    assert pr.reconcile(store, FakeLlm(), language="ko")["asked"] == 1


def test_bad_json_and_unknown_ids_are_skipped(store) -> None:
    rows = three_pairs(store)
    broken = pr.reconcile(store, FakeLlm("I think they are the same."), language="ko")
    assert broken["skip_reasons"] == {pr.SKIP_BAD_JSON: 3} and broken["update"] == broken["delete"] == 0
    unknown = pr.reconcile(store, FakeLlm(answer(
        {"pair": [rows["delete"][0], 99999], "action": "delete", "keep_id": rows["delete"][0]},
        {"pair": list(rows["update"]), "action": "delete", "keep_id": 99999},
        {"pair": list(rows["keep"]), "action": "update", "keep_id": rows["keep"][0], "text": ""})), language="ko")
    assert unknown["skip_reasons"] == {pr.SKIP_UNKNOWN_PAIR: 1, pr.SKIP_BAD_DECISION: 2, pr.SKIP_NO_ANSWER: 1}
    assert all(store.preference(OWNER, pref_id) for pair in rows.values() for pref_id in pair)


def test_user_said_row_is_not_removed_for_an_agent_inferred_row(store) -> None:
    left, right = plane(0, 0.85)
    said = add(store, OWNER, "Answer in Korean", left, basis="user_said")
    inferred = add(store, OWNER, "Answer in Korean when coding at work", right, basis="agent_inferred",
                   situation="coding")
    llm = FakeLlm(answer({"pair": [said, inferred], "action": "delete", "keep_id": inferred}),
                  answer({"pair": [said, inferred], "action": "update", "keep_id": inferred, "text": "x"}))
    first = pr.reconcile(store, llm, language="ko")
    second = pr.reconcile(store, llm, language="ko")
    assert first["skip_reasons"] == second["skip_reasons"] == {pr.SKIP_PROTECTED: 1}
    assert store.preference(OWNER, said)["text"] == "Answer in Korean"
    assert store.preference(OWNER, inferred)["text"] == "Answer in Korean when coding at work"


def test_llm_failure_is_reported_and_changes_nothing(store) -> None:
    rows = three_pairs(store)
    result = pr.reconcile(store, FailingLlm(), language="ko")
    assert result["errors"] == ["RuntimeError: provider down"] and result["skip_reasons"] == {pr.SKIP_LLM_ERROR: 3}
    assert all(store.preference(OWNER, pref_id) for pair in rows.values() for pref_id in pair)


def test_pairs_are_sent_in_batches_of_32(store, monkeypatch) -> None:
    monkeypatch.setattr(pr.ReconcileTuning, "PAIRS_PER_CALL", 2)
    three_pairs(store)
    llm = FakeLlm()
    assert pr.reconcile(store, llm, language="ko")["calls"] == 2 and [len(json.loads(
        call[1]["content"].split("\n", 1)[1])) for call in llm.calls] == [2, 1]


# language and switch ------------------------------------------------------------------------------------------
def test_prompt_names_the_extraction_language(store, tmp_path: Path) -> None:
    assert "Write every new text in Korean." in pr.system_prompt("ko")
    assert "Write every new text in English." in pr.system_prompt("en")
    three_pairs(store)
    NeospikeConfig(extraction_language="en").save(tmp_path)
    llm = FakeLlm()
    assert pr.run(store, tmp_path, llm)["language"] == "en"
    assert "in English." in llm.calls[0][0]["content"]


def test_switched_off_stage_calls_no_llm(store, tmp_path: Path) -> None:
    three_pairs(store)
    NeospikeConfig(pref_reconcile_enabled=False).save(tmp_path)
    llm = FakeLlm()
    assert pr.run(store, tmp_path, llm) == {"disabled": True} and llm.calls == []


def test_extraction_language_setting(tmp_path: Path) -> None:
    assert NeospikeConfig.load(tmp_path).extraction_language == "en"  # English by default
    NeospikeConfig(extraction_language="ko").save(tmp_path)
    assert NeospikeConfig.load(tmp_path).extraction_language == "ko"
    assert "in English." in pr.system_prompt("unknown")
    with pytest.raises(ValueError, match="extraction_language must be one of ko, en"):
        NeospikeConfig.from_values({"extraction_language": "fr"})


def test_dashboard_shows_english_when_nothing_is_saved(http, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(api.shutil, "which", FixedAnswer(None))
    monkeypatch.setattr(api.subprocess, "run", FakeRun(FakeCompleted()))
    reply = http.get("/setup/status")
    assert reply.status_code == 200 and reply.json()["config"]["extraction_language"] == "en"


def test_dashboard_saves_and_shows_the_language(http) -> None:  # noqa: F811
    target = hermes_home() / "neospike" / "config.json"
    assert http.post("/setup/config", json={"turn_trace_enabled": False, "extraction_language": "en"}).status_code == 200
    assert json.loads(target.read_text(encoding="utf-8"))["extraction_language"] == "en"
    assert http.post("/setup/config", json={"turn_trace_enabled": True}).status_code == 200  # older client
    assert json.loads(target.read_text(encoding="utf-8"))["extraction_language"] == "en"
    assert http.post("/setup/config", json={"extraction_language": "ko"}).status_code == 200  # Config tab
    stored = json.loads(target.read_text(encoding="utf-8"))
    assert stored["extraction_language"] == "ko" and stored["turn_trace_enabled"] is True
    assert http.post("/setup/config", json={"extraction_language": "en"}).status_code == 200
    refused = http.post("/setup/config", json={"extraction_language": "fr"})
    assert refused.status_code == 422 and "extraction_language" in json.dumps(refused.json())
    assert NeospikeConfig.load(hermes_home()).extraction_language == "en"


# batch stage --------------------------------------------------------------------------------------------------
def test_daily_batch_runs_pref_reconcile_after_embed_pending() -> None:
    sys.path.insert(0, str(DAILY_SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("daily_pref_reconcile", DAILY_SCRIPT)
        daily = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(daily)
    finally:
        sys.path.remove(str(DAILY_SCRIPT.parent))
    names = [name for name, _ in daily.stages(Path("home"))]
    assert names.index("pref-reconcile") == names.index("embed-pending") + 1
    assert dict(daily.stages(Path("home")))["pref-reconcile"] == ["pref-reconcile"]


def test_constraint_is_not_merged_into_a_common_row(store) -> None:
    left, right = plane(0, 0.85)
    common = add(store, OWNER, "Answer in Korean", left)
    constraint = add(store, OWNER, "Answer in Korean when coding at work", right, situation="coding",
                     pref_type="constraint")
    llm = FakeLlm(answer({"pair": [common, constraint], "action": "update", "keep_id": common,
                          "text": "Always answer in Korean"}))
    result = pr.reconcile(store, llm, language="ko")
    assert result["skip_reasons"] == {pr.SKIP_CONSTRAINT_COMMON: 1}
    assert store.preference(OWNER, common)["text"] == "Answer in Korean"
    assert store.preference(OWNER, constraint)["situation"] == "coding"
