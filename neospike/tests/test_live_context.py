"""What Hermes resends (state.db active user rows) is not recalled or
searched again, builtin mirrors stay out of the prefetch, and a <neospike_profile> block goes out once per
conversation context. Fake state.db and a keyword embedder; no LLM, no CLI."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import pytest

from neospike import conversations as conv
from neospike.config import NeospikeConfig, write_memory_rule
from neospike.provider import (CRITERIA_CHANGED_NOTICE, PROFILE_CLOSE, PROFILE_HEADER, PROFILE_INFERRED, PROFILE_OPEN,
                               PROFILE_STATED, NeospikeProvider, ProfileTuning, injected_part)
from neospike.preferences import BASIS_AGENT_INFERRED, PREF_TYPES
from neospike.recall_log import REASON_IN_CONTEXT, REASON_SOURCE_CAP
from neospike.store import HISTORY_OMIT

SID = "20261118_live"
OTHER_SID = "20261101_other"
NAMESPACE = "default:owner"
KEYWORDS = ("coffee", "tea", "python", "guitar", "rust", "bicycle", "noodle")
DIMENSIONS = len(KEYWORDS) + 1  # the last axis: no keyword
CUT_AGO_S = 1000.0
BLOCK_TAG = re.compile(r"<(neospike_\w+)>")  # opening tag of a prefetch block
PREFETCH_BLOCKS = {"neospike_profile", "neospike_recall"}  # plus neospike_situation when a situation matches
RULES = "Keep every build decision of the Falcon project as its own memory item."
LONG_WORDS = 70


class KeywordEmbedder:
    """Unit vector over the keywords a text contains; a text without one sits on the last axis."""

    dimensions = DIMENSIONS

    def embed_query(self, text: str) -> list[float]:
        return self.vector(text)

    def embed_document(self, text: str, **_kwargs) -> list[float]:
        return self.vector(text)

    @staticmethod
    def vector(text: str) -> list[float]:
        lowered = text.casefold()
        axes = [float(word in lowered) for word in KEYWORDS]
        axes.append(0.0 if any(axes) else 1.0)
        norm = sum(axes) ** 0.5
        return [value / norm for value in axes]


class WordCounter:
    """One token per whitespace word (no tokenizer.json needed)."""

    @staticmethod
    def count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def cut(text: str, max_tokens: int) -> list[str]:
        return []


def broken_lane(*_args, **_kwargs) -> list:
    raise RuntimeError("note lane down")


# --- fixtures and helpers ---------------------------------------------------------------------------------------

def make_state(home: Path) -> None:
    """Hermes' state.db: the sessions columns and the messages columns the provider reads."""
    with closing(sqlite3.connect(home / "state.db")) as connection, connection:
        connection.executescript(
            "CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, git_repo_root TEXT, cwd TEXT);"
            "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
            "role TEXT NOT NULL, content TEXT, api_content TEXT, active INTEGER NOT NULL DEFAULT 1, "
            "compacted INTEGER NOT NULL DEFAULT 0, timestamp REAL);")


def add_row(home: Path, content: str, api_content: str | None, *, session_id: str = SID, role: str = "user",
            active: int = 1, compacted: int = 0, timestamp: float | None = None) -> int:
    with closing(sqlite3.connect(home / "state.db")) as connection, connection:
        return int(connection.execute(
            "INSERT INTO messages(session_id,role,content,api_content,active,compacted,timestamp) "
            "VALUES (?,?,?,?,?,?,?)",
            (session_id, role, content, api_content, active, compacted,
             time.time() if timestamp is None else timestamp)).lastrowid)


def deliver(home: Path, block: str, *, content: str = "the question", header: str = "") -> int:
    """The user row Hermes stores when the prefetch arrived in time: its text, then the memory context."""
    return add_row(home, content, f"{header}{content}\n\n<memory-context>\n{block}\n</memory-context>")


def compact(home: Path, at: float) -> None:
    """A compaction summarized the conversation's messages up to `at`."""
    add_row(home, "old question", None, active=0, compacted=1, timestamp=at)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    NeospikeConfig(conversation_capture=False).save(tmp_path)
    return tmp_path


@pytest.fixture
def provider(home: Path):
    instance = NeospikeProvider(embedder=KeywordEmbedder())
    instance.initialize(SID, hermes_home=str(home), agent_identity="default", user_id="owner",
                        agent_context="primary", cwd=str(home))
    yield instance
    instance.shutdown()


def remember(provider: NeospikeProvider, text: str, *, origin: str = "agent:20261001_old", memory_kind: str = "fact",
             basis: str | None = "user_said", created_at: float | None = None, importance: float = 0.5) -> int:
    """A stored memory item as neospike_memory_save writes it (without the save-time tidy). A preference is a
    common row of the preferences table since schema v23 (a constraint always needs a situation: refused;
    its pref id is returned; basis None = an
    older row, stored agent_inferred)."""
    store = provider._memory_db
    if memory_kind in PREF_TYPES:
        fields = {"type": memory_kind, "text": text, "situation": "", "basis": basis or BASIS_AGENT_INFERRED,
                  "subject": "s", "importance": importance}
        pref_id = store.add_preference(NAMESPACE, fields, embedding=KeywordEmbedder.vector(text),
                                       situation_embedding=None)
        assert pref_id is not None
        if created_at is not None:
            with store._lock, store._conn:
                store._conn.execute("UPDATE preferences SET created_at=? WHERE id=?", (created_at, pref_id))
        return pref_id
    metadata = {"scope": "global", "subject": "s", "durability": "stable", "tags": [],
                "session_id": origin.split(":", 1)[1], "memory_kind": memory_kind}
    if basis is not None:
        metadata["basis"] = basis
    memory_id = store.insert(NAMESPACE, text, KeywordEmbedder.vector(text), source=origin, kind="memory_item",
                             importance=importance, metadata=metadata, dedupe_key=f"test\0{origin}\0{text}")
    assert memory_id is not None
    if created_at is not None:
        with store._lock, store._conn:
            store._conn.execute("UPDATE items SET created_at=? WHERE id=?", (created_at, memory_id))
    return memory_id


def line(memory_id: int, text: str, origin: str = "agent:20261001_old") -> str:
    return f"- [id:{memory_id} memory_item {origin}] {text}"


def pref_line(pref_id: int, text: str, pref_type: str = "preference") -> str:
    """A <neospike_profile> line (the id is pref:N; no origin since 2026-11-20)."""
    return f"- [pref:{pref_id} {pref_type}] {text}"


def search(provider: NeospikeProvider, query: str, limit: int = 5) -> dict:
    return json.loads(provider.handle_tool_call("neospike_memory_search", {"query": query, "limit": limit}))


def conversation_search(provider: NeospikeProvider, query: str, limit: int = 5) -> dict:
    """The search tool's in-context rules over the lanes with the conversation lane on (the tool itself leaves the
    conversation lane out; the prefetch runs it when conversation_prefetch is on)."""
    live = provider._read_live_context(SID, messages=True)
    hits = provider._fused_search(query, limit=limit, conversations=True, history=HISTORY_OMIT, live=live)
    kept, in_context = provider._split_in_context(hits, live)
    return {"results": kept, "omitted_in_context": len(in_context) + provider._chat_omitted}


def log_entries(home: Path) -> list[dict]:
    return [json.loads(row) for path in sorted((home / "neospike" / "logs").glob("recall-*.jsonl"))
            for row in path.read_text(encoding="utf-8").splitlines()]


# 1. a delivered block's memory lines are not sent again --------------------------------------------------------
def test_delivered_line_is_not_recalled_again(home, provider):
    make_state(home)
    text = "The user drinks coffee black."
    expected = line(remember(provider, text), text)
    first = provider.prefetch("coffee?", session_id=SID)
    assert expected in first
    deliver(home, first)
    assert provider.prefetch("coffee?", session_id=SID).count(expected) == 0


# 2. a block Hermes did not deliver (no api_content) goes out again ---------------------------------------------
def test_undelivered_line_goes_out_again(home, provider):
    make_state(home)
    text = "The user drinks coffee black."
    expected = line(remember(provider, text), text)
    assert expected in provider.prefetch("coffee?", session_id=SID)
    add_row(home, "coffee?", None)
    assert expected in provider.prefetch("coffee?", session_id=SID)


# 3. the profile block comes back once its row left the active context ------------------------------------------
def test_inactive_row_brings_the_profile_back(home, provider):
    make_state(home)
    remember(provider, "Prefers guitar examples.", memory_kind="preference")
    first = provider.prefetch("anything", session_id=SID)
    assert first.count(PROFILE_OPEN) == 1
    row = deliver(home, first)
    assert PROFILE_OPEN not in provider.prefetch("anything", session_id=SID)
    with closing(sqlite3.connect(home / "state.db")) as connection, connection:
        connection.execute("UPDATE messages SET active=0 WHERE id=?", (row,))
    assert provider.prefetch("anything", session_id=SID).count(PROFILE_OPEN) == 1


# 4. without a readable state.db: recall as before, the profile once per conversation in this process ------------
@pytest.mark.parametrize("state", ["missing", "old_columns"])
def test_unreadable_state_keeps_recall_and_sends_the_profile_once(home, provider, state):
    if state == "old_columns":
        with closing(sqlite3.connect(home / "state.db")) as connection, connection:
            connection.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, "
                               "content TEXT, api_content TEXT)")
    fact, own = "The user drinks coffee black.", "The coffee grinder setting is 12."
    fact_line = line(remember(provider, fact), fact)
    own_line = line(remember(provider, own, origin=f"agent:{SID}"), own, f"agent:{SID}")
    remember(provider, "Prefers guitar examples.", memory_kind="preference")
    first = provider.prefetch("coffee?", session_id=SID)
    second = provider.prefetch("coffee?", session_id=SID)
    assert first.count(PROFILE_OPEN) == 1 and PROFILE_OPEN not in second
    assert fact_line in first and fact_line in second  # in-context check off
    assert own_line in first and own_line in second  # own-save check off


# 5. an edited memory has a new line and goes out again ---------------------------------------------------------
def test_edited_memory_goes_out_again(home, provider):
    make_state(home)
    text, edited = "The user drinks coffee black.", "The user drinks coffee with milk."
    memory_id = remember(provider, text)
    deliver(home, provider.prefetch("coffee?", session_id=SID))
    store = provider._memory_db
    with store._lock, store._conn:
        store._conn.execute("UPDATE items SET text=? WHERE id=?", (edited, memory_id))
    assert line(memory_id, edited) in provider.prefetch("coffee?", session_id=SID)


# 6. this conversation's memories: after the cut left out, before the cut kept ----------------------------------
def test_memories_saved_here_after_the_cut_are_left_out(home, provider):
    make_state(home)
    now = time.time()
    compact(home, now - CUT_AGO_S)
    before = "The rust toolchain was pinned last month."
    before_line = line(remember(provider, before, origin=f"agent:{SID}", created_at=now - 2 * CUT_AGO_S), before,
                       f"agent:{SID}")
    reply = json.loads(provider.handle_tool_call("neospike_memory_save", {
        "text": "The build now uses the rust nightly toolchain for every single release candidate.", "scope": "global",
        "subject": "build", "durability": "stable", "kind": "fact", "basis": "user_said"}))
    assert reply["stored"] is True and "deleted" not in reply  # a wider word gap: listed as older, not merged
    recall = provider.prefetch("rust?", session_id=SID)
    assert before_line in recall and "rust nightly" not in recall
    found = search(provider, "rust?")
    assert [hit["text"] for hit in found["results"]] == [before] and found["omitted_in_context"] == 1


# 7. search tool: this conversation's excerpts still in the context (every message in an active state.db row) ----
def _conversation(provider: NeospikeProvider, session_id: str, times: list[float], *,
                  words: int = LONG_WORDS) -> list[tuple[str, str]]:
    """Messages alternating bicycle / filler, each with its own tag word; the bicycle ones are at seq 1, 3, 5 ...
    with the given created_at times (neospike's capture time). Returns the (role, content) turns in seq order."""
    turns, stamps = [], []
    for index, stamp in enumerate(times):
        turns += [("user" if index % 2 == 0 else "assistant", " ".join(["bicycle"] * words) + f" m{index}a"),
                  ("assistant" if index % 2 == 0 else "user", " ".join(["filler"] * words) + f" m{index}b")]
        stamps += [stamp, stamp]
    store = provider._memory_db
    conv.ConversationStore(store).append(session_id, turns)
    with store._lock, store._conn:
        conversation_id = store._conn.execute("SELECT id FROM conversations WHERE session_id=?",
                                              (session_id,)).fetchone()[0]
        store._conn.executemany("UPDATE conversation_messages SET created_at=? WHERE conversation_id=? AND seq=?",
                                [(stamp, conversation_id, seq) for seq, stamp in enumerate(stamps, 1)])
    conv.ConversationIndexer(store, KeywordEmbedder(), token_counter=WordCounter(), max_tokens=120,
                             context_tokens=0).index_pending()
    return turns


def to_state(home: Path, turns: list[tuple[str, str]], seqs: range, *, active: int, compacted: int,
             timestamp: float) -> None:
    """state.db rows of the messages at seqs (1-based), as Hermes keeps them."""
    for seq in seqs:
        role, content = turns[seq - 1]
        add_row(home, content, None, role=role, active=active, compacted=compacted, timestamp=timestamp)


def test_search_leaves_out_live_excerpts_and_counts_them(home, provider):
    make_state(home)
    now = time.time()
    cut = now - CUT_AGO_S
    turns = _conversation(provider, SID, [cut - 500, cut + 300, now])
    to_state(home, turns, range(1, 3), active=0, compacted=1, timestamp=cut)  # seq 1-2 summarized
    to_state(home, turns, range(3, 7), active=1, compacted=0, timestamp=now)  # seq 3-6 still in context
    _conversation(provider, OTHER_SID, [now - 9000, now - 8000, now - 7000])
    wide = conversation_search(provider, "bicycle", limit=4)
    origins = {(hit["session_id"], hit["seq_start"]) for hit in wide["results"]}
    assert origins == {(SID, 1), (OTHER_SID, 1), (OTHER_SID, 3), (OTHER_SID, 5)}
    assert wide["omitted_in_context"] == 2
    narrow = conversation_search(provider, "bicycle", limit=3)
    assert len(narrow["results"]) == 3 and narrow["omitted_in_context"] == 2
    assert all(hit["session_id"] != SID or hit["seq_start"] == 1 for hit in narrow["results"])
    others = conversation_search(provider, "bicycle", limit=2)
    assert len(others["results"]) == 2


def test_excerpt_with_one_summarized_message_is_found(home, provider):
    """A chunk over several messages: one of them only in a compacted row (no active copy) keeps the chunk."""
    make_state(home)
    now = time.time()
    turns = _conversation(provider, SID, [now, now], words=8)
    chunk = [tuple(row) for row in provider._memory_db._conn.execute(
        "SELECT seq_start, seq_end FROM conversation_chunks ch JOIN conversations c ON c.id=ch.conversation_id "
        "WHERE c.session_id=? AND seq_start>0", (SID,))]
    assert chunk == [(1, 4)]  # one chunk over all four messages
    to_state(home, turns, range(1, 2), active=0, compacted=1, timestamp=now - CUT_AGO_S)
    to_state(home, turns, range(2, 5), active=1, compacted=0, timestamp=now)
    found = conversation_search(provider, "bicycle")
    assert [(hit["session_id"], hit["seq_start"]) for hit in found["results"]] == [(SID, 1)]
    assert found["omitted_in_context"] == 0
    to_state(home, turns, range(1, 2), active=1, compacted=0, timestamp=now)  # now every message is resent
    found = conversation_search(provider, "bicycle")
    assert found["results"] == [] and found["omitted_in_context"] == 1


def test_active_copy_next_to_compacted_original_leaves_the_excerpt_out(home, provider):
    make_state(home)
    now = time.time()
    cut = now - CUT_AGO_S
    turns = _conversation(provider, SID, [cut - 500, cut - 400])
    to_state(home, turns, range(1, 5), active=0, compacted=1, timestamp=cut)
    to_state(home, turns, range(1, 3), active=1, compacted=0, timestamp=cut)  # compaction kept seq 1-2 as copies
    found = conversation_search(provider, "bicycle")
    assert [(hit["session_id"], hit["seq_start"]) for hit in found["results"]] == [(SID, 3)]
    assert found["omitted_in_context"] == 1


def test_late_capture_time_does_not_decide(home, provider):
    """created_at well past the state.db time (a turn longer than 120 s): judged by content only."""
    make_state(home)
    now = time.time()
    cut = now - CUT_AGO_S
    late = cut + 300  # past cut + the former 120 s margin
    turns = _conversation(provider, SID, [late, late, late])
    to_state(home, turns, range(1, 3), active=0, compacted=1, timestamp=cut)  # summarized: found
    to_state(home, turns, range(3, 5), active=0, compacted=1, timestamp=cut)
    to_state(home, turns, range(3, 5), active=1, compacted=0, timestamp=cut)  # active copies: left out
    to_state(home, turns, range(5, 7), active=1, compacted=0, timestamp=now)  # live: left out
    found = conversation_search(provider, "bicycle")
    assert [(hit["session_id"], hit["seq_start"]) for hit in found["results"]] == [(SID, 1)]
    assert found["omitted_in_context"] == 2


# 8. builtin mirrors: not in the prefetch (the next rank fills the slot), found by the search tool --------------
def test_builtin_mirror_is_out_of_the_prefetch_only(home, provider):
    texts = [f"Coffee fact number {index}." for index in range(8)]
    lines = [line(remember(provider, text, origin=f"agent:2026100{index}_x"), text, f"agent:2026100{index}_x")
             for index, text in enumerate(texts)]
    mirror = "The user likes coffee with oat milk."
    provider.on_memory_write("add", "memory", mirror)
    recall = provider.prefetch("coffee?", session_id=SID)
    assert mirror not in recall and all(expected in recall for expected in lines)
    found = search(provider, "coffee?", limit=20)
    assert mirror in [hit["text"] for hit in found["results"]]


# 9. the profile block --------------------------------------------------------------------------------------------
def test_profile_block_groups_orders_and_drops_mirror_duplicates(home, provider):
    now = time.time()
    inferred = remember(provider, "Seems to like noodle metaphors.", memory_kind="preference",
                        basis="agent_inferred", created_at=now - 50)
    old = remember(provider, "Likes bicycle analogies.", memory_kind="preference", basis=None, created_at=now - 100)
    preference = remember(provider, "Prefers guitar examples.", memory_kind="preference")
    remember(provider, "Likes tea in the afternoon.", memory_kind="preference")
    remember(provider, "The coffee machine is broken.", memory_kind="fact")
    provider.on_memory_write("add", "user", "The user drinks tea every afternoon.")
    block = provider.prefetch("", session_id=SID)
    assert block == "\n".join([
        PROFILE_OPEN, PROFILE_HEADER,
        PROFILE_STATED, pref_line(preference, "Prefers guitar examples."),
        PROFILE_INFERRED, pref_line(inferred, "Seems to like noodle metaphors."),
        pref_line(old, "Likes bicycle analogies."),
        PROFILE_CLOSE])


def test_profile_block_cap_drops_the_tail_and_logs_the_count(home, provider):
    body = "Prefer this long phrasing rule " + "x" * 180
    for index in range(30):
        remember(provider, f"{body} {index}.", memory_kind="preference")
    text = provider.prefetch("anything", session_id=SID)
    block = text[:text.index(PROFILE_CLOSE) + len(PROFILE_CLOSE)]
    shown = block.count("\n- [pref:")
    assert len(block) <= ProfileTuning.MAX_CHARS and 0 < shown < 30
    assert log_entries(home)[-1]["profile"] == {"lines": shown, "over_limit": 30 - shown}


def test_no_profile_lines_no_block(home, provider):
    remember(provider, "The coffee machine is broken.", memory_kind="fact")
    assert PROFILE_OPEN not in provider.prefetch("coffee?", session_id=SID)
    assert provider.prefetch("", session_id=SID) == ""


# 10. profile lines are not recalled in the same prefetch nor later ---------------------------------------------
def test_profile_lines_stay_out_of_recall(home, provider):
    make_state(home)
    text = "Prefers coffee without sugar."
    expected = pref_line(remember(provider, text, memory_kind="preference"), text)
    first = provider.prefetch("coffee?", session_id=SID)
    assert first.count(expected) == 1 and "<neospike_recall>" not in first
    deliver(home, first)
    assert provider.prefetch("coffee?", session_id=SID).count(expected) == 0


# 11. an image turn (header, then the text, then the injected blocks) ------------------------------------------
def test_image_turn_rows_count_as_in_context(home, provider):
    make_state(home)
    text = "The user drinks coffee black."
    expected = line(remember(provider, text), text)
    first = provider.prefetch("coffee?", session_id=SID)
    deliver(home, first, content="what about this coffee photo?", header="[Image #1: photo.png]\n")
    assert provider.prefetch("coffee?", session_id=SID).count(expected) == 0
    assert injected_part("question", "[img]\nquestion\n\nblock") == "\n\nblock"
    assert injected_part("missing", "[img]\nquestion") == "[img]\nquestion"


# 12. recall log: in-context items are dropped as in_context; injected = the lines carried ----------------------
def test_recall_log_reasons(home, provider):
    make_state(home)
    seen = remember(provider, "Coffee seen before.", origin="agent:20261002_a")
    deliver(home, line(seen, "Coffee seen before.", "agent:20261002_a"))
    same_source = [remember(provider, f"Coffee item {index}.", origin="agent:20261003_b") for index in range(3)]
    recall = provider.prefetch("coffee?", session_id=SID)
    entry = log_entries(home)[-1]
    injected = [item["id"] for item in entry["injected"]]
    reasons = {item["id"]: item["reason"] for item in entry["dropped"]}
    assert reasons[seen] == REASON_IN_CONTEXT and seen not in injected
    assert len(injected) == 2 and set(injected) < set(same_source)
    [capped] = set(same_source) - set(injected)
    assert reasons[capped] == REASON_SOURCE_CAP
    assert recall.count("\n- [id:") == len(injected)


# 13. a degraded lane leaves no text in the recall block --------------------------------------------------------
def test_degraded_lane_is_not_written_into_the_block(home, provider, monkeypatch):
    text = "The user drinks coffee black."
    expected = line(remember(provider, text), text)
    monkeypatch.setattr(provider, "_note_lane", broken_lane)
    recall = provider.prefetch("coffee?", session_id=SID)
    assert expected in recall and "Neospike degraded" not in recall
    assert log_entries(home)[-1]["unavailable_lanes"] == ["note"]


# 14. output order: rules notice -> profile -> recall; no pattern block -------------------------------
def test_output_order(home, provider):
    remember(provider, "Prefers guitar examples.", memory_kind="preference")
    remember(provider, "The user drinks coffee black.")
    provider.system_prompt_block()
    write_memory_rule(home, RULES)
    text = provider.prefetch("coffee?", session_id=SID)
    places = [text.index(CRITERIA_CHANGED_NOTICE), text.index(PROFILE_OPEN), text.index("<neospike_recall>")]
    assert places == sorted(places) and places[0] == 0 and set(BLOCK_TAG.findall(text)) == PREFETCH_BLOCKS


# 7. in-context matching beyond the whole line: a clipped note line and the same body in another line format ------------
def test_clipped_note_line_counts_as_in_context():
    from neospike.provider import LiveContext, SearchResult, _clip_note_line, recall_line
    note = SearchResult(7, "A long note body. " * 40, "note:7.12", 0.9, "note")
    full = recall_line(note)
    clipped = _clip_note_line(full, note, 300)
    assert clipped.endswith("…") and len(clipped) < len(full)
    live = LiveContext(SID, f"\n<neospike_recall>\n{clipped}\n</neospike_recall>\n", 0.0)
    assert not live.holds(full) and live.holds_clipped(full) and live.has_seen(full, note.text)
    other = SearchResult(8, "A different long note body. " * 40, "note:8.1", 0.9, "note")
    assert not live.has_seen(recall_line(other), other.text)


def test_same_body_in_an_older_line_format_is_not_sent_again(home, provider):
    make_state(home)
    text = "The user drinks coffee black, every morning before the first meeting."
    memory_id = remember(provider, text)
    old_format = f"- [id:{memory_id} memory_item agent:20261001_old extra] {text}"  # a format that changed since
    deliver(home, f"<neospike_recall>\n{old_format}\n</neospike_recall>")
    assert text not in provider.prefetch("coffee?", session_id=SID)


def test_short_body_elsewhere_in_the_context_still_goes_out(home, provider):
    make_state(home)
    text = "The user drinks coffee black."  # under MIN_TEXT_CHARS: a body this short is not matched by text
    expected = line(remember(provider, text), text)
    deliver(home, f"<neospike_recall>\n- [note:3.1] {text}\n</neospike_recall>")
    assert expected in provider.prefetch("coffee?", session_id=SID)
