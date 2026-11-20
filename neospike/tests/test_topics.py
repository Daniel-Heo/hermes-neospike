"""Topics. No LLM is called.

A topic is a tag; no topic page is written. A test that needs an existing page writes it directly (_topic_page)."""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import pytest

from neospike import topics
from neospike.config import EMBEDDING_DIMENSIONS, NeospikeConfig
from neospike.notes import NoteService
from neospike.provider import NeospikeProvider
from neospike.store import MemoryStore
from neospike.tests.support_items import members_of, note_rows, topic_labels, topic_names
from neospike.tests.test_remember_dedupe import no_remember_tidy  # noqa: F401 (fixture)

# These toy embedders give distinct memories one vector; post-save dedupe is not under test here.
pytestmark = pytest.mark.usefixtures("no_remember_tidy")

NS = "test:topics"
DIMS = 8
PAGE_AXIS = DIMS - 1  # note chunk vectors of the store tests; never a memory axis


class PageEmbedder:
    """Note chunk vectors for the store tests (memories get explicit vectors)."""

    def embed_document(self, text: str, **_kwargs) -> list[float]:
        return _axis(PAGE_AXIS)


def _axis(index: int) -> list[float]:
    vector = [0.0] * DIMS
    vector[index] = 1.0
    return vector


def _mix(cosine: float, main: int = 0, other: int = 1) -> list[float]:
    vector = [0.0] * DIMS
    vector[main], vector[other] = cosine, math.sqrt(1 - cosine * cosine)
    return vector


def _store(tmp_path: Path, **config) -> MemoryStore:
    directory = tmp_path / "neospike"
    directory.mkdir(parents=True, exist_ok=True)
    if config:
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return MemoryStore(directory / "memory.db", dimensions=DIMS)


def _remember(store: MemoryStore, text: str, vector: list[float], subject: str = "release approval") -> int:
    memory_id = store.insert(NS, text, vector, source="agent:s", kind="memory_item",
                             metadata={"scope": "global", "subject": subject, "durability": "ongoing", "tags": [],
                                       "session_id": "s"})
    assert memory_id is not None
    return memory_id


def _topic_from(store: MemoryStore, texts: list[str], vector: list[float]) -> tuple[str, list[int]]:
    """A topic made of these memories (the topic_new apply step; discovery itself: test_topic_discovery)."""
    ids = [_remember(store, text, vector) for text in texts]
    with store._lock, store._conn:
        topic_id = topics._create_topic(store._conn, topics._member_rows(store._conn, ids), time.time())
    return topic_id, ids


def _topic_page(store: MemoryStore, topic_id: str, embedder, *, summary: str = "Summary",
                 decisions: str = "Decision") -> int:
    """A topic page (agent note, topic tag, attached to the topic by a topic link)."""
    with store._lock:
        member_ids = topics._topic_member_ids(store._conn, topic_id)
        name = topic_id
    evidence = "\n".join(f"- [[mem:{member_id}]] evidence" for member_id in member_ids) or "(없음)"
    body = (f"## 요약\n{summary}\n\n## 현재 결정\n{decisions}\n\n## 바뀐 결정\n(없음)\n\n"
            f"## 근거 기억\n{evidence}\n")
    saved = NoteService(store, embedder).save(f"{topics.TOPIC_TITLE_PREFIX}{name}", body, tags=["topic"],
                                              author="agent")
    with store._lock, store._conn:
        store._conn.execute("INSERT INTO item_links(src_id,relation,label) VALUES (?,'topic',?)", (saved["id"], topic_id))
    return int(saved["id"])


CLEAN_TIME = 1.0  # a topic "clean" since: its links and members dated back to this time


def _clean(store: MemoryStore, name: str) -> None:
    """v22 keeps no dirty flag: date the topic's links and members back so a later change shows in its time."""
    with store._lock, store._conn:
        store._conn.execute("UPDATE item_links SET created_at=? WHERE relation='topic' AND label=?", (CLEAN_TIME, name))
        store._conn.execute("UPDATE items SET updated_at=? WHERE id IN (SELECT src_id FROM item_links "
                            "WHERE relation='topic' AND label=?)", (CLEAN_TIME, name))


def _dirty(store: MemoryStore, name: str) -> bool:
    """Changed since _clean: the computed topic time (topics.topic_changed_at_sql) moved."""
    changed = store._conn.execute(f"SELECT {topics.topic_changed_at_sql('?')}", (name,)).fetchone()[0]
    return changed is not None and float(changed) > CLEAN_TIME


# 1. assignment (no topic on save; joins only through links) ---------------------------
def test_stored_memory_has_no_topic_and_name_stays(tmp_path):
    store = _store(tmp_path)
    topic_id, _ = _topic_from(store, ["approval mark A", "approval mark B", "approval mark C"], _axis(0))
    _clean(store, topic_id)  # start from a clean topic
    name = topic_id
    near = _remember(store, "almost the same memory", _mix(0.99))
    assert near not in topic_labels(store._conn)
    assert not _dirty(store, topic_id)
    far = _remember(store, "far away memory", _mix(0.3, other=2))
    assert topics.assign_all(store, NS)["members"] == [{"memory_id": near, "topic_id": topic_id}]
    assert _dirty(store, topic_id)
    assert far not in topic_labels(store._conn)
    # the name set at creation never changes when the members do.
    assert topic_labels(store._conn)[near] == name and topic_names(store._conn) == [name]
    store.close()


# 2. discovery -------------------------------------------------------------------------------
def test_discover_six_linked_make_one_topic_five_make_none(tmp_path):
    """A linked group of more than 5 memories becomes a new topic at once (no candidate step)."""
    store = _store(tmp_path)
    for index in range(1, 6):
        _remember(store, f"otter {index}", _mix(0.95, 0, index), subject="otter sighting")
    assert topics.discover(store, NS)["topic_new"] == 0
    _remember(store, "otter six", _mix(0.95, 0, 6), subject="otter count")
    assert topics.discover(store, NS)["topic_new"] == 1
    names = topic_names(store._conn)
    assert len(names) == 1 and names[0].startswith("otter")
    assert topics.discover(store, NS)["topic_new"] == 0  # the members have their topic now
    store.close()


# 3. no topic page is written ------------------------------------------------------
def test_new_topic_has_no_page(tmp_path):
    store = _store(tmp_path)
    topic_id, ids = _topic_from(store, ["Use OTTER mark", "Mark lives in config", "Mark is checked by CI"], _axis(0))
    assert sorted(members_of(store._conn, topic_id)) == sorted(ids)  # only the memories carry the topic link
    assert note_rows(store._conn) == []
    store.close()


def test_long_note_line_is_clipped_not_dropped():
    """A note chunk longer than the recall room is cut to it with "…"; memories still drop."""
    from neospike.store import SearchResult
    short = SearchResult(1, "short memory", "agent:s", 1.0, "memory_item")
    note = SearchResult(7, "y" * 5000, "note:7 #근거 기억", 0.9, "note")
    long_memory = SearchResult(2, "z" * 5000, "agent:s", 0.8, "memory_item")
    text = NeospikeProvider._render_recall([short, note, long_memory], max_chars=1000)
    lines = text.split("\n")
    assert len(text) - len("\n</neospike_recall>") == 1000
    assert lines[2].endswith("short memory") and lines[3].startswith("- [note:7 #근거 기억] yyy")
    assert lines[3].endswith("y…") and lines[4] == "</neospike_recall>"
    full = NeospikeProvider._render_recall([short, note], max_chars=1000 + 5000)
    assert full.split("\n")[3] == "- [note:7 #근거 기억] " + "y" * 5000  # fits: unchanged
    tiny = NeospikeProvider._render_recall([short, note], max_chars=len(text.split("\n- [note")[0]) + 40)
    assert "note:7" not in tiny  # less than the minimum body text left: dropped as before


def test_correction_and_supersede_mark_topic_dirty(tmp_path):
    store = _store(tmp_path)
    topic_id, ids = _topic_from(store, ["Use OTTER mark", "Mark lives in config", "Mark is checked by CI"], _axis(0))
    _clean(store, topic_id)
    row = store.fetch_item_for_update(NS, ids[1], project="", session_id="s")
    assert store.update_memory_item(NS, ids[1], expected_text=row["text"], expected_row=row,
                                    item={"text": "Mark lives in settings"}, embedding=_axis(0),
                                    project="", session_id="s") == "updated"
    assert _dirty(store, topic_id)
    store.close()


# 7. no real LLM ------------------------------------------------------------------------------
def test_topics_module_calls_no_network():
    """Topics are never handed to an LLM; the module has no network or process call."""
    source = Path(topics.__file__).read_text(encoding="utf-8-sig")
    assert not re.search(r"anthropic|openai|urllib|requests|httpx|http\.client|subprocess", source)


# 6. prefetch de-duplication -----------------------------------------------------------
KEYWORDS = ("alpha", "beta", "gamma", "delta", "epsilon")


class AxisEmbedder:
    dimensions = EMBEDDING_DIMENSIONS

    def embed_document(self, text: str, **_kwargs) -> list[float]:
        axes = [float(word in text) for word in KEYWORDS]
        if not any(axes):
            axes = [0.0] * len(KEYWORDS) + [1.0]
        norm = math.sqrt(sum(axes))  # a unit vector (axes are 0/1)
        return [value / norm for value in axes] + [0.0] * (self.dimensions - len(axes))

    embed_query = embed_document

    def take_warnings(self) -> list[str]:
        return []


class RecallCapture:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def __call__(self, query: str, session_id: str, results: list[dict], *_dropped_and_profile) -> None:
        # _log_recall also takes the dropped items and the profile counts; results = the lines shown.
        self.results = list(results)


def test_topic_page_never_in_search_or_recall(tmp_path, monkeypatch):
    """A topic has no content: a TOPIC page (an old page, or titled TOPIC: with the topic tag after its topic
    is gone) never comes up, and its evidence memories keep their slots."""
    home = tmp_path / "home"
    NeospikeConfig().save(home)
    provider = NeospikeProvider(embedder=AxisEmbedder())
    provider.initialize("session", hermes_home=str(home), cwd=str(home))
    try:
        def remember(text: str) -> int:
            reply = json.loads(provider.handle_tool_call("neospike_memory_save", {
                "text": text, "scope": "global", "subject": "item", "durability": "ongoing", "kind": "fact", "basis": "user_said"}))
            assert reply == {"stored": True}
            return int(provider._memory_db._conn.execute("SELECT MAX(id) FROM items WHERE type='memory'").fetchone()[0])

        store = provider._memory_db
        evidence = [remember("alpha first memory item"), remember("alpha second memory item")]
        with store._lock, store._conn:
            topic_id = topics._create_topic(store._conn, topics._member_rows(store._conn, evidence), time.time())
        page = _topic_page(store, topic_id, provider._vectorizer, summary="alpha beta", decisions="alpha beta")
        orphan = NoteService(store, provider._vectorizer).save(f"{topics.TOPIC_TITLE_PREFIX}gone", "alpha beta page",
                                                             tags=["topic"], author="agent")["id"]
        plain = NoteService(store, provider._vectorizer).save("Other note", "beta alpha plain note", author="agent")["id"]
        capture = RecallCapture()
        monkeypatch.setattr(provider, "_log_recall", capture)
        text = provider.prefetch("alpha beta", session_id="session")
        note_ids = {row["id"] for row in capture.results if row.get("kind") == "note"}
        assert note_ids == {plain} and f"note:{page}" not in text and f"note:{orphan}" not in text
        assert set(evidence) <= {row["id"] for row in capture.results if row.get("kind") not in {"note", "conversation"}}
        found = json.loads(provider.handle_tool_call("neospike_memory_search", {"query": "alpha beta", "limit": 20}))
        assert {row["id"] for row in found["results"] if row.get("kind") == "note"} == {plain}
    finally:
        provider.shutdown()


# CLI wiring ------------------------------------------------------------------------------------
class CliEmbedder(AxisEmbedder):
    def __init__(self, *_args, **_kwargs) -> None:
        pass


def _cli(home: Path, monkeypatch, capsys, *arguments: str, stdin: str | None = None, code: int = 0) -> dict:
    import io
    import sys

    from neospike import cli
    monkeypatch.setattr(cli, "EmbeddingClient", CliEmbedder)
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(home), "--owner", NS, *arguments])
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(stdin.encode("utf-8")), encoding="utf-8"))
    assert cli.main() == code
    return json.loads(capsys.readouterr().out)


def test_cli_topic_pipeline(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    NeospikeConfig().save(home)
    store = MemoryStore(home / "neospike" / "memory.db", dimensions=AxisEmbedder.dimensions)
    for text in ("alpha rule one", "alpha rule two", "alpha rule three", "alpha rule four", "alpha rule five",
                 "alpha rule six"):  # more than 5 linked memories make a topic
        store.insert(NS, text, AxisEmbedder().embed_document(text), source="agent:s", kind="memory_item",
                     metadata={"scope": "global", "subject": "alpha rule", "durability": "ongoing", "session_id": "s"})
    store.close()
    discovered = _cli(home, monkeypatch, capsys, "topic-discover")  # created at once
    assert discovered["topic_new"] == 1 and len(discovered["topic_ids"]) == 1
    assigned = _cli(home, monkeypatch, capsys, "topic-assign")
    assert (assigned["link_checked"], assigned["assigned"]) == (0, 0)
