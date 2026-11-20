"""The real-data adapter path end to end on a synthetic HERMES_HOME (no real data is ever read).

The temporary home holds the three kinds of content the product stores — memories (one valuable line each),
notes (a slightly longer text) and conversations (raw user/assistant turns, conversation_prefetch on) — and
`neospike eval run / sweep / seal / test` run through cli.main on it: snapshot (read-only), every lane
including conv, default output root (redirected to tmp_path), protocol.json and the test ledger.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from neospike import NeospikeProvider, cli
from neospike.evaluation import command, protocol

SESSION = "eval-synthetic"
INDEX_WAIT_SECONDS = 10
SWEEP_GRID = "0.6:0.8:0.1"
ALL_LANES = "prefetch,memory,note,conv"
WAL_SIDE_FILES = ("neospike/memory.db-wal", "neospike/memory.db-shm")
# One axis per topic word; a text without any topic word falls on the last axis.
TOPICS = ("ferry", "compost", "violin", "glacier", "lighthouse", "orchard")

MEMORIES = {
    "ferry": "The island ferry leaves pier 3 at 7:40 every weekday.",
    "lighthouse": "The lighthouse key is kept in the red box by the boathouse door.",
}
NOTES = {
    "compost": ("Compost routine",
                ("Turn the compost pile every second Sunday. Keep it damp like a wrung sponge, add dry leaves "
                 "when it smells sour, and move finished compost to the vegetable beds in early spring.")),
    "orchard": ("Orchard pruning",
                ("Prune the orchard apple trees in late winter while they are dormant. Remove crossing branches "
                 "first, then open the centre so light reaches the fruit; never cut more than a quarter.")),
}
CONVERSATIONS = {
    "violin": ("Which violin strings did we settle on for the student violin?",
               "We settled on synthetic-core violin strings, medium tension, replaced every six months."),
    "glacier": ("How long is the glacier hike we planned?",
                "The glacier hike is 14 km with a guide; we start at the hut at six in the morning."),
}


class TopicEmbedder:
    """Deterministic topic-axis embedder: cosine 1 within one topic, 0 across topics."""

    dimensions = len(TOPICS) + 1

    def embed_query(self, text, **_kwargs):
        folded = str(text).casefold()
        vector = [1.0 if topic in folded else 0.0 for topic in TOPICS]
        vector.append(0.0 if any(vector) else 1.0)
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    def embed_document(self, text, **_kwargs):
        return self.embed_query(text)

    def take_warnings(self):
        return []


class _EmbedderFactory:
    """Stands in for command._real_embedder: never an inference client."""

    def __call__(self, _home):
        return TopicEmbedder()


def _build_home(home: Path) -> dict[str, str]:
    """Fill a fresh home through the provider; returns {topic: question-set id}."""
    folder = home / "neospike"
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps({"conversation_prefetch": True}), encoding="utf-8")
    provider = NeospikeProvider(embedder=TopicEmbedder())
    provider.initialize(SESSION, hermes_home=str(home), agent_identity="default", user_id="owner",
                        agent_context="primary", cwd=str(home))
    refs: dict[str, str] = {}
    try:
        for topic, text in MEMORIES.items():
            saved = json.loads(provider.handle_tool_call("neospike_memory_save", {
                "text": text, "scope": "global", "subject": topic, "durability": "stable", "kind": "fact", "basis": "user_said"}))
            assert saved == {"stored": True}, saved
            [memory_id] = [row[0] for row in provider._memory_db._conn.execute(
                "SELECT id FROM items WHERE type='memory' AND text=?", (text,))]
            refs[topic] = f"mem:{memory_id}"
        for topic, (title, body) in NOTES.items():
            note = provider._core.save_note(title, body, author="human")
            refs[topic] = f"note:{note['id']}"
        for topic, (user, assistant) in CONVERSATIONS.items():
            provider.sync_turn(user, assistant, session_id=f"{SESSION}-{topic}")
        assert provider._talk_indexer.wait_idle(INDEX_WAIT_SECONDS)
        rows = provider._memory_db._conn.execute("SELECT id, title FROM conversations").fetchall()
        for topic in CONVERSATIONS:
            [conversation_id] = [row[0] for row in rows if topic in str(row[1]).casefold()]
            refs[topic] = f"conv:{conversation_id}"
    finally:
        provider.shutdown()
    return refs


def _question(qid, split, text, gold=(), kind="normal"):
    return {"id": qid, "split": split, "type": kind, "question": text, "gold": list(gold), "lang": "en"}


def _write_set(path: Path, refs: dict[str, str]) -> None:
    rows = [
        _question("m1", "dev", "When does the ferry leave?", [refs["ferry"]]),
        _question("m2", "test", "Where is the lighthouse key?", [refs["lighthouse"]]),
        _question("n1", "dev", "How often do I turn the compost?", [refs["compost"]]),
        _question("n2", "test", "When should the orchard be pruned?", [refs["orchard"]]),
        _question("c1", "dev", "Which violin strings did we pick?", [refs["violin"]]),
        _question("c2", "test", "How long is the glacier hike?", [refs["glacier"]]),
        _question("x1", "dev", "What is the wifi password at the office?", kind="no_answer"),
        _question("x2", "test", "Who won the chess match last night?", kind="no_answer"),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8-sig")


def _cli(capsys, home: Path, *arguments) -> tuple[int, dict]:
    code = cli.main(["--home", str(home), "eval", *arguments])
    return code, json.loads(capsys.readouterr().out)


def _tree_digest(root: Path) -> dict[str, str]:
    """{relative path: SHA-256} of every file under root."""
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def synthetic_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))  # the default home is the synthetic one, never the user's
    monkeypatch.setattr(command, "RESULTS_ROOT", tmp_path / "results")  # never the package folder in a test
    monkeypatch.setattr(command, "_real_embedder", _EmbedderFactory())
    refs = _build_home(home)
    question_set = tmp_path / "questions.jsonl"
    _write_set(question_set, refs)
    return home, question_set


def test_real_adapter_runs_run_sweep_seal_test_on_a_synthetic_home(synthetic_home, tmp_path, capsys):
    home, question_set = synthetic_home
    before = _tree_digest(home)
    source = ["--set", str(question_set)]

    code, run = _cli(capsys, home, "run", *source, "--lanes", ALL_LANES)
    assert (code, run["source"]) == (0, "real")
    assert not [note for note in run["warnings"] if "conv lane skipped" in note]
    for split in ("dev", "test"):
        prefetch = run["lanes"]["prefetch"]["points"]["default"][split]
        assert (prefetch["recall"], prefetch["fpr"]) == (1.0, 0.0), (split, prefetch)
    for lane, question_id in (("memory", "m1"), ("note", "n1"), ("conv", "c1")):
        dev = run["lanes"][lane]["points"]["default"]["dev"]
        assert dev["fpr"] == 0.0 and dev["recall"] > 0, (lane, question_id, dev)
    run_directory = Path(run["out"])
    assert run_directory.parent == tmp_path / "results" / command.RUNS_NAME

    code, sweep = _cli(capsys, home, "sweep", *source, "--lanes", ALL_LANES, "--grid", SWEEP_GRID)
    assert code == 0 and sweep["verification"]
    assert Path(sweep["out"]).parent == run_directory.parent

    code, seal = _cli(capsys, home, "seal", *source, "--lanes", ALL_LANES, "--grid", SWEEP_GRID)
    sealed = tmp_path / "results" / command.PROTOCOL_NAME
    assert code == 0 and seal["splits"] == {"dev": 4, "test": 4} and sealed.is_file()

    code, tested = _cli(capsys, home, "test", *source)
    assert code == 0 and tested["ledger"]["runs"] == 1 and tested["changed_since_seal"] == []
    assert len(protocol.read_ledger(protocol.ledger_path(sealed))) == 1

    # The snapshot path never changes a file of the evaluated home; SQLite readers may only add the empty WAL
    # side files of a WAL database (memory.db-wal stays 0 bytes).
    after = _tree_digest(home)
    assert {name: after[name] for name in before} == before
    added = set(after) - set(before)
    assert added <= set(WAL_SIDE_FILES), added
    wal = home / WAL_SIDE_FILES[0]
    assert not wal.exists() or wal.stat().st_size == 0
