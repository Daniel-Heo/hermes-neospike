"""Data sources of the evaluation: both yield a Prepared snapshot for lanes.Session.

- RealSource: a live HERMES_HOME copied read-only into a temporary home (eval_recall.snapshot_home); question
  gold ids are real ids (mem:N, note:N); gold bodies come from the snapshot, else an older gold database.
- SyntheticSource: a fixture corpus (fictional content, kept in the repository) written into a temporary home;
  gold and stale entries name corpus keys and are resolved to the ids the build produced. The fixture embedder
  (ConceptEmbedder) maps lexicon concepts in Korean and English to shared axes plus one background axis, so
  cross-language questions match and unrelated texts keep a non-zero cosine like bge-m3. For regression tests
  only: thresholds are never decided on it.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..eval_recall import Question, gold_bodies, parse_question, snapshot_home
from ..service import NeospikeService
from ..store import (
    CREATED_BY_HUMAN,
    DEFAULT_NAMESPACE,
    ORIGIN_MIGRATION,
    MemoryStore,
    memory_db_path,
    pick_namespace,
)

FIXTURE_SOURCE = "eval-fixture"
FIXTURE_AUTHOR = "human"
KIND_MEMORY, KIND_NOTE = "memory", "note"
TEMP_PREFIX = "neospike-eval-"


@dataclass
class Prepared:
    home: Path  # the snapshot home every lane reads
    questions: list[Question]
    bodies: dict[str, str]
    namespace: str
    embedder: Any


class _TemporaryHome:
    """A temporary directory holding <tmp>/home; removed on exit."""

    def __init__(self, work_directory: Path | None) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix=TEMP_PREFIX, dir=work_directory,
                                                      ignore_cleanup_errors=True)
        self.home = Path(self._temporary.name) / "home"

    def cleanup(self) -> None:
        self._temporary.cleanup()


class RealSource:
    """with RealSource(home, questions, embedder=...) as prepared: read-only snapshot of a live home."""

    def __init__(self, home: Path, questions: list[Question], *, embedder: Any, namespace: str | None = None,
                 work_directory: Path | None = None, gold_database: Path | None = None) -> None:
        self.source = Path(home).expanduser()
        self.questions, self.embedder, self.namespace = questions, embedder, namespace
        self.work_directory, self.gold_database = work_directory, gold_database
        self._temporary: _TemporaryHome | None = None

    def __enter__(self) -> Prepared:
        if not memory_db_path(self.source).is_file():
            raise ValueError("No neospike/memory.db at the selected home")
        self._temporary = _TemporaryHome(self.work_directory)
        snapshot = self._temporary.home
        snapshot_home(self.source, snapshot)
        namespace = pick_namespace(self.namespace, memory_db_path(snapshot))
        sources = [memory_db_path(snapshot), *([Path(self.gold_database)] if self.gold_database else [])]
        return Prepared(snapshot, self.questions, gold_bodies(self.questions, sources), namespace, self.embedder)

    def __exit__(self, *_exc: object) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()


# -- synthetic fixture -------------------------------------------------------------------------------------------

class ConceptEmbedder:
    """Deterministic fixture embedder. lexicon.json: {"background": b, "concepts": {name: [terms...]}}.
    A text gets 1 on each concept axis one of whose terms (casefolded) it contains, and b on a background axis;
    the vector is L2-normalised. Two texts sharing no concept keep cosine b^2 / (|u| |v|)."""

    def __init__(self, lexicon: dict[str, Any]) -> None:
        concepts = lexicon.get("concepts")
        background = lexicon.get("background")
        if not isinstance(concepts, dict) or not concepts or not isinstance(background, (int, float)):
            raise ValueError("lexicon needs a 'concepts' object and a numeric 'background'")
        self.concepts = [(name, tuple(term.casefold() for term in terms)) for name, terms in concepts.items()]
        self.background = float(background)
        self.dimensions = len(self.concepts) + 1

    @classmethod
    def load(cls, path: Path) -> ConceptEmbedder:
        return cls(json.loads(Path(path).read_text(encoding="utf-8-sig")))

    def embed_query(self, text: str, **_kwargs: Any) -> list[float]:
        folded = str(text).casefold()
        vector = [1.0 if any(term in folded for term in terms) else 0.0 for _name, terms in self.concepts]
        vector.append(self.background)
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    def embed_document(self, text: str, **kwargs: Any) -> list[float]:
        return self.embed_query(text, **kwargs)

    def take_warnings(self) -> list[str]:
        return []


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path).name} line {number}: invalid JSON ({exc.msg})") from None
            if not isinstance(row, dict):  # ValueError, not TypeError: callers catch ValueError
                raise ValueError(f"{Path(path).name} line {number}: expected a JSON object")  # noqa: TRY004
            rows.append(row)
    return rows


def _memory_metadata(project: str) -> dict[str, str]:
    return {"scope": "project" if project else "global", "subject": "eval", "durability": "stable", "session_id": ""}


def _link_supersede(store: MemoryStore, next_id: int, prev_id: int) -> None:
    """supersedes(next -> prev) as one write transaction (the store's link step; the product writes it
    inside replace_item_with_new, which needs a live edit session)."""
    with store._lock:
        try:
            store._begin_write()
            verdict = store._insert_supersede_link(next_id, prev_id, created_by=CREATED_BY_HUMAN,
                                                   origin=ORIGIN_MIGRATION, session_id="", evidence="",
                                                   now=time.time())
            if not verdict["linked"]:
                raise ValueError(f"fixture supersede mem:{next_id} -> mem:{prev_id} refused: {verdict['reason']}")
            store._conn.commit()
        except BaseException:
            store._abort()
            raise


def build_corpus(corpus: list[dict[str, Any]], home: Path, embedder: Any,
                 namespace: str = DEFAULT_NAMESPACE) -> dict[str, str]:
    """Write the corpus into home/neospike/memory.db; returns {corpus key: 'mem:N' | 'note:N'}."""
    store = MemoryStore(memory_db_path(home), dimensions=embedder.dimensions)
    refs: dict[str, str] = {}
    try:
        service = NeospikeService(store=store, embedder=embedder, namespace=namespace)
        supersedes: list[tuple[str, str]] = []
        for row in corpus:
            key, kind = row.get("key"), row.get("kind")
            if not isinstance(key, str) or not key or key in refs:
                raise ValueError(f"corpus key {key!r} must be a unique non-empty string")
            if kind == KIND_MEMORY:
                text, project = str(row["text"]), str(row.get("project", ""))
                memory_id = store.insert(namespace, text, embedder.embed_document(text), source=FIXTURE_SOURCE,
                                         kind="memory_item", importance=float(row.get("importance", 0.5)),
                                         project=project, metadata=_memory_metadata(project))
                if memory_id is None:
                    raise ValueError(f"corpus key {key}: duplicate memory text")
                refs[key] = f"mem:{memory_id}"
                if row.get("supersedes"):
                    supersedes.append((key, str(row["supersedes"])))
            elif kind == KIND_NOTE:
                note = service.save_note(str(row["title"]), str(row["body"]), author=FIXTURE_AUTHOR)
                refs[key] = f"note:{note['id']}"
            else:
                raise ValueError(f"corpus key {key}: kind must be {KIND_MEMORY} or {KIND_NOTE}")
        for new_key, old_key in supersedes:
            if old_key not in refs:
                raise ValueError(f"corpus key {new_key}: supersedes unknown key {old_key}")
            _link_supersede(store, int(refs[new_key].split(":")[1]), int(refs[old_key].split(":")[1]))
    finally:
        store.close()
    return refs


def resolve_questions(rows: list[dict[str, Any]], refs: dict[str, str]) -> list[Question]:
    """Fixture question rows (gold/stale name corpus keys) -> Questions with the built ids."""
    questions, seen = [], set()
    for number, row in enumerate(rows, 1):
        resolved = dict(row)
        for name in ("gold", "stale"):
            keys = row.get(name, [])
            missing = [key for key in keys if key not in refs] if isinstance(keys, list) else []
            if missing:
                raise ValueError(f"question line {number}: unknown corpus keys {missing}")
            if isinstance(keys, list) and name in row:
                resolved[name] = [refs[key] for key in keys]
        question = parse_question(resolved, number)
        if question.id in seen:
            raise ValueError(f"question line {number}: duplicate question id {question.id}")
        seen.add(question.id)
        questions.append(question)
    return questions


class SyntheticSource:
    """with SyntheticSource(corpus, questions, lexicon) as prepared: a temporary home built from the fixture."""

    def __init__(self, corpus_path: Path, questions_path: Path, lexicon_path: Path, *,
                 work_directory: Path | None = None) -> None:
        self.corpus_path, self.questions_path, self.lexicon_path = corpus_path, questions_path, lexicon_path
        self.work_directory = work_directory
        self._temporary: _TemporaryHome | None = None
        self.refs: dict[str, str] = {}

    def __enter__(self) -> Prepared:
        embedder = ConceptEmbedder.load(self.lexicon_path)
        self._temporary = _TemporaryHome(self.work_directory)
        home = self._temporary.home
        self.refs = build_corpus(read_jsonl(self.corpus_path), home, embedder)
        questions = resolve_questions(read_jsonl(self.questions_path), self.refs)
        return Prepared(home, questions, gold_bodies(questions, [memory_db_path(home)]), DEFAULT_NAMESPACE, embedder)

    def __exit__(self, *_exc: object) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()
