"""Recall question-set scorer.

Runs every question through the prefetch retrieval path (NeospikeProvider._fused_search,
RECALL_INJECT_LIMIT, the same thresholds). The live databases are only read: they are copied
with SQLite's backup API from read-only connections into a temporary home, and every
search (including access bookkeeping) runs against that copy.

Hit rule: a gold memory is hit when its body (whitespace normalised) is contained in a result's
text, so a memory that became an automatic-note chunk still counts. The id rule is reported beside it
(id_*), and the injected block (the prefetch text, _render_recall) is scored with the body rule too
(block_*). A gold body comes from the evaluated home, else from gold_database (an older copy of the DB,
read-only); a gold id without a body is judged by id only.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .recall_log import item_ref
from .store import memory_db_path, pick_namespace

SPLITS = ("dev", "test")
TYPE_NORMAL = "normal"
TYPE_CHANGED = "changed"
TYPE_NO_ANSWER = "no_answer"
TYPES = (TYPE_NORMAL, TYPE_CHANGED, TYPE_NO_ANSWER)
GOLD_TYPES = (TYPE_NORMAL, TYPE_CHANGED)
# Optional fields (sets without them stay valid): lang, no_answer_kind (no_answer only), category.
LANGUAGES = ("ko", "en", "mixed")
NO_ANSWER_KINDS = ("near", "far")
CATEGORY_CROSS_LANGUAGE = "cross_language"
QUESTION_FIELDS = {"id", "split", "type", "question", "gold", "stale", "project", "lang", "no_answer_kind",
                   "category"}
REQUIRED_FIELDS = {"id", "split", "type", "question", "gold"}
# Read-only snapshot of these neospike files.
SNAPSHOT_DATABASES = ("memory.db",)
EVAL_SESSION_ID = ""  # no session authority: session-scoped items stay hidden, as in a fresh session
EVAL_AGENT_CONTEXT = "eval"  # not "primary": provider writes are disabled
COMPARE_EPSILON = 1e-9
BASELINE_SPLIT = "test"


@dataclass(frozen=True)
class Question:
    id: str
    split: str
    type: str
    question: str
    gold: tuple[str, ...]
    stale: tuple[str, ...] = ()
    project: str = ""
    lang: str = ""
    no_answer_kind: str = ""
    category: str = ""


@dataclass
class QuestionScore:
    question: Question
    refs: list[str]
    recall: float | None = None
    reciprocal_rank: float | None = None
    stale_hits: list[str] = field(default_factory=list)
    false_injection: int = 0
    id_recall: float | None = None
    id_reciprocal_rank: float | None = None
    block_recall: float | None = None
    false_block_lines: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.question.split, "type": self.question.type, "results": self.refs,
            "recall_at_8": self.recall, "reciprocal_rank": self.reciprocal_rank,
            "id_recall_at_8": self.id_recall, "id_reciprocal_rank": self.id_reciprocal_rank,
            "block_recall": self.block_recall, "stale_hits": self.stale_hits,
            "false_injection": self.false_injection, "false_block_lines": self.false_block_lines,
        }


def normalized(text: Any) -> str:
    """Whitespace-normalised text (the body rule)."""
    return " ".join(str(text).split())


def _first_and_found(question: Question, entries: list[Any], hit: Any) -> tuple[float, float]:
    """(recall, reciprocal rank) of the gold ids over ranked entries; hit(gold, entry) -> bool."""
    found, first = set(), None
    for rank, entry in enumerate(entries, 1):
        hits = {gold for gold in question.gold if hit(gold, entry)}
        if hits and first is None:
            first = rank
        found |= hits
    return len(found) / len(question.gold), 0.0 if first is None else 1.0 / first


class _IdHit:
    def __call__(self, gold: str, ref: str) -> bool:
        return gold == ref


class _BodyHit:
    """Body rule: the gold body inside the entry text; a gold id without a body falls back to the id rule."""

    def __init__(self, bodies: dict[str, str]) -> None:
        self.bodies = bodies

    def __call__(self, gold: str, entry: tuple[str, str]) -> bool:
        ref, text = entry
        body = self.bodies.get(gold)
        return gold == ref if not body else body in text


def _id_list(value: Any, name: str, line: int) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"line {line}: {name} must be a list of id strings")
    return tuple(value)


def parse_question(raw: Any, line: int) -> Question:
    if not isinstance(raw, dict):
        raise ValueError(f"line {line}: expected a JSON object")
    unknown = raw.keys() - QUESTION_FIELDS
    missing = REQUIRED_FIELDS - raw.keys()
    if unknown or missing:
        raise ValueError(f"line {line}: unknown fields {sorted(unknown)}, missing fields {sorted(missing)}")
    if not isinstance(raw["id"], str) or not raw["id"]:
        raise ValueError(f"line {line}: id must be a non-empty string")
    if raw["split"] not in SPLITS:
        raise ValueError(f"line {line}: split must be one of {SPLITS}")
    if raw["type"] not in TYPES:
        raise ValueError(f"line {line}: type must be one of {TYPES}")
    if not isinstance(raw["question"], str) or not raw["question"].strip():
        raise ValueError(f"line {line}: question must be a non-empty string")
    gold = _id_list(raw["gold"], "gold", line)
    stale = _id_list(raw.get("stale", []), "stale", line)
    if raw["type"] in GOLD_TYPES and not gold:
        raise ValueError(f"line {line}: {raw['type']} questions need at least one gold id")
    if raw["type"] == TYPE_NO_ANSWER and gold:
        raise ValueError(f"line {line}: no_answer questions must have an empty gold list")
    if stale and raw["type"] != TYPE_CHANGED:
        raise ValueError(f"line {line}: stale is only for changed questions")
    project = raw.get("project") or ""
    if not isinstance(project, str):
        raise ValueError(f"line {line}: project must be a string")
    lang, kind, category = (raw.get(name) or "" for name in ("lang", "no_answer_kind", "category"))
    if lang and lang not in LANGUAGES:
        raise ValueError(f"line {line}: lang must be one of {LANGUAGES}")
    if kind and (kind not in NO_ANSWER_KINDS or raw["type"] != TYPE_NO_ANSWER):
        raise ValueError(f"line {line}: no_answer_kind must be one of {NO_ANSWER_KINDS}, on no_answer questions only")
    if not isinstance(category, str):
        raise ValueError(f"line {line}: category must be a string")
    return Question(raw["id"], raw["split"], raw["type"], raw["question"], gold, stale, project, lang, kind, category)


def load_question_set(path: Path) -> list[Question]:
    questions: list[Question] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, text in enumerate(handle, 1):
            if not text.strip():
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {number}: invalid JSON ({exc.msg})") from None
            question = parse_question(raw, number)
            if question.id in seen:
                raise ValueError(f"line {number}: duplicate question id {question.id}")
            seen.add(question.id)
            questions.append(question)
    return questions


def score_question(question: Question, refs: list[str], texts: list[str] | None = None,
                   gold_bodies: dict[str, str] | None = None, block: str | None = None) -> QuestionScore:
    """refs/texts: the ranked results (item_ref, text). recall/reciprocal_rank use the body rule (texts given;
    else the id rule), id_* the id rule, block_recall the body rule over the injected block. Stale hits
    count result ids."""
    score = QuestionScore(question, refs)
    lines = 0 if block is None else max(0, block.count(_BLOCK_LINE))
    if question.type == TYPE_NO_ANSWER:
        score.false_injection, score.false_block_lines = len(refs), lines
        return score
    score.id_recall, score.id_reciprocal_rank = _first_and_found(question, refs, _IdHit())
    if texts is None:
        score.recall, score.reciprocal_rank = score.id_recall, score.id_reciprocal_rank
    else:
        body_hit = _BodyHit(gold_bodies or {})
        score.recall, score.reciprocal_rank = _first_and_found(
            question, list(zip(refs, (normalized(text) for text in texts))), body_hit)
        if block is not None:
            score.block_recall, _ = _first_and_found(question, [("", normalized(block))], body_hit)
    if question.type == TYPE_CHANGED:
        stale = set(question.stale)
        score.stale_hits = sorted(stale & set(refs))
    return score


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(scores: Iterable[QuestionScore]) -> dict[str, Any]:
    scores = list(scores)
    with_gold = [item for item in scores if item.question.type in GOLD_TYPES]
    blocks = [item.block_recall for item in with_gold if item.block_recall is not None]
    return {
        "count": len(scores),
        "recall_at_8": _mean([item.recall for item in with_gold]),
        "mrr": _mean([item.reciprocal_rank for item in with_gold]),
        "id_recall_at_8": _mean([item.id_recall for item in with_gold]),
        "id_mrr": _mean([item.id_reciprocal_rank for item in with_gold]),
        "block_recall": _mean(blocks),
        "stale_injected": sum(len(item.stale_hits) for item in scores),
        "false_injection": sum(item.false_injection for item in scores),
        "false_block_lines": sum(item.false_block_lines for item in scores),
        "per_question": {item.question.id: item.as_dict() for item in scores},
    }


def compare_to_baseline(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Gate on the test split: any drop in recall@8 or rise in injections regresses."""
    try:
        now, base = current["splits"][BASELINE_SPLIT], baseline["splits"][BASELINE_SPLIT]
    except (KeyError, TypeError):
        raise ValueError("baseline comparison needs the test split in both results") from None
    reasons = []
    if now["recall_at_8"] is not None and base.get("recall_at_8") is not None \
            and now["recall_at_8"] < base["recall_at_8"] - COMPARE_EPSILON:
        reasons.append("recall_at_8")
    for name in ("false_injection", "stale_injected"):
        if now[name] > base.get(name, 0):
            reasons.append(name)
    return {"regressed": bool(reasons), "reasons": reasons,
            "baseline": {name: base.get(name) for name in ("recall_at_8", "mrr", "false_injection", "stale_injected")}}


def snapshot_home(source_home: Path, target_home: Path) -> None:
    """Copy the neospike state needed for search; the source is opened read-only."""
    source, target = Path(source_home) / "neospike", Path(target_home) / "neospike"
    target.mkdir(parents=True, exist_ok=True)
    config_path = source / "config.json"
    if config_path.is_file():
        shutil.copyfile(config_path, target / "config.json")
    for name in SNAPSHOT_DATABASES:
        if (source / name).is_file():
            _copy_database(source / name, target / name)


def _copy_database(source: Path, target: Path) -> None:
    reader = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)
    try:
        writer = sqlite3.connect(target)
        try:
            reader.backup(writer)
        finally:
            writer.close()
    finally:
        reader.close()


def _namespace_parts(namespace: str) -> tuple[str, str]:
    identity, separator, user_id = namespace.partition(":")
    if not separator or not identity or not user_id:
        raise ValueError("namespace must look like '<agent_identity>:<user_id>'")
    return identity, user_id


_BLOCK_LINE = "\n- ["  # one injected line of the prefetch block


def gold_bodies(questions: Iterable[Question], databases: Iterable[Path]) -> dict[str, str]:
    """mem:N gold id -> normalised body, from the first database (opened read-only) that still has it."""
    wanted = {gold for question in questions for gold in question.gold if gold.startswith("mem:")}
    bodies: dict[str, str] = {}
    for database in databases:
        missing = sorted(wanted - bodies.keys())
        if not missing or not Path(database).is_file():
            continue
        reader = sqlite3.connect(f"{Path(database).resolve().as_uri()}?mode=ro", uri=True)
        try:
            for ref in missing:
                row = reader.execute("SELECT text FROM items WHERE id=? AND type='memory'",
                                     (int(ref.split(":", 1)[1]),)).fetchone()
                if row is not None and normalized(row[0]):
                    bodies[ref] = normalized(row[0])
        finally:
            reader.close()
    return bodies


class Evaluator:
    """Owns a provider bound to a snapshot home; call close() (or use as a context manager)."""

    def __init__(self, snapshot: Path, *, embedder: Any,
                 namespace: str | None = None, gold_bodies: dict[str, str] | None = None) -> None:
        from .provider import NeospikeProvider
        identity, user_id = _namespace_parts(pick_namespace(namespace, memory_db_path(snapshot)))
        self.provider = NeospikeProvider(embedder=embedder)
        self.gold_bodies = dict(gold_bodies or {})
        self.provider.initialize(EVAL_SESSION_ID, hermes_home=str(snapshot), cwd=str(snapshot),
                                 agent_identity=identity, user_id=user_id, agent_context=EVAL_AGENT_CONTEXT)

    def __enter__(self) -> "Evaluator":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def close(self) -> None:
        self.provider.shutdown()

    def results(self, question: Question) -> tuple[list[dict[str, Any]], str, list[str]]:
        """(ranked results, the injected block as prefetch formats it, warnings)."""
        from .provider import RECALL_INJECT_LIMIT
        from .store import SearchResult
        provider = self.provider
        root = Path(question.project).expanduser().resolve(strict=False) if question.project else None
        # Same scope fields prefetch's _sync_project_scope sets from a session's project.
        provider._project_dir = str(root) if root else ""
        provider._core.allowed_roots = [root] if root else []
        provider._core.session_id = EVAL_SESSION_ID
        results = provider._fused_search(question.question, limit=RECALL_INJECT_LIMIT)
        block = provider._render_recall([SearchResult(item["id"], item["text"], item["origin"], item["score"],
                                                       item.get("kind", "episodic")) for item in results])
        return results, block, list(provider._search_notes)

    def refs(self, question: Question) -> tuple[list[str], list[str]]:
        results, _, warnings = self.results(question)
        return [item_ref(item) for item in results], warnings

    def run(self, questions: Iterable[Question]) -> tuple[list[QuestionScore], list[str]]:
        scores, warnings = [], []
        for question in questions:
            results, block, found = self.results(question)
            scores.append(score_question(question, [item_ref(item) for item in results],
                                         [str(item.get("text", "")) for item in results], self.gold_bodies, block))
            warnings.extend(f"{question.id}: {warning}" for warning in found)
        return scores, warnings


def evaluate(home: Path, questions: list[Question], *, embedder: Any,
             namespace: str | None = None, split: str | None = None,
             work_directory: Path | None = None, gold_database: Path | None = None) -> dict[str, Any]:
    """gold_database: an older copy of memory.db read (read-only) for gold bodies the home no longer has."""
    selected = [item for item in questions if split is None or item.split == split]
    with tempfile.TemporaryDirectory(prefix="neospike-eval-", dir=work_directory,
                                     ignore_cleanup_errors=True) as temporary:
        snapshot = Path(temporary) / "home"
        snapshot_home(Path(home).expanduser(), snapshot)
        # Default: the live provider namespace of the evaluated home (read from the snapshot copy).
        namespace = pick_namespace(namespace, memory_db_path(snapshot))
        sources = [memory_db_path(snapshot), *([Path(gold_database)] if gold_database else [])]
        bodies = gold_bodies(selected, sources)
        with Evaluator(snapshot, embedder=embedder, namespace=namespace,
                       gold_bodies=bodies) as runner:
            scores, warnings = runner.run(selected)
    splits = {name: summarize(item for item in scores if item.question.split == name)
              for name in SPLITS if any(item.split == name for item in selected)}
    gold_ids = {gold for item in selected for gold in item.gold}
    return {"splits": splits, "warnings": warnings,
            "gold_without_body": sorted(gold_ids - bodies.keys())}
