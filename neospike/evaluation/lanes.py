"""Search lanes the evaluation scores and the threshold sweep.

Every lane calls the product search code on a snapshot home:
- prefetch: NeospikeProvider._fused_search (the injected recall, k = RECALL_INJECT_LIMIT), via eval_recall.Evaluator
- memory: mcp_server.search_memory_and_formula_readonly (the memory and automatic-note part lane of MCP mem_search,
  k = DEFAULT_RESULTS)
- note: notes.search_note_parts with formula=False (the note and doc part lane of MCP mem_search, k = DEFAULT_RESULTS)

Threshold None runs the product default untouched. A sweep threshold is applied where the product reads it at call
time: MemoryStore.semantic_threshold (memory and note-chunk admission of prefetch) and the threshold
arguments of the MCP functions. Every grid point is therefore the real path; only the query embedding is computed
once (CachingEmbedder). The design's "collect at the lowest threshold, re-filter per grid point" is not used: the
result rows expose the blended rank score, not the admission cosine (store.py search, SearchResult), and the
superseded-slot swap (store.py _swap_superseded_hits) depends on which rows were admitted, so re-filtering would
reimplement the store. verify_default() still compares the sweep's default point with an untouched run.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Self

from ..eval_recall import EVAL_SESSION_ID, Evaluator, Question
from ..recall_log import item_ref
from ..store import DEFAULT_SEMANTIC_CUTOFF, memory_db_path
from .core import Judged, judge

LANE_PREFETCH, LANE_MEMORY, LANE_NOTE = "prefetch", "memory", "note"
LANES = (LANE_PREFETCH, LANE_MEMORY, LANE_NOTE)
DEFAULT_LANES = (LANE_PREFETCH, LANE_MEMORY, LANE_NOTE)
DEFAULT_KEY = "default"  # report key of the untouched product threshold


class SweepTuning:
    """Threshold grid (bge-m3 cosines sit high; the old 0.30-0.80/0.025 grid was too coarse)."""
    GRID_START = 0.50
    GRID_STOP = 0.75
    GRID_STEP = 0.01
    GRID_DIGITS = 4


def parse_grid(text: str) -> list[float]:
    """'start:stop:step' (inclusive stop) -> ascending thresholds."""
    try:
        start, stop, step = (float(part) for part in text.split(":"))
    except ValueError:
        raise ValueError("grid must look like start:stop:step, e.g. 0.50:0.75:0.01") from None
    if not (0 < start <= stop <= 1) or step <= 0:
        raise ValueError("grid needs 0 < start <= stop <= 1 and step > 0")
    count = round((stop - start) / step) + 1
    return [round(start + index * step, SweepTuning.GRID_DIGITS) for index in range(count)
            if start + index * step <= stop + 1e-9]


def default_grid() -> list[float]:
    return parse_grid(f"{SweepTuning.GRID_START}:{SweepTuning.GRID_STOP}:{SweepTuning.GRID_STEP}")


def parse_lanes(text: str | None) -> tuple[str, ...]:
    if not text:
        return DEFAULT_LANES
    names = tuple(dict.fromkeys(part.strip() for part in text.split(",") if part.strip()))
    unknown = [name for name in names if name not in LANES]
    if unknown or not names:
        raise ValueError(f"unknown lanes {unknown}; choose from {', '.join(LANES)}")
    return names


class CachingEmbedder:
    """Wraps an embedder: each query text is embedded once for the whole run (every lane, every grid point)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.dimensions = inner.dimensions
        self._queries: dict[str, list[float]] = {}

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        vector = self._queries.get(text)
        if vector is None:
            vector = list(self.inner.embed_query(text, **kwargs))
            self._queries[text] = vector
        return vector

    def embed_document(self, text: str, **kwargs: Any) -> list[float]:
        return self.inner.embed_document(text, **kwargs)

    def take_warnings(self) -> list[str]:
        take = getattr(self.inner, "take_warnings", None)
        warnings = take() if callable(take) else []
        return list(warnings) if isinstance(warnings, list) else []


def _memory_ref(item: dict[str, Any]) -> str:
    """Memory-lane hit (MCP mem_search's memory and automatic-note part lane) -> question-set id (an automatic-note
    part counts as its note)."""
    if item.get("kind") == "note_chunk":
        return f"note:{item.get('note_id')}"
    return f"mem:{item.get('id')}"


def _note_ref(item: dict[str, Any]) -> str:
    return f"note:{item.get('id')}"


def _texts(items: Iterable[dict[str, Any]]) -> list[str]:
    return [str(item.get("text", "")) for item in items]


class Session:
    """Lanes bound to one snapshot home. Use as a context manager; the snapshot is opened by the provider
    (prefetch, conv) and through a read-only connection (MCP lanes)."""

    def __init__(self, home: Path, *, embedder: Any, namespace: str, bodies: dict[str, str]) -> None:
        self.home = Path(home)
        self.embedder = embedder if isinstance(embedder, CachingEmbedder) else CachingEmbedder(embedder)
        self.namespace = namespace
        self.bodies = dict(bodies)
        self.warnings: list[str] = []
        self.evaluator = Evaluator(self.home, embedder=self.embedder, namespace=namespace, gold_bodies=bodies)
        path = memory_db_path(self.home)
        self.connection = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()
        self.evaluator.close()

    @property
    def provider(self) -> Any:
        return self.evaluator.provider

    def product_default(self, lane: str) -> float:
        from .. import mcp_server
        if lane == LANE_MEMORY:
            return float(mcp_server.MEMORY_SEARCH_COSINE_THRESHOLD)
        if lane == LANE_NOTE:
            return float(mcp_server.SEMANTIC_THRESHOLD)
        return float(DEFAULT_SEMANTIC_CUTOFF)

    def search(self, lane: str, question: Question, threshold: float | None) -> tuple[list[str], list[str]]:
        """(refs, texts) of one lane, cut to its k, at threshold (None: the product default)."""
        if lane == LANE_PREFETCH:
            return self._with_store_threshold(threshold, _PrefetchCall(self, question))
        from .. import mcp_server
        from ..notes import search_note_parts
        vector = self.embedder.embed_query(question.question)
        limit = mcp_server.DEFAULT_RESULTS
        if lane == LANE_MEMORY:
            extra = {} if threshold is None else {"threshold": threshold}
            hits = mcp_server.search_memory_and_formula_readonly(
                self.connection, self.namespace, vector, dimensions=self.embedder.dimensions, limit=limit,
                project=_project_root(question), session_id=EVAL_SESSION_ID, **extra)
            return [_memory_ref(item) for item in hits], _texts(hits)
        if lane == LANE_NOTE:
            hits = search_note_parts(self.connection, question.question, vector, dimensions=self.embedder.dimensions,
                                      semantic_threshold=self.product_default(lane) if threshold is None else threshold,
                                      limit=limit, tag=None, formula=False)
            return [_note_ref(item) for item in hits], _texts(hits)
        raise ValueError(f"unknown lane {lane}")

    def _with_store_threshold(self, threshold: float | None, call: Any) -> tuple[list[str], list[str]]:
        store = self.provider._memory_db
        saved = store.semantic_threshold
        if threshold is not None:
            store.semantic_threshold = threshold
        try:
            return call()
        finally:
            store.semantic_threshold = saved


def _project_root(question: Question) -> str:
    return str(Path(question.project).expanduser().resolve(strict=False)) if question.project else ""


class _PrefetchCall:
    def __init__(self, session: Session, question: Question) -> None:
        self.session, self.question = session, question

    def __call__(self) -> tuple[list[str], list[str]]:
        results, _block, warnings = self.session.evaluator.results(self.question)
        self.session.warnings.extend(f"{self.question.id}: {warning}" for warning in warnings)
        return [item_ref(item) for item in results], _texts(results)


def threshold_key(threshold: float | None) -> str:
    if threshold is None:
        return DEFAULT_KEY
    return f"{threshold:.{SweepTuning.GRID_DIGITS}f}".rstrip("0").rstrip(".")


def collect(session: Session, questions: Sequence[Question], lanes: Sequence[str],
            thresholds: Sequence[float | None]) -> dict[str, dict[str, list[Judged]]]:
    """{lane: {threshold key: [Judged per question]}}."""
    output: dict[str, dict[str, list[Judged]]] = {}
    for lane in lanes:
        output[lane] = {}
        for threshold in thresholds:
            judged = []
            for question in questions:
                refs, texts = session.search(lane, question, threshold)
                judged.append(judge(question, refs, texts, session.bodies))
            output[lane][threshold_key(threshold)] = judged
    return output


def verify_default(session: Session, questions: Sequence[Question], swept: dict[str, dict[str, list[Judged]]],
                   ) -> dict[str, Any]:
    """Check: per lane, the sweep point equal to the product default must return the same refs as an
    untouched run. Lanes whose default is not on the grid report 'not_on_grid'."""
    report: dict[str, Any] = {}
    for lane, points in swept.items():
        key = threshold_key(session.product_default(lane))
        if key not in points:
            report[lane] = {"status": "not_on_grid", "threshold": key}
            continue
        untouched = collect(session, questions, [lane], [None])[lane][DEFAULT_KEY]
        differing = [now.question.id for now, base in zip(points[key], untouched) if now.refs != base.refs]
        report[lane] = {"status": "ok" if not differing else "mismatch", "threshold": key, "differing": differing}
    return report
