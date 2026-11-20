"""Scoring core of the search evaluation (design EVAL_PROTOCOL_DESIGN.md sections 1, 3, 5).

One scorer for every data source: the real-data snapshot and the synthetic fixture feed the same functions.
Recall@k is hit-any (1 when a gold item is among the k returned results), MRR@k the reciprocal rank of the first
gold item, no-answer FPR the share of no_answer questions with at least one result, irrelevant injections the
mean count of non-gold results on gold questions, stale injections the count of `stale` ids returned on changed
questions. Gold matching is the eval_recall body rule: a gold memory body inside the result text,
else the id.
"""
from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from operator import attrgetter, itemgetter
from typing import Any

from ..eval_recall import GOLD_TYPES, TYPE_CHANGED, TYPE_NO_ANSWER, Question, normalized


class GateTuning:
    """Pre-registered decision values (design sections 1 and 3)."""
    BOOTSTRAP_RESAMPLES = 2000
    BOOTSTRAP_SEED = 443
    CI_LEVEL = 0.95
    TARGET_RECALL = 0.80  # absolute dev target, prefetch Recall@8
    TARGET_FPR = 0.10  # absolute dev target, no-answer FPR
    SELECT_DELTA = 0.05  # candidates keep Recall@k >= best - delta
    LEDGER_WARN_RUNS = 5  # test runs on one sealed set before a new test set is advised


@dataclass(frozen=True)
class Judged:
    """One question on one lane at one threshold."""
    question: Question
    refs: tuple[str, ...]
    hit: int | None = None  # gold questions only
    reciprocal_rank: float | None = None
    irrelevant: int = 0  # gold questions: results matching no gold item
    stale: tuple[str, ...] = ()
    false_positive: bool | None = None  # no_answer questions only

    def as_row(self) -> dict[str, Any]:
        question = self.question
        return {"id": question.id, "split": question.split, "type": question.type, "lang": question.lang,
                "no_answer_kind": question.no_answer_kind, "category": question.category,
                "refs": list(self.refs), "hit": self.hit, "reciprocal_rank": self.reciprocal_rank,
                "irrelevant": self.irrelevant, "stale": list(self.stale), "false_positive": self.false_positive}


class _GoldMatch:
    """gold id -> does this (ref, normalised text) entry hold it: body rule, id rule without a body."""

    def __init__(self, bodies: dict[str, str]) -> None:
        self.bodies = bodies

    def __call__(self, gold: str, ref: str, text: str) -> bool:
        body = self.bodies.get(gold)
        return gold == ref if not body else body in text


def judge(question: Question, refs: Sequence[str], texts: Sequence[str] | None = None,
          bodies: dict[str, str] | None = None) -> Judged:
    """refs/texts: the ranked results already cut to the lane's k. texts None: id rule only."""
    refs = tuple(refs)
    if question.type == TYPE_NO_ANSWER:
        return Judged(question, refs, false_positive=bool(refs))
    match = _GoldMatch((bodies or {}) if texts is not None else {})
    plain = [normalized(text) for text in texts] if texts is not None else [""] * len(refs)
    first, irrelevant = None, 0
    for rank, (ref, text) in enumerate(zip(refs, plain), 1):
        if any(match(gold, ref, text) for gold in question.gold):
            first = rank if first is None else first
        else:
            irrelevant += 1
    stale = tuple(sorted(set(question.stale) & set(refs))) if question.type == TYPE_CHANGED else ()
    return Judged(question, refs, hit=int(first is not None), reciprocal_rank=0.0 if first is None else 1.0 / first,
                  irrelevant=irrelevant, stale=stale)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(items: Iterable[Judged]) -> dict[str, Any]:
    items = list(items)
    gold = [item for item in items if item.question.type in GOLD_TYPES]
    empty = [item for item in items if item.question.type == TYPE_NO_ANSWER]
    return {
        "count": len(items), "gold_count": len(gold), "no_answer_count": len(empty),
        "recall": _mean([item.hit for item in gold]),
        "mrr": _mean([item.reciprocal_rank for item in gold]),
        "fpr": _mean([float(item.false_positive) for item in empty]),
        "irrelevant": _mean([float(item.irrelevant) for item in gold]),
        "stale_injected": sum(len(item.stale) for item in items),
    }


class _GroupKey:
    """Breakdown key of a question: language, type (no_answer split near/far) or category."""

    def __init__(self, axis: str) -> None:
        self.axis = axis

    def __call__(self, question: Question) -> str:
        if self.axis == "lang":
            return question.lang or "unknown"
        if self.axis == "category":
            return question.category or "none"
        return f"{question.type}:{question.no_answer_kind}" if question.no_answer_kind else question.type


BREAKDOWN_AXES = ("lang", "type", "category")


def breakdown(items: Iterable[Judged]) -> dict[str, dict[str, dict[str, Any]]]:
    """{axis: {group: summarize(group)}} for every BREAKDOWN_AXES axis."""
    items = list(items)
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for axis in BREAKDOWN_AXES:
        key = _GroupKey(axis)
        groups: dict[str, list[Judged]] = {}
        for item in items:
            groups.setdefault(key(item.question), []).append(item)
        result[axis] = {name: summarize(group) for name, group in sorted(groups.items())}
    return result


def split_report(items: Iterable[Judged]) -> dict[str, Any]:
    items = list(items)
    return {**summarize(items), "by": breakdown(items)}


# -- paired bootstrap -------------------------------------------------------------------------------------------

def _percentile(sorted_values: Sequence[float], share: float) -> float:
    """Linear-interpolated percentile of an ascending list (share in [0, 1])."""
    position = share * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def paired_bootstrap(current: Sequence[float], baseline: Sequence[float], *,
                     resamples: int = GateTuning.BOOTSTRAP_RESAMPLES, seed: int = GateTuning.BOOTSTRAP_SEED,
                     level: float = GateTuning.CI_LEVEL) -> dict[str, Any]:
    """Percentile CI of mean(current - baseline) over questions resampled with replacement (pairs kept)."""
    if len(current) != len(baseline):
        raise ValueError("paired bootstrap needs equally long, aligned lists")
    if not current:
        return {"n": 0, "diff": None, "low": None, "high": None}
    diffs = [float(now) - float(base) for now, base in zip(current, baseline)]
    count = len(diffs)
    draw = random.Random(seed)
    means = sorted(sum(diffs[draw.randrange(count)] for _ in range(count)) / count for _ in range(resamples))
    tail = (1.0 - level) / 2.0
    return {"n": count, "diff": sum(diffs) / count, "low": _percentile(means, tail),
            "high": _percentile(means, 1.0 - tail)}


def _paired(current: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]], name: str) -> tuple[list, list]:
    ids = sorted(qid for qid in current.keys() & baseline.keys()
                 if current[qid].get(name) is not None and baseline[qid].get(name) is not None)
    return [current[qid][name] for qid in ids], [baseline[qid][name] for qid in ids]


def gate(current: dict[str, dict[str, Any]], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Test regression gate. current/baseline: {question id: as_row()} of one lane and split.
    Regressed when the 95% CI upper bound of the Recall@k difference is below 0, or stale injections grew."""
    recall = paired_bootstrap(*_paired(current, baseline, "hit"))
    rows_now, rows_base = _paired(current, baseline, "false_positive")
    fpr = paired_bootstrap([float(value) for value in rows_now], [float(value) for value in rows_base])
    shared = current.keys() & baseline.keys()
    stale_now = sum(len(current[qid].get("stale") or ()) for qid in shared)
    stale_base = sum(len(baseline[qid].get("stale") or ()) for qid in shared)
    reasons = []
    if recall["high"] is not None and recall["high"] < 0:
        reasons.append("recall_ci_below_zero")
    if stale_now > stale_base:
        reasons.append("stale_injected")
    return {"regressed": bool(reasons), "reasons": reasons, "recall_diff": recall, "fpr_diff": fpr,
            "stale_injected": {"current": stale_now, "baseline": stale_base},
            "unpaired": sorted(current.keys() ^ baseline.keys())}


def targets(summary: dict[str, Any], *, recall: float = GateTuning.TARGET_RECALL,
            fpr: float = GateTuning.TARGET_FPR) -> dict[str, Any]:
    """Absolute dev targets (prefetch Recall@8 >= 0.80, FPR <= 0.10); None when a metric has no questions."""
    recall_ok = None if summary["recall"] is None else summary["recall"] >= recall
    fpr_ok = None if summary["fpr"] is None else summary["fpr"] <= fpr
    return {"recall_target": recall, "fpr_target": fpr, "recall_met": recall_ok, "fpr_met": fpr_ok}


# -- threshold selection ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepRow:
    threshold: float
    recall: float | None
    fpr: float | None
    irrelevant: float | None
    mrr: float | None = None
    pareto: bool = field(default=False, compare=False)


def _orderable(value: float | None) -> float:
    return 0.0 if value is None else value


def _dominates(left: SweepRow, right: SweepRow) -> bool:
    """left at least as good on (recall up, fpr down, irrelevant down) and better on one."""
    a = (_orderable(left.recall), -_orderable(left.fpr), -_orderable(left.irrelevant))
    b = (_orderable(right.recall), -_orderable(right.fpr), -_orderable(right.irrelevant))
    return all(x >= y for x, y in zip(a, b)) and a != b


def mark_pareto(rows: Sequence[SweepRow]) -> list[SweepRow]:
    return [SweepRow(row.threshold, row.recall, row.fpr, row.irrelevant, row.mrr,
                     pareto=not any(_dominates(other, row) for other in rows)) for row in rows]


class _SelectKey:
    """Sort key of the rule: lowest (FPR, irrelevant), then the highest threshold."""

    def __call__(self, row: SweepRow) -> tuple[float, float, float]:
        return _orderable(row.fpr), _orderable(row.irrelevant), -row.threshold


def select_threshold(rows: Sequence[SweepRow], *, delta: float = GateTuning.SELECT_DELTA) -> dict[str, Any]:
    """Keep Recall@k >= best - delta; take the lowest (FPR, irrelevant); ties: higher threshold.
    The pick is a candidate only: the user confirms it."""
    scored = [row for row in rows if row.recall is not None]
    if not scored:
        return {"candidate": None, "best_recall": None, "delta": delta}
    best = max(scored, key=attrgetter("recall")).recall
    kept = [row for row in scored if row.recall >= best - delta - 1e-12]
    pick = min(kept, key=_SelectKey())
    return {"candidate": pick.threshold, "best_recall": best, "delta": delta,
            "kept": sorted(row.threshold for row in kept)}


def sweep_rows(per_threshold: dict[float, dict[str, Any]]) -> list[SweepRow]:
    """{threshold: summarize()} -> rows (ascending threshold) with Pareto flags."""
    rows = [SweepRow(threshold, summary["recall"], summary["fpr"], summary["irrelevant"], summary["mrr"])
            for threshold, summary in sorted(per_threshold.items(), key=itemgetter(0))]
    return mark_pareto(rows)
