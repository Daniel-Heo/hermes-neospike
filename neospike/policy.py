from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any


# One memory item may hold this many characters; longer text is filed as an agent note.
ITEM_TEXT_CEILING = 2000
_SUBJECT_CEILING = 200
_TAG_CEILING = 64
_TAG_COUNT_CEILING = 8
_DURABILITY_CHOICES = ("transient", "ongoing", "stable")
# What an item is and where it came from. Tool argument -> metadata_json key (column `kind` stays memory_item).
KIND_CHOICES = ("fact", "decision", "preference", "constraint")
BASIS_CHOICES = ("user_said", "agent_inferred")
ITEM_LABELS = (("kind", "memory_kind", KIND_CHOICES), ("basis", "basis", BASIS_CHOICES))


class AdmitVerdict(Enum):
    BLOCK = "block"
    SKIP = "skip"
    INDEX = "index"


_CREDENTIAL_RES = tuple(re.compile(p) for p in (
    r"(?i)\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+",
    r"\bsk-[A-Za-z0-9_-]{8,}\b",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"\b(?:ghp|github_pat|xox[baprs]|AKIA)[A-Za-z0-9_-]{12,}\b",
    r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b",
    # HTTP-log headers get pasted; whitespace is collapsed first, so a folded header matches as well.
    r"(?i)\b(?:proxy-)?authorization\s*:\s*(?:basic|bearer|token)\s+\S+",
    r"(?i)\bx-(?:api[-_]?key|auth(?:entication)?)\s*:\s*\S+",
))

_PROMPT_ATTACK_RES = tuple(re.compile(p) for p in (
    r"(?i)\bignore (?:all |any )?(?:previous|prior|system) instructions\b",
    r"(?i)\b(?:reveal|print|exfiltrate|show) (?:the )?(?:system prompt|hidden instructions|credentials|secrets)\b",
    r"(?i)\byou are now (?:in|a|an)\b",
))

_FILTER_RES = _CREDENTIAL_RES + _PROMPT_ATTACK_RES


def _trips_filter(text: str) -> bool:
    flat = " ".join(text.split())
    return any(rx.search(flat) for rx in _FILTER_RES)


def flagged_line_numbers(text: str) -> list[int]:
    """1-based numbers of lines that trip the filter; a pattern folded over two lines flags both
    (screen_text collapses whitespace across the whole text); unlocatable hits flag every line."""
    rows = text.splitlines()
    hits = [n for n, row in enumerate(rows, 1) if _trips_filter(row)]
    if hits or not _trips_filter(text):
        return hits
    for n in range(1, len(rows)):
        if _trips_filter(rows[n - 1] + " " + rows[n]):
            hits += (n, n + 1)
    return sorted(set(hits)) or list(range(1, len(rows) + 1))


def screen_text(text: str) -> AdmitVerdict:
    flat = " ".join(text.split())
    if _trips_filter(flat):
        return AdmitVerdict.BLOCK
    if len(flat) < 12 or len(flat.split()) < 3:
        return AdmitVerdict.SKIP
    return AdmitVerdict.INDEX


def item_schema_fields(*, text_key: str = "text", text_cap: int | None = ITEM_TEXT_CEILING) -> dict[str, Any]:
    """JSON-schema fields of a semantic memory item, shared by remember and explicit correction."""
    text_schema: dict[str, Any] = {"type": "string",
                                   "description": "A self-contained natural-language memory item, not a transcript quote."}
    if text_cap is not None:
        text_schema["maxLength"] = text_cap
    unit = {"type": "number", "minimum": 0, "maximum": 1}
    return {
        text_key: text_schema,
        "subject": {"type": "string", "minLength": 1, "maxLength": _SUBJECT_CEILING},
        "durability": {"type": "string", "enum": list(_DURABILITY_CHOICES)},
        "kind": {"type": "string", "enum": list(KIND_CHOICES), "description": "What the item is."},
        "basis": {"type": "string", "enum": list(BASIS_CHOICES),
                  "description": "user_said only for the user's own words, else agent_inferred."},
        "importance": dict(unit),
        "tags": {"type": "array", "items": {"type": "string", "maxLength": _TAG_CEILING},
                 "maxItems": _TAG_COUNT_CEILING},
    }


def _checked_tags(raw: dict[str, Any]) -> list[str]:
    given = raw.get("tags", [])
    if not isinstance(given, list) or not all(isinstance(t, str) for t in given):
        raise ValueError("Memory tags must be an array of strings")
    tags = [t.strip() for t in given if t.strip()]
    if len(tags) > _TAG_COUNT_CEILING or any(len(t) > _TAG_CEILING for t in tags):
        raise ValueError("Memory tags must contain at most 8 values of at most 64 characters")
    # Labels are short but go through the same secret/injection screen.
    if any(screen_text(t) is AdmitVerdict.BLOCK for t in tags):
        raise ValueError("Memory item was rejected by ingestion policy")
    return tags


def _checked_labels(raw: dict[str, Any], *, labels_required: bool) -> dict[str, str]:
    """kind/basis (ITEM_LABELS) under their metadata keys; an omitted one is left out unless required."""
    labels: dict[str, str] = {}
    for arg, meta_key, choices in ITEM_LABELS:
        if arg not in raw:
            if labels_required:
                raise ValueError(f"Memory {arg} is required: one of {', '.join(choices)}")
            continue
        if raw[arg] not in choices:
            raise ValueError(f"Memory {arg} must be one of {', '.join(choices)}")
        labels[meta_key] = raw[arg]
    return labels


def check_memory_item(raw: dict[str, Any], *, partial_defaults: bool = False,
                      char_cap: int | None = ITEM_TEXT_CEILING, labels_required: bool = False) -> dict[str, Any]:
    """Validate the semantic fields only (ownership never comes from model input).
    partial_defaults: a correction keeps the stored value of every omitted optional field.
    labels_required: the save tool needs kind and basis; other writers may omit them.
    Whitespace is trimmed; negation is never interpreted or removed."""
    if not all(isinstance(raw.get(k), str) for k in ("text", "subject", "durability")):
        raise ValueError("Memory text, subject, and durability must be strings")
    text, subject, durability = raw["text"].strip(), raw["subject"].strip(), raw["durability"]
    if durability not in _DURABILITY_CHOICES:
        raise ValueError("Invalid memory durability")
    if not subject or len(subject) > _SUBJECT_CEILING:
        raise ValueError("Memory subject must contain 1 to 200 characters")
    if char_cap is not None and len(text) > char_cap:
        raise ValueError(f"Memory item exceeds the {char_cap}-character limit")
    if screen_text(text) is not AdmitVerdict.INDEX or screen_text(subject) is AdmitVerdict.BLOCK:
        raise ValueError("Memory item was rejected by ingestion policy")
    checked: dict[str, Any] = {"text": text, "subject": subject, "durability": durability}
    for name, fallback in (("importance", 0.6), ("confidence", 1.0)):
        if partial_defaults and name not in raw:
            continue
        number = raw.get(name, fallback)
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 <= number <= 1 \
                or not math.isfinite(number):
            raise ValueError(f"Memory {name} must be a finite number from 0 to 1")
        checked[name] = float(number)
    if not partial_defaults or "tags" in raw:
        checked["tags"] = _checked_tags(raw)
    checked.update(_checked_labels(raw, labels_required=labels_required))
    return checked
