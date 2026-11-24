"""Masking of secrets and prompt-injection lines in user text (patterns of policy.py and this module).

Never refuses: secrets become [REDACTED], injection lines [BLOCKED LINE]; mask_text returns the count.
"""
from __future__ import annotations

import re

from . import policy

REDACTED = "[REDACTED]"
BLOCKED_LINE = "[BLOCKED LINE]"
_PRIVATE_KEY_MARKER = "PRIVATE KEY"
# Extra secret patterns beside policy.py's.
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(api[_-]?key|token|password|secret|authorization)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:ghp|github_pat|xox[baprs]|AKIA)[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"-----BEGIN [^-]+ PRIVATE KEY-----.*?-----END [^-]+ PRIVATE KEY-----", re.DOTALL),
)


def _mask_patterns() -> tuple:
    """policy.py + this module's secret patterns, ordered so no pattern strands part of a secret:
    whole private-key blocks first (policy matches only the BEGIN line), then policy's header
    patterns (a folded `Authorization:` / `Bearer x` keeps its token otherwise), then this module's."""
    blocks = tuple(pattern for pattern in _SECRET_PATTERNS if _PRIVATE_KEY_MARKER in pattern.pattern)
    rest = tuple(pattern for pattern in _SECRET_PATTERNS if _PRIVATE_KEY_MARKER not in pattern.pattern)
    return blocks + tuple(policy._CREDENTIAL_RES) + rest


_SECRET_MASKS = _mask_patterns()
_INJECTION_MASKS = tuple(policy._PROMPT_ATTACK_RES)


def _is_injection_line(line: str) -> bool:
    cleaned = " ".join(line.split())
    return any(pattern.search(cleaned) for pattern in _INJECTION_MASKS)


def mask_text(text: str) -> tuple[str, int]:
    """(masked text, number of masks). Injection lines are replaced whole, then secrets."""
    count = 0
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if _is_injection_line(line):
            lines[index] = BLOCKED_LINE
            count += 1
    masked = "\n".join(lines)
    for pattern in _SECRET_MASKS:
        masked, hits = pattern.subn(REDACTED, masked)
        count += hits
    return masked, count
