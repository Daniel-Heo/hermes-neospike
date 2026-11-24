from __future__ import annotations

import re
from typing import Callable

CountFn = Callable[[str], int]
CutFn = Callable[[str, int], list[str]]
Piece = tuple[str, str]  # (separator placed before the text, text)


_ATX_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_CODE_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
_EMPTY_LINE_RUN_RE = re.compile(r"\n[ \t]*\n")
# Sentence end: . ? ! (Korean '다.' included) or CJK stops, followed by whitespace.
_SENTENCE_GAP = re.compile(r"(?<=[.?!。！？])\s+")
# Split levels, coarse to fine; the joiner placed before a piece cut at each level.
_BY_PARAGRAPH, _BY_LINE, _BY_SENTENCE, _BY_TOKEN = range(4)
_LEVEL_JOINERS = ("\n\n", "\n", " ", "")


def _emit_section(sections: list[tuple[str, str]], trail: str, lines: list[str]) -> None:
    body = "\n".join(lines).strip()
    if body:
        sections.append((trail, body))


def split_by_headings(text: str) -> list[tuple[str, str]]:
    """(heading_trail, section text) per `#` heading; fenced code is never a heading.

    heading_trail joins the open heading stack with '>' (e.g. 'Title>Sub').
    Text before the first heading gets an empty trail.
    """
    sections: list[tuple[str, str]] = []
    open_headings: list[tuple[int, str]] = []
    trail, lines, fenced = "", [], False
    for line in text.splitlines():
        if _CODE_FENCE_RE.match(line):
            fenced = not fenced
        found = None if fenced else _ATX_HEADING_RE.match(line)
        if found is None:
            lines.append(line)
            continue
        _emit_section(sections, trail, lines)
        depth = len(found.group(1))
        while open_headings and open_headings[-1][0] >= depth:
            open_headings.pop()
        open_headings.append((depth, found.group(2).strip()))
        trail = ">".join(name for _, name in open_headings)
        lines = [line]
    _emit_section(sections, trail, lines)
    return sections


def count_headings(text: str) -> int:
    total, fenced = 0, False
    for line in text.splitlines():
        if _CODE_FENCE_RE.match(line):
            fenced = not fenced
        elif not fenced and _ATX_HEADING_RE.match(line):
            total += 1
    return total


class _Budget:
    def __init__(self, count: CountFn, limit: int, cutter: CutFn | None) -> None:
        self.count = count
        self.limit = limit
        self.cutter = cutter

    def fits(self, text: str) -> bool:
        return self.count(text) <= self.limit

    def cut(self, text: str) -> list[str]:
        """Split one over-long sentence into fitting pieces at token boundaries."""
        if self.cutter is not None:
            pieces = [piece for piece in self.cutter(text, self.limit) if piece]
            if pieces and all(self.fits(piece) for piece in pieces):
                return pieces
        # Fallback without a tokenizer: longest fitting character prefix (binary search).
        pieces, begin = [], 0
        while begin < len(text):
            low, high = begin + 1, len(text)
            while low < high:
                mid = (low + high + 1) // 2
                if self.fits(text[begin:mid]):
                    low = mid
                else:
                    high = mid - 1
            pieces.append(text[begin:low])
            begin = low
        return pieces


def _glue(pieces: list[Piece]) -> str:
    """The first piece's separator is dropped."""
    return "".join(text if i == 0 else joiner + text for i, (joiner, text) in enumerate(pieces))


def sentence_list(text: str) -> list[str]:
    """Sentences ending in . ? ! (incl. Korean '다.') or CJK full stops; whitespace-separated."""
    return [part.strip() for part in _SENTENCE_GAP.split(text) if part.strip()]


def _split_at(text: str, level: int) -> list[str]:
    if level == _BY_PARAGRAPH:
        parts = _EMPTY_LINE_RUN_RE.split(text)
    elif level == _BY_LINE:
        parts = text.split("\n")
    else:
        return sentence_list(text)
    return [part.strip() for part in parts if part.strip()]


def _atomize_text(text: str, budget: _Budget, level: int, joiner: str, out: list[Piece]) -> None:
    """Split at the coarsest boundary (paragraph, line, sentence) whose pieces fit;
    only a single sentence over the budget is cut at token boundaries."""
    if budget.fits(text):
        out.append((joiner, text))
        return
    if level >= _BY_TOKEN:
        for i, piece in enumerate(budget.cut(text)):
            out.append((joiner if i == 0 else _LEVEL_JOINERS[_BY_TOKEN], piece))
        return
    parts = _split_at(text, level)
    if len(parts) <= 1:
        _atomize_text(text, budget, level + 1, joiner, out)
        return
    for i, part in enumerate(parts):
        _atomize_text(part, budget, level + 1, joiner if i == 0 else _LEVEL_JOINERS[level], out)


def _trailing_overlap(pieces: list[Piece], budget: _Budget, overlap_tokens: int) -> list[Piece]:
    """Trailing whole pieces (sentences or larger) within overlap_tokens; never a partial sentence."""
    tail: list[Piece] = []
    if overlap_tokens <= 0:
        return tail
    for piece in reversed(pieces):
        grown = [piece, *tail]
        if budget.count(_glue(grown)) > overlap_tokens:
            break
        tail = grown
    return tail


def _pack_atoms(body: str, budget: _Budget, overlap_tokens: int) -> list[str]:
    """Pieces of one over-budget text: paragraph/line/sentence atoms packed greedily, each within
    the budget; consecutive pieces repeat up to overlap_tokens of whole trailing atoms."""
    atoms: list[Piece] = []
    _atomize_text(body, budget, _BY_PARAGRAPH, "", atoms)
    packed: list[str] = []
    current: list[Piece] = []
    for atom in atoms:
        if not current or budget.fits(_glue([*current, atom])):
            current.append(atom)
            continue
        packed.append(_glue(current))
        carry = _trailing_overlap(current, budget, overlap_tokens)
        while carry and not budget.fits(_glue([*carry, atom])):
            carry = carry[1:]
        current = [*carry, atom]
    if current:
        packed.append(_glue(current))
    return packed


def pack_markdown_chunks(
    text: str, count_tokens: CountFn, *, max_tokens: int, overlap_tokens: int = 64,
    cut_tokens: CutFn | None = None,
) -> list[tuple[str, str]]:
    """(heading_trail, text) chunks per heading section, each within max_tokens.

    A section over max_tokens splits at paragraph, line, then sentence
    boundaries; only one sentence over max_tokens is cut at token boundaries
    (cut_tokens, else a character binary search). Consecutive chunks of one
    section repeat up to overlap_tokens of whole trailing sentences.
    count_tokens must count what is embedded (prefix included).
    """
    if max_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Expected max_tokens > overlap_tokens >= 0")
    budget = _Budget(count_tokens, max_tokens, cut_tokens)
    chunks: list[tuple[str, str]] = []
    for trail, body in split_by_headings(text):
        if budget.fits(body):
            chunks.append((trail, body))
        else:
            chunks.extend((trail, part) for part in _pack_atoms(body, budget, overlap_tokens))
    return chunks


# --- exact parts of a document (v22 items rows) -------------------------------------------------------------
# A document is stored as consecutive parts whose concatenation is the body byte for byte (no overlap): each
# heading section starts a new part; a section over the budget splits at paragraph, line, then sentence ends,
# and a single sentence over the budget at token boundaries. Whitespace stays with the part before it.
_PARAGRAPH_END_RE = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")
_SENTENCE_END_RE = re.compile(r"(?<=[.?!。！？])\s+")


def _heading_starts(text: str) -> list[tuple[int, str]]:
    """(offset, heading trail) of every heading section start; offset 0 always (empty trail before a heading)."""
    starts: list[tuple[int, str]] = [(0, "")]
    open_headings: list[tuple[int, str]] = []
    fenced = False
    offset = 0
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if _CODE_FENCE_RE.match(bare):
            fenced = not fenced
        found = None if fenced else _ATX_HEADING_RE.match(bare)
        if found is not None:
            depth = len(found.group(1))
            while open_headings and open_headings[-1][0] >= depth:
                open_headings.pop()
            open_headings.append((depth, found.group(2).strip()))
            trail = ">".join(name for _, name in open_headings)
            if offset == 0:
                starts[0] = (0, trail)
            else:
                starts.append((offset, trail))
        offset += len(line)
    return starts


def _boundaries(text: str, start: int, end: int, level: int) -> list[int]:
    """Cut offsets strictly inside (start, end) at the given level (after the separator)."""
    piece = text[start:end]
    if level == _BY_PARAGRAPH:
        cuts = [start + found.end() for found in _PARAGRAPH_END_RE.finditer(piece)]
    elif level == _BY_LINE:
        cuts = [start + index + 1 for index, char in enumerate(piece) if char == "\n"]
    else:
        cuts = [start + found.end() for found in _SENTENCE_END_RE.finditer(piece)]
    return [cut for cut in cuts if start < cut < end]


def _exact_cut(budget: _Budget, text: str) -> list[str]:
    """Contiguous fitting pieces of one over-long sentence whose concatenation is the text."""
    pieces = budget.cut(text)
    if "".join(pieces) == text:
        return pieces
    budget_without_cutter = _Budget(budget.count, budget.limit, None)
    return budget_without_cutter.cut(text)


def _exact_atoms(text: str, start: int, end: int, budget: _Budget, level: int, out: list[tuple[int, int]]) -> None:
    if budget.fits(text[start:end]):
        out.append((start, end))
        return
    if level >= _BY_TOKEN:
        position = start
        for piece in _exact_cut(budget, text[start:end]):
            out.append((position, position + len(piece)))
            position += len(piece)
        return
    cuts = _boundaries(text, start, end, level)
    if not cuts:
        _exact_atoms(text, start, end, budget, level + 1, out)
        return
    edges = [start, *cuts, end]
    for low, high in zip(edges, edges[1:]):
        _exact_atoms(text, low, high, budget, level + 1, out)


def split_document_parts(text: str, count_tokens: CountFn, *, max_tokens: int,
                         cut_tokens: CutFn | None = None) -> list[tuple[str, str]]:
    """(heading_trail, part) of a document body, each part within max_tokens (whitespace-only parts are merged
    into the part before them), and "".join(parts) == text. An empty body has no parts."""
    if max_tokens < 1:
        raise ValueError("Expected max_tokens >= 1")
    budget = _Budget(count_tokens, max_tokens, cut_tokens)
    starts = _heading_starts(text)
    spans: list[tuple[str, int, int]] = []
    for index, (start, trail) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
        if start >= end:
            continue
        atoms: list[tuple[int, int]] = []
        _exact_atoms(text, start, end, budget, _BY_PARAGRAPH, atoms)
        low, high = atoms[0]
        for atom_start, atom_end in atoms[1:]:
            if budget.fits(text[low:atom_end]):
                high = atom_end
                continue
            spans.append((trail, low, high))
            low, high = atom_start, atom_end
        spans.append((trail, low, high))
    # A whitespace-only span joins the part before it (only its trailing whitespace can pass the budget, which
    # the embedder may cut without losing text); at the start of the body it joins the first part when that
    # still fits, else it stays a part of its own.
    parts: list[tuple[str, str]] = []
    lead = ""
    for trail, low, high in spans:
        piece = text[low:high]
        if not piece.strip():
            if parts:
                parts[-1] = (parts[-1][0], parts[-1][1] + piece)
            else:
                lead += piece
            continue
        if lead and budget.fits(lead + piece):
            piece, lead = lead + piece, ""
        elif lead:
            parts.append(("", lead))
            lead = ""
        parts.append((trail, piece))
    if lead:
        parts.append(("", lead))
    return parts


