from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from .notes import NoteService
from .policy import ITEM_TEXT_CEILING, check_memory_item
from .store import AGENT_EDITOR, HISTORY_COLLAPSE, MemoryStore, SearchResult


MODE_CORRECT_INPLACE = "correct"
MODE_SUPERSEDE_NEW = "supersede"
VALID_UPDATE_MODES = (MODE_CORRECT_INPLACE, MODE_SUPERSEDE_NEW)

# Too long a correction is split at its last blank line, else its last sentence end, else at the limit.
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n")
_SENTENCE_BOUNDARY = re.compile(r"[.!?。！？](?=\s)")
_SUBJECT_LIMIT = 200
_ID_CEILING = 2**63 - 1
_EXPECTED_TEXT_CEILING = 2000
_FILE_SIZE_CEILING = 5_000_000
_UPDATABLE_FIELDS = frozenset(("subject", "durability", "importance", "confidence", "tags", "kind", "basis"))


def _last_cut_point(window: str) -> int:
    """Where to cut `window`: last paragraph break (not at 0), else end of the last sentence, else 0."""
    breaks = [m.start() for m in _PARAGRAPH_BREAK.finditer(window) if m.start() > 0]
    if breaks:
        return breaks[-1]
    ends = [m.end() for m in _SENTENCE_BOUNDARY.finditer(window)]
    return ends[-1] if ends else 0


def cut_memory_text(text: str, limit: int = ITEM_TEXT_CEILING) -> list[str]:
    """Stripped parts of at most `limit` characters; cut at a paragraph, else a sentence, else the limit."""
    parts: list[str] = []
    remainder = text.strip()
    while len(remainder) > limit:
        # one character past the limit: a break right after the limit-th character still fits
        cut_at = _last_cut_point(remainder[: limit + 1]) or limit
        head = remainder[:cut_at].strip()
        if head:
            parts.append(head)
        remainder = remainder[cut_at:].strip()
    if remainder:
        parts.append(remainder)
    return parts


def numbered_subject(subject: str, index: int, grand_total: int) -> str:
    """Subject plus ' (i/N)', trimmed so the whole stays inside the subject limit."""
    tag = f" ({index}/{grand_total})"
    return subject.strip()[: _SUBJECT_LIMIT - len(tag)].rstrip() + tag


_DENIED_FILENAMES = frozenset((".env", ".env.local", ".env.production", "id_rsa", "id_ed25519"))
_INDEXABLE_SUFFIXES = frozenset(
    (".md", ".txt", ".rst", ".py", ".js", ".ts", ".tsx", ".json", ".yaml", ".yml", ".toml")
)


class NeospikeService:
    def __init__(
        self,
        *,
        store: MemoryStore,
        embedder: Any,
        namespace: str,
        allowed_roots: list[Path] | None = None,
    ) -> None:
        self.store, self.embedder, self.namespace = store, embedder, namespace
        self.allowed_roots = [Path(entry).expanduser().resolve() for entry in allowed_roots or ()]
        self.session_id = ""
        self._notes: NoteService | None = None
        self._query_cache: tuple[int, str, list[float]] | None = None

    @property
    def notes(self) -> NoteService:
        if self._notes is None or self._notes.embedder is not self.embedder:
            self._notes = NoteService(self.store, self.embedder)
        return self._notes

    def _project_scope(self) -> str:
        """The first allowed root is the project of every capture and lookup; none -> ''."""
        return str(self.allowed_roots[0]) if self.allowed_roots else ""

    def _embed_query_cached(self, query: str) -> list[float]:
        """Memory and notes branches of one query share a single embedding."""
        owner = id(self.embedder)
        if self._query_cache is not None:
            cached_owner, cached_query, cached_vector = self._query_cache
            if cached_owner == owner and cached_query == query:
                return cached_vector
        vec = list(self.embedder.embed_query(query))
        self._query_cache = (owner, query, vec)
        return vec

    def cached_query_vector(self, query: str) -> list[float] | None:
        """The vector the last search of this query already computed; None without one (no embedding call)."""
        if self._query_cache is None:
            return None
        cached_owner, cached_query, cached_vector = self._query_cache
        return cached_vector if cached_owner == id(self.embedder) and cached_query == query else None

    def save_note(
        self,
        title: Any,
        note_body: Any,
        *,
        tags: Any = None,
        project: Any = "",
        author: str = "agent",
        expected_rev: int | None = None,
        note_id: int | None = None,
    ) -> dict:
        return self.notes.save(
            title, note_body, tags=tags, project=project, author=author, expected_rev=expected_rev, note_id=note_id
        )

    def read_note(
        self, *, note_id: int | None = None, title: str | None = None, include_deleted: bool = False
    ) -> dict:
        return self.notes.read(note_id=note_id, title=title, include_deleted=include_deleted)

    def delete_note(self, note_id: int, *, author: str, expected_rev: int | None) -> dict:
        return self.notes.delete(note_id, author=author, expected_rev=expected_rev)

    def restore_note(
        self, note_id: int, revision_no: int, *, author: str = "human", expected_rev: int | None = None
    ) -> dict:
        return self.notes.restore(note_id, revision_no, author=author, expected_rev=expected_rev)

    def search_notes(
        self, query: str, *, limit: int = 5, project: str | None = None, formula: bool | None = None
    ) -> list[dict]:
        """formula=True keeps only automatic-note chunks, False the other notes, None everything."""
        if not query.strip() or not self.notes.has_searchable():
            return []
        vec = self._embed_query_cached(query)
        return self.notes.search(query, vec, limit=limit, project=project, formula=formula)

    def _prepare_pieces(self, new_text: Any, fields: dict[str, Any]) -> list[dict]:
        """Validated memory items for new_text: one, or several when it exceeds the item limit.
        Every piece is validated before anything is written."""
        parts = cut_memory_text(new_text) if isinstance(new_text, str) else [new_text]
        several = len(parts) > 1
        if several and not isinstance(fields.get("subject"), str):
            raise ValueError("Memory text, subject, and durability must be strings")
        prepared = []
        for position, part in enumerate(parts, 1):
            extra = {"subject": numbered_subject(fields["subject"], position, len(parts))} if several else {}
            prepared.append(check_memory_item({"text": part, **fields, **extra}, partial_defaults=True))
        return prepared

    def update_memory_item(
        self,
        mid: int,
        *,
        expected_text: str,
        new_text: str,
        mode: str = MODE_CORRECT_INPLACE,
        evidence: str = "",
        edited_by: str = AGENT_EDITOR,
        **fields: Any,
    ) -> dict:
        """Validate and embed outside SQLite, then write against the row read in the preflight.

        mode=correct overwrites in place (the old text is kept as an item_revisions entry);
        mode=supersede stores new_text as a new item and marks the old one invalid.
        expected_text is a content compare-and-set, not a revision token, so an A -> B -> A
        change finished before the preflight read goes unnoticed.
        """
        if mode not in VALID_UPDATE_MODES:
            raise ValueError(f"mode must be one of {', '.join(VALID_UPDATE_MODES)}")
        if type(mid) is not int or not 1 <= mid <= _ID_CEILING:
            raise ValueError("Memory ID must be an integer from 1 to " + str(_ID_CEILING))
        if not isinstance(expected_text, str) or not expected_text.strip() or len(expected_text) > _EXPECTED_TEXT_CEILING:
            raise ValueError("expected_text must contain the exact " "previously retrieved memory text")
        if fields.keys() - _UPDATABLE_FIELDS:
            raise ValueError("Unknown atomic-update " "fields")
        first, *extra_items = self._prepare_pieces(new_text, fields)
        # Authority comes from the service's own state, never from the semantic item.
        project, session_id = self._project_scope(), self.session_id
        preflight = self.store.fetch_item_for_update(self.namespace, mid, project=project, session_id=session_id)
        if preflight is None:
            return {"updated": False, "reason": "unavailable"}
        if preflight["text"] != expected_text:
            return {"updated": False, "reason": "conflict"}
        common = {
            "expected_text": expected_text, "expected_row": preflight, "item": first,
            "project": project, "session_id": session_id, "split_items": extra_items,
        }
        try:
            common["embedding"] = self.embedder.embed_document(first["text"])
            common["split_embeddings"] = [list(self.embedder.embed_document(one["text"])) for one in extra_items]
            extra_ids: list = []
            if mode == MODE_SUPERSEDE_NEW:
                outcome = self.store.replace_item_with_new(self.namespace, mid, evidence=evidence, **common)
                extra_ids = outcome.pop("split_ids", [])
            else:
                status = self.store.update_memory_item(
                    self.namespace, mid, edited_by=edited_by, split_ids=extra_ids, **common
                )
                outcome = {"updated": True, "id": mid} if status == "updated" else {"updated": False, "reason": status}
            if extra_items and (outcome.get("updated") or outcome.get("superseded")):
                outcome["ids"] = [outcome["id"], *extra_ids]
        except Exception as err:
            # Embedder and SQLite errors can carry payloads or private paths, so they are not passed on.
            raise RuntimeError("Atomic memory update " "failed; no change was saved") from err
        return outcome

    def search(self, query: str, *, limit: int = 5, history: str = HISTORY_COLLAPSE,
               skip_kinds: Sequence[str] = ()) -> list[SearchResult]:
        """history is one of store.HISTORY_*. The default (collapse) is the prefetch treatment and is left out
        of the call, so a store with the pre-v9 search signature still serves prefetch. skip_kinds: memory kinds
        left out before the limit (passed only when non-empty, for the same reason)."""
        extra = _history_kwargs(history)
        if skip_kinds:
            extra["skip_kinds"] = tuple(skip_kinds)
        return self.store.search(
            self.namespace,
            query,
            self._embed_query_cached(query),
            limit=limit,
            project=self._project_scope(),
            session_id=self.session_id,
            **extra,
        )

def _history_kwargs(history: str) -> dict[str, str]:
    """Keyword for search(): present only for a non-default history mode (see NeospikeService.search)."""
    return {} if history == HISTORY_COLLAPSE else {"history": history}
