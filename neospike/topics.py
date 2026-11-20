"""Topic pages. No LLM is called here.

Daily cron order (no LLM step):
    1. ``topic-assign``    neighbour links of the live memories not checked yet (or whose body
                           changed); a memory without a topic linked to a topic member joins that topic
                           (closest link wins), repeated along chains. A stored memory gets no topic.
    2. ``topic-discover``  linked groups of more than 5 memories without a topic -> new topics, created at
                           once (nothing is recorded for review or undo).

Import rule: numpy is required and imported at module top; without it the import raises ImportError
with the pip command, so the provider fails to load visibly. store.py imports this module lazily; notes is imported inside functions.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import threading
import time
from collections import Counter
from typing import Any, Iterable

try:
    import numpy as np
except ImportError as error:  # no slow fallback; the provider load fails with this message
    raise ImportError("neospike needs numpy. Environment without numpy: "
                      + (", ".join(p for p in sys.path if p.endswith(("site-packages", "dist-packages"))) or sys.prefix)
                      + ". Run `python install.py` again from the neospike repository.") from error


from .store import (FORMULA_AUTHOR, ITEM_MEMORY, NOTE_CHUNKS_SQL, RELATION_TOPIC, note_head_sql,
                    vector_present_sql)
from .store import topic_order as store_topic_order

TOPIC_TITLE_PREFIX = "TOPIC: "


class _Tuning:
    """Starting values without a measurement (the thresholds themselves live in config.json)."""
    # neighbour links (fixed; no config UI)
    LINK_COSINE = 0.6  # every live memory links to other live memories (topic or not) at cos >= this
    LINK_NEIGHBORS = 3  # at most this many links chosen when the memory is checked
    GROUP_MAX_WITHOUT_TOPIC = 5  # a linked group of more memories than this becomes a topic
    LINK_BLOCK_ROWS = 256  # checked memories scored per matrix product (bounds the score matrix)
    SUBJECT_WEIGHT = 2  # a subject word counts this many times a body word when naming
    NAME_MAX_CHARS = 60
    # main formula (fixed; no config UI)
    NOTE_MEMBER_LIMIT = 10  # a topic with more live members than this becomes an automatic note
    ABSORB_COSINE = 0.8  # a live memory joins the automatic note whose closest chunk has cos >= this
    ABSORB_BLOCK_ROWS = 256  # memories scored per matrix product in note-absorb
    FOLLOW_UP_JOIN_SECONDS = 5.0  # shutdown wait for the save follow-up thread (as the conversation indexer)


_NAME_WORD = re.compile(r"[^\W\d_][\w-]+", flags=re.UNICODE)
_STOPWORDS = frozenset(
    "the and for with that this from are was were has have not but you your our use uses used into onto "
    "then than when what which will would can could should its it's they them their there here about also "
    "only just more most some such very each other".split()
)


# A memory that can belong to a topic: an active (not superseded) memory item.
# An embedding-pending memory (no vector yet) is not live here until embed-pending fills it, so links,
# joining, discovery, absorb, dedupe and conversion never read its empty BLOB.
# Preferences/constraints are not items rows since v23 (preferences.py), so none is ever assignable.
_ASSIGNABLE_SQL = (f"m.type='{ITEM_MEMORY}' AND m.kind='memory_item' AND m.invalid_at IS NULL AND m.archived_at IS NULL "
                   f"AND {vector_present_sql('m.embedding')}")


class TopicError(ValueError):
    pass


# vectors (numpy only) ------------------------------------------------------------------
FLOAT32_BYTES = 4


def _unit_rows(blobs: list[bytes]) -> tuple[list[Any], list[bool]]:
    """(float64 unit rows aligned with blobs, kept mask) of float32 blobs; zero or odd-width vectors are None."""
    if not blobs:
        return [], []
    width = len(blobs[0]) // FLOAT32_BYTES
    valid = [index for index, blob in enumerate(blobs)
             if blob is not None and width and len(blob) == width * FLOAT32_BYTES]
    rows: list[Any] = [None] * len(blobs)
    if valid:
        matrix = np.frombuffer(b"".join(blobs[index] for index in valid), dtype=np.float32)
        matrix = matrix.reshape(len(valid), width).astype(np.float64)
        # stored vectors are already unit vectors (normalize_rows at write); only all-zero rows drop.
        nonzero = matrix.any(axis=1)
        for position, index in enumerate(valid):
            if nonzero[position]:
                rows[index] = matrix[position]
    return rows, [row is not None for row in rows]


def nearest_links(rows: list[Any], sources: list[int]) -> list[tuple[int, int, float]]:
    """(source, other, cos) for each source index: up to LINK_NEIGHBORS other rows with cos >= LINK_COSINE,
    closest first (ties: lower index). None rows never link."""
    kept = [index for index, row in enumerate(rows) if row is not None]
    sources = [index for index in sources if rows[index] is not None]
    if len(kept) < 2 or not sources:
        return []
    matrix = np.asarray([rows[index] for index in kept], dtype=np.float64)
    position = {index: place for place, index in enumerate(kept)}
    pairs: list[tuple[int, int, float]] = []
    for start in range(0, len(sources), _Tuning.LINK_BLOCK_ROWS):
        block = sources[start:start + _Tuning.LINK_BLOCK_ROWS]
        scores = np.asarray([rows[index] for index in block], dtype=np.float64) @ matrix.T
        scores[np.arange(len(block)), [position[index] for index in block]] = -np.inf
        for line, index in enumerate(block):
            hits = np.flatnonzero(scores[line] >= _Tuning.LINK_COSINE)
            if not hits.size:
                continue
            chosen = hits[np.lexsort((hits, -scores[line, hits]))][:_Tuning.LINK_NEIGHBORS]
            pairs.extend((index, kept[int(column)], float(scores[line, column])) for column in chosen)
    return pairs


# topics as links (v22) -----------------------------------------------------------------------------------
# A topic has no row: it is the item_links rows of relation 'topic' whose label is its name (src_id = a member
# memory, dst_id NULL; a TOPIC page note may carry one too). A memory is in at most one topic. A topic without
# links does not exist; its state (when it last changed) is read from its links and members, never stored.
_TOPIC_LINK = f"relation='{RELATION_TOPIC}'"


def mark_dirty(connection: sqlite3.Connection, topic_names: Iterable[str], now: float) -> int:
    """Kept for callers: a topic's change time is computed (topic_changed_at), nothing is stored."""
    return 0


def drop_empty_topics(connection: sqlite3.Connection, topic_names: Iterable[str] | None = None) -> list[str]:
    """Kept for callers: a topic without member links is gone by itself (nothing to delete)."""
    return []


def member_topics(connection: sqlite3.Connection, memory_ids: Iterable[int]) -> list[str]:
    """Names of the topics the given items are in."""
    ids = sorted({int(memory_id) for memory_id in memory_ids})
    names: set[str] = set()
    for start in range(0, len(ids), _SQL_IDS_BATCH):
        batch = ids[start:start + _SQL_IDS_BATCH]
        names.update(str(row[0]) for row in connection.execute(
            f"SELECT DISTINCT label FROM item_links WHERE {_TOPIC_LINK} AND src_id IN ({','.join('?' * len(batch))})",
            batch))
    return sorted(names)


def touch_memories(connection: sqlite3.Connection, memory_ids: Iterable[int], now: float) -> list[str]:
    """A member was corrected/superseded/restored: the names of its topics (their change time follows the
    member's updated_at by itself)."""
    return member_topics(connection, memory_ids)


def topic_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(f"SELECT 1 FROM item_links WHERE {_TOPIC_LINK} AND label=? LIMIT 1",
                              (str(name),)).fetchone() is not None


def set_topic(connection: sqlite3.Connection, memory_id: int, name: str, now: float) -> None:
    """memory_id joins topic name (its earlier topic link, if any, is replaced)."""
    connection.execute(f"DELETE FROM item_links WHERE {_TOPIC_LINK} AND src_id=?", (int(memory_id),))
    connection.execute(f"INSERT INTO item_links(src_id,relation,label,created_at) VALUES (?,'{RELATION_TOPIC}',?,?)",
                       (int(memory_id), str(name), now))


def unused_topic_name(connection: sqlite3.Connection, base: str) -> str:
    """base, or `base (N)` with the smallest N no topic has (a topic is identified by its name)."""
    if not topic_exists(connection, base):
        return base
    number = 2
    while topic_exists(connection, f"{base} ({number})"):
        number += 1
    return f"{base} ({number})"


def topic_order(connection: sqlite3.Connection) -> dict[str, int]:
    """name -> the id of its oldest link (creation order of the live topics; the old lower-topic-id rule)."""
    return store_topic_order(connection)


def topic_changed_at_sql(label_sql: str) -> str:
    """SQL value: when a topic last changed = the newest of its links' created_at and its members' updated_at."""
    return (f"(SELECT MAX(MAX(t.created_at), COALESCE(MAX(m.updated_at), 0)) FROM item_links t "
            f"LEFT JOIN items m ON m.id=t.src_id WHERE t.{_TOPIC_LINK} AND t.label={label_sql})")


# hand-edited membership -----------------------------------------------------------
# One meta row holds the marks, so no schema step: {"pinned": {"<mem>": "<topic>"}, "excluded": {"<mem>": [topic, ...]}}
# (topic = its name). pinned: put in by hand; formula-migrate's rebuild keeps it in that topic. excluded: taken
# out by hand; topic-assign and the rebuild never put it back into those topics. A mark on a gone topic or memory
# is inert.
MANUAL_META_KEY = "topic_manual"


class ManualMarks:
    def __init__(self, raw: Any) -> None:
        raw = raw if isinstance(raw, dict) else {}
        pinned, excluded = raw.get("pinned"), raw.get("excluded")
        self.pinned: dict[str, str] = {str(key): str(value) for key, value in (pinned or {}).items()
                                       if isinstance(value, str)} if isinstance(pinned, dict) else {}
        self.excluded: dict[str, list[str]] = {
            str(key): sorted({str(item) for item in value if isinstance(item, str)})
            for key, value in (excluded or {}).items() if isinstance(value, list)} if isinstance(excluded, dict) else {}

    def excluded_from(self, memory_id: int) -> list[str]:
        return self.excluded.get(str(int(memory_id)), [])

    def pinned_to(self, memory_id: int) -> str | None:
        return self.pinned.get(str(int(memory_id)))

    def pin(self, memory_id: int, topic: str) -> None:
        key = str(int(memory_id))
        self.pinned[key] = str(topic)
        kept = [item for item in self.excluded.get(key, []) if item != str(topic)]
        if kept:
            self.excluded[key] = kept
        else:
            self.excluded.pop(key, None)

    def exclude(self, memory_id: int, topic: str) -> None:
        key = str(int(memory_id))
        if self.pinned.get(key) == str(topic):
            del self.pinned[key]
        self.excluded[key] = sorted({*self.excluded.get(key, []), str(topic)})

    def as_json(self) -> str:
        return json.dumps({"pinned": self.pinned, "excluded": self.excluded}, sort_keys=True, ensure_ascii=False)


def manual_marks(connection: sqlite3.Connection) -> ManualMarks:
    row = connection.execute("SELECT value FROM meta_entries WHERE key=?", (MANUAL_META_KEY,)).fetchone()
    try:
        return ManualMarks(json.loads(row[0]) if row else {})
    except json.JSONDecodeError:
        return ManualMarks({})


def save_manual_marks(connection: sqlite3.Connection, marks: ManualMarks) -> None:
    connection.execute("INSERT OR REPLACE INTO meta_entries(key,value) VALUES (?,?)", (MANUAL_META_KEY, marks.as_json()))


def topic_of_note(connection: sqlite3.Connection, note_id: int) -> str:
    row = connection.execute(f"SELECT label FROM item_links WHERE {_TOPIC_LINK} AND src_id=? ORDER BY id LIMIT 1",
                             (int(note_id),)).fetchone()
    if row is None:
        raise TopicError(f"note {note_id} is not the page of a live topic")
    return str(row[0])


def live_topic(connection: sqlite3.Connection, topic: str) -> str:
    if not isinstance(topic, str) or not topic_exists(connection, topic):
        raise TopicError(f"topic {topic} is not a live topic")
    return topic


def _topic_target(connection: sqlite3.Connection, topic: str | None, note_id: int | None) -> str | None:
    if topic is not None:
        return live_topic(connection, topic)
    return None if note_id is None else topic_of_note(connection, note_id)


def move_members(store: Any, namespace: str, memory_ids: list[int], embedder: Any, *, target_note: int | None = None,
                 remove_from_note: int | None = None, target_topic: str | None = None,
                 remove_from_topic: str | None = None, now: float | None = None) -> dict:
    """topic-member-move: put memories into a topic by hand (target_topic / target_note) or take
    them out of one (remove_from_topic / remove_from_note). A topic has no page, so the dashboard
    names it by its name; the note forms are kept for callers that still use them. The marks keep cron from undoing it."""
    if sum(value is not None for value in (target_note, remove_from_note, target_topic, remove_from_topic)) != 1:
        raise TopicError("give exactly one of a target topic or the topic to remove from")
    current = time.time() if now is None else now
    ids = list(dict.fromkeys(int(memory_id) for memory_id in memory_ids))
    with store._lock, store._conn:
        connection = store._conn
        marks = manual_marks(connection)
        touched: set[str] = set()
        target = _topic_target(connection, target_topic, target_note)
        if target is not None:
            marks_sql = ",".join("?" * len(ids)) or "NULL"
            usable = {int(row[0]) for row in connection.execute(
                f"SELECT m.id FROM items m WHERE m.owner=? AND m.id IN ({marks_sql}) AND {_ASSIGNABLE_SQL}",
                (namespace, *ids))}
            done = [memory_id for memory_id in ids if memory_id in usable]
            touched.update(member_topics(connection, done))
            for memory_id in done:
                set_topic(connection, memory_id, target, current)
                marks.pin(memory_id, target)
            if done:
                touched.add(target)
        else:
            source = _topic_target(connection, remove_from_topic, remove_from_note)
            members = {int(row[0]) for row in connection.execute(
                f"SELECT src_id FROM item_links WHERE {_TOPIC_LINK} AND label=?", (source,))}
            done = [memory_id for memory_id in ids if memory_id in members]
            connection.executemany(f"DELETE FROM item_links WHERE {_TOPIC_LINK} AND src_id=? AND label=?",
                                   [(memory_id, source) for memory_id in done])
            for memory_id in done:
                marks.exclude(memory_id, source)
            if done:
                touched.add(source)
        save_manual_marks(connection, marks)
    # No topic page is rewritten; the "evidence" key keeps the output shape.
    evidence = {"pages": [], "rewritten": 0}
    return {"moved": len(done), "moved_ids": done, "skipped_ids": [memory_id for memory_id in ids if memory_id not in done],
            "dirty_topics": sorted(touched), "evidence": evidence}


# naming (code, most frequent word of member subjects/bodies) -------------------------------
def topic_name(subjects: list[str], texts: list[str]) -> str:
    counts: Counter[str] = Counter()
    first: dict[str, int] = {}
    position = 0
    for weight, values in ((_Tuning.SUBJECT_WEIGHT, subjects), (1, texts)):
        for value in values:
            for word in _NAME_WORD.findall(value or ""):
                folded = word.casefold()
                if folded in _STOPWORDS:
                    continue
                counts[folded] += weight
                first.setdefault(folded, position)
                position += 1
    if not counts:
        return "topic"
    best = min(counts, key=_NameRank(counts, first))
    return best[:_Tuning.NAME_MAX_CHARS]


class _NameRank:
    """Sort key: most frequent first, then first appearance (function object, no lambda)."""
    def __init__(self, counts: Counter, first: dict[str, int]) -> None:
        self.counts, self.first = counts, first

    def __call__(self, word: str) -> tuple[int, int]:
        return -self.counts[word], self.first[word]


def _subject(metadata_json: str) -> str:
    try:
        value = json.loads(metadata_json or "{}").get("subject", "")
    except (json.JSONDecodeError, AttributeError):
        return ""
    return value if isinstance(value, str) else ""


def _member_rows(connection: sqlite3.Connection, memory_ids: list[int]) -> list[sqlite3.Row]:
    if not memory_ids:
        return []
    return connection.execute(
        f"SELECT id,text,metadata_json,embedding,created_at,updated_at,superseded_by,invalid_at FROM items "
        f"WHERE id IN ({','.join('?' * len(memory_ids))}) ORDER BY id", memory_ids).fetchall()


def _name_of(rows: list[sqlite3.Row]) -> str:
    return topic_name([_subject(row["metadata_json"]) for row in rows], [row["text"] for row in rows])


# links and joining (stage 1) --------------------------------------------------------
# Every live memory (_ASSIGNABLE_SQL) is checked once per body (content_hash) and links to its closest live
# neighbours, topic or not. Links stay when a memory gets a topic; only a deleted memory (or a changed body,
# for its own links) loses them. A memory without a topic joins the topic of its closest linked topic member.
def _config(store: Any) -> Any:
    return store.read_neospike_config()


Link = tuple[int, int, float]  # (low id, high id, cos)


def _live(connection: sqlite3.Connection, namespace: str) -> list[sqlite3.Row]:
    """Live memories of a namespace in id order: the link pool."""
    return connection.execute(
        f"SELECT m.id,m.text_hash,m.text,m.metadata_json,m.embedding FROM items m WHERE m.owner=? "
        f"AND {_ASSIGNABLE_SQL} ORDER BY m.id", (namespace,)).fetchall()


def _pool_pairs(pool: list[sqlite3.Row], sources: list[int]) -> list[Link]:
    """nearest_links of the pool rows at *sources* as (low id, high id, cos), one row per pair."""
    vectors, _ = _unit_rows([row["embedding"] for row in pool])
    pairs: dict[tuple[int, int], float] = {}
    for left, right, score in nearest_links(vectors, sources):
        a, b = int(pool[left]["id"]), int(pool[right]["id"])
        pairs[(min(a, b), max(a, b))] = score
    return [(low, high, score) for (low, high), score in sorted(pairs.items())]


def _link_plan(connection: sqlite3.Connection,
               namespace: str) -> tuple[list[sqlite3.Row], list[int], list[int], list[Link]]:
    """(live memories, ids to check, ids among them checked before with another body, their new links).
    Only memories never checked or checked with another text_hash are scored."""
    pool = _live(connection, namespace)
    checked = {int(row[0]): row[1] for row in connection.execute(
        "SELECT memory_id,text_hash FROM memory_checks WHERE stage='topic'")}
    fresh = [index for index, row in enumerate(pool)
             if int(row["id"]) not in checked or checked[int(row["id"])] != row["text_hash"]]
    if not fresh:
        return pool, [], [], []
    ids = [int(pool[index]["id"]) for index in fresh]
    return pool, ids, [memory_id for memory_id in ids if memory_id in checked], _pool_pairs(pool, fresh)


def _delete_links_of(connection: sqlite3.Connection, memory_ids: list[int]) -> None:
    for start in range(0, len(memory_ids), _SQL_IDS_BATCH):
        batch = memory_ids[start:start + _SQL_IDS_BATCH]
        marks = ",".join("?" * len(batch))
        connection.execute(f"DELETE FROM memory_pairs WHERE low_id IN ({marks}) OR high_id IN ({marks})",
                           (*batch, *batch))


def _save_links(connection: sqlite3.Connection, pool: list[sqlite3.Row], checked: list[int], changed: list[int],
                pairs: list[Link], now: float) -> int:
    """Drop the old links of the changed memories, store the new links and the check marks (with the body
    hash); returns the links added."""
    _delete_links_of(connection, changed)
    added = connection.executemany(
        "INSERT OR IGNORE INTO memory_pairs(low_id,high_id,cosine,created_at) VALUES (?,?,?,?)",
        [(low, high, score, now) for low, high, score in pairs]).rowcount if pairs else 0
    hashes = {int(row["id"]): row["text_hash"] for row in pool}
    connection.executemany(
        "INSERT OR REPLACE INTO memory_checks(memory_id,stage,checked_at,text_hash) VALUES (?,'topic',?,?)",
        [(memory_id, now, hashes[memory_id]) for memory_id in checked])
    return max(added, 0)


def refresh_links(connection: sqlite3.Connection, namespace: str,
                  now: float) -> tuple[list[sqlite3.Row], int, int]:
    """The incremental link step inside the caller's transaction: (live memories, memories checked, links added)."""
    pool, checked, changed, pairs = _link_plan(connection, namespace)
    return pool, len(checked), _save_links(connection, pool, checked, changed, pairs, now) if checked else 0


def _stored_links(connection: sqlite3.Connection, live_ids: set[int], *, drop: Iterable[int] = (),
                  extra: Iterable[Link] = ()) -> list[Link]:
    """Stored links between live memories (minus those touching *drop*) plus *extra*."""
    dropped = set(drop)
    links = [(int(low), int(high), float(score)) for low, high, score in connection.execute(
        "SELECT low_id,high_id,cosine FROM memory_pairs ORDER BY low_id,high_id")
        if int(low) in live_ids and int(high) in live_ids and int(low) not in dropped and int(high) not in dropped]
    links.extend(link for link in extra if link[0] in live_ids and link[1] in live_ids)
    return links


def _live_topic_links(connection: sqlite3.Connection) -> dict[int, str]:
    """item id -> topic name of every topic link."""
    return {int(row[0]): str(row[1]) for row in connection.execute(
        f"SELECT src_id,label FROM item_links WHERE {_TOPIC_LINK} ORDER BY id")}


def _with_topic(connection: sqlite3.Connection) -> set[int]:
    return {int(row[0]) for row in connection.execute(f"SELECT src_id FROM item_links WHERE {_TOPIC_LINK}")}


class _TopicRank:
    """Tie order of topics: their creation order (oldest link id), then the name."""

    def __init__(self, order: dict | None) -> None:
        self.order = order or {}

    def __call__(self, topic: Any) -> tuple[int, str]:
        return self.order.get(topic, 0), str(topic)


def join_plan(links: Iterable[Link], members: dict[int, Any], marks: ManualMarks,
              order: dict[str, int] | None = None) -> dict[int, Any]:
    """{memory id: topic} of the memories without a topic that join one (pure). A memory linked to members
    of several topics takes the topic of its highest-cosine link (tie: the older topic, order); joined memories pass
    their topic on along chains, round by round, until nothing joins. A topic the memory was taken out of by
    hand is never joined. Topics never merge: a member never changes topic."""
    rank = _TopicRank(order)
    neighbours: dict[int, list[tuple[float, int]]] = {}
    for low, high, score in links:
        neighbours.setdefault(low, []).append((score, high))
        neighbours.setdefault(high, []).append((score, low))
    topic_of = dict(members)
    joined: dict[int, int] = {}
    frontier = sorted(topic_of)
    while frontier:
        best: dict[int, tuple[float, int]] = {}
        for member in frontier:
            topic = topic_of[member]
            for score, other in neighbours.get(member, ()):
                if other in topic_of or topic in marks.excluded_from(other):
                    continue
                held = best.get(other)
                if held is None or score > held[0] or (score == held[0] and rank(topic) < rank(held[1])):
                    best[other] = (score, topic)
        for memory_id, (_, topic) in best.items():
            topic_of[memory_id] = joined[memory_id] = topic
        frontier = sorted(best)
    return joined


def assign_all(store: Any, namespace: str, *, now: float | None = None) -> dict:
    """topic-assign: the incremental link step, then every live memory without a topic that is
    linked (directly or along a chain of memories without a topic) to a topic member joins (join_plan)."""
    current = time.time() if now is None else now
    with store._lock, store._conn:
        connection = store._conn
        pool, checked, added = refresh_links(connection, namespace, current)
        live_ids = {int(row["id"]) for row in pool}
        members = {memory_id: topic_id for memory_id, topic_id in _live_topic_links(connection).items()
                   if memory_id in live_ids}
        joined = join_plan(_stored_links(connection, live_ids), members, manual_marks(connection),
                           topic_order(connection))
        for memory_id, topic in sorted(joined.items()):
            set_topic(connection, memory_id, topic, current)
    return {"link_checked": checked, "links_added": added, "assigned": len(joined),
            "members": [{"memory_id": key, "topic_id": value} for key, value in sorted(joined.items())]}


# new topics (stage 2) ------------------------------------------------------------------------
class _Forest:
    """Union-find over memory ids (path halving; the smaller root id wins)."""

    def __init__(self, ids: Iterable[int]) -> None:
        self.parent = {memory_id: memory_id for memory_id in ids}

    def find(self, memory_id: int) -> int:
        parent = self.parent
        while parent[memory_id] != memory_id:
            parent[memory_id] = parent[parent[memory_id]]
            memory_id = parent[memory_id]
        return memory_id

    def union(self, a: int, b: int) -> None:
        if a in self.parent and b in self.parent:
            low, high = sorted((self.find(a), self.find(b)))
            self.parent[high] = low


def _components(ids: Iterable[int], links: Iterable[Link]) -> list[list[int]]:
    """Connected groups of *ids* over the links between them (links to other ids ignored); each group in id
    order, groups in order of their first id."""
    forest = _Forest(ids)
    for low, high, _ in links:
        forest.union(low, high)
    groups: dict[int, list[int]] = {}
    for memory_id in sorted(forest.parent):
        groups.setdefault(forest.find(memory_id), []).append(memory_id)
    return sorted(groups.values(), key=_FIRST_ID)


def _first_id(group: list[int]) -> int:
    return group[0]


_FIRST_ID = _first_id


def _linked_groups(pool: list[sqlite3.Row], links: Iterable[Link]) -> list[list[sqlite3.Row]]:
    """Groups of pool memories joined by links, kept when larger than GROUP_MAX_WITHOUT_TOPIC; members and
    groups in id order. Links to memories outside pool are ignored."""
    by_id = {int(row["id"]): row for row in pool}
    return [[by_id[memory_id] for memory_id in group] for group in _components(by_id, links)
            if len(group) > _Tuning.GROUP_MAX_WITHOUT_TOPIC]


def _live_topics(connection: sqlite3.Connection) -> list[dict]:
    """Every topic in creation order: {"id": name, "name": name} (a topic is identified by its name)."""
    order = topic_order(connection)
    return [{"id": name, "name": name} for name in sorted(order, key=order.__getitem__)]


def discover(store: Any, namespace: str, *, now: float | None = None) -> dict:
    """topic-discover: a new topic for each linked group of more than GROUP_MAX_WITHOUT_TOPIC memories without a
    topic (vector math only; the incremental link step runs first). The topic is created at once, in
    the same transaction; nothing is proposed or recorded. ``topic_ids`` lists the new topics."""
    current = time.time() if now is None else now
    created: list[str] = []
    with store._lock:
        connection = store._conn
        with connection:
            pool, checked, links_added = refresh_links(connection, namespace, current)
        with connection:
            taken = _with_topic(connection)
            free = [row for row in pool if int(row["id"]) not in taken]
            links = _stored_links(connection, {int(row["id"]) for row in free})
            for group in _linked_groups(free, links):
                ids = [int(row["id"]) for row in group]
                created.append(_create_topic(connection, _member_rows(connection, ids), current))
    return {"topic_new": len(created), "topic_ids": created, "link_checked": checked, "links_added": links_added}


def _create_topic(connection: sqlite3.Connection, rows: list[sqlite3.Row], now: float) -> str:
    """A new topic of these memories, named from them (a name another topic has gets ' (N)')."""
    name = unused_topic_name(connection, _name_of(rows))
    for row in rows:
        set_topic(connection, int(row["id"]), name, now)
    return name


# main formula ---------------------------------------------------------------------
# A topic is a tag (no page). note-absorb (daily, before topic-assign): a live memory whose closest automatic
# note chunk has cos >= ABSORB_COSINE becomes a chunk of that note and the memory is deleted. topic-to-note
# (daily, after topic-discover): a topic with more than NOTE_MEMBER_LIMIT live members becomes one automatic
# note of their bodies; those memories, the topic's superseded members and the topic are deleted.
# formula-migrate: the one-off move of an existing home to this formula. Automatic notes: notes.py.
FORMULA_EDITOR = "formula"  # item_revisions.edited_by of an automatic chunk change
MEMBER_VALID, MEMBER_INVALID, MEMBER_ARCHIVED, MEMBER_OTHER = "valid", "invalid", "archived", "other"
_SQL_IDS_BATCH = 500


def _delete_memories(store: Any, memory_ids: Iterable[int], now: float) -> int:
    """store.delete's clean-up for many memories inside the caller's transaction (lock held): the rows
    (tombstones by trigger), topic membership, links, discovery and absorb marks."""
    ids = sorted({int(memory_id) for memory_id in memory_ids})
    removed = 0
    for start in range(0, len(ids), _SQL_IDS_BATCH):
        batch = ids[start:start + _SQL_IDS_BATCH]
        removed += store._conn.execute(f"DELETE FROM items WHERE type='{ITEM_MEMORY}' AND id IN ({','.join('?' * len(batch))})",
                                       batch).rowcount
    for start in range(0, len(ids), _SQL_IDS_BATCH):
        store._drop_memory_links(ids[start:start + _SQL_IDS_BATCH], now)
    return removed


def _states_of(connection: sqlite3.Connection, member_ids: Iterable[int], namespace: str) -> dict[str, list[int]]:
    """Memories by state, ids ascending: valid (_ASSIGNABLE_SQL in namespace), invalid (superseded),
    archived (a human archive; never deleted), other (another kind or namespace)."""
    states: dict[str, list[int]] = {MEMBER_VALID: [], MEMBER_INVALID: [], MEMBER_ARCHIVED: [], MEMBER_OTHER: []}
    ids = sorted({int(memory_id) for memory_id in member_ids})
    for start in range(0, len(ids), _SQL_IDS_BATCH):
        batch = ids[start:start + _SQL_IDS_BATCH]
        for memory_id, state in connection.execute(
                f"SELECT m.id, CASE WHEN m.archived_at IS NOT NULL THEN '{MEMBER_ARCHIVED}' "
                f"WHEN m.owner=? AND {_ASSIGNABLE_SQL} THEN '{MEMBER_VALID}' "
                f"WHEN m.invalid_at IS NOT NULL THEN '{MEMBER_INVALID}' ELSE '{MEMBER_OTHER}' END "
                f"FROM items m WHERE m.type='{ITEM_MEMORY}' AND m.id IN ({','.join('?' * len(batch))}) ORDER BY m.id",
                (namespace, *batch)):
            states[state].append(int(memory_id))
    return states


def _topic_member_ids(connection: sqlite3.Connection, topic: str) -> list[int]:
    """Memory members of a topic (a TOPIC page note's link is not a member)."""
    return [int(row[0]) for row in connection.execute(
        f"SELECT l.src_id FROM item_links l JOIN items m ON m.id=l.src_id AND m.type='{ITEM_MEMORY}' "
        f"WHERE l.{_TOPIC_LINK} AND l.label=? ORDER BY l.src_id", (topic,))]


def _plan_of(topic_id: str | None, name: str, states: dict[str, list[int]], skip: frozenset[int]) -> dict | None:
    valid = [memory_id for memory_id in states[MEMBER_VALID] if memory_id not in skip]
    if len(valid) <= _Tuning.NOTE_MEMBER_LIMIT:
        return None
    invalid = [memory_id for memory_id in states[MEMBER_INVALID] if memory_id not in skip]
    return {"topic_id": topic_id, "name": name, "valid": len(valid), "invalid": len(invalid),
            "archived": len(states[MEMBER_ARCHIVED]), "other": len(states[MEMBER_OTHER]),
            "note_memory_ids": valid, "deleted_memory_ids": sorted([*valid, *invalid]),
            "untagged_memory_ids": sorted([*states[MEMBER_ARCHIVED], *states[MEMBER_OTHER]])}


def _note_plan(connection: sqlite3.Connection, topic: dict, namespace: str,
               skip: frozenset[int] = frozenset()) -> dict | None:
    """What topic-to-note does with one topic (None at or below the limit). skip: memories another step
    deletes first (formula-migrate de-duplication)."""
    states = _states_of(connection, _topic_member_ids(connection, topic["id"]), namespace)
    return _plan_of(topic["id"], topic["name"], states, skip)


def _convert_topic(store: Any, topic: dict, namespace: str, now: float) -> dict | None:
    """One topic -> one automatic note, inside the caller's write transaction."""
    from .notes import MEMORY_SOURCE_COLUMNS, chunk_source_json, create_formula_note
    connection = store._conn
    plan = _note_plan(connection, topic, namespace)
    if plan is None:
        return None
    ids = plan["note_memory_ids"]
    items = []
    for start in range(0, len(ids), _SQL_IDS_BATCH):
        batch = ids[start:start + _SQL_IDS_BATCH]
        items.extend((row["text"], row["embedding"], chunk_source_json(row)) for row in connection.execute(
            f"SELECT text,embedding,{MEMORY_SOURCE_COLUMNS} FROM items "
            f"WHERE id IN ({','.join('?' * len(batch))}) ORDER BY id", batch))
    _delete_memories(store, plan["deleted_memory_ids"], now)
    connection.execute(f"DELETE FROM item_links WHERE {_TOPIC_LINK} AND label=?",  # archived/other: untagged
                       (plan["topic_id"],))
    note = create_formula_note(connection, topic["name"], items, now)
    return {**{key: plan[key] for key in ("topic_id", "name", "valid", "invalid", "archived", "other")},
            "note": note, "deleted_memory_ids": plan["deleted_memory_ids"]}


def topic_to_note(store: Any, namespace: str, *, now: float | None = None) -> dict:
    """topic-to-note: every live topic with more than NOTE_MEMBER_LIMIT live members becomes an automatic
    note (chunks = member bodies by id, embeddings copied, project ''); one transaction per topic, so a
    failed topic leaves its memories, the topic and the notes as they were and the next topic still runs."""
    current = time.time() if now is None else now
    converted: list[dict] = []
    errors: list[dict] = []
    with store._lock:
        connection = store._conn
        for topic in _live_topics(connection):
            try:
                connection.execute("BEGIN IMMEDIATE")
                done = _convert_topic(store, topic, namespace, current)
                connection.commit()
            except (sqlite3.Error, ValueError, RuntimeError) as error:  # NoteError is a ValueError
                if connection.in_transaction:
                    connection.rollback()
                errors.append({"topic_id": topic["id"], "error": f"{type(error).__name__}: {error}"})
                continue
            if done is not None:
                converted.append(done)
    return {"converted": len(converted), "topics": converted, "errors": errors, "limit": _Tuning.NOTE_MEMBER_LIMIT}


# note-absorb ---------------------------------------------------------------------------------
class _AbsorbScan:
    """Scores memories against chunk vectors; counts every (memory, chunk) pair scored."""

    def __init__(self) -> None:
        self.compared = 0

    def best(self, memory_rows: list[Any], chunk_rows: list[Any]) -> list[tuple[int, float]]:
        """(chunk position, cos) of the closest chunk per memory row (first maximum); (-1, -inf) when none."""
        if not memory_rows or not chunk_rows:
            return [(-1, -np.inf)] * len(memory_rows)
        chunks = np.asarray(chunk_rows, dtype=np.float64)
        result: list[tuple[int, float]] = []
        for start in range(0, len(memory_rows), _Tuning.ABSORB_BLOCK_ROWS):
            block = np.asarray(memory_rows[start:start + _Tuning.ABSORB_BLOCK_ROWS], dtype=np.float64)
            scores = block @ chunks.T
            self.compared += int(scores.size)
            columns = np.argmax(scores, axis=1)
            result.extend((int(column), float(scores[line, column])) for line, column in enumerate(columns))
        return result


def _absorb_plan(memories: list[sqlite3.Row], marks: list[int], chunks: list[sqlite3.Row],
                 scan: _AbsorbScan) -> dict[int, int]:
    """{memory id: note id} of the memories to absorb. Each memory is scored only against the chunks above its
    mark (those it was not compared with); then, until nothing changes, the memories left are scored against
    the memories absorbed the round before (they become chunks of those notes in this run)."""
    chunk_vectors, _ = _unit_rows([row["embedding"] for row in chunks])
    memory_vectors, _ = _unit_rows([row["embedding"] for row in memories])
    absorbed: dict[int, int] = {}
    groups: dict[int, list[int]] = {}
    for position, mark in enumerate(marks):
        if memory_vectors[position] is not None:
            groups.setdefault(mark, []).append(position)
    fresh: list[tuple[Any, int]] = []
    for mark in sorted(groups):
        targets = [index for index, row in enumerate(chunks)
                   if int(row["id"]) > mark and chunk_vectors[index] is not None]
        positions = groups[mark]
        best = scan.best([memory_vectors[position] for position in positions],
                         [chunk_vectors[index] for index in targets])
        for position, (column, score) in zip(positions, best):
            if column >= 0 and score >= _Tuning.ABSORB_COSINE:
                note_id = int(chunks[targets[column]]["note_id"])
                absorbed[int(memories[position]["id"])] = note_id
                fresh.append((memory_vectors[position], note_id))
    while fresh:
        left = [position for position, row in enumerate(memories)
                if memory_vectors[position] is not None and int(row["id"]) not in absorbed]
        best = scan.best([memory_vectors[position] for position in left], [vector for vector, _ in fresh])
        added: list[tuple[Any, int]] = []
        for position, (column, score) in zip(left, best):
            if column >= 0 and score >= _Tuning.ABSORB_COSINE:
                absorbed[int(memories[position]["id"])] = fresh[column][1]
                added.append((memory_vectors[position], fresh[column][1]))
        fresh = added
    return absorbed


def _unchanged(connection: sqlite3.Connection, rows: list[sqlite3.Row], namespace: str) -> list[sqlite3.Row]:
    """The planned memories still live with the same text (one edited meanwhile is left for the next run)."""
    ids = [int(row["id"]) for row in rows]
    current = {int(row[0]): row[1] for row in connection.execute(
        f"SELECT m.id,m.text_hash FROM items m WHERE m.owner=? AND {_ASSIGNABLE_SQL} "
        f"AND m.id IN ({','.join('?' * len(ids))})", (namespace, *ids))}
    return [row for row in rows if current.get(int(row["id"])) == row["text_hash"]]


def absorb(store: Any, namespace: str, *, now: float | None = None) -> dict:
    """note-absorb: live memories (_ASSIGNABLE_SQL) against the chunks of live automatic notes. A memory whose
    best chunk has cos >= ABSORB_COSINE is appended to that chunk's note (body as a chunk, embedding copied,
    no embedding call) and deleted. One chunk edit per note per run (one revision), all in one transaction.
    Increments: memory_checks (stage absorb) keeps per memory its text_hash and the highest chunk id it was
    compared with, so an unchanged second run scores nothing."""
    from .notes import (MEMORY_SOURCE_COLUMNS, apply_formula_chunk_ops, chunk_source_json, formula_chunk_text,
                        formula_chunk_watermark)
    current = time.time() if now is None else now
    with store._lock:
        connection = store._conn
        chunks = connection.execute(
            f"SELECT c.id,c.note_id,c.embedding FROM {NOTE_CHUNKS_SQL} c JOIN items n ON n.id=c.note_id "
            "WHERE n.author=? AND n.deleted_at IS NULL ORDER BY c.id", (FORMULA_AUTHOR,)).fetchall()
        watermark = formula_chunk_watermark(connection)
        memories = [row for row in connection.execute(
            f"SELECT m.text_hash,m.text,m.embedding,{','.join('m.' + name for name in MEMORY_SOURCE_COLUMNS.split(','))} "
            f"FROM items m WHERE m.owner=? AND {_ASSIGNABLE_SQL} ORDER BY m.id", (namespace,))
            if formula_chunk_text(row["text"])]
        checked = {int(row[0]): (row[1], int(row[2])) for row in connection.execute(
            "SELECT memory_id,text_hash,chunk_id FROM memory_checks WHERE stage='absorb'")}
    marks = []
    for row in memories:
        record = checked.get(int(row["id"]))
        marks.append(record[1] if record is not None and record[0] == row["text_hash"] else 0)
    scan = _AbsorbScan()
    plan = _absorb_plan(memories, marks, chunks, scan)
    by_note: dict[int, list[sqlite3.Row]] = {}
    for row in memories:
        if int(row["id"]) in plan:
            by_note.setdefault(plan[int(row["id"])], []).append(row)
    notes_done: list[dict] = []
    skipped: list[int] = []
    with store._lock:
        connection = store._conn
        try:
            connection.execute("BEGIN IMMEDIATE")
            moved = formula_chunk_watermark(connection) != watermark  # a chunk was written meanwhile
            deleted: list[int] = []
            for note_id in sorted(by_note):
                rows = _unchanged(connection, by_note[note_id], namespace)
                live = connection.execute(f"SELECT 1 FROM items n WHERE n.id=? AND {note_head_sql('n')} AND n.author=? "
                                          "AND n.deleted_at IS NULL", (note_id, FORMULA_AUTHOR)).fetchone()
                if live is None:
                    rows = []
                skipped.extend(int(row["id"]) for row in by_note[note_id] if row not in rows)
                if not rows:
                    continue
                done = apply_formula_chunk_ops(
                    connection, note_id,
                    add=[(formula_chunk_text(row["text"]), row["embedding"], chunk_source_json(row)) for row in rows],
                    editor=FORMULA_EDITOR, now=current)
                notes_done.append({"id": note_id, "title": done["title"], "rev": done["rev"], "added": len(rows),
                                   "memory_ids": [int(row["id"]) for row in rows]})
                deleted.extend(int(row["id"]) for row in rows)
            _delete_memories(store, deleted, current)
            mark = watermark if moved else formula_chunk_watermark(connection)
            records = [(int(row["id"]), row["text_hash"], mark) for row in memories
                       if int(row["id"]) not in plan and checked.get(int(row["id"])) != (row["text_hash"], mark)]
            connection.executemany(
                "INSERT OR REPLACE INTO memory_checks(memory_id,stage,text_hash,chunk_id) VALUES (?,'absorb',?,?)", records)
            connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
    return {"memories": len(memories), "chunks": len(chunks), "compared": scan.compared,
            "absorbed": sum(item["added"] for item in notes_done), "notes": notes_done,
            "skipped_ids": sorted(skipped), "threshold": _Tuning.ABSORB_COSINE}


# save follow-up ----------------------------------------------------------------------
def after_save(store: Any, namespace: str, *, now: float | None = None) -> dict:
    """What follows a save, once the saving turn is over: note-absorb, then topic-assign. Both are incremental
    (memory_checks), so nothing new means nothing done. Not run inside the turn: absorb deletes memory rows that
    the same turn's update/reconcile may still name."""
    return {"absorb": absorb(store, namespace, now=now), "assign": assign_all(store, namespace, now=now)}


class SaveFollowUp:
    """after_save on a daemon thread, woken at the end of a turn that saved or changed a memory (the
    ConversationIndexer shape: start at initialize, notify per turn, stop at shutdown)."""

    def __init__(self, store: Any, namespace: str) -> None:
        self.store = store
        self.namespace = namespace
        self.last_error: str | None = None
        self.runs = 0
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="neospike-save-follow-up", daemon=True)
        self._thread.start()

    def notify(self) -> None:
        self._idle.clear()
        self._wake.set()

    def wait_idle(self, timeout: float) -> bool:
        return self._idle.wait(timeout)

    def stop(self, timeout: float = _Tuning.FOLLOW_UP_JOIN_SECONDS) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            if self._stop.is_set():
                break
            self._wake.clear()
            try:
                after_save(self.store, self.namespace)
                self.runs += 1
                self.last_error = None
            except Exception as exc:  # Retried after the next saving turn, or by the daily embed-pending stage.
                self.last_error = f"{type(exc).__name__}: {exc}"
            if not self._wake.is_set():
                self._idle.set()
        self._idle.set()
