"""Sphere map of memories and notes. Read-only: takes a mode=ro connection.

Layout: every item vector (memory embedding; note = normalized mean of its chunk vectors) is
a unit vector (stored vectors are normalized at write), centred, reduced to 3 dimensions with numpy PCA (SVD), and the 3-d point is
normalized onto the unit sphere. Axis signs are fixed (largest-|value| loading of each component
is positive), so the same data always gives the same picture.
Folder position: the mean of its members' full-dimension vectors, projected with the same PCA
(same mean and axes) and normalized. The projection is affine, so this equals the mean of the
members' un-normalized projections; the layout keeps those and needs no numpy to place folders.
Unit 3-d coordinates are never averaged.
Neighbours: top 5 by full-dimension cosine.
Cache: the layout (keys, coordinates, projections, neighbours) is keyed by a sha256 of the
item keys and raw float32 vector bytes, hashed without numpy. It lives in memory and, with a
cache_dir, in `map-<hash>.json`; a hit reads the file and never imports numpy. Other map files are
deleted when a new one is written. Folders, links, titles and the recent flag are read every call.
Topic points: one point per live topic with members on the map, placed like a folder marker (mean
of the members' projections, normalized); read every call, never cached, so a member change moves it next time.
Folder recency: a folder's use time is the latest max(accessed_at, updated_at) of its members;
within map_recent_days -> recent, within map_stale_days -> mid, otherwise (or no members) old.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import folders as folder_family
from .store import (ITEM_MEMORY, NOTE_CHUNKS_SQL, NOTE_HEAD_SQL, RELATION_TOPIC, note_head_sql, topic_handles,
                    vector_present_sql)
from .config import MAP_RECENT_DAYS_DEFAULT, MAP_STALE_DAYS_DEFAULT

if TYPE_CHECKING:  # numpy is imported only where the layout is computed (cache miss)
    import numpy as np

MAP_MEMORY_KINDS = ("memory_item", "durable")  # the dashboard Memory tab's items
TOPIC_LANE = "topic"  # topic points; key "topic:<id>"
NEIGHBOR_COUNT = 5
PCA_DIMENSIONS = 3
TITLE_CHARS = 80
DEFAULT_RECENT_DAYS = MAP_RECENT_DAYS_DEFAULT
DEFAULT_STALE_DAYS = MAP_STALE_DAYS_DEFAULT
RECENCY_RECENT = "recent"
RECENCY_MID = "mid"
RECENCY_OLD = "old"
SECONDS_PER_DAY = 86400.0
NEIGHBOR_BLOCK_ROWS = 512
ZERO_NORM = 1e-12
FALLBACK_POINT = (0.0, 0.0, 1.0)
CACHE_ENTRIES = 4
TASK_TITLE_PREFIX = "TASK_"
FLOAT32_BYTES = 4
HASH_VERSION = b"neospike-map/v1\n"
CACHE_FORMAT = 1
CACHE_DIRECTORY = "map-cache"  # never ".cache": that is the embedding daemon's private folder
CACHE_FILE_PREFIX = "map-"
CACHE_FILE_SUFFIX = ".json"
# A memory is superseded while its items.invalid_at is set.
INVALID_COLUMN = "invalid_at"
# Drawn links: only refers and supersedes rows of item_links are links on the map.
LINK_RELATIONS = ("refers", "supersedes")


@dataclass(frozen=True)
class _Item:
    key: str
    lane: str
    item_id: int
    kind: str
    title: str
    updated_at: float
    accessed_at: float


@dataclass(frozen=True)
class _Inputs:
    """Map candidates in map order (memories by id, then notes by id) and their raw float32 vectors."""
    items: list[_Item]
    blobs: list[tuple[bytes, ...]]  # one vector for a memory, the chunk vectors (by ord) for a note
    dimensions: int
    digest: str


@dataclass(frozen=True)
class Geometry:
    mean: "np.ndarray"        # (d,) mean of the normalized item vectors
    axes: "np.ndarray"        # (3, d) PCA components, sign-fixed
    coords: "np.ndarray"      # (n, 3) unit vectors
    neighbors: "np.ndarray"   # (n, k) item indices, best first


# Layout dict (also the cache file body): format, hash, keys, coords, projected, neighbors (keys).
_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def recency(used_at: float | None, now: float, recent_window_days: float, stale_window_days: float) -> str:
    """recent / mid / old for a use time (seconds); None (no members) is old."""
    if used_at is None:
        return RECENCY_OLD
    age = now - used_at
    if age <= recent_window_days * SECONDS_PER_DAY:
        return RECENCY_RECENT
    if age <= stale_window_days * SECONDS_PER_DAY:
        return RECENCY_MID
    return RECENCY_OLD


def cache_directory(home: Path) -> Path:
    """<HERMES_HOME>/neospike/map-cache, where the map files live."""
    return Path(home) / "neospike" / CACHE_DIRECTORY


def cache_file(cache_dir: Path, digest: str) -> Path:
    return Path(cache_dir) / f"{CACHE_FILE_PREFIX}{digest}{CACHE_FILE_SUFFIX}"


def empty_map(recent_window_days: float = DEFAULT_RECENT_DAYS,
              stale_window_days: float = DEFAULT_STALE_DAYS) -> dict[str, Any]:
    return {"points": [], "folders": [], "hash": None, "recent_days": recent_window_days,
            "stale_days": stale_window_days, "cached": False}


def _clip(text: str) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= TITLE_CHARS else flat[:TITLE_CHARS - 1] + "…"


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _dimensions(connection: sqlite3.Connection) -> int | None:
    row = connection.execute("SELECT value FROM meta_entries WHERE key='embedding_dimensions'").fetchone()
    return int(row[0]) if row is not None else None


def _has_table(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _memory_rows(connection: sqlite3.Connection, vector_bytes: int) -> tuple[list[_Item], list[tuple[bytes, ...]]]:
    kinds = ",".join("?" * len(MAP_MEMORY_KINDS))
    items, blobs = [], []
    for row in connection.execute(
        f"SELECT id,text,embedding,updated_at,accessed_at FROM items WHERE type='{ITEM_MEMORY}' AND kind IN ({kinds}) "
        f"AND {vector_present_sql()} ORDER BY id", MAP_MEMORY_KINDS,
    ):
        blob = bytes(row[2] or b"")
        if len(blob) != vector_bytes:
            continue
        items.append(_Item(f"mem:{int(row[0])}", "memory", int(row[0]), "memory", _clip(row[1]),
                           float(row[3] or 0.0), float(row[4] or 0.0)))
        blobs.append((blob,))
    return items, blobs


def _note_rows(connection: sqlite3.Connection, vector_bytes: int) -> tuple[list[_Item], list[tuple[bytes, ...]]]:
    """Notes with at least one chunk of the map's dimension. A note whose chunk sum is zero is
    dropped later, by the layout (that needs the vector arithmetic)."""
    chunks: dict[int, list[bytes]] = {}
    for note_id, blob in connection.execute(
        f"SELECT c.note_id,c.embedding FROM {NOTE_CHUNKS_SQL} c JOIN items n ON n.id=c.note_id "
        "WHERE n.deleted_at IS NULL ORDER BY c.note_id,c.ord"
    ):
        blob = bytes(blob or b"")
        if len(blob) == vector_bytes:
            chunks.setdefault(int(note_id), []).append(blob)
    items, blobs = [], []
    for row in connection.execute(
        f"SELECT id,title,author,updated_at,accessed_at FROM items WHERE {NOTE_HEAD_SQL} AND deleted_at IS NULL "
        "ORDER BY id"
    ):
        parts = chunks.get(int(row[0]))
        if not parts:
            continue
        kind = "task" if str(row[1]).startswith(TASK_TITLE_PREFIX) else str(row[2])
        items.append(_Item(f"note:{int(row[0])}", "note", int(row[0]), kind, str(row[1]),
                           float(row[3] or 0.0), float(row[4] or 0.0)))
        blobs.append(tuple(parts))
    return items, blobs


def input_hash(dimensions: int, keys: list[str], blobs: list[tuple[bytes, ...]]) -> str:
    """sha256 of the keys and raw vector bytes in map order; no numpy."""
    digest = hashlib.sha256(HASH_VERSION)
    digest.update(b"%d\n" % dimensions)
    for key, group in zip(keys, blobs):
        digest.update(b"%s\n%d\n" % (key.encode("utf-8"), len(group)))
        for blob in group:
            digest.update(blob)
    return digest.hexdigest()


def _read_inputs(connection: sqlite3.Connection, dimensions: int) -> _Inputs:
    vector_bytes = dimensions * FLOAT32_BYTES
    memory_points, memory_blobs = _memory_rows(connection, vector_bytes)
    note_items, note_blobs = _note_rows(connection, vector_bytes)
    items = memory_points + note_items
    blobs = memory_blobs + note_blobs
    return _Inputs(items, blobs, dimensions, input_hash(dimensions, [item.key for item in items], blobs))


def pca_axes(matrix: "np.ndarray") -> tuple["np.ndarray", "np.ndarray"]:
    """(mean, 3×d sign-fixed principal axes) of the rows. Missing components (rank < 3) are zero rows."""
    import numpy as np
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axes = np.zeros((PCA_DIMENSIONS, matrix.shape[1]))
    count = min(PCA_DIMENSIONS, vt.shape[0])
    axes[:count] = vt[:count]
    for index in range(count):
        pivot = int(np.argmax(np.abs(axes[index])))
        if axes[index, pivot] < 0:
            axes[index] = -axes[index]
    return mean, axes


def to_sphere(vectors: "np.ndarray", mean: "np.ndarray", axes: "np.ndarray") -> "np.ndarray":
    """Project full-dimension rows with the map's PCA and normalize onto the unit sphere."""
    import numpy as np
    projected = (np.atleast_2d(vectors) - mean) @ axes.T
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    unit = np.divide(projected, norms, out=np.zeros_like(projected), where=norms > ZERO_NORM)
    unit[norms[:, 0] <= ZERO_NORM] = FALLBACK_POINT
    return unit


def top_neighbors(unit_rows: "np.ndarray", count: int = NEIGHBOR_COUNT) -> "np.ndarray":
    """Indices of the `count` most cosine-similar other rows (score desc, index asc), blockwise."""
    import numpy as np
    total = unit_rows.shape[0]
    width = min(count, max(total - 1, 0))
    result = np.zeros((total, width), dtype=np.int64)
    if width == 0:
        return result
    for start in range(0, total, NEIGHBOR_BLOCK_ROWS):
        stop = min(start + NEIGHBOR_BLOCK_ROWS, total)
        scores = unit_rows[start:stop] @ unit_rows.T
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        # The width-th largest score of each row; every candidate at or above it is sorted exactly.
        cutoff = np.partition(scores, total - width, axis=1)[:, total - width]
        for offset in range(stop - start):
            row = scores[offset]
            candidates = np.flatnonzero(row >= cutoff[offset])
            # lexsort: last key is primary -> score desc, then index asc for ties.
            order = candidates[np.lexsort((candidates, -row[candidates]))]
            result[start + offset] = order[:width]
    return result


def compute_geometry(unit_rows: "np.ndarray") -> Geometry:
    """Rows are unit vectors already: stored memory vectors (normalize_rows at write) and normalized note means."""
    mean, axes = pca_axes(unit_rows)
    return Geometry(mean, axes, to_sphere(unit_rows, mean, axes), top_neighbors(unit_rows))


def _compute_layout(inputs: _Inputs) -> dict[str, Any]:
    """Cache miss: the only path that imports numpy."""
    import numpy as np
    keys, vectors = [], []
    for item, group in zip(inputs.items, inputs.blobs):
        total = np.frombuffer(group[0], dtype=np.float32).astype(np.float64)
        if item.lane == "note":
            for blob in group[1:]:
                total += np.frombuffer(blob, dtype=np.float32)
            norm = float(np.linalg.norm(total))
            if not norm > ZERO_NORM or not np.isfinite(norm):
                continue
            total = total / norm
        keys.append(item.key)
        vectors.append(total)
    layout: dict[str, Any] = {"format": CACHE_FORMAT, "hash": inputs.digest, "keys": keys,
                              "coords": [], "projected": [], "neighbors": []}
    if not vectors:
        return layout
    unit_rows = np.vstack(vectors)
    geometry = compute_geometry(unit_rows)
    layout["coords"] = geometry.coords.tolist()
    layout["projected"] = ((unit_rows - geometry.mean) @ geometry.axes.T).tolist()
    layout["neighbors"] = [[keys[other] for other in row] for row in geometry.neighbors.tolist()]
    return layout


def _valid_layout(layout: Any, inputs: _Inputs) -> bool:
    if not isinstance(layout, dict) or layout.get("format") != CACHE_FORMAT or layout.get("hash") != inputs.digest:
        return False
    keys = layout.get("keys")
    if not isinstance(keys, list):
        return False
    count = len(keys)
    return all(isinstance(layout.get(name), list) and len(layout[name]) == count
               for name in ("coords", "projected", "neighbors")) and set(keys) <= {item.key for item in inputs.items}


def _read_cache_file(path: Path, inputs: _Inputs) -> dict[str, Any] | None:
    try:
        layout = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return layout if _valid_layout(layout, inputs) else None


def _write_cache_file(cache_dir: Path, layout: dict[str, Any]) -> None:
    """Atomic write of map-<hash>.json, then removal of every other map file. Best effort: a
    read-only or full disk only costs the next call a recompute."""
    target = cache_file(cache_dir, layout["hash"])
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{CACHE_FILE_PREFIX}", suffix=".tmp", dir=cache_dir)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(layout, handle, separators=(",", ":"))
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        for stale in cache_dir.glob(f"{CACHE_FILE_PREFIX}*{CACHE_FILE_SUFFIX}"):
            if stale.name != target.name:
                stale.unlink(missing_ok=True)
    except OSError:
        return


def _remember(layout: dict[str, Any]) -> None:
    with _CACHE_LOCK:
        _CACHE[layout["hash"]] = layout
        _CACHE.move_to_end(layout["hash"])
        while len(_CACHE) > CACHE_ENTRIES:
            _CACHE.popitem(last=False)


def _layout(inputs: _Inputs, cache_dir: Path | None) -> tuple[dict[str, Any], bool]:
    """(layout, cached): memory, then map-<hash>.json, then compute (and write the file)."""
    with _CACHE_LOCK:
        cached = _CACHE.get(inputs.digest)
        if cached is not None:
            _CACHE.move_to_end(inputs.digest)
            return cached, True
    if cache_dir is not None:
        stored = _read_cache_file(cache_file(cache_dir, inputs.digest), inputs)
        if stored is not None:
            _remember(stored)
            return stored, True
    layout = _compute_layout(inputs)
    _remember(layout)
    if cache_dir is not None:
        _write_cache_file(Path(cache_dir), layout)
    return layout, False


def clear_cache() -> None:
    """Drop the in-memory layouts (map files on disk stay)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def superseded_memory_ids(connection: sqlite3.Connection) -> set[int]:
    if not _has_table(connection, "items"):
        return set()
    return {int(row[0]) for row in connection.execute(
        f"SELECT id FROM items WHERE type='{ITEM_MEMORY}' AND {INVALID_COLUMN} IS NOT NULL")}


def _link_graph(connection: sqlite3.Connection, keys: set[str]) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Undirected links between map items, and backlink counts (distinct live sources, self excluded)."""
    adjacency: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    if not _has_table(connection, "item_links"):
        return adjacency, {}
    for src_type, src_id, dst_type, dst_id in connection.execute(
        f"SELECT CASE s.type WHEN '{ITEM_MEMORY}' THEN 'memory' ELSE 'note' END,l.src_id,"
        f"CASE d.type WHEN '{ITEM_MEMORY}' THEN 'memory' ELSE 'note' END,l.dst_id FROM item_links l "
        "JOIN items s ON s.id=l.src_id JOIN items d ON d.id=l.dst_id "
        f"WHERE (s.type='{ITEM_MEMORY}' OR ({note_head_sql('s')} AND s.deleted_at IS NULL))"
        f" AND l.relation IN ({','.join('?' * len(LINK_RELATIONS))}) ORDER BY l.id",
        LINK_RELATIONS,
    ):
        source = folder_family.item_key(src_type, src_id)
        target = folder_family.item_key(dst_type, dst_id)
        if source == target:
            continue
        sources.setdefault(target, set()).add(source)
        if source in keys and target in keys:
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)
    return adjacency, {key: len(value) for key, value in sources.items()}


def _point(item: _Item, xyz: list[float], neighbors: list[str], links: list[str], backlinks: int,
           folder_id: int | None, recent: bool, superseded: bool) -> dict[str, Any]:
    return {
        "key": item.key, "id": item.item_id, "lane": item.lane, "kind": item.kind, "title": item.title,
        "folder_id": folder_id, "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]),
        "backlinks": backlinks, "neighbors": neighbors, "links": links, "recent": recent,
        "superseded": superseded, "updated_at": item.updated_at,
    }


def build_map(connection: sqlite3.Connection, *, recent_window_days: float = DEFAULT_RECENT_DAYS,
              stale_window_days: float = DEFAULT_STALE_DAYS, now: float | None = None,
              cache_dir: Path | None = None) -> dict[str, Any]:
    """The /map payload. Never writes the database: run it on a mode=ro connection.
    cache_dir: where map-<hash>.json is kept (None: in-memory cache only)."""
    started = time.perf_counter()
    now = time.time() if now is None else now
    empty = empty_map(recent_window_days, stale_window_days)
    windows = (now, recent_window_days, stale_window_days)
    if not _has_table(connection, "items") or not _has_table(connection, "meta_entries"):
        return {**empty, "folders": _folder_entries(connection, [], {}, None, windows)}
    dimensions = _dimensions(connection)
    if dimensions is None:
        return empty
    inputs = _read_inputs(connection, dimensions)
    if not inputs.items:
        return {**empty, "folders": _folder_entries(connection, [], {}, None, windows)}
    layout, cached = _layout(inputs, cache_dir)
    rows = {key: index for index, key in enumerate(layout["keys"])}
    items = [item for item in inputs.items if item.key in rows]
    keys = set(rows)
    members = folder_family.folder_members(connection)
    adjacency, backlink_counts = _link_graph(connection, keys)
    superseded = superseded_memory_ids(connection)
    window = recent_window_days * SECONDS_PER_DAY
    points = []
    for item in items:
        row = rows[item.key]
        points.append(_point(
            item, layout["coords"][row], list(layout["neighbors"][row]),
            sorted(adjacency.get(item.key, ())), backlink_counts.get(item.key, 0),
            members.get((item.lane, item.item_id)),
            now - max(item.accessed_at, item.updated_at) <= window,
            item.lane == "memory" and item.item_id in superseded,
        ))
    projected = [layout["projected"][rows[item.key]] for item in items]
    points.extend(topic_points(connection, items, projected))
    return {"points": points, "folders": _folder_entries(connection, items, members, projected, windows),
            "hash": inputs.digest, "recent_days": recent_window_days, "stale_days": stale_window_days,
            "cached": cached, "seconds": round(time.perf_counter() - started, 4)}


def _sphere_point(vector: list[float]) -> list[float]:
    """A 3-d projection onto the unit sphere (FALLBACK_POINT at the origin), like to_sphere."""
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm > ZERO_NORM:
        return list(FALLBACK_POINT)
    return [value / norm for value in vector]


def topic_key(topic_id: int) -> str:
    return f"{TOPIC_LANE}:{int(topic_id)}"



def topic_points(connection: sqlite3.Connection, items: list[_Item], projected: list[list[float]]) -> list[dict[str, Any]]:
    """One point per live topic, at the projected mean of its member memories on the map (the folder
    marker rule, no numpy). A topic with no member on the map gets no point. links stay empty here; the
    dashboard route fills them from the shared edge set."""
    if not _has_table(connection, "item_links"):
        return []
    handles = topic_handles(connection)
    row_of = {item.item_id: index for index, item in enumerate(items) if item.lane == "memory"}
    rows_by_topic: dict[int, list[int]] = {}
    for memory_id, name in connection.execute(
            f"SELECT src_id,label FROM item_links WHERE relation='{RELATION_TOPIC}' ORDER BY src_id"):
        row = row_of.get(int(memory_id))
        if row is not None:
            rows_by_topic.setdefault(handles[str(name)], []).append(row)
    # A topic's time: the newest of its links and its members' updated_at (no stored topic state, v22).
    changed = {str(row[0]): float(row[1]) for row in connection.execute(
        "SELECT l.label, MAX(MAX(l.created_at), COALESCE(MAX(m.updated_at), 0)) FROM item_links l "
        f"LEFT JOIN items m ON m.id=l.src_id WHERE l.relation='{RELATION_TOPIC}' GROUP BY l.label")}
    points = []
    for name, topic_id in handles.items():  # creation order
        updated_at = changed.get(name)
        rows = rows_by_topic.get(int(topic_id))
        if not rows:
            continue
        mean = [sum(projected[row][axis] for row in rows) / len(rows) for axis in range(PCA_DIMENSIONS)]
        xyz = _sphere_point(mean)
        points.append({
            "key": topic_key(topic_id), "id": int(topic_id), "lane": TOPIC_LANE, "kind": TOPIC_LANE, "title": str(name),
            "folder_id": None, "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]), "backlinks": 0,
            "neighbors": [], "links": [], "recent": False, "superseded": False, "updated_at": updated_at,
            "count": len(rows), "topic": True,
        })
    return points


def _folder_entries(connection: sqlite3.Connection, items: list[_Item], members: dict[tuple[str, int], int],
                    projected: list[list[float]] | None,
                    windows: tuple[float, float, float]) -> list[dict[str, Any]]:
    """Live folders with the count of their items on the map, the projected member mean and the recency.
    projected[i] is item i's un-normalized PCA projection; their mean is the projection of the vector mean."""
    now, recent_window_days, stale_window_days = windows
    rows_by_folder: dict[int, list[int]] = {}
    for index, item in enumerate(items):
        folder_id = members.get((item.lane, item.item_id))
        if folder_id is not None:
            rows_by_folder.setdefault(folder_id, []).append(index)
    entries = []
    for folder in folder_family.list_folders(connection):
        rows = rows_by_folder.get(folder["id"], [])
        used_at = max((max(items[row].accessed_at, items[row].updated_at) for row in rows), default=None)
        entry = {"id": folder["id"], "name": folder["name"], "color": folder["color"], "origin": folder["origin"],
                 "count": len(rows), "x": None, "y": None, "z": None, "used_at": used_at,
                 "recency": recency(used_at, now, recent_window_days, stale_window_days)}
        if rows and projected is not None:
            mean = [sum(projected[row][axis] for row in rows) / len(rows) for axis in range(PCA_DIMENSIONS)]
            xyz = _sphere_point(mean)
            entry.update(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        entries.append(entry)
    return entries
