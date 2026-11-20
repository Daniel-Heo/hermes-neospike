"""Folders of knowledge items (memories and notes).

- One level only; an item belongs to at most one folder (PRIMARY KEY(lane, item_id)).
- folder-delete moves the folder to the trash (deleted_at) and returns its members to unfiled.
- origin: 'human' or 'auto'. A human edit of an auto folder records origin='human', so automation
  proposes instead of editing it. Automation passes actor='auto' and keeps the origin.
- Member colour: the Okabe-Ito colours without yellow and green, first unused one; after the six,
  golden-angle hues that skip the yellow and green ranges (those belong to folder markers and the
  recent ring).
Writes take a MemoryStore connection (CLI single-writer path); reads take any connection.
"""
from __future__ import annotations

import colorsys
import re
import sqlite3
import time
from typing import Any, Iterable

ORIGINS = ("human", "auto")
ACTORS = ORIGINS
LANE_PREFIXES = {"mem": "memory", "note": "note"}
LANE_KEYS = {"memory": "mem", "note": "note"}
MEMBER_PALETTE = ("#E69F00", "#56B4E9", "#0072B2", "#D55E00", "#CC79A7", "#000000")
GOLDEN_ANGLE_DEGREES = 137.5
GOLDEN_START_DEGREES = 0.0
GOLDEN_SATURATION = 0.65
GOLDEN_LIGHTNESS = 0.50
GOLDEN_MAX_STEPS = 4096
# Hue ranges (degrees, inclusive) reserved for the folder marker (yellow) and the recent ring (green).
RESERVED_HUES = ((40.0, 65.0), (90.0, 160.0))
MAX_NAME_CHARS = 100
MAX_ITEMS_PER_MOVE = 10_000
_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_ITEM_KEY = re.compile(r"^(mem|note):(\d{1,19})$")

FOLDERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    color TEXT NOT NULL,
    created_by TEXT NOT NULL DEFAULT 'human' CHECK (created_by IN ('human','auto')),
    created_at REAL NOT NULL,
    updated_at REAL,
    deleted_at REAL
);
CREATE TABLE IF NOT EXISTS folder_items (
    folder_id INTEGER NOT NULL REFERENCES folders(id),
    item_type TEXT NOT NULL CHECK (item_type IN ('memory','note')),
    item_id INTEGER NOT NULL,
    PRIMARY KEY (item_type, item_id)
);
CREATE INDEX IF NOT EXISTS folder_items_folder ON folder_items(folder_id);
"""
FOLDER_TABLES = ("folders", "folder_items")


class FolderError(ValueError):
    code = "invalid"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": str(self), **self.details}


class FolderNotFound(FolderError):
    code = "not_found"


# -- colours ------------------------------------------------------------------------------------

def hue_degrees(color: str) -> float:
    red, green, blue = (int(color[index:index + 2], 16) / 255.0 for index in (1, 3, 5))
    hue, _, _ = colorsys.rgb_to_hls(red, green, blue)
    return hue * 360.0


def hue_reserved(hue: float) -> bool:
    return any(low <= hue <= high for low, high in RESERVED_HUES)


def _hsl_hex(hue: float) -> str:
    red, green, blue = colorsys.hls_to_rgb(hue / 360.0, GOLDEN_LIGHTNESS, GOLDEN_SATURATION)
    return "#{:02X}{:02X}{:02X}".format(round(red * 255), round(green * 255), round(blue * 255))


def next_color(used: Iterable[str]) -> str:
    """First unused palette colour, then golden-angle hues outside the reserved ranges."""
    taken = {color.upper() for color in used}
    for color in MEMBER_PALETTE:
        if color not in taken:
            return color
    for step in range(1, GOLDEN_MAX_STEPS + 1):
        hue = (GOLDEN_START_DEGREES + step * GOLDEN_ANGLE_DEGREES) % 360.0
        if hue_reserved(hue):
            continue
        candidate = _hsl_hex(hue)
        # Rounding can move a hue next to a boundary; check the stored colour, not the angle.
        if candidate not in taken and not hue_reserved(hue_degrees(candidate)):
            return candidate
    raise FolderError("No unused folder colour is left")


def validated_color(color: Any) -> str:
    if not isinstance(color, str) or not _COLOR.match(color):
        raise FolderError("Folder colour must be #RRGGBB")
    return color.upper()


def validated_name(name: Any) -> str:
    cleaned = " ".join(str(name or "").split())
    if not cleaned:
        raise FolderError("Folder name must not be empty")
    if len(cleaned) > MAX_NAME_CHARS:
        raise FolderError(f"Folder name is longer than {MAX_NAME_CHARS} characters")
    return cleaned


def parse_item_keys(text: Any) -> list[tuple[str, int]]:
    """'mem:1,note:3' -> [('memory', 1), ('note', 3)]; duplicates keep their first position."""
    keys = [part.strip() for part in str(text or "").split(",") if part.strip()]
    if not keys:
        raise FolderError("Give at least one item as mem:<id> or note:<id>")
    if len(keys) > MAX_ITEMS_PER_MOVE:
        raise FolderError(f"At most {MAX_ITEMS_PER_MOVE} items per move")
    parsed: dict[tuple[str, int], None] = {}
    for key in keys:
        match = _ITEM_KEY.match(key)
        if match is None or int(match.group(2)) < 1:
            raise FolderError(f"Invalid item '{key}': use mem:<id> or note:<id>")
        parsed[(LANE_PREFIXES[match.group(1)], int(match.group(2)))] = None
    return list(parsed)


def item_key(lane: str, item_id: int) -> str:
    return f"{LANE_KEYS[lane]}:{int(item_id)}"


# -- reads (any connection, mode=ro included) ---------------------------------------------------

def has_folder_tables(connection: sqlite3.Connection) -> bool:
    found = {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('folders','folder_items')")}
    return found == set(FOLDER_TABLES)


def _folder_dict(row: sqlite3.Row | tuple, count: int) -> dict:
    folder_id, name, color, origin, created_at, updated_at, deleted_at = tuple(row)
    return {"id": int(folder_id), "name": name, "color": color, "origin": origin, "created_at": created_at,
            "updated_at": updated_at, "deleted_at": deleted_at, "count": count}


_FOLDER_COLUMNS = "id,name,color,created_by,created_at,updated_at,deleted_at"
# Members whose item still exists (a deleted memory or trashed note no longer counts). Literal SQL over the v22
# items table (store.py imports this module, so its constants are not imported here).
_MEMORY_EXISTS = "EXISTS (SELECT 1 FROM items m WHERE m.id={id} AND m.type='memory')"
_NOTE_LIVE = ("EXISTS (SELECT 1 FROM items n WHERE n.id={id} AND n.type IN ('note','doc') AND n.title IS NOT NULL "
              "AND n.deleted_at IS NULL)")
_LIVE_MEMBER = (
    f"((fi.item_type='memory' AND {_MEMORY_EXISTS.format(id='fi.item_id')}) OR "
    f"(fi.item_type='note' AND {_NOTE_LIVE.format(id='fi.item_id')}))"
)


def list_folders(connection: sqlite3.Connection, *, include_deleted: bool = False) -> list[dict]:
    if not has_folder_tables(connection):
        return []
    counts = dict(connection.execute(
        f"SELECT fi.folder_id, COUNT(*) FROM folder_items fi WHERE {_LIVE_MEMBER} GROUP BY fi.folder_id").fetchall())
    where = "" if include_deleted else " WHERE deleted_at IS NULL"
    rows = connection.execute(f"SELECT {_FOLDER_COLUMNS} FROM folders{where} ORDER BY id").fetchall()
    return [_folder_dict(row, int(counts.get(row[0], 0))) for row in rows]


def folder_members(connection: sqlite3.Connection) -> dict[tuple[str, int], int]:
    """(lane, item_id) -> folder_id for live folders."""
    if not has_folder_tables(connection):
        return {}
    return {(lane, int(item_id)): int(folder_id) for folder_id, lane, item_id in connection.execute(
        "SELECT fi.folder_id, fi.item_type, fi.item_id FROM folder_items fi JOIN folders f ON f.id=fi.folder_id "
        "WHERE f.deleted_at IS NULL")}


# -- writes (MemoryStore connection; callers hold store._lock) ----------------------------------

def _live_folder(connection: sqlite3.Connection, folder_id: int) -> tuple:
    row = connection.execute(f"SELECT {_FOLDER_COLUMNS} FROM folders WHERE id=? AND deleted_at IS NULL",
                             (folder_id,)).fetchone()
    if row is None:
        raise FolderNotFound(f"Folder {folder_id} not found")
    return tuple(row)


def _touch(connection: sqlite3.Connection, folder_ids: Iterable[int], actor: str, now: float) -> None:
    """Bump updated_at; a human edit turns an auto folder into a human one."""
    for folder_id in sorted(set(folder_ids)):
        if actor == "human":
            connection.execute("UPDATE folders SET updated_at=?, created_by='human' WHERE id=?", (now, folder_id))
        else:
            connection.execute("UPDATE folders SET updated_at=? WHERE id=?", (now, folder_id))


def _checked_actor(actor: str) -> str:
    if actor not in ACTORS:
        raise FolderError(f"actor must be one of {list(ACTORS)}")
    return actor


def _live_colors(connection: sqlite3.Connection) -> list[str]:
    return [row[0] for row in connection.execute("SELECT color FROM folders WHERE deleted_at IS NULL")]


def create_folder(connection: sqlite3.Connection, name: Any, *, color: Any = None, actor: str = "human") -> dict:
    """origin follows the actor: a folder made by automation is 'auto'."""
    origin = _checked_actor(actor)
    cleaned = validated_name(name)
    with connection:
        chosen = validated_color(color) if color else next_color(_live_colors(connection))
        now = time.time()
        cursor = connection.execute(
            "INSERT INTO folders(name,color,created_by,created_at,updated_at) VALUES (?,?,?,?,?)",
            (cleaned, chosen, origin, now, now))
        return _folder_dict(_live_folder(connection, int(cursor.lastrowid)), 0)


def update_folder(connection: sqlite3.Connection, folder_id: int, *, name: Any = None, color: Any = None,
                  actor: str = "human") -> dict:
    actor = _checked_actor(actor)
    if name is None and color is None:
        raise FolderError("Give a new name or colour")
    with connection:
        _live_folder(connection, folder_id)
        if name is not None:
            connection.execute("UPDATE folders SET name=? WHERE id=?", (validated_name(name), folder_id))
        if color is not None:
            connection.execute("UPDATE folders SET color=? WHERE id=?", (validated_color(color), folder_id))
        _touch(connection, [folder_id], actor, time.time())
    return _one(connection, folder_id)


def _one(connection: sqlite3.Connection, folder_id: int) -> dict:
    for folder in list_folders(connection, include_deleted=True):
        if folder["id"] == folder_id:
            return folder
    raise FolderNotFound(f"Folder {folder_id} not found")


def _missing_items(connection: sqlite3.Connection, items: list[tuple[str, int]]) -> list[str]:
    missing = []
    for lane, item_id in items:
        if lane == "memory":
            found = connection.execute(f"SELECT 1 WHERE {_MEMORY_EXISTS.format(id='?')}", (item_id,)).fetchone()
        else:
            found = connection.execute(f"SELECT 1 WHERE {_NOTE_LIVE.format(id='?')}", (item_id,)).fetchone()
        if found is None:
            missing.append(item_key(lane, item_id))
    return missing


def move_items(connection: sqlite3.Connection, folder_id: int | None, items: list[tuple[str, int]], *,
               actor: str = "human") -> dict:
    """Put items into one folder (None: unfiled). An item already in another folder leaves it."""
    actor = _checked_actor(actor)
    with connection:
        if folder_id is not None:
            _live_folder(connection, folder_id)
        missing = _missing_items(connection, items)
        if missing:
            raise FolderNotFound("Items not found", items=missing)
        previous: set[int] = set()
        moved = 0
        for lane, item_id in items:
            row = connection.execute("SELECT folder_id FROM folder_items WHERE item_type=? AND item_id=?",
                                     (lane, item_id)).fetchone()
            current = None if row is None else int(row[0])
            if current == folder_id:
                continue
            if current is not None:
                previous.add(current)
                # Explicit DELETE (not INSERT OR REPLACE): the delete trigger records a tombstone.
                connection.execute("DELETE FROM folder_items WHERE item_type=? AND item_id=?", (lane, item_id))
            if folder_id is not None:
                connection.execute("INSERT INTO folder_items(folder_id,item_type,item_id) VALUES (?,?,?)",
                                   (folder_id, lane, item_id))
            moved += 1
        touched = previous | ({folder_id} if folder_id is not None else set())
        if moved:
            _touch(connection, touched, actor, time.time())
    return {"folder_id": folder_id, "moved": moved, "items": [item_key(lane, item_id) for lane, item_id in items],
            "left": sorted(previous)}


def delete_folder(connection: sqlite3.Connection, folder_id: int, *, actor: str = "human") -> dict:
    """Trash the folder; its members become unfiled. No item is deleted."""
    actor = _checked_actor(actor)
    with connection:
        _live_folder(connection, folder_id)
        released = connection.execute("DELETE FROM folder_items WHERE folder_id=?", (folder_id,)).rowcount
        now = time.time()
        connection.execute("UPDATE folders SET deleted_at=? WHERE id=?", (now, folder_id))
        _touch(connection, [folder_id], actor, now)
    return {"id": folder_id, "deleted": True, "released": int(released)}
