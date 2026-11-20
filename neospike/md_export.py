"""Notes as Markdown files for use elsewhere.

The server only builds file contents; the browser writes them into a folder the user picks
(or downloads them as one zip). Nothing here writes to disk or to the database.
"""
from __future__ import annotations

import io
import json
import re
import sqlite3
import time
import zipfile
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Iterable

from neospike import folders as folder_family
from neospike.notes import note_body
from neospike.store import NOTE_HEAD_SQL

EXPORT_SOURCE = "neospike export"
TOPIC_FOLDER = "topics"
TOPIC_TITLE_PREFIX = "TOPIC: "
MARKDOWN_SUFFIX = ".md"
EMPTY_NAME = "_"
REPLACEMENT = "_"
ZIP_NAME = "notes-export.zip"
_SQL_BATCH = 500
# Characters Windows refuses in a file name, plus control characters; trailing dots and spaces are dropped by Windows.
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_TRAILING = re.compile(r"[. ]+$")
# Windows device names are reserved even with an extension (CON.md).
_RESERVED_STEMS = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{n}" for n in range(1, 10)), *(f"LPT{n}" for n in range(1, 10))})
_NOTE_COLUMNS = "id,title,tags_json,project,author,rev,created_at,updated_at"


class ExportPathError(ValueError):
    """A built path would leave the chosen folder."""


def safe_name(text: str) -> str:
    """One path component that is valid on Windows, macOS and Linux."""
    name = _INVALID_CHARS.sub(REPLACEMENT, str(text))
    trailing = _TRAILING.search(name)
    if trailing:
        name = name[:trailing.start()] + REPLACEMENT * len(trailing.group())
    if not name:
        return EMPTY_NAME
    if name.split(".", 1)[0].upper() in _RESERVED_STEMS:
        name = REPLACEMENT + name
    return name


def checked_path(path: str) -> str:
    """Reject absolute paths, drive letters, backslashes and '.'/'..' components."""
    if not path or "\\" in path or ":" in path or path.startswith("/"):
        raise ExportPathError(f"Unsafe export path: {path!r}")
    parts = PurePosixPath(path).parts
    if not parts or any(part in ("", ".", "..") for part in parts) or "/".join(parts) != path:
        raise ExportPathError(f"Unsafe export path: {path!r}")
    return path


def _iso(epoch: float | None) -> str:
    if epoch is None:
        return ""
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _quoted(value: str) -> str:
    # A JSON string is a valid YAML double-quoted scalar.
    return json.dumps(value, ensure_ascii=False)


def note_markdown(note: sqlite3.Row | dict) -> str:
    tags = json.loads(note["tags_json"])
    lines = [
        "---",
        f"id: {int(note['id'])}",
        f"title: {_quoted(note['title'])}",
        "tags: [" + ", ".join(_quoted(tag) for tag in tags) + "]",
        f"project: {_quoted(note['project'] or '')}",
        f"author: {note['author']}",
        f"rev: {int(note['rev'])}",
        f"created: {_iso(note['created_at'])}",
        f"updated: {_iso(note['updated_at'])}",
        f"source: {EXPORT_SOURCE}",
        "---",
        "",
    ]
    return "\n".join(lines) + note["body"]


def _with_body(connection: sqlite3.Connection, row: sqlite3.Row) -> dict:
    record = {key: row[key] for key in row.keys()}
    record["body"] = note_body(connection, int(row["id"]), str(row["author"] or ""))
    return record


def _live_notes(connection: sqlite3.Connection, ids: list[int] | None) -> list[dict]:
    live = f"FROM items WHERE {NOTE_HEAD_SQL} AND deleted_at IS NULL"
    if ids is None:
        return [_with_body(connection, row) for row in
                connection.execute(f"SELECT {_NOTE_COLUMNS} {live} ORDER BY id").fetchall()]
    rows: list[dict] = []
    unique = sorted(set(ids))
    for start in range(0, len(unique), _SQL_BATCH):
        batch = unique[start:start + _SQL_BATCH]
        rows.extend(_with_body(connection, row) for row in connection.execute(
            f"SELECT {_NOTE_COLUMNS} {live} AND id IN ({','.join('?' * len(batch))}) ORDER BY id",
            batch,
        ).fetchall())
    return rows


def _folder_dirs(connection: sqlite3.Connection) -> dict[int, str]:
    """note id -> sanitized folder name, for notes filed in a live folder."""
    names = {folder["id"]: safe_name(folder["name"]) for folder in folder_family.list_folders(connection)}
    return {item_id: names[folder_id] for (lane, item_id), folder_id in folder_family.folder_members(connection).items()
            if lane == "note" and folder_id in names}


def _directory(note: sqlite3.Row, topic_ids: set[int], folder_dirs: dict[int, str]) -> str:
    note_id = int(note["id"])
    if note_id in topic_ids or str(note["title"]).startswith(TOPIC_TITLE_PREFIX):
        return TOPIC_FOLDER
    return folder_dirs.get(note_id, "")


def export_files(connection: sqlite3.Connection, ids: list[int] | None = None,
                 topic_ids: Iterable[int] = ()) -> list[dict[str, str]]:
    """[{path, content}] for live notes (all of them when ids is None), oldest id first.

    path is '<dir>/<title>.md'; a name that clashes (case-insensitively, as on Windows and macOS)
    with an earlier note's gets ' (note-<id>)'.
    """
    rows = _live_notes(connection, ids)
    topics = {int(note_id) for note_id in topic_ids}
    folder_dirs = _folder_dirs(connection) if folder_family.has_folder_tables(connection) else {}
    taken: set[str] = set()
    files: list[dict[str, str]] = []
    for note in rows:
        directory = _directory(note, topics, folder_dirs)
        prefix = directory + "/" if directory else ""
        stem = safe_name(note["title"])
        path = prefix + stem + MARKDOWN_SUFFIX
        if path.casefold() in taken:
            stem = f"{stem} (note-{int(note['id'])})"
            path = prefix + stem + MARKDOWN_SUFFIX
        while path.casefold() in taken:  # only if another title already reads '<title> (note-<id>)'
            stem += REPLACEMENT
            path = prefix + stem + MARKDOWN_SUFFIX
        taken.add(path.casefold())
        files.append({"path": checked_path(path), "content": note_markdown(note)})
    return files


def zip_bytes(files: list[dict[str, str]]) -> bytes:
    buffer = io.BytesIO()
    stamp = time.localtime()[:6]
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in files:
            info = zipfile.ZipInfo(checked_path(item["path"]), date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, item["content"].encode("utf-8"))
    return buffer.getvalue()
