"""Notes exported as Markdown files (the browser writes them; the server only builds them)."""
from __future__ import annotations

import io
import json
import sqlite3
import zipfile

import pytest
from fastapi.testclient import TestClient

from neospike import md_export
from neospike.folders import FOLDERS_SCHEMA
from neospike.store import ITEM_LINKS_DDL, ITEMS_DDL
from neospike.tests.test_dashboard_notes import _create, client, home  # noqa: F401  (pytest fixtures)

CREATED = 1_762_732_800.0  # 2025-11-10T00:00:00Z


def _db() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(ITEMS_DDL + ITEM_LINKS_DDL + FOLDERS_SCHEMA)
    return connection


PART_ID_OFFSET = 1000  # the one part row of note N is item N + PART_ID_OFFSET


def _note(connection: sqlite3.Connection, note_id: int, title: str, body: str = "body", *, tags=(), deleted: bool = False) -> None:
    """A v22 document: its head row (the note id) and one part row holding the body, chained by a next link."""
    columns = "(id,type,owner,title,text,text_hash,project,tags_json,author,kind,embedding,created_at,updated_at,rev,deleted_at)"
    connection.execute(f"INSERT INTO items{columns} VALUES (?,'note','',?,'','',?,?,'agent','note',X'',?,?,2,?)",
                       (note_id, title, "proj", json.dumps(list(tags)), CREATED, CREATED + 60,
                        CREATED if deleted else None))
    connection.execute(f"INSERT INTO items{columns} VALUES (?,'note','',NULL,?,'',?,'[]','agent','note',X'',?,?,2,NULL)",
                       (note_id + PART_ID_OFFSET, body, "proj", CREATED, CREATED + 60))
    connection.execute("INSERT INTO item_links(src_id,dst_id,relation) VALUES (?,?,'next')",
                       (note_id, note_id + PART_ID_OFFSET))


@pytest.mark.parametrize("title, name", [
    ("a/b", "a_b"), ("x: y", "x_ y"), ('<>:"/\\|?*', "_________"), ("end. ", "end__"), ("CON", "_CON"),
    ("con.notes", "_con.notes"), ("...", "___"), ("한글 제목", "한글 제목"),
])
def test_safe_name_replaces_windows_invalid_characters(title: str, name: str) -> None:
    assert md_export.safe_name(title) == name


@pytest.mark.parametrize("path", ["../a.md", "a/../b.md", "/a.md", "C:/a.md", "a\\b.md", "a//b.md", "./a.md", "a/", ""])
def test_checked_path_rejects_paths_that_leave_the_folder(path: str) -> None:
    with pytest.raises(md_export.ExportPathError):
        md_export.checked_path(path)


def test_export_names_dirs_collisions_and_frontmatter() -> None:
    connection = _db()
    _note(connection, 1, "a/b", "# A\n\n[[x: y]] link\n", tags=["t1", "태그"])
    _note(connection, 2, "a_b")             # same file name as 'a/b' after sanitising
    _note(connection, 3, "x: y")
    _note(connection, 4, "TOPIC: tree", "tree body")
    _note(connection, 5, "A_B")             # case-insensitive clash (Windows/macOS)
    _note(connection, 6, "Filed", "in a folder")
    _note(connection, 7, "Gone", deleted=True)
    _note(connection, 8, "page of topic")  # a topic page whose title lacks the prefix
    connection.execute("INSERT INTO folders VALUES (1,'Work: 2026','#123456','human',?,NULL,NULL)", (CREATED,))
    connection.execute("INSERT INTO folder_items VALUES (1,'note',6)")

    files = md_export.export_files(connection, None, topic_ids={8})
    assert [item["path"] for item in files] == [
        "a_b.md", "a_b (note-2).md", "x_ y.md", "topics/TOPIC_ tree.md", "A_B (note-5).md", "Work_ 2026/Filed.md",
        "topics/page of topic.md"]
    first = files[0]["content"].split("\n")
    assert first[:11] == ["---", "id: 1", 'title: "a/b"', 'tags: ["t1", "태그"]', 'project: "proj"', "author: agent",
                          "rev: 2", "created: 2025-11-10T00:00:00Z", "updated: 2025-11-10T00:01:00Z",
                          "source: neospike export", "---"]
    assert files[0]["content"].endswith("---\n# A\n\n[[x: y]] link\n")  # body as stored

    picked = md_export.export_files(connection, [6, 7, 3, 6])
    assert [item["path"] for item in picked] == ["x_ y.md", "Work_ 2026/Filed.md"]  # trashed 7 skipped, 6 once


def test_zip_holds_the_same_files() -> None:
    files = [{"path": "topics/TOPIC_ tree.md", "content": "트리"}, {"path": "a.md", "content": "a"}]
    with zipfile.ZipFile(io.BytesIO(md_export.zip_bytes(files))) as archive:
        assert archive.namelist() == ["topics/TOPIC_ tree.md", "a.md"]
        assert archive.read("topics/TOPIC_ tree.md").decode("utf-8") == "트리"
    with pytest.raises(md_export.ExportPathError):
        md_export.zip_bytes([{"path": "../evil.md", "content": "x"}])


def test_export_routes_return_files_and_zip(client: TestClient) -> None:  # noqa: F811
    one = _create(client, "Plan: A/B", "# Plan\n\nFirst.")
    two = _create(client, "Plan_ A_B", "Second.")
    assert client.get("/notes").json()["total"] == 2

    every = client.get("/notes/export").json()
    assert [item["path"] for item in every] == ["Plan_ A_B.md", f"Plan_ A_B (note-{two['id']}).md"]
    assert every[0]["content"].startswith(f"---\nid: {one['id']}\ntitle: \"Plan: A/B\"\n")

    only = client.get("/notes/export", params={"ids": str(two["id"])}).json()
    assert [item["path"] for item in only] == ["Plan_ A_B.md"]  # no clash when exported alone
    assert client.get("/notes/export", params={"ids": "1,x"}).status_code == 422
    assert client.get("/notes/export", params={"ids": "0"}).status_code == 422

    response = client.get("/notes/export.zip")
    assert response.status_code == 200 and response.headers["content-type"] == "application/zip"
    assert "notes-export.zip" in response.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert sorted(archive.namelist()) == sorted(item["path"] for item in every)


def test_export_before_the_database_exists(client: TestClient) -> None:  # noqa: F811
    assert client.get("/notes/export").json() == []
    with zipfile.ZipFile(io.BytesIO(client.get("/notes/export.zip").content)) as archive:
        assert archive.namelist() == []
