"""독립 실행 CLI의 권한 범위: 작업 디렉터리가 기준이며 호출자가 알려주는 세션/PWD는 권한이 아니다."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from neospike import cli
from neospike.config import EMBEDDING_DIMENSIONS, NeospikeConfig
from neospike.store import MemoryStore

OWNER = "default:local"
FOREIGN_OWNER = "other:local"
UNKNOWN_ID_OFFSET = 100


class _ConstantEmbedder:
    """모든 질의를 같은 단위 벡터로 만든다(검색 범위만 시험하려는 용도)."""

    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, *args, **kwargs):
        pass

    def embed_query(self, text):
        return [1.0] + [0.0] * (self.dimensions - 1)


@dataclass
class CliWorld:
    home: Path
    db: Path
    proj_a: Path
    proj_b: Path
    ids: dict


def _insert(store, owner, label, scope, project):
    vector = _ConstantEmbedder().embed_query("approval")
    added = store.add(
        owner, f"The {label} project requires the copper approval mark.", vector,
        source="agent:session-a", kind="memory_item", project=project,
        metadata={"scope": scope, "session_id": "session-a"},
    )
    assert added
    return store._conn.execute("SELECT max(id) FROM items WHERE type='memory'").fetchone()[0]


@pytest.fixture
def world(tmp_path, monkeypatch) -> CliWorld:
    home = tmp_path / "cli-home"
    proj_a, proj_b = tmp_path / "alpha", tmp_path / "beta"
    proj_a.mkdir()
    proj_b.mkdir()
    NeospikeConfig().save(home)
    db = home / "neospike" / "memory.db"
    store = MemoryStore(db, dimensions=EMBEDDING_DIMENSIONS)
    layout = (
        ("in_alpha", OWNER, "project", str(proj_a.resolve())),
        ("in_beta", OWNER, "project", str(proj_b.resolve())),
        ("in_session", OWNER, "session", ""),
        ("everywhere", OWNER, "global", ""),
        ("foreign", FOREIGN_OWNER, "global", ""),
    )
    ids = {label: _insert(store, owner, label, scope, project) for label, owner, scope, project in layout}
    store.close()
    # state.db에 세션이 기록되어 있어도 그것이 CLI의 권한 근거는 아니다.
    with closing(sqlite3.connect(home / "state.db")) as conn, conn:
        conn.execute("CREATE TABLE sessions(id TEXT PRIMARY KEY, cwd TEXT, git_repo_root TEXT)")
        conn.execute("INSERT INTO sessions VALUES (?,?,NULL)", ("session-a", str(proj_a.resolve())))
    monkeypatch.chdir(proj_a)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(cli, "EmbeddingClient", _ConstantEmbedder)
    return CliWorld(home, db, proj_a, proj_b, ids)


def run_cli(world, monkeypatch, capsys, *words):
    """CLI를 실행하고 stdout JSON을 돌려준다. 종료 코드는 0이어야 한다."""
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(world.home), *map(str, words)])
    assert cli.main() == 0
    return json.loads(capsys.readouterr().out)


def table_state(db):
    with closing(sqlite3.connect(db)) as conn, conn:
        return (
            conn.execute("SELECT * FROM items WHERE type='memory' ORDER BY id").fetchall(),
            conn.execute("SELECT rowid, text FROM items_fts ORDER BY rowid").fetchall(),
        )


def found_ids(result):
    return {row["id"] for row in result["results"]}


def test_delete_through_symlinked_cwd_ignores_stale_pwd_and_clears_fts(world, monkeypatch, capsys, tmp_path):
    link = tmp_path / "link-to-alpha"
    try:
        link.symlink_to(world.proj_a, target_is_directory=True)
    except OSError:
        pytest.skip("symlink 생성 권한 없음")
    monkeypatch.chdir(link)
    monkeypatch.setenv("PWD", str(world.proj_b))

    outcome = run_cli(world, monkeypatch, capsys, "memory-delete", "--id", world.ids["in_alpha"])

    rows, fts = table_state(world.db)
    assert outcome == {"removed": True}
    assert world.ids["in_alpha"] not in {row[0] for row in rows}
    assert world.ids["in_alpha"] not in {row[0] for row in fts}
    assert len(rows) == len(fts) == 4


def test_delete_authority_is_the_row_itself_and_unknown_id_is_not_removed(world, monkeypatch, capsys):
    # memory-delete의 권한은 행 자신의 project/session 이다.
    monkeypatch.setenv("PWD", str(world.proj_b))

    results = [
        run_cli(world, monkeypatch, capsys, "memory-delete", "--id", world.ids["in_beta"]),
        run_cli(world, monkeypatch, capsys, "memory-delete", "--id", world.ids["in_session"]),
        run_cli(world, monkeypatch, capsys, "memory-delete", "--id", max(world.ids.values()) + UNKNOWN_ID_OFFSET),
    ]

    assert results == [{"removed": True}, {"removed": True}, {"removed": False}]


def test_deprecated_command_and_flag_names_still_work_and_warn(world, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "neospike", "--home", str(world.home), "--namespace", OWNER, "forget", str(world.ids["everywhere"]),
    ])

    assert cli.main() == 0
    captured = capsys.readouterr()

    assert json.loads(captured.out) == {"removed": True}
    assert "'forget' is renamed to 'memory-delete'" in captured.err
    assert "'--namespace' is renamed to '--owner'" in captured.err


def test_search_follows_cwd_and_not_session_env_or_pwd(world, monkeypatch, capsys):
    monkeypatch.setenv("PWD", str(world.proj_b))
    monkeypatch.setenv("HERMES_SESSION_ID", "session-a")

    from_alpha = run_cli(world, monkeypatch, capsys, "memory-search", "copper approval", "--limit", 10)
    monkeypatch.chdir(world.proj_b)
    from_beta = run_cli(world, monkeypatch, capsys, "memory-search", "copper approval", "--limit", 10)

    assert found_ids(from_alpha) == {world.ids["in_alpha"], world.ids["everywhere"]}
    assert found_ids(from_beta) == {world.ids["in_beta"], world.ids["everywhere"]}


def test_global_row_can_be_deleted_from_any_project(world, monkeypatch, capsys):
    monkeypatch.chdir(world.proj_b)

    assert run_cli(world, monkeypatch, capsys, "memory-delete", "--id", world.ids["everywhere"]) == {"removed": True}


def test_row_of_another_owner_is_refused_and_leaves_tables_unchanged(world, monkeypatch, capsys):
    before = table_state(world.db)

    outcome = run_cli(world, monkeypatch, capsys, "memory-delete", "--id", world.ids["foreign"])

    assert outcome == {"removed": False}
    assert table_state(world.db) == before
