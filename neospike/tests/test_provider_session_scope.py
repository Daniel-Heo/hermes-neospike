"""Per-session project scope of NeospikeProvider (state.db lookups, revocation, concurrency)."""
from __future__ import annotations

import json
import threading
from functools import partial
from pathlib import Path

import pytest
from neospike.tests.support_provider import OWNER_NS, boot_provider, write_sessions_db

JOIN_TIMEOUT_S = 5
SLOW_SEARCH_S = 0.15
PROJECT_NOTE = "The project keeps a Python index."


@pytest.fixture()
def two_projects(tmp_path: Path) -> tuple[Path, Path]:
    one = tmp_path / "proj-one"
    two = tmp_path / "proj-two"
    one.mkdir()
    two.mkdir()
    write_sessions_db(tmp_path, [("sess-1", str(one)), ("sess-2", str(two))])
    return one, two


def _add_project_note(provider, project: Path, session_id: str) -> None:
    provider._memory_db.add(
        OWNER_NS,
        PROJECT_NOTE,
        [0, 1, 0],
        source="test",
        kind="memory_item",
        importance=1.0,
        project=str(project),
        metadata={"scope": "project", "session_id": session_id},
    )


def _search_results(provider) -> list:
    return json.loads(provider.handle_tool_call("neospike_memory_search", {"query": "Python index"}))["results"]


def _run_together(*targets) -> None:
    threads = [threading.Thread(target=target) for target in targets]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=JOIN_TIMEOUT_S)
    assert not any(thread.is_alive() for thread in threads)


def test_state_db_overrides_stale_cwd_given_at_init(tmp_path: Path) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    write_sessions_db(tmp_path, [("sess-1", str(new))])
    provider = boot_provider(tmp_path, "sess-1", cwd=old)
    _add_project_note(provider, new, "sess-1")

    text = provider.prefetch("Python index", session_id="sess-1")

    assert "keeps a Python index" in text
    assert provider._project_dir == str(new)
    assert provider._core.allowed_roots == [new.resolve()]


def test_switching_to_cwdless_session_drops_project_access(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    write_sessions_db(tmp_path, [("desk", str(proj)), ("chat", None)])
    provider = boot_provider(tmp_path, "desk", cwd=proj)
    _add_project_note(provider, proj, "desk")
    assert len(_search_results(provider)) >= 1

    provider.on_session_switch("chat")

    assert _search_results(provider) == []
    assert provider._project_dir == ""
    assert provider._core.allowed_roots == []


def test_switching_to_unregistered_session_drops_project_access(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    write_sessions_db(tmp_path, [("desk", str(proj))])
    provider = boot_provider(tmp_path, "desk", cwd=proj)
    service = provider._core
    assert service.allowed_roots == [proj.resolve()]

    provider.on_session_switch("never-registered")

    assert (provider._project_dir, service.allowed_roots) == ("", [])


def test_parallel_prefetches_each_see_their_own_roots(tmp_path: Path, two_projects: tuple[Path, Path]) -> None:
    one, two = two_projects
    provider = boot_provider(tmp_path, "sess-1", cwd=one)
    service = provider._core
    first_in = threading.Event()
    roots_seen: dict[str, list[Path]] = {}

    def slow_first_search(query: str, *, limit: int) -> list:
        if not first_in.is_set():
            first_in.set()
            threading.Event().wait(SLOW_SEARCH_S)
        roots_seen[query] = list(service.allowed_roots)
        return []

    service.search = slow_first_search

    def call(query: str, session: str) -> None:
        provider.prefetch(query, session_id=session)

    t1 = threading.Thread(target=call, args=("q-one", "sess-1"))
    t2 = threading.Thread(target=call, args=("q-two", "sess-2"))
    t1.start()
    assert first_in.wait(timeout=2)
    t2.start()
    t1.join(timeout=JOIN_TIMEOUT_S)
    t2.join(timeout=JOIN_TIMEOUT_S)

    assert roots_seen == {"q-one": [one.resolve()], "q-two": [two.resolve()]}


def test_prefetch_for_other_session_retargets_later_tool_calls(
    tmp_path: Path, two_projects: tuple[Path, Path]
) -> None:
    one, two = two_projects
    provider = boot_provider(tmp_path, "sess-1", cwd=one)

    provider.prefetch("Python", session_id="sess-2")
    provider.handle_tool_call("neospike_memory_search", {"query": "Python"})

    assert (provider._active_sid, provider._project_dir) == ("sess-2", str(two.resolve()))
    assert provider._core.session_id == "sess-2"


def test_parallel_tool_calls_keep_the_session_of_their_prefetch(
    tmp_path: Path, two_projects: tuple[Path, Path]
) -> None:
    one, two = two_projects
    provider = boot_provider(tmp_path, "sess-1", cwd=one)
    rendezvous = threading.Barrier(2)
    outcome: dict[str, dict] = {}

    def echo_scope(tool: str, args: dict, **kwargs) -> str:
        return json.dumps({"project": provider._project_dir, "session": provider._core.session_id})

    provider._dispatch_tool = echo_scope

    def worker(session: str) -> None:
        provider.prefetch("Python", session_id=session)
        rendezvous.wait(timeout=3)
        outcome[session] = json.loads(provider.handle_tool_call("neospike_memory_search", {}))

    _run_together(partial(worker, "sess-1"), partial(worker, "sess-2"))

    assert outcome["sess-1"] == {"project": str(one.resolve()), "session": "sess-1"}
    assert outcome["sess-2"] == {"project": str(two.resolve()), "session": "sess-2"}


def test_parallel_saves_are_attributed_to_the_calling_session(tmp_path: Path) -> None:
    write_sessions_db(tmp_path, [("sess-1", None), ("sess-2", None)])
    provider = boot_provider(tmp_path, "sess-1")
    rendezvous = threading.Barrier(2)
    replies: list[dict] = []

    def worker(session: str) -> None:
        provider.prefetch("Python", session_id=session)
        rendezvous.wait(timeout=3)
        reply = provider.handle_tool_call(
            "neospike_memory_save",
            {"text": f"{session} indexes with Python.", "scope": "session", "subject": "indexing", "durability": "ongoing", "kind": "fact", "basis": "user_said"},
        )
        replies.append(json.loads(reply))

    _run_together(partial(worker, "sess-1"), partial(worker, "sess-2"))

    assert replies == [{"stored": True}, {"stored": True}]
    rows = provider._memory_db._conn.execute("SELECT origin, metadata_json FROM items WHERE type='memory'").fetchall()
    pairs = sorted((origin, json.loads(meta)["session_id"]) for origin, meta in rows)
    assert pairs == [("agent:sess-1", "sess-1"), ("agent:sess-2", "sess-2")]


def test_project_scoped_save_is_refused_when_session_has_no_project(tmp_path: Path) -> None:
    write_sessions_db(tmp_path, [("chat", None)])
    provider = boot_provider(tmp_path, "chat")

    reply = json.loads(
        provider.handle_tool_call(
            "neospike_memory_save",
            {"text": "Indexing runs on Python.", "scope": "project", "subject": "indexing", "durability": "stable", "kind": "fact", "basis": "user_said"},
        )
    )

    assert reply == {"error": "Project-scoped memory requires an active project"}
    assert provider._memory_db.count(OWNER_NS) == 0
