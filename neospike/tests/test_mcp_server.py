"""Contract tests: MCP stdio server for Claude Code, TASK CLI helpers, dashboard env.

MCP tests start the real server process through the SDK stdio client. The server
and its CLI write child use the deterministic bag-of-words embedder instead of
the embedding daemon. Without the optional `mcp` extra those tests are skipped.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neospike.config import NeospikeConfig
from neospike.database import canonical_database_path
from neospike.notes import NoteService, append_report_section, owned_paths
from neospike.notes import head_row
from neospike.tests.support_items import chunk_rows
from neospike.store import MemoryStore
from neospike.tests.test_dashboard_notes import _CHILD_BOOT
from neospike.tests.test_notes import DIMENSIONS, BowEmbedder

HAS_MCP = importlib.util.find_spec("mcp") is not None
needs_mcp = pytest.mark.skipif(not HAS_MCP, reason="optional extra [mcp] is not installed")
REPO = Path(__file__).resolve().parents[2]
EXPECTED_TOOLS = {"note_read", "mem_search", "note_related", "note_append_report"}
RETIRED_TOOLS = {"memory_search", "note_search", "doc_search"}  # one mem_search
# Server process: stub Hermes core, swap the daemon for BowEmbedder, point CLI writes at the
# same stubbed CLI child the dashboard tests use, then run the real server entrypoint.
_SERVER_BOOT = (
    "import sys,types;"
    "a=types.ModuleType('agent');m=types.ModuleType('agent.memory_provider');"
    "m.MemoryProvider=type('MemoryProvider',(),{});a.memory_provider=m;"
    "sys.modules.setdefault('agent',a);sys.modules.setdefault('agent.memory_provider',m);"
    "from neospike import mcp_server as s;from neospike.tests.test_notes import BowEmbedder;"
    f"s.EmbeddingClient=BowEmbedder;s.CLI_PREFIX=[sys.executable,'-c',{_CHILD_BOOT!r}];"
    "sys.exit(s.main(sys.argv[1:]))"
)

TASK_BODY = (
    "# TASK_101 — MCP 연동\n"
    "\n"
    "## 배경\n"
    "Claude Code가 노트를 도구로 읽는다. 한국어 본문과\r\n윈도 줄바꿈이 섞여 있다.\n"
    "\n"
    "## 소유 파일\n"
    "- `neospike/mcp_server.py`\n"
    "- `neospike/cli.py`: CLI만\n"
    "\n"
    "## 보고\n"
    "- 이전 보고: 착수했다.\n"
    "\n"
    "## 판정 지표\n"
    "1. 바이트 단위로 같다.  \n"
    "\t끝 줄 공백 유지\n"
)
PLAIN_BODY = "# 회의 메모\n\n## 내용\n일반 노트는 보고를 받을 수 없다. 이 문장은 충분히 길다.\n"


def _db(home: Path) -> Path:
    return canonical_database_path(home / "neospike", "memory")


def _save(home: Path, title: str, body: str, tags: list[str]) -> dict:
    store = MemoryStore(_db(home), dimensions=DIMENSIONS)
    try:
        return NoteService(store, BowEmbedder()).save(title, body, tags=tags, author="agent")
    finally:
        store.close()


def _row(home: Path, title: str) -> tuple:
    store = MemoryStore(_db(home), dimensions=DIMENSIONS)
    try:
        with store._lock:
            head = head_row(store._conn, title=title)
            note = (head["body"], head["rev"], head["updated_at"])
            revisions = store._conn.execute("SELECT COUNT(*) FROM item_revisions").fetchone()[0]
            chunks = len(chunk_rows(store._conn))
        return note, revisions, chunks
    finally:
        store.close()


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    home = tmp_path / "hermes"
    NeospikeConfig().save(home)
    _save(home, "TASK_101", TASK_BODY, ["task", "status:active"])
    _save(home, "회의 메모", PLAIN_BODY, ["meeting"])
    return home


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO), env.get("PYTHONPATH"))))
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class _Scenario:
    """Runs steps(session) against a freshly started server process."""

    def __init__(self, home: Path, steps) -> None:
        self.home = home
        self.steps = steps
        self.initialized = None  # the server's initialize result

    async def __call__(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        parameters = StdioServerParameters(command=sys.executable, args=["-c", _SERVER_BOOT, "--home", str(self.home)],
                                           env=_child_env(), cwd=str(REPO))
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                self.initialized = await session.initialize()
                return await self.steps(session)


def _run(home: Path, steps):
    import anyio
    return anyio.run(_Scenario(home, steps))


async def _call(session, name: str, arguments: dict) -> tuple[bool, dict]:
    result = await session.call_tool(name, arguments)
    text = "".join(part.text for part in result.content if getattr(part, "type", "") == "text")
    # ToolError text may carry an "Error executing tool <name>: " prefix before the JSON.
    return bool(result.is_error), json.loads(text[text.find("{"):])


async def _list_tool_names(session) -> list[str]:
    return [tool.name for tool in (await session.list_tools()).tools]


async def _read_steps(session) -> dict:
    by_title = await _call(session, "note_read", {"id_or_title": "TASK_101"})
    by_id = await _call(session, "note_read", {"id_or_title": str(by_title[1]["note"]["id"])})
    missing = await _call(session, "note_read", {"id_or_title": "TASK_999"})
    return {"title": by_title, "id": by_id, "missing": missing}


async def _append_steps(session) -> dict:
    report = "- 결과: 합격.\n- 측정: 357 passed\r\n- 다음: 없음"
    task = await _call(session, "note_append_report", {"title": "TASK_101", "text": report})
    plain = await _call(session, "note_append_report", {"title": "회의 메모", "text": report})
    escape = await _call(session, "note_append_report", {"title": "TASK_101", "text": "## 소유 파일\n- `x.py`"})
    read = await _call(session, "note_read", {"id_or_title": "TASK_101"})
    related = await _call(session, "note_related", {"id_or_title": "TASK_101"})
    search = await _call(session, "mem_search", {"query": "이전 보고 착수했다 결과 합격 측정 357 passed", "tag": "task", "limit": 20})
    memory = await _call(session, "mem_search", {"query": "anything", "limit": 3})
    return {"task": task, "plain": plain, "escape": escape, "read": read, "related": related,
            "search": search, "memory": memory, "report": report}


async def _secret_steps(session) -> tuple[bool, dict]:
    return await _call(session, "note_append_report",
                       {"title": "TASK_101", "text": "배포 결과\napi_key = sk-abcdefghijklmnop1234"})


@needs_mcp
def test_tool_list_is_exactly_the_four_tools_with_mem_search(home: Path) -> None:
    names = _run(home, _list_tool_names)
    assert sorted(names) == sorted(EXPECTED_TOOLS) and len(names) == 4
    assert "mem_search" in names and not RETIRED_TOOLS & set(names)


@needs_mcp
def test_initialize_reports_the_plugin_version(home: Path) -> None:
    import anyio
    scenario = _Scenario(home, _list_tool_names)
    anyio.run(scenario)
    manifest = (REPO / "neospike" / "plugin.yaml").read_text(encoding="utf-8")
    version = next(line.split(":", 1)[1].strip() for line in manifest.splitlines() if line.startswith("version:"))
    assert version and scenario.initialized.server_info.name == "neospike"
    assert scenario.initialized.server_info.version == version


@needs_mcp
def test_note_read_returns_the_saved_body_byte_for_byte(home: Path) -> None:
    result = _run(home, _read_steps)
    ok_title, by_title = result["title"]
    ok_id, by_id = result["id"]
    assert not ok_title and not ok_id
    note = by_title["note"]
    assert note["body"].encode("utf-8") == TASK_BODY.encode("utf-8")
    assert by_id["note"]["body"].encode("utf-8") == TASK_BODY.encode("utf-8")
    assert note["rev"] == 1 and note["author"] == "agent" and note["tags"] == ["task", "status:active"]
    failed, missing = result["missing"]
    assert failed and missing["error"] == "not_found"


@needs_mcp
def test_append_report_touches_only_the_report_section_and_raises_rev(home: Path) -> None:
    plain_before = _row(home, "회의 메모")[0]
    result = _run(home, _append_steps)
    failed, task = result["task"]
    assert not failed and task["note"]["rev"] == 2 and task["note"]["previous_rev"] == 1
    body = result["read"][1]["note"]["body"]
    assert result["read"][1]["note"]["rev"] == 2
    report_end = TASK_BODY.index("착수했다.") + len("착수했다.")
    tail_start = TASK_BODY.index("## 판정 지표")
    # Every byte up to the old report text and every byte from the next section on is unchanged.
    assert body.startswith(TASK_BODY[:report_end]) and body.endswith(TASK_BODY[tail_start:])
    assert body == TASK_BODY[:report_end] + "\n\n" + result["report"] + "\n\n" + TASK_BODY[tail_start:]
    failed_plain, plain = result["plain"]
    assert failed_plain and plain["error"] == "forbidden"
    assert _row(home, "회의 메모")[0] == plain_before
    failed_escape, escape = result["escape"]
    assert failed_escape and escape["error"] == "invalid"
    assert not result["related"][0] and "backlinks" in result["related"][1]
    assert not result["search"][0] and result["search"][1]["results"]
    assert all(hit["title"] == "TASK_101" and hit["kind"] == "note" for hit in result["search"][1]["results"])
    assert not result["memory"][0] and result["memory"][1] == {"results": []}


@needs_mcp
def test_secret_report_is_rejected_and_the_database_is_unchanged(home: Path) -> None:
    before = _row(home, "TASK_101")
    failed, payload = _run(home, _secret_steps)
    assert failed and payload["error"] == "policy" and payload["lines"] == [2]
    assert _row(home, "TASK_101") == before


def _cli(home: Path, *arguments: str) -> dict:
    result = subprocess.run([sys.executable, "-m", "neospike.cli", "--home", str(home), *arguments],
                            capture_output=True, timeout=120, env=_child_env(), cwd=str(REPO))
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    return json.loads(result.stdout.decode("utf-8"))


def test_task_owners_finds_overlaps_between_active_tasks_and_task_next(home: Path) -> None:
    _save(home, "TASK_007", "# TASK_007\n\n## 소유 파일\n- 수정\n  - `neospike/CLI.py`\n  - `neospike\\notes.py`\n"
          "- 수정 금지\n  - `neospike/store.py`\n\n## 작업\n- 정리한다. 충분히 긴 설명 문장.\n", ["task", "status:active"])
    _save(home, "TASK_012", "# TASK_012\n\n## 소유 파일\n- `neospike/notes.py`: 일부\n- `neospike/store.py`\n"
          "\n## 작업\n- 다른 일을 한다. 충분히 긴 설명.\n", ["task", "status:active"])
    _save(home, "TASK_020", "# TASK_020\n\n## 소유 파일\n- `neospike/mcp_server.py`\n\n## 작업\n- 끝난 일. 충분히 긴 설명.\n",
          ["task", "status:done"])
    owners = _cli(home, "task-owners")
    assert [task["title"] for task in owners["tasks"]] == ["TASK_007", "TASK_012", "TASK_101"]
    assert owners["tasks"][0]["files"] == ["neospike/CLI.py", "neospike/notes.py"]
    overlaps = {entry["path"].casefold(): entry["tasks"] for entry in owners["overlaps"]}
    assert overlaps == {"neospike/notes.py": ["TASK_007", "TASK_012"], "neospike/cli.py": ["TASK_007", "TASK_101"]}
    assert _cli(home, "task-next") == {"number": 102, "title": "TASK_102", "max": 101}


def test_report_section_is_created_at_the_end_when_missing() -> None:
    assert append_report_section("# T\n\n## 작업\n- a", "완료") == "# T\n\n## 작업\n- a\n\n## Report\n\n완료\n"
    assert append_report_section("# T\n\n## 작업\n- a", "done", language="en") == "# T\n\n## 작업\n- a\n\n## Report\n\ndone\n"
    assert append_report_section("# T\n\n## 작업\n- a", "완료", language="ko") == "# T\n\n## 작업\n- a\n\n## 보고\n\n완료\n"
    assert append_report_section("# T\n## 보고", "완료") == "# T\n## 보고\n\n완료\n"
    fenced = "## 보고\n```\n## 가짜 제목\n```\n\n## 다음\n"
    assert append_report_section(fenced, "x") == "## 보고\n```\n## 가짜 제목\n```\n\nx\n\n## 다음\n"
    assert owned_paths("## 소유 파일\n```\n- `fake.py`\n```\n- `real.py`\n") == ["real.py"]


def test_existing_english_report_section_takes_the_report_in_any_case() -> None:
    body = "# T\n\n## Report\n- earlier\n\n## Next\n- n\n"
    for language in ("ko", "en"):
        assert append_report_section(body, "x", language=language) == "# T\n\n## Report\n- earlier\n\nx\n\n## Next\n- n\n"
    upper = "# T\n\n## REPORT\n- earlier\n"
    assert append_report_section(upper, "x", language="ko") == "# T\n\n## REPORT\n- earlier\n\nx\n"


@pytest.mark.parametrize("language, heading", [("ko", "## 보고"), (None, "## Report")])
def test_service_titles_a_new_report_section_in_the_home_language(tmp_path: Path, language, heading) -> None:
    home = tmp_path / "hermes"
    if language is not None:
        NeospikeConfig(extraction_language=language).save(home)
    store = MemoryStore(_db(home), dimensions=DIMENSIONS)
    try:
        notes = NoteService(store, BowEmbedder())
        notes.save("TASK_102", "# TASK_102\n\n## 작업\n- 충분히 긴 설명 문장이다.\n", tags=["task"], author="agent")
        notes.append_report("TASK_102", "done", hermes_home=home)
        body = notes.read(title="TASK_102")["body"]
    finally:
        store.close()
    assert body.endswith("\n" + heading + "\n\ndone\n") and body.count("## ") == 2


def test_clean_env_forwards_model_and_inference_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from neospike.dashboard import plugin_api as api
    monkeypatch.setenv("HERMES_BGE_M3_DIR", r"D:\models\bge-m3-onnx")
    monkeypatch.setenv("NEOSPIKE_INFERENCE_HOME", r"D:\shared\hermes")
    environment = api._subprocess_env()
    assert environment["HERMES_BGE_M3_DIR"] == r"D:\models\bge-m3-onnx"
    assert environment["NEOSPIKE_INFERENCE_HOME"] == r"D:\shared\hermes"
