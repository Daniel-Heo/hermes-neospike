"""Contract tests: the MCP HTTP server for Claude Code and Hermes, TASK CLI helpers, dashboard env.

MCP tests start the real server process on a free test port and talk to it through the SDK streamable-http
client with the token header. The server uses the deterministic bag-of-words embedder instead of the
embedding daemon.
Without the optional `mcp` extra those tests are skipped.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neospike.database import canonical_database_path
from neospike.notes import NoteService, owned_paths
from neospike.store import MemoryStore
from neospike.mcp_headers import authorization_headers
from neospike.mcp_http import server_url
from neospike.tests.test_mcp_http import TOKEN, RunningServer, prepare_home, stop_all
from neospike.tests.test_notes import DIMENSIONS, BowEmbedder

HAS_MCP = importlib.util.find_spec("mcp") is not None
needs_mcp = pytest.mark.skipif(not HAS_MCP, reason="optional extra [mcp] is not installed")
REPO = Path(__file__).resolve().parents[2]
EXPECTED_TOOLS = {"note_read", "mem_search", "mem_period", "chat_search", "chat_read", "chat_period"}
RETIRED_TOOLS = {"memory_search", "note_search", "doc_search", "note_related"}  # one mem_search; note_related left
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


@pytest.fixture()
def home(tmp_path: Path):
    home = tmp_path / "hermes"
    home.mkdir()
    prepare_home(home)
    _save(home, "TASK_101", TASK_BODY, ["task", "status:active"])
    _save(home, "회의 메모", PLAIN_BODY, ["meeting"])
    yield home
    stop_all(home)


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO), env.get("PYTHONPATH"))))
    env["PYTHONIOENCODING"] = "utf-8"
    return env


class _Scenario:
    """Runs steps(session) against a freshly started server process (HTTP, with the token header)."""

    def __init__(self, home: Path, steps) -> None:
        self.home = home
        self.steps = steps
        self.port = 0
        self.initialized = None  # the server's initialize result

    async def __call__(self):
        import httpx2
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        async with httpx2.AsyncClient(headers=authorization_headers(TOKEN)) as client:
            async with streamable_http_client(server_url(self.port), http_client=client) as (read, write):
                async with ClientSession(read, write) as session:
                    self.initialized = await session.initialize()
                    return await self.steps(session)

    def run(self):
        import anyio
        with RunningServer(self.home) as record:
            self.port = record.port
            return anyio.run(self)


def _run(home: Path, steps):
    return _Scenario(home, steps).run()


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


async def _search_steps(session) -> dict:
    search = await _call(session, "mem_search", {"query": "이전 보고 착수했다", "tag": "task", "limit": 20})
    memory = await _call(session, "mem_search", {"query": "anything", "limit": 3})
    return {"search": search, "memory": memory}


# The agreed wording, kept here as an independent copy so a change to the server text fails a test.
EXPECTED_INSTRUCTIONS = """Neospike is the user's long-term memory for this machine: past decisions, memories, notes (TASK_ work orders, agent and automatic notes), docs (notes the user wrote, read-only) and the full Hermes chat history. All tools are read-only.

Search here when you: start a task or a new session; are about to decide, or ask the user, something that may already be decided (settings, thresholds, names, designs, fixes, rules); see a work order mention an earlier TASK, decision or bug; need what happened before (yesterday, last week, an earlier attempt, an error seen before).

Pick the tool by what you know:
1. The meaning, not the exact words -> mem_search (memories, notes and docs by meaning). Always first for decisions; cite mem:N or the note title.
2. An exact word, number, file name, command or error text from past work -> chat_search (substring search of user and assistant message text; not tool call arguments or tool results).
3. Only the time ("yesterday", "the last 2 days") -> chat_period for conversations (with a per-conversation summary) or mem_period for memories and notes created or changed then.
4. Then read in full: note_read for a note or doc body, chat_read for one message with its tool results and neighbours.

Order: mem_search first; go to chat_search or chat_period only for original wording or evidence; then chat_read. Start with a small limit and narrow with since/until, role or session instead of raising it. A newer decision overrides an older one; superseded memories appear only with include_history.

mem_search and mem_period search every project and show each memory's project; weigh a memory from another project accordingly."""
# Each tool description opens with when to use it, in these words.
EXPECTED_FIRST_SENTENCES = {
    "mem_search": "Find past decisions, memories, notes and docs by meaning; use first before deciding or asking "
                  "about anything that may already be decided.",
    "chat_search": "Find exact words, numbers, file names, commands or error texts in the text of past Hermes "
                   "user and assistant messages.",
    "chat_period": "List what happened in a time range (e.g. yesterday) without a query: conversations with a "
                   "summary, then messages.",
    "mem_period": "List memories, notes and docs created or changed in a time range (e.g. the last 2 days), "
                  "without a query.",
    "note_read": "Read the full body of one note or doc (a TASK_ work order, an agent note, a note the user wrote) "
                 "by id or exact title.",
    "chat_read": "Read one past chat message in full with its tool calls, their results and neighbouring messages.",
}
CLIENT_TEXT_LIMIT = 4096  # Claude Code cuts server instructions and preloaded tool descriptions here
ALWAYS_LOAD_KEY = "anthropic/alwaysLoad"


async def _list_tools(session) -> list:
    return list((await session.list_tools()).tools)


@needs_mcp
def test_initialize_sends_the_agreed_instructions_within_the_client_limit(home: Path) -> None:
    scenario = _Scenario(home, _list_tool_names)
    scenario.run()
    instructions = scenario.initialized.instructions
    assert instructions == EXPECTED_INSTRUCTIONS
    assert len(instructions) <= CLIENT_TEXT_LIMIT


@needs_mcp
def test_tool_descriptions_open_with_when_to_use_and_fit_the_client_limit(home: Path) -> None:
    tools = _run(home, _list_tools)
    descriptions = {tool.name: tool.description for tool in tools}
    assert set(descriptions) == set(EXPECTED_FIRST_SENTENCES)
    for name, first in EXPECTED_FIRST_SENTENCES.items():
        assert descriptions[name].startswith(first + " "), name
        assert len(descriptions[name]) <= CLIENT_TEXT_LIMIT, name


@needs_mcp
def test_every_tool_is_marked_always_load(home: Path) -> None:
    tools = _run(home, _list_tools)
    assert len(tools) == 6
    for tool in tools:
        assert tool.meta is not None and tool.meta.get(ALWAYS_LOAD_KEY) is True, tool.name


@needs_mcp
def test_tool_list_is_exactly_the_six_read_only_tools(home: Path) -> None:
    names = _run(home, _list_tool_names)
    assert sorted(names) == sorted(EXPECTED_TOOLS) and len(names) == 6
    assert "mem_search" in names and not RETIRED_TOOLS & set(names)


@needs_mcp
def test_initialize_reports_the_plugin_version(home: Path) -> None:
    scenario = _Scenario(home, _list_tool_names)
    scenario.run()
    manifest = (REPO / "neospike" / "plugin.yaml").read_text(encoding="utf-8")
    version = next(line.split(":", 1)[1].strip() for line in manifest.splitlines() if line.startswith("version:"))
    assert version and scenario.initialized.server_info.name == "neospike-mcp"
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
def test_tagged_mem_search_finds_the_task_note(home: Path) -> None:
    result = _run(home, _search_steps)
    assert not result["search"][0] and result["search"][1]["results"]
    assert all(hit["title"] == "TASK_101" and hit["kind"] == "note" for hit in result["search"][1]["results"])
    assert not result["memory"][0] and result["memory"][1] == {"results": []}


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


def test_owned_paths_skip_fenced_code() -> None:
    assert owned_paths("## 소유 파일\n```\n- `fake.py`\n```\n- `real.py`\n") == ["real.py"]


def test_clean_env_forwards_model_and_inference_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from neospike.dashboard import plugin_api as api
    monkeypatch.setenv("HERMES_BGE_M3_DIR", r"D:\models\bge-m3-onnx")
    monkeypatch.setenv("NEOSPIKE_INFERENCE_HOME", r"D:\shared\hermes")
    environment = api._subprocess_env()
    assert environment["HERMES_BGE_M3_DIR"] == r"D:\models\bge-m3-onnx"
    assert environment["NEOSPIKE_INFERENCE_HOME"] == r"D:\shared\hermes"
