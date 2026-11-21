# neospike windows patch v1
"""Dashboard setup API for Neospike.

One background job at a time runs the long steps. Reported values are observed
file and process state only; a started command is never counted as success.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

# Hermes imports this file as a loose module, not as a package member; putting just the plugin's
# parent directory on sys.path makes `neospike.*` imports behave the same here and in plain Python.
_PKG_PARENT_DIR = str(Path(__file__).resolve().parents[2])
if _PKG_PARENT_DIR not in sys.path:
    sys.path.insert(0, _PKG_PARENT_DIR)

from neospike._wincompat import chmod as _win_chmod, default_hermes_home
from neospike.config import (DEFAULT_SAVE_CRITERIA, ENABLED_KEY, EXTRACTION_LANGUAGE_DEFAULT, OLD_CONFIG_KEYS,
                             REMOVED_CONFIG_KEYS, SaveCriteriaLimits, parse_extraction_language, read_enabled, read_memory_rule,
                             write_memory_rule)
from neospike.database import canonical_database_path
from neospike import hermes_runtime
from neospike import inference  # stdlib only at import; release_daemon for the on/off switch
from neospike import runtime_check  # stdlib only; its checks run in the Hermes runtime
from neospike.notes import (
    NoteError, head_row, list_notes, note_history, note_record, note_related, note_titles,
)
from neospike import conversations as conversation_family
from neospike import folders as folder_family
from neospike import md_export
from neospike import preferences as pref_rows  # stdlib only (numpy-free)
from neospike.sphere_map import TOPIC_LANE, topic_key  # numpy-free at import (topic points)
from neospike.store import (ITEM_MEMORY, NOTE_AUTHOR_NAMES, NOTE_HEAD_SQL, CONVERSATION_TABLE_NAMES, RELATION_NEXT,
                             RELATION_REFERS, RELATION_SUPERSEDES, RELATION_TOPIC, SQLITE_BUSY_TIMEOUT_SECONDS,
                             build_fts_match, memory_metadata, note_head_sql, topic_handles, topic_name_of)

router = APIRouter()
_JOB_LOCK = threading.Lock()
_JOB_TABLE: dict[str, dict[str, Any]] = {}

# BAAI/bge-m3 int8 ONNX, pinned revision, not gated (MIT).
# The model file lands in onnx/ and is moved next to tokenizer.json (embedder.ONNX_MODEL_FILE / MODEL_FOLDER).
MODEL_REPO = "Xenova/bge-m3"
MODEL_REVISION = "4de13258303883538bd53b696b452bf8099f0858"
MODEL_ONNX_NAME = "sentence_transformers_int8.onnx"
MODEL_DOWNLOADS = ["onnx/" + MODEL_ONNX_NAME, "tokenizer.json"]
MODEL_FOLDER = "bge-m3-onnx"
# Import names of plugin.yaml python_dependencies (faiss-cpu imports as faiss, huggingface-hub as
# huggingface_hub). Hermes's package manager installs them; the dashboard only checks that they import.
DEPENDENCY_MODULES = ("numpy", "faiss", "tokenizers", "onnxruntime", "huggingface_hub")


def _home_dir_path() -> Path:
    return Path(default_hermes_home()).expanduser().resolve()


HF_CLI_MODULE = "huggingface_hub.cli.hf"  # the `hf` console script's module (plugin.yaml huggingface-hub)
NEOSPIKE_CLI_MODULE = "neospike.cli"


def _runtime_python_path() -> Path | None:
    """The Hermes runtime python (hermes_runtime.current_runtime_python), None when there is none."""
    try:
        return Path(hermes_runtime.current_runtime_python(_home_dir_path()))
    except hermes_runtime.HermesRuntimeNotFound:
        return None


def _runtime_command(target: list[str]) -> list[str] | None:
    """*target* (`module, args...`) through hermes_launch.py of this plugin copy on the
    runtime python, so it imports from the Hermes package manager's environment; None without a runtime."""
    python = _runtime_python_path()
    if python is None or not python.is_file():
        return None
    return hermes_runtime.launch_command(_home_dir_path(), target, python=str(python))


# Neospike settings read by child processes (CLI, inference daemon); dropping one makes the
# child look for models or the daemon in the default place (embedder.py, _wincompat.py).
_PASSTHROUGH_ENV = ("HERMES_BGE_M3_DIR", "NEOSPIKE_INFERENCE_HOME", "NEOSPIKE_FORCE_TCP")


def _subprocess_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in ("HOME", "PATH", "LANG", "LC_ALL", "TMPDIR", "USERPROFILE", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "HF_HOME", "PYTHONUTF8", "HERMES_PYTHON", *_PASSTHROUGH_ENV) if key in os.environ}
    env["HERMES_HOME"] = str(_home_dir_path())
    return env


def _run_logged(command: list[str], job: dict[str, Any]) -> None:
    job["detail"] = f"Running {Path(command[0]).name} {' '.join(command[1:3])}"
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             env=_subprocess_env())
    # Output is drained and dropped: it may contain tokens or user IDs.
    for _ in child.stdout or ():
        pass
    exit_code = child.wait()
    if exit_code != 0:
        raise RuntimeError(f"Command exited with status {exit_code}. See job log for details.")


def _job_thread_body(worker, job: dict[str, Any]) -> None:
    try:
        worker(job)
        outcome = {"state": "complete", "detail": "Completed successfully"}
    except Exception as failure:
        outcome = {"state": "failed", "detail": str(failure)}
    with _JOB_LOCK:
        job.update(outcome)


def _start_job(kind: str, worker) -> dict[str, str]:
    with _JOB_LOCK:
        if any(entry["state"] == "running" for entry in _JOB_TABLE.values()):
            raise HTTPException(409, "Another setup operation is already running")
        # Keep at most 20 jobs: drop finished ones from the oldest end.
        while len(_JOB_TABLE) >= 20:
            first_id = next(iter(_JOB_TABLE))
            if _JOB_TABLE[first_id]["state"] == "running":
                break
            del _JOB_TABLE[first_id]
        new_id = uuid.uuid4().hex
        job: dict[str, Any] = {"id": new_id, "kind": kind, "state": "running", "detail": "Starting…", "log": ""}
        _JOB_TABLE[new_id] = job
    threading.Thread(target=_job_thread_body, args=(worker, job), name=f"neospike-{kind}", daemon=True).start()
    return {"job_id": new_id}


def _model_files_ready() -> bool:
    model_dir = _home_dir_path() / "models" / MODEL_FOLDER
    return all((model_dir / Path(name).name).is_file() for name in MODEL_DOWNLOADS)


def _flatten_model_file(model_dir: Path) -> None:
    """hf download keeps the repo's onnx/ folder; the embedder reads the model next to tokenizer.json."""
    nested = model_dir / MODEL_DOWNLOADS[0]
    if nested.is_file():
        os.replace(nested, model_dir / MODEL_ONNX_NAME)


def _read_json_dict(path: Path, encoding: str = "utf-8") -> dict[str, Any]:
    """The JSON object in *path*; {} when the file is missing, unreadable or not an object."""
    try:
        data = json.loads(path.read_text(encoding=encoding))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _try_run(argv: list[str], timeout: int, **options: Any) -> subprocess.CompletedProcess | None:
    """The finished child, or None when it timed out or could not start."""
    try:
        return subprocess.run(argv, timeout=timeout, env=options.pop("env", None) or _subprocess_env(), **options)
    except (subprocess.TimeoutExpired, OSError):
        return None


def _database_summary(db_file: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"exists": db_file.is_file(), "healthy": None, "embedding_dimensions": None,
                               "memory_count": None}
    if not summary["exists"]:
        return summary
    try:
        with closing(sqlite3.connect(f"file:{db_file.as_posix()}?mode=ro", uri=True)) as conn:
            names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            summary["healthy"] = "items" in names
            if "items" in names:
                summary["memory_count"] = conn.execute(
                    f"SELECT COUNT(*) FROM items WHERE type='{ITEM_MEMORY}'").fetchone()[0]
            if "meta_entries" in names:
                found = conn.execute("SELECT value FROM meta_entries WHERE key='embedding_dimensions'").fetchone()
                summary["embedding_dimensions"] = int(found[0]) if found else None
    except (sqlite3.Error, ValueError, OSError):
        summary["healthy"] = False
    return summary


def _collect_status() -> dict[str, Any]:
    home = _home_dir_path()
    model_ready = _model_files_ready()
    saved = _read_json_dict(home / "neospike" / "config.json") if (home / "neospike" / "config.json").is_file() else {}
    hermes_exe = shutil.which("hermes")
    provider = None
    if hermes_exe:
        asked = _try_run([hermes_exe, "config", "get", "memory.provider"], 20, capture_output=True, text=True)
        if asked is not None and asked.returncode == 0:
            provider = asked.stdout.strip().strip('"')
    import_check = _runtime_command([runtime_check.MODULE, runtime_check.CHECK_IMPORTS, *DEPENDENCY_MODULES])
    # A path can exist but still be an invalid or stale executable, so the import is actually tried.
    imported = _try_run(import_check, 30, capture_output=True) if import_check else None
    platform_module = __import__("platform")
    return {
        "platform": {"system": platform_module.system(), "machine": platform_module.machine()},
        "hermes_found": bool(hermes_exe), "python_found": import_check is not None,
        "dependencies_installed": imported is not None and imported.returncode == 0,
        "text_model_downloaded": model_ready, "text_model_ready": model_ready,
        "provider": provider, "active": provider == "neospike",
        "config": {
            ENABLED_KEY: read_enabled(home),
            "turn_trace_enabled": _saved_setting(saved, "turn_trace_enabled", False),
            "extraction_language": _saved_setting(saved, "extraction_language", EXTRACTION_LANGUAGE_DEFAULT),
        },
        "database": _database_summary(canonical_database_path(home / "neospike", "memory")),
        "disk": {"free_bytes": shutil.disk_usage(home).free},
    }


class ModelsBody(BaseModel):
    """Empty (text model only); a `visual` field from an old client is ignored."""


class ConfigBody(BaseModel):
    # None (omitted): the saved value stays. Setup saves only the debug switch, Config only the language.
    turn_trace_enabled: bool | None = None
    extraction_language: str | None = None
    # The on/off switch (Config tab); false pauses Neospike in this home and asks the shared daemon to stop.
    enabled: bool | None = None


def _saved_setting(saved: dict[str, Any], key: str, default: Any = None) -> Any:
    """saved[key], else its v21 key (config.OLD_CONFIG_KEYS: a config.json not moved yet), else default."""
    if key in saved:
        return saved[key]
    old = next((old for old, new in OLD_CONFIG_KEYS.items() if new == key), None)
    return saved.get(old, default) if old else default


class ConfirmBody(BaseModel):
    confirm: bool = False


@router.get("/status")
@router.get("/setup/status")
def setup_status_route():
    return _collect_status()


@router.get("/jobs/{job_id}")
def job_get_route(job_id: str):
    with _JOB_LOCK:
        found = _JOB_TABLE.get(job_id)
        if not found:
            raise HTTPException(404, "Unknown job")
        return dict(found)


@router.get("/setup/progress")
def progress_route(job_id: str | None = None):
    with _JOB_LOCK:
        if job_id is None:
            return {"jobs": [dict(entry) for entry in list(_JOB_TABLE.values())[-20:]]}
        if job_id not in _JOB_TABLE:
            raise HTTPException(404, "Unknown job")
        return dict(_JOB_TABLE[job_id])


@router.post("/setup/models", status_code=202)
def models_route(request: ModelsBody):
    if _runtime_command([HF_CLI_MODULE]) is None:
        raise HTTPException(409, "Hermes Python environment was not found")

    def worker(job):
        target_dir = _home_dir_path() / "models" / MODEL_FOLDER
        target_dir.mkdir(exist_ok=True, parents=True)
        _run_logged(_runtime_command([HF_CLI_MODULE, "download", MODEL_REPO, *MODEL_DOWNLOADS,
                                      "--revision", MODEL_REVISION, "--local-dir", str(target_dir)]), job)
        _flatten_model_file(target_dir)
    return _start_job("download", worker)


def _overlay_config(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """The form fields over the existing config.json, so keys the form does not show (map_recent_days,
    map_stale_days, recall log, conversation capture) survive a dashboard save."""
    merged = {**_read_json_dict(path), **payload}
    merged.pop("embedding_dimensions", None)  # fixed in code; an old key is dropped on save
    for key in (*OLD_CONFIG_KEYS, *REMOVED_CONFIG_KEYS):  # v21 and removed keys go on the next save
        merged.pop(key, None)
    return merged


@router.post("/setup/config")
def config_route(request: ConfigBody):
    payload = {key: value for key, value in request.model_dump().items() if value is not None}
    if "extraction_language" in payload:
        try:
            payload["extraction_language"] = parse_extraction_language(payload["extraction_language"])
        except ValueError as exc:
            raise HTTPException(422, {"error": "invalid", "message": str(exc)}) from exc
    _save_config_overlay(_home_dir_path() / "neospike" / "config.json", payload)
    reply = {"saved": True, "config": payload, "reindex_performed": False}
    if payload.get(ENABLED_KEY) is False:
        reply["daemon"] = _release_daemon(_home_dir_path())
    return reply


def _release_daemon(home: Path) -> dict[str, Any]:
    """inference.release_daemon for the switch turned off: {"state": stopped | not_running | in_use | error,
    "users": the homes that still use the daemon, "message": why it could not be reached}."""
    try:
        state, users = inference.release_daemon(home)
    except OSError as exc:
        return {"state": "error", "users": [], "message": str(exc)}
    return {"state": state, "users": [str(user) for user in users]}


def _save_config_overlay(path: Path, payload: dict[str, Any]) -> None:
    """payload over config.json, written privately and atomically (other keys survive)."""
    path.parent.mkdir(exist_ok=True, parents=True)
    scratch = path.with_suffix(".json.tmp")
    scratch.write_text(json.dumps(_overlay_config(path, payload), indent=2) + "\n", encoding="utf-8")
    _win_chmod(scratch, 0o600)
    os.replace(scratch, path)


# ---- Memory creation rules (user request 2026-11-14): <home>/neospike/MEMORY_RULE.md, shown and saved whole in
# the Config tab; the provider puts it into its system prompt and an open conversation gets a change once in front
# of its next recall.
class RuleBody(BaseModel):
    text: str


def _rule_payload() -> dict[str, Any]:
    return {"text": read_memory_rule(_home_dir_path()), "default": DEFAULT_SAVE_CRITERIA,
            "max_chars": SaveCriteriaLimits.MAX_ENABLED_CHARS}


@router.get("/config/memory-rule")
def rule_get_route():
    return _rule_payload()


@router.put("/config/memory-rule")
def rule_put_route(request: RuleBody):
    try:
        write_memory_rule(_home_dir_path(), request.text)
    except ValueError as exc:
        raise HTTPException(422, {"error": "invalid", "message": str(exc)}) from exc
    return {"saved": True, **_rule_payload()}


# Model picker " API" mark. agent.credential_pool (public plugin API) gives each credential an auth_type; it turns an
# anthropic Claude subscription token (sk-ant-oat...) into "oauth" although that provider also takes API keys.
AUTH_TYPE_API_KEY = "api_key"  # agent.credential_pool.AUTH_TYPE_API_KEY
SUBSCRIPTION_PROVIDERS = frozenset({"openai-codex"})  # ChatGPT subscription: never an API key
PROVIDER_LIST_SEPARATOR = ","
MAX_PROVIDER_QUERY = 200  # slugs per request; the Hermes picker lists far fewer


def api_key_authenticated(provider: str, auth_types: list[str]) -> bool:
    """True only when every credential Hermes holds for *provider* is an API key (any OAuth token wins)."""
    if provider in SUBSCRIPTION_PROVIDERS or not auth_types:
        return False
    return all(kind == AUTH_TYPE_API_KEY for kind in auth_types)


def _pool_auth_types(provider: str) -> list[str]:
    from agent.credential_pool import load_pool  # Hermes process only (public plugin API module)
    return [str(entry.auth_type) for entry in load_pool(provider).entries()]


@router.get("/config/api-key-providers")
def api_key_providers_route(providers: str = ""):
    slugs = [slug.strip() for slug in providers.split(PROVIDER_LIST_SEPARATOR) if slug.strip()][:MAX_PROVIDER_QUERY]
    marked = []
    for slug in dict.fromkeys(slugs):
        try:
            auth_types = _pool_auth_types(slug)
        except Exception:  # no pool for this slug or no Hermes runtime: shown without the mark
            continue
        if api_key_authenticated(slug, auth_types):
            marked.append(slug)
    return {"providers": marked}


def _activate_provider():
    launcher = shutil.which("hermes")
    if not launcher:
        raise HTTPException(409, "Hermes CLI was not found on PATH")
    done = subprocess.run([launcher, "config", "set", "memory.provider", "neospike"],
                          capture_output=True, text=True, timeout=30, env=_subprocess_env())
    if done.returncode:
        raise HTTPException(500, "Activation failed; check Hermes configuration permissions")
    return {"active": True, "restart_required": True}


@router.post("/setup/activate")
def activate_route(request: ConfirmBody):
    if not request.confirm:
        raise HTTPException(400, "Activation requires explicit confirmation")
    return _activate_provider()


@router.post("/health")
def health_route():
    state = _collect_status()
    problems = []
    if not state["dependencies_installed"]:
        problems.append("Runtime dependencies are unavailable")
    if not state.get("text_model_ready", state["text_model_downloaded"]):
        problems.append("Text model files are incomplete")
    if not state["active"]:
        problems.append("memory.provider is not neospike")
    provider_ok = False
    if not problems:
        probe = _runtime_command([runtime_check.MODULE, runtime_check.CHECK_PROVIDER])
        probed = _try_run(probe, 90, capture_output=True, text=True, env=_subprocess_env()) if probe else None
        provider_ok = probed is not None and probed.returncode == 0
        if not provider_ok:
            problems.append("Provider availability check failed; inspect local Dashboard logs")
    return {"healthy": not problems and provider_ok, "provider_available": provider_ok, "problems": problems,
            "warnings": [], "restart_required_after_activation": True}


# --- Notes / Memory -------------------------------------------------------
# Single writer: this process opens the DB read-only (mode=ro) and never embeds. Every write,
# and the hybrid note search (it needs a query embedding), runs the CLI in a subprocess.

CLI_TIMEOUT_S = 180
NOTE_SEARCH_CHUNK_CAP = 50
NOTE_AUTHOR_SEPARATOR = ","  # ?authors=agent,formula
CONV_SEARCH_CAP = 20
ITEM_LIST_CAP = 200
LISTED_ITEM_KINDS = ("memory_item", "durable")
_NOTE_HTTP_CODES = {"not_found": 404, "conflict": 409, "forbidden": 403, "policy": 422, "invalid": 422}


def _cli_argv_base() -> list[str]:
    command = _runtime_command([NEOSPIKE_CLI_MODULE])
    if command is None:
        raise HTTPException(409, "Hermes Python environment was not found")
    return command


def _cli_call(arguments: list[str], payload: dict[str, Any] | None = None, *, namespace: str | None = None) -> tuple[int, dict[str, Any]]:
    """Run one CLI command with JSON on stdin; return (exit code, parsed JSON). Output is never echoed."""
    command = [*_cli_argv_base(), "--home", str(_home_dir_path())]
    if namespace is not None:
        command += ["--owner", namespace]
    env = _subprocess_env()
    # hermes_launch.py of this copy puts its plugins folder on sys.path; JSON is UTF-8 on every console code page.
    env["PYTHONIOENCODING"] = "utf-8"
    stdin = b"" if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        result = subprocess.run([*command, *arguments], input=stdin, capture_output=True, timeout=CLI_TIMEOUT_S, env=env)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise HTTPException(503, "Neospike command could not run; check the Hermes Python environment") from exc
    text = result.stdout.decode("utf-8", errors="replace").strip()
    for candidate in (text, text.rsplit("\n", 1)[-1]):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return result.returncode, parsed
    raise HTTPException(502, f"Neospike command failed with status {result.returncode}; see dashboard logs")


class _RoConnection:
    """`with _RoConnection() as conn`: a mode=ro connection, or None when the DB does not exist yet."""

    def __init__(self) -> None:
        self.connection: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection | None:
        path = canonical_database_path(_home_dir_path() / "neospike", "memory")
        if not path.is_file():
            return None
        self.connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=SQLITE_BUSY_TIMEOUT_SECONDS)
        self.connection.row_factory = sqlite3.Row
        return self.connection

    def __exit__(self, *exc_info: Any) -> None:
        if self.connection is not None:
            self.connection.close()


def _tables_exist(connection: sqlite3.Connection | None, *names: str) -> bool:
    if connection is None:
        return False
    found = {row[0] for row in connection.execute(
        f"SELECT name FROM sqlite_master WHERE type='table' AND name IN ({','.join('?' * len(names))})", names,
    )}
    return found == set(names)


def _fetch_note(connection: sqlite3.Connection | None, note_id: int, *, include_deleted: bool = False) -> dict[str, Any]:
    if not _tables_exist(connection, "items", "item_links"):
        raise HTTPException(404, "Note not found")
    try:
        return note_record(connection, note_id=note_id, include_deleted=include_deleted)
    except NoteError as exc:
        raise HTTPException(_NOTE_HTTP_CODES.get(exc.code, 422), exc.as_dict()) from exc


def _note_reply(code: int, result: dict[str, Any], note_id: int | None = None) -> dict[str, Any]:
    """Map a note CLI reply to HTTP. A 409 carries the current note so the editor can merge."""
    if code == 0 and "note" in result:
        return result["note"]
    status = _NOTE_HTTP_CODES.get(str(result.get("error")), 422)
    detail = {key: value for key, value in result.items() if key in {"error", "message", "current_rev", "lines", "fields", "title"}}
    if status == 409 and note_id is not None:
        with _RoConnection() as connection:
            try:
                current = _fetch_note(connection, note_id, include_deleted=True)
                detail["current"] = {key: current[key] for key in ("id", "title", "body", "tags", "project", "rev", "deleted_at")}
            except HTTPException:
                pass
    raise HTTPException(status, detail)


class NoteCreateBody(BaseModel):
    title: str
    body: str
    tags: list[str] = Field(default_factory=list)
    project: str = ""


class NoteUpdateBody(BaseModel):
    title: str
    body: str
    expected_rev: int
    tags: list[str] | None = None
    project: str | None = None


class NoteRestoreBody(BaseModel):
    rev: int
    expected_rev: int | None = None


class ItemUpdateBody(BaseModel):
    expected_text: str
    new_text: str
    subject: str | None = None
    tags: list[str] | None = None


def _parse_authors(authors: str) -> tuple[str, ...] | None:
    """'' -> None (every author); 'agent,formula' -> ('agent', 'formula'); an unknown author -> 422.
    The Docs tab asks for 'human' (docs = notes the user wrote), the Notes tab for 'agent,formula'."""
    if not authors.strip():
        return None
    parsed = tuple(dict.fromkeys(part.strip() for part in authors.split(NOTE_AUTHOR_SEPARATOR) if part.strip()))
    unknown = [name for name in parsed if name not in NOTE_AUTHOR_NAMES]
    if unknown or not parsed:
        raise HTTPException(422, f"Invalid note author: {authors!r}")
    return parsed


def _author_note_ids(connection: sqlite3.Connection, authors: tuple[str, ...], ids: list[int] | None) -> list[int]:
    """ids (in their order) written by one of authors; with ids None, every live such note, newest first."""
    marks = ",".join("?" * len(authors))
    if ids is None:
        return [int(row[0]) for row in connection.execute(
            f"SELECT id FROM items WHERE {NOTE_HEAD_SQL} AND deleted_at IS NULL AND author IN ({marks}) "
            "ORDER BY updated_at DESC, id DESC", authors)]
    kept = {int(row[0]) for row in connection.execute(
        f"SELECT id FROM items WHERE {NOTE_HEAD_SQL} AND author IN ({marks})", authors)}
    return [note_id for note_id in ids if note_id in kept]


@router.get("/notes")
def notes_list_route(q: str = "", tag: str = "", project: str = "", authors: str = ""):
    author_set = _parse_authors(authors)
    ids: list[int] | None = None
    if q.strip():
        arguments = ["note-search", q, "--limit", str(NOTE_SEARCH_CHUNK_CAP)]
        if project:
            arguments += ["--project", project]
        code, result = _cli_call(arguments)
        if code:
            raise HTTPException(_NOTE_HTTP_CODES.get(str(result.get("error")), 422), result.get("message", "Search failed"))
        ids = list(dict.fromkeys(int(hit["id"]) for hit in result.get("results", [])))
    with _RoConnection() as connection:
        if not _tables_exist(connection, "items"):
            return {"notes": []}
        # total: every live note of the asked authors, whatever the other filters (the Export button counts it).
        if author_set is None:
            total = int(connection.execute(
                f"SELECT COUNT(*) FROM items WHERE {NOTE_HEAD_SQL} AND deleted_at IS NULL").fetchone()[0])
        else:
            ids = _author_note_ids(connection, author_set, ids)
            total = len(_author_note_ids(connection, author_set, None))
        return {"notes": list_notes(connection, tag=tag or None, project=project or None, ids=ids), "total": total}


@router.get("/notes/titles")
def note_titles_get(prefix: str = ""):
    with _RoConnection() as connection:
        if not _tables_exist(connection, "items"):
            return {"titles": []}
        return {"titles": note_titles(connection, prefix)}


@router.post("/notes")
def note_create_route(request: NoteCreateBody):
    payload = {"title": request.title, "body": request.body, "tags": request.tags, "project": request.project, "author": "human"}
    return _note_reply(*_cli_call(["note-save"], payload))


# ---- Markdown export: the browser writes these into a folder the user picks; nothing is written here. ----
def _parse_export_ids(ids: str) -> list[int] | None:
    """'' -> None (every live note); '3,5' -> [3, 5]; anything else -> 422."""
    if not ids.strip():
        return None
    parsed: list[int] = []
    for part in ids.split(","):
        text = part.strip()
        if not text.isdigit() or int(text) < 1:
            raise HTTPException(422, f"Invalid note id: {part!r}")
        parsed.append(int(text))
    return parsed


def _collect_export_files(ids: str, authors: str = "") -> list[dict[str, str]]:
    """ids given: those notes; none: every live note, or with authors every live note of those authors."""
    note_ids = _parse_export_ids(ids)
    author_set = _parse_authors(authors)
    with _RoConnection() as connection:
        if not _tables_exist(connection, "items"):
            return []
        if note_ids is None and author_set is not None:
            note_ids = _author_note_ids(connection, author_set, None)
        return md_export.export_files(connection, note_ids, _live_topic_note_ids(connection))


@router.get("/notes/export")
def notes_export_route(ids: str = "", authors: str = ""):
    return _collect_export_files(ids, authors)


@router.get("/notes/export.zip")
def notes_zip_route(ids: str = "", authors: str = ""):
    return Response(md_export.zip_bytes(_collect_export_files(ids, authors)), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{md_export.ZIP_NAME}"'})


@router.get("/notes/{note_id}")
def note_get_route(note_id: int):
    with _RoConnection() as connection:
        return _fetch_note(connection, note_id)


@router.put("/notes/{note_id}")
def note_put_route(note_id: int, request: NoteUpdateBody):
    with _RoConnection() as connection:
        current = _fetch_note(connection, note_id, include_deleted=True)
    payload = {
        "id": note_id, "title": request.title, "body": request.body, "author": "human", "expected_rev": request.expected_rev,
        "tags": current["tags"] if request.tags is None else request.tags,
        "project": current["project"] if request.project is None else request.project,
    }
    return _note_reply(*_cli_call(["note-save"], payload), note_id)


@router.delete("/notes/{note_id}")
def note_delete_route(note_id: int, expected_rev: int | None = None):
    if expected_rev is None:
        with _RoConnection() as connection:
            expected_rev = _fetch_note(connection, note_id)["rev"]
    return _note_reply(*_cli_call(["note-delete", "--id", str(note_id), "--expected-rev", str(expected_rev)]), note_id)


@router.get("/notes/{note_id}/related")
def note_related_route(note_id: int):
    with _RoConnection() as connection:
        _fetch_note(connection, note_id, include_deleted=True)
        return note_related(connection, note_id)


# ---- Map connections: one edge set feeds both the map's lines and the side panel's Connected list. ----
_REL_TOPIC_MEMBER = "topic member"
_REL_PIECE = "piece"  # the next chain of split memory pieces (item_links relation next between memories)
_EXCERPT_LEN = 120
_KIND_RANK = {"topic": 0, "note": 1, "memory": 2}
# (relation, this item is the edge source) -> the relation as seen from this item.
_RELATION_TEXT = {
    (RELATION_REFERS, True): "refers", (RELATION_REFERS, False): "referenced by",
    (RELATION_SUPERSEDES, True): "supersedes", (RELATION_SUPERSEDES, False): "superseded by",
    (_REL_TOPIC_MEMBER, True): _REL_TOPIC_MEMBER, (_REL_TOPIC_MEMBER, False): _REL_TOPIC_MEMBER,
    (_REL_PIECE, True): "next piece", (_REL_PIECE, False): "previous piece",
}
# A live end of a drawn link: a memory, or a document head that is not trashed. lane: memory or note.
_LIVE_END_SQL = (f"(i.type='{ITEM_MEMORY}' OR ({note_head_sql('i')} AND i.deleted_at IS NULL))")
_LANE_OF_SQL = f"(CASE i.type WHEN '{ITEM_MEMORY}' THEN 'memory' ELSE 'note' END)"


def _graph_edges(connection: sqlite3.Connection | None) -> list[tuple[str, str, str]]:
    """(source key, target key, relation): refers/supersedes links whose two ends are live items, the topic
    links of memories (topic:<handle> -> memory), and the next chain of split memory texts (piece -> next
    piece). Self links are dropped."""
    edges: list[tuple[str, str, str]] = []
    if not _tables_exist(connection, "items", "item_links"):
        return edges
    # A reverted supersedes link keeps its row with reverted_at set; it no longer connects anything.
    for row in connection.execute(
        f"SELECT {_LANE_OF_SQL.replace('i.', 's.')},l.src_id,{_LANE_OF_SQL.replace('i.', 'd.')},l.dst_id,l.relation "
        "FROM item_links l JOIN items s ON s.id=l.src_id JOIN items d ON d.id=l.dst_id "
        "WHERE l.relation IN (?,?) AND l.reverted_at IS NULL AND "
        + _LIVE_END_SQL.replace("i.", "s.") + " AND " + _LIVE_END_SQL.replace("i.", "d.") + " ORDER BY l.id",
        (RELATION_REFERS, RELATION_SUPERSEDES),
    ):
        edges.append((folder_family.item_key(row[0], row[1]), folder_family.item_key(row[2], row[3]), row[4]))
    handles = topic_handles(connection)
    for name, memory_id in connection.execute(
        f"SELECT l.label,l.src_id FROM item_links l JOIN items m ON m.id=l.src_id AND m.type='{ITEM_MEMORY}' "
        f"WHERE l.relation='{RELATION_TOPIC}' ORDER BY l.id"
    ):
        edges.append((topic_key(handles[str(name)]), folder_family.item_key("memory", memory_id), _REL_TOPIC_MEMBER))
    for piece_id, next_id in connection.execute(
        f"SELECT l.src_id,l.dst_id FROM item_links l JOIN items a ON a.id=l.src_id AND a.type='{ITEM_MEMORY}' "
        f"JOIN items b ON b.id=l.dst_id AND b.type='{ITEM_MEMORY}' WHERE l.relation='{RELATION_NEXT}' ORDER BY l.id"
    ):
        edges.append((folder_family.item_key("memory", piece_id), folder_family.item_key("memory", next_id),
                      _REL_PIECE))
    return [edge for edge in edges if edge[0] != edge[1]]


def _table_col_names(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def _apply_map_links(points: list[dict[str, Any]], edges: list[tuple[str, str, str]]) -> None:
    """Rewrites each point's `links` (the lines drawn on the map) from the shared edge set, undirected,
    restricted to keys on the map — the same edges the Connected list reads."""
    keys = {point["key"] for point in points}
    adjacency: dict[str, set[str]] = {}
    for source, target, _relation in edges:
        if source in keys and target in keys:
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)
    for point in points:
        point["links"] = sorted(adjacency.get(point["key"], ()))


def _one_line_excerpt(text: str) -> str:
    return " ".join(str(text).split())[:_EXCERPT_LEN]


def _ids_in_lane(keys: list[str], lane: str) -> list[int]:
    prefix = (TOPIC_LANE if lane == TOPIC_LANE else folder_family.LANE_KEYS[lane]) + ":"
    return [int(key[len(prefix):]) for key in keys if key.startswith(prefix)]


def _rows_for_ids(connection: sqlite3.Connection, sql: str, ids: list[int]) -> list[sqlite3.Row]:
    return connection.execute(sql.format(marks=",".join("?" * len(ids))), ids).fetchall() if ids else []


def connected_items(connection: sqlite3.Connection, lane: str, item_id: int) -> list[dict[str, Any]]:
    """Every live item connected to (lane, item_id), one entry per item with all its relations."""
    key = topic_key(item_id) if lane == TOPIC_LANE else folder_family.item_key(lane, item_id)
    relations: dict[str, list[str]] = {}
    for source, target, relation in _graph_edges(connection):
        if key not in (source, target):
            continue
        other, label = (target, _RELATION_TEXT[(relation, True)]) if source == key else (source, _RELATION_TEXT[(relation, False)])
        labels = relations.setdefault(other, [])
        if label not in labels:
            labels.append(label)
    keys = list(relations)
    found: dict[str, dict[str, Any]] = {}
    # `text` is the full body / memory text for the panel's read-only "all connected text" box.
    for note_id in _ids_in_lane(keys, "note"):
        note = head_row(connection, note_id=note_id)
        if note is not None:
            found[folder_family.item_key("note", note_id)] = {"lane": "note", "id": note_id, "title": str(note["title"]),
                                                              "kind": "note", "text": str(note["body"])}
    for row in _rows_for_ids(connection, f"SELECT id,text FROM items WHERE type='{ITEM_MEMORY}' AND id IN ({{marks}})",
                             _ids_in_lane(keys, "memory")):
        memory_id = int(row[0])
        found[folder_family.item_key("memory", memory_id)] = {"lane": "memory", "id": memory_id, "title": _one_line_excerpt(row[1]),
                                                              "kind": "memory", "text": str(row[1])}
    names = {handle: name for name, handle in topic_handles(connection).items()}
    for handle in _ids_in_lane(keys, TOPIC_LANE):
        if handle in names:
            found[topic_key(handle)] = {"lane": TOPIC_LANE, "id": handle, "title": names[handle], "kind": "topic",
                                        "text": ""}
    items = [{"key": other, **found[other], "relations": relations[other]} for other in keys if other in found]
    return sorted(items, key=_connection_sort_key)


def _connection_sort_key(item: dict[str, Any]) -> tuple[int, int]:
    return _KIND_RANK[item["kind"]], item["id"]


@router.get("/connections/{lane}/{item_id}")
def connections_get_route(lane: str, item_id: int):
    """The map side panel's Connected list. Read-only. Lane "topic" lists the topic's
    members and carries its title (read-only)."""
    if lane not in folder_family.LANE_KEYS and lane != TOPIC_LANE:
        raise HTTPException(404, "Unknown item type")
    with _RoConnection() as connection:
        found = None
        if _tables_exist(connection, "items", "item_links"):
            if lane == TOPIC_LANE:
                found = topic_name_of(connection, item_id)
            else:
                where = f"type='{ITEM_MEMORY}'" if lane == "memory" else NOTE_HEAD_SQL
                row = connection.execute(f"SELECT id FROM items WHERE id=? AND {where}", (item_id,)).fetchone()
                found = None if row is None else int(row[0])
        if found is None:
            raise HTTPException(404, "Item not found")
        result = {"key": topic_key(item_id) if lane == TOPIC_LANE else folder_family.item_key(lane, item_id),
                  "items": connected_items(connection, lane, item_id), "is_topic": lane == TOPIC_LANE}
        if lane == TOPIC_LANE:
            result["title"] = str(found)
        return result


def _live_topic_note_ids(connection: sqlite3.Connection | None) -> set[int]:
    if not _tables_exist(connection, "items", "item_links"):
        return set()
    return {int(row[0]) for row in connection.execute(
        f"SELECT n.id FROM item_links t JOIN items n ON n.id=t.src_id AND {note_head_sql('n')} AND n.deleted_at IS NULL "
        f"WHERE t.relation='{RELATION_TOPIC}'")}


# ---- Hand-edited topic members: Move to... in the topic panel, drag a point onto a topic. ----
class TopicMoveBody(BaseModel):
    """Topics are addressed by their handle (to_topic / remove_from_topic: the topic id of /topics and the map);
    no field renames a topic."""
    items: list[str] = Field(default_factory=list)
    to_topic: int | None = None  # topic handle to put the items into
    remove_from_topic: int | None = None  # topic handle to take them out of
    folder_id: int | None = None  # with remove_from_topic and to_folder: then file them in this folder (None = Unfiled)
    to_folder: bool = False


@router.get("/topics")
def topics_list_route():
    """Live topics (title = the topic name), for the Move to... menu. Read-only."""
    with _RoConnection() as connection:
        if not _tables_exist(connection, "items", "item_links"):
            return {"topics": []}
        handles = topic_handles(connection)
        counts = dict(connection.execute(
            f"SELECT l.label,COUNT(*) FROM item_links l JOIN items m ON m.id=l.src_id AND m.type='{ITEM_MEMORY}' "
            f"WHERE l.relation='{RELATION_TOPIC}' GROUP BY l.label").fetchall())
        rows = sorted(handles.items(), key=_topic_list_order)
        return {"topics": [{"id": handle, "title": name, "members": int(counts.get(name, 0))} for name, handle in rows]}


def _topic_list_order(entry: tuple[str, int]) -> tuple[str, int]:
    return entry


@router.post("/topics/members/move")
def topic_move_route(request: TopicMoveBody):
    if not request.items or (request.to_topic is None) == (request.remove_from_topic is None) \
            or (request.to_folder and request.remove_from_topic is None):
        raise HTTPException(422, {"error": "invalid", "message": "Select items and exactly one topic target"})
    with _RoConnection() as connection:
        handle = request.to_topic if request.to_topic is not None else request.remove_from_topic
        name = topic_name_of(connection, handle) if _tables_exist(connection, "item_links") else None
    if name is None:
        raise HTTPException(404, {"error": "not_found", "message": f"topic {handle} is not a live topic"})
    target = ["--to-topic", name] if request.to_topic is not None else ["--remove-from-topic", name]
    result = _folder_reply(*_cli_call(["topic-member-move", "--items", ",".join(request.items), *target]))
    if request.to_folder:
        folder = "none" if request.folder_id is None else str(request.folder_id)
        result["folder"] = _folder_reply(*_cli_call(["folder-move", "--folder", folder, "--items", ",".join(request.items)]))
    return result


@router.get("/notes/{note_id}/revisions")
def note_history_route(note_id: int):
    with _RoConnection() as connection:
        _fetch_note(connection, note_id, include_deleted=True)
        return note_history(connection, note_id)


@router.post("/notes/{note_id}/restore")
def note_restore_route(note_id: int, request: NoteRestoreBody):
    arguments = ["note-restore", "--id", str(note_id), "--rev", str(request.rev)]
    if request.expected_rev is not None:
        arguments += ["--expected-rev", str(request.expected_rev)]
    return _note_reply(*_cli_call(arguments), note_id)


# ---- Conversations: read-only. Search needs a query embedding, so it runs the CLI. ----

@router.get("/conversations")
def convs_list_route():
    with _RoConnection() as connection:
        if not _tables_exist(connection, *CONVERSATION_TABLE_NAMES):
            return {"conversations": []}
        return {"conversations": conversation_family.list_conversations(connection)}


@router.get("/conversations/search")
def convs_search_route(q: str = ""):
    if not q.strip():
        return {"results": []}
    code, result = _cli_call(["conv-search", q, "--limit", str(CONV_SEARCH_CAP)])
    if code:
        raise HTTPException(422, result.get("message", "Search failed"))
    return {"results": result.get("results", [])}


@router.get("/conversations/{conversation_id}")
def conv_get_route(conversation_id: int):
    with _RoConnection() as connection:
        if not _tables_exist(connection, *CONVERSATION_TABLE_NAMES):
            raise HTTPException(404, "Conversation not found")
        try:
            return conversation_family.conversation_record(connection, conversation_id)
        except conversation_family.ConversationError as exc:
            raise HTTPException(404, "Conversation not found") from exc


def _memory_row_dict(row: sqlite3.Row) -> dict[str, Any]:
    metadata = memory_metadata(row)
    return {"id": int(row["id"]), "owner": row["owner"], "text": row["text"], "kind": row["kind"],
            "project": row["project"], "scope": metadata.get("scope"), "subject": metadata.get("subject"),
            "tags": metadata.get("tags") or [], "created_at": row["created_at"], "editable": row["kind"] == "memory_item",
            "invalid_at": row["invalid_at"], "superseded_by": row["superseded_by"],
            "archived": row["archived_at"] is not None, "archived_at": row["archived_at"]}


_MEMORY_COLS = ("m.id,m.owner,m.text,m.kind,m.project,m.scope,m.metadata_json,m.created_at,m.invalid_at,"
                 "m.superseded_by,m.archived_at")


def _memory_select_cols(connection: sqlite3.Connection) -> str:
    return _MEMORY_COLS


def _memory_text_of(connection: sqlite3.Connection, memory_id: int) -> str | None:
    row = connection.execute(f"SELECT text FROM items WHERE id=? AND type='{ITEM_MEMORY}'", (memory_id,)).fetchone()
    return None if row is None else row[0]


@router.get("/memories/{item_id}")
def item_get_route(item_id: int):
    """One memory row for the editor (the map opens items one at a time)."""
    with _RoConnection() as connection:
        row = None
        if _tables_exist(connection, "items"):
            row = connection.execute(f"SELECT {_memory_select_cols(connection)} FROM items m WHERE m.id=? "
                                     f"AND m.type='{ITEM_MEMORY}'", (item_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "Memory item not found")
        return _memory_row_dict(row)


@router.get("/memories")
def items_list_route(q: str = ""):
    """Memory items and promoted memories; q is a lexical (FTS5) filter, no embedding here."""
    with _RoConnection() as connection:
        if not _tables_exist(connection, "items", "items_fts"):
            return {"items": []}
        columns = _memory_select_cols(connection)
        kinds = ",".join("?" * len(LISTED_ITEM_KINDS))
        live = f"m.type='{ITEM_MEMORY}' AND m.kind IN ({kinds})"
        values: list[Any] = [*LISTED_ITEM_KINDS]
        expression = build_fts_match(q) if q.strip() else ""
        if expression:
            rows = connection.execute(
                f"SELECT {columns} FROM items_fts JOIN items m ON m.id=items_fts.rowid "
                f"WHERE items_fts MATCH ? AND {live} ORDER BY bm25(items_fts) LIMIT ?",
                (expression, *values, ITEM_LIST_CAP),
            ).fetchall()
        elif q.strip():
            rows = []
        else:
            rows = connection.execute(
                f"SELECT {columns} FROM items m WHERE {live} ORDER BY m.created_at DESC, m.id DESC LIMIT ?",
                (*values, ITEM_LIST_CAP),
            ).fetchall()
        return {"items": [_memory_row_dict(row) for row in rows]}


def _memory_ns_and_text(item_id: int) -> tuple[str, str]:
    """(namespace, current text) of one memory row; 404 when absent."""
    with _RoConnection() as connection:
        row = None
        if _tables_exist(connection, "items"):
            row = connection.execute(f"SELECT owner,text FROM items WHERE id=? AND type='{ITEM_MEMORY}'",
                                     (item_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Memory item not found")
    return row["owner"], row["text"]


@router.put("/memories/{item_id}")
def item_put_route(item_id: int, request: ItemUpdateBody):
    namespace, current_text = _memory_ns_and_text(item_id)
    payload: dict[str, Any] = {"id": item_id, "expected_text": request.expected_text, "new_text": request.new_text}
    if request.subject is not None:
        payload["subject"] = request.subject
    if request.tags is not None:
        payload["tags"] = request.tags
    code, result = _cli_call(["memory-update"], payload, namespace=namespace)
    if code:
        raise HTTPException(422, {"error": "invalid", "message": result.get("message", "Memory item update was rejected")})
    if result.get("updated"):
        return result
    if result.get("reason") == "conflict":
        raise HTTPException(409, {"error": "conflict", "message": "Memory item changed; reload it", "current_text": current_text})
    raise HTTPException(409, {"error": "unavailable", "message": "This memory cannot be edited here (only agent-written memory items are editable)"})


class ItemRestoreBody(BaseModel):
    rev: int


@router.get("/memories/{item_id}/history")
def item_history_route(item_id: int):
    """Supersedes links on either side (with evidence) and the stored revisions of one memory."""
    with _RoConnection() as connection:
        if not _tables_exist(connection, "items") or connection.execute(
                f"SELECT 1 FROM items WHERE id=? AND type='{ITEM_MEMORY}'", (item_id,)).fetchone() is None:
            raise HTTPException(404, "Memory item not found")
        links: list[dict[str, Any]] = []
        if _tables_exist(connection, "item_links"):
            for row in connection.execute(
                "SELECT id,src_id,dst_id,created_by,origin,session_id,evidence,created_at,reverted_at FROM item_links "
                "WHERE relation=? AND (src_id=? OR dst_id=?) ORDER BY id",
                (RELATION_SUPERSEDES, item_id, item_id),
            ):
                links.append({**dict(row), "new_id": int(row["src_id"]), "old_id": int(row["dst_id"]),
                              "new_text": _memory_text_of(connection, int(row["src_id"])),
                              "old_text": _memory_text_of(connection, int(row["dst_id"]))})
        revisions: list[dict[str, Any]] = []
        if _tables_exist(connection, "item_revisions"):
            revisions = [dict(row) for row in connection.execute(
                "SELECT rev,text,edited_at,edited_by,session_id FROM item_revisions WHERE item_id=? ORDER BY rev DESC",
                (item_id,))]
        return {"id": item_id, "supersedes": links, "revisions": revisions}


@router.post("/links/{link_id}/revert")
def link_revert_route(link_id: int):
    code, result = _cli_call(["link-revert", "--link", str(link_id)])
    if code or not result.get("reverted"):
        raise HTTPException(404 if result.get("reason") == "not_found" else 422,
                            {"error": result.get("reason") or result.get("error") or "invalid",
                             "message": result.get("message", "Supersedes link was not reverted")})
    return result


@router.post("/memories/{item_id}/restore")
def item_restore_route(item_id: int, request: ItemRestoreBody):
    namespace, _ = _memory_ns_and_text(item_id)
    code, result = _cli_call(["memory-restore", "--id", str(item_id), "--rev", str(request.rev)], namespace=namespace)
    if code or not result.get("restored"):
        status = 409 if result.get("reason") == "conflict" else 404 if result.get("reason") == "not_found" else 422
        raise HTTPException(status, {"error": result.get("reason") or result.get("error") or "invalid",
                                     "message": result.get("message", "Memory revision was not restored")})
    return result


@router.post("/memories/{item_id}/archive")
def item_archive_route(item_id: int):
    """Human archive: out of prefetch, still shown here and in direct search."""
    namespace, _ = _memory_ns_and_text(item_id)
    code, result = _cli_call(["memory-archive", "--id", str(item_id)], namespace=namespace)
    if code or not result.get("archived"):
        raise HTTPException(409, "Memory item is already archived")
    return {"archived": True, "id": item_id}


@router.post("/memories/{item_id}/unarchive")
def item_unarchive_route(item_id: int):
    """Back into prefetch; a search hit alone never does this."""
    namespace, _ = _memory_ns_and_text(item_id)
    code, result = _cli_call(["memory-unarchive", "--id", str(item_id)], namespace=namespace)
    if code or not result.get("unarchived"):
        raise HTTPException(409, "Memory item is not archived")
    return {"unarchived": True, "id": item_id}


@router.delete("/memories/{item_id}")
def item_delete_route(item_id: int):
    namespace, _ = _memory_ns_and_text(item_id)
    code, result = _cli_call(["memory-delete", "--id", str(item_id)], namespace=namespace)
    if code or not result.get("removed"):
        raise HTTPException(404, "Memory item was not deleted")
    return {"removed": True, "id": item_id}


# ---- Preferences tab: preferences/constraints. Reads use mode=ro; every write runs the CLI
# (pref-add / pref-update / pref-delete), which embeds. A row added here is basis user_said. ----

_PREF_HTTP_CODES = {"not_found": 404, "conflict": 409, "duplicate": 409, "unavailable": 404,
                    "situation_required": 422, "invalid": 422}


class PreferenceCreateBody(BaseModel):
    type: str
    text: str
    situation: str = ""


class PreferenceUpdateBody(BaseModel):
    expected_text: str
    type: str | None = None
    text: str | None = None
    situation: str | None = None


def _pref_reply(code: int, result: dict[str, Any]) -> dict[str, Any]:
    if code:
        detail = {key: value for key, value in result.items() if key in {"error", "message", "current_text"}}
        raise HTTPException(_PREF_HTTP_CODES.get(str(result.get("error")), 422), detail or "Preference command failed")
    return result


def _preference_owner(pref_id: int) -> str:
    """The namespace of one preferences row; 404 when absent."""
    with _RoConnection() as connection:
        row = None
        if connection is not None and pref_rows.table_present(connection):
            row = pref_rows.get_preference(connection, pref_id)
    if row is None:
        raise HTTPException(404, "Preference not found")
    return str(row["owner"])


@router.get("/preferences")
def preferences_list_route():
    """Every preference/constraint, newest change first (no vectors)."""
    with _RoConnection() as connection:
        if connection is None or not pref_rows.table_present(connection):
            return {"preferences": []}
        return {"preferences": pref_rows.list_preferences(connection)}


@router.post("/preferences")
def preference_create_route(request: PreferenceCreateBody):
    payload = {"type": request.type, "text": request.text, "situation": request.situation,
               "basis": pref_rows.BASIS_USER_SAID}
    return _pref_reply(*_cli_call(["pref-add"], payload))


@router.put("/preferences/{pref_id}")
def preference_put_route(pref_id: int, request: PreferenceUpdateBody):
    namespace = _preference_owner(pref_id)
    payload: dict[str, Any] = {"id": pref_id, "expected_text": request.expected_text}
    for name in ("type", "text", "situation"):
        value = getattr(request, name)
        if value is not None:
            payload[name] = value
    return _pref_reply(*_cli_call(["pref-update"], payload, namespace=namespace))


@router.delete("/preferences/{pref_id}")
def preference_delete_route(pref_id: int):
    namespace = _preference_owner(pref_id)
    code, result = _cli_call(["pref-delete", "--id", str(pref_id)], namespace=namespace)
    if code or not result.get("removed"):
        raise HTTPException(404, "Preference was not deleted")
    return {"removed": True, "id": pref_id}


# ---- Sphere map and folders. Reads use mode=ro; every folder write runs the CLI. ----

_FOLDER_HTTP_CODES = {"not_found": 404, "invalid": 422}
_MAP_FAIL_CODE = 503


class FolderCreateBody(BaseModel):
    name: str
    color: str | None = None


class FolderUpdateBody(BaseModel):
    name: str | None = None
    color: str | None = None


class FolderMoveBody(BaseModel):
    folder_id: int | None = None
    items: list[str] = Field(default_factory=list)


def _folder_reply(code: int, result: dict[str, Any]) -> dict[str, Any]:
    if code:
        detail = {key: value for key, value in result.items() if key in {"error", "message", "items"}}
        raise HTTPException(_FOLDER_HTTP_CODES.get(str(result.get("error")), 422), detail or "Folder command failed")
    return result


def _name_color_args(name: str | None, color: str | None) -> list[str]:
    # --key=value keeps a value that starts with '-' from being read as an option.
    options = [] if name is None else [f"--name={name}"]
    return options + ([] if color is None else [f"--color={color}"])


@router.get("/map")
def map_get_route():
    """Computed by the CLI child: numpy lives in the plugin Python, not in this dashboard process."""
    code, result = _cli_call(["map"])
    if code or "points" not in result:
        raise HTTPException(_MAP_FAIL_CODE, str(result.get("message") or "The map could not be computed"))
    # The lines use the Connected list's edges (adds the topic links and the piece chain).
    # Topic points (lane "topic", topic=true) come from the CLI; they are the drop targets for memories.
    with _RoConnection() as connection:
        _apply_map_links(result["points"], _graph_edges(connection))
    return result


@router.get("/folders")
def folders_list_route():
    with _RoConnection() as connection:
        if connection is None:
            return {"folders": []}
        return {"folders": folder_family.list_folders(connection)}


@router.post("/folders")
def folder_create_route(request: FolderCreateBody):
    return _folder_reply(*_cli_call(["folder-create", *_name_color_args(request.name, request.color)]))["folder"]


@router.put("/folders/{folder_id}")
def folder_put_route(folder_id: int, request: FolderUpdateBody):
    arguments = ["folder-update", "--id", str(folder_id), *_name_color_args(request.name, request.color)]
    return _folder_reply(*_cli_call(arguments))["folder"]


@router.delete("/folders/{folder_id}")
def folder_delete_route(folder_id: int):
    return _folder_reply(*_cli_call(["folder-delete", "--id", str(folder_id)]))


@router.post("/folders/move")
def folder_move_route(request: FolderMoveBody):
    if not request.items:
        raise HTTPException(422, {"error": "invalid", "message": "Select at least one item"})
    target = "none" if request.folder_id is None else str(request.folder_id)
    return _folder_reply(*_cli_call(["folder-move", "--folder", target, "--items", ",".join(request.items)]))
