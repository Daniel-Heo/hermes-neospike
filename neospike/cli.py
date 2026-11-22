# neospike windows patch v1
from __future__ import annotations
from ._wincompat import default_hermes_home as _windows_default_home

import argparse
import functools
import json
import sqlite3
import sys
from pathlib import Path

from .inference import EmbeddingClient, EmbeddingUnavailable
from . import backup as backup_tools
from . import topics as topic_ops
from . import conversations as conv_ops
from . import folders as folder_ops
from .config import EMBEDDING_DIMENSIONS, NeospikeConfig
from .database import canonical_database_path
from .evaluation.command import register_eval_cli, run_eval_cli
from .notes import (ORIGIN_CONVERSATION, ORIGIN_HUMAN, NoteError, embed_pending, next_task_title, task_owners,
                    tidy_saved_preference)
from .renames import forward_old_options
from .recall_log import LOG_DIRECTORY, append_failure
from .service import NeospikeService
from . import preferences as pref_ops
from .store import (
    HISTORY_OMIT, HISTORY_SHOW, HUMAN_EDITOR,
    SQLITE_BUSY_TIMEOUT_SECONDS, MemoryStore, memory_db_path, memory_metadata,
    pick_namespace,
)


_NOTE_VERBS = {"note-save", "note-read", "note-search", "note-delete", "note-restore",
                  "task-owners", "task-next"}
_NOTE_SAVE_KEYS = {"id", "title", "body", "tags", "project", "author", "expected_rev", "origin"}
_ITEM_VERBS = {"memory-update", "memory-delete"}
# Dashboard Preferences tab: JSON in, JSON out; a row added here is basis user_said.
_PREF_VERBS = {"pref-add", "pref-update", "pref-delete", "pref-list"}
_PREF_ADD_KEYS = {"type", "text", "situation", "basis", "subject", "importance"}
_PREF_UPDATE_KEYS = {"id", "expected_text", *_PREF_ADD_KEYS}
# Supersede / revision / link maintenance. JSON out; exit 1 on refusal.
_LINK_VERBS = {"link-revert", "memory-revisions", "memory-restore"}
_TOPIC_VERBS = {"topic-assign", "topic-discover", "topic-member-move", "note-absorb", "topic-to-note"}
PREF_RECONCILE_VERB = "pref-reconcile"  # runs as a Hermes idle task; the CLI only explains that
EMBED_PENDING_VERB = "embed-pending"  # daily stage; ends with the save follow-up (note-absorb, topic-assign)
EMBED_PENDING_EDITED_BY = topic_ops.FORMULA_EDITOR  # edited_by of a chunk that takes a filled duplicate's body
_CONV_VERBS = {"conv-index", "conv-search", "conv-import", "conv-read"}
_FOLDER_VERBS = {"folder-create", "folder-update", "folder-move", "folder-delete", "folder-list"}
_NO_FOLDER_WORD = "none"
_ITEM_UPDATE_KEYS = {"id", "expected_text", "new_text", "subject", "durability", "importance", "confidence", "tags"}
_EXIT_REFUSED = 1
SCHEMA_MIGRATE_VERB = "schema-migrate"
# Old (v21) command -> current command, still accepted with a warning on stderr. forget took the id as a
# positional argument; memory-delete takes --id (row authority, as item-delete).
OLD_COMMANDS = {
    "item-update": "memory-update", "item-delete": "memory-delete", "forget": "memory-delete",
    "archive": "memory-archive", "unarchive": "memory-unarchive",
    "restore": "backup-restore", "supersede-revert": "link-revert",
    "search": "memory-search", "status": "memory-count",
}
_POSITIONAL_ID_COMMANDS = {"forget"}
_GLOBAL_VALUE_OPTIONS = {"--home", "--owner"}


def _emit_json(payload: dict) -> None:
    # UTF-8 bytes regardless of the console code page; the dashboard reads this pipe.
    sys.stdout.buffer.write(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()


def _load_json_payload(path: str | None, fields: set[str], command: str) -> dict:
    raw = Path(path).read_bytes() if path else sys.stdin.buffer.read()
    request = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(request, dict) or request.keys() - fields:
        raise NoteError(f"{command} expects a JSON object with fields {sorted(fields)}")
    return request


def _note_save_payload(path: str | None) -> dict:
    return _load_json_payload(path, _NOTE_SAVE_KEYS, "note-save")


def _run_note_cmd(args: argparse.Namespace, store: MemoryStore) -> int:
    """Note verbs: JSON in, JSON out. Author defaults to human (the dashboard/editor path, origin human)."""
    try:
        if args.command in {"task-owners", "task-next"}:
            with store._lock:
                _emit_json(task_owners(store._conn) if args.command == "task-owners" else next_task_title(store._conn))
            return 0
        if args.command in {"note-read", "note-delete"}:
            reader = NeospikeService(store=store, embedder=None, namespace=args.namespace)
            if args.command == "note-read":
                _emit_json({"note": reader.read_note(note_id=args.id, title=args.title)})
            else:
                _emit_json({"note": reader.delete_note(args.id, author="human", expected_rev=args.expected_rev)})
            return 0
        embedder = EmbeddingClient(Path(args.home).expanduser())
        service = NeospikeService(store=store, embedder=embedder, namespace=args.namespace)
        if args.command == "note-search":
            _emit_json({"results": service.search_notes(args.query, limit=args.limit, project=args.project)})
            return 0
        if args.command == "note-restore":
            _emit_json({"note": service.restore_note(args.id, args.rev, author="human", expected_rev=args.expected_rev)})
            return 0
        request = _note_save_payload(args.input)
        author = request.get("author", "human")
        # origin: cron/topic callers pass "automation"; a human note then refuses the write
        # Default follows the author.
        origin = request.get("origin", ORIGIN_HUMAN if author == "human" else ORIGIN_CONVERSATION)
        saved = service.notes.save(
            request.get("title"), request.get("body"), tags=request.get("tags"), project=request.get("project", ""),
            author=author, expected_rev=request.get("expected_rev"), note_id=request.get("id"),
            origin=origin, namespace=args.namespace,
        )
        _emit_json({"note": saved})
        return 0
    except NoteError as exc:
        _emit_json(exc.as_dict())
        return _EXIT_REFUSED
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        _emit_json({"error": "invalid", "message": f"{type(exc).__name__}: {exc}"})
        return _EXIT_REFUSED


def _row_authority(store: MemoryStore, namespace: str, memory_id: int) -> tuple[str, str, str, dict] | None:
    """(project, session_id, kind, metadata) of one memory row, or None when it does not exist.

    The dashboard is the local owner, so it may change an item of any scope:
    it borrows the row's own project/session identity as the write authority.
    """
    with store._lock:
        row = store._conn.execute(
            "SELECT project,kind,scope,metadata_json FROM items WHERE owner=? AND id=? AND type='memory'",
            (namespace, memory_id),
        ).fetchone()
    if row is None:
        return None
    metadata = memory_metadata(row)
    session = metadata.get("session_id")
    return row["project"], session if isinstance(session, str) else "", row["kind"], metadata


def _run_item_cmd(args: argparse.Namespace, store: MemoryStore) -> int:
    """Dashboard memory-item writes. JSON in, JSON out, exit 1 on invalid input."""
    try:
        if args.command == "memory-delete":
            authority = _row_authority(store, args.namespace, args.id)
            removed = authority is not None and store.delete(
                args.namespace, args.id, project=authority[0], session_id=authority[1],
            )
            _emit_json({"removed": removed})
            return 0
        request = _load_json_payload(args.input, _ITEM_UPDATE_KEYS, "memory-update")
        memory_id = request.pop("id", None)
        if type(memory_id) is not int:
            raise ValueError("memory-update requires an integer id")
        authority = _row_authority(store, args.namespace, memory_id)
        if authority is None or authority[2] != "memory_item":
            _emit_json({"updated": False, "reason": "unavailable"})
            return 0
        # Omitted semantic fields keep the stored values (the editor changes text first).
        for field in ("subject", "durability"):
            if field not in request and isinstance(authority[3].get(field), str):
                request[field] = authority[3][field]
        embedder = EmbeddingClient(Path(args.home).expanduser())
        service = NeospikeService(store=store, embedder=embedder, namespace=args.namespace)
        # The exact stored string: Path()/resolve() could rewrite separators or case, and the
        # service only reads str(allowed_roots[0]) on this path.
        service.allowed_roots = [authority[0]] if authority[0] else []
        service.session_id = authority[1]
        _emit_json(service.update_memory_item(
            memory_id, expected_text=request.pop("expected_text", None), new_text=request.pop("new_text", None),
            edited_by=HUMAN_EDITOR, **request,
        ))
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        _emit_json({"error": "invalid", "message": str(exc)})
        return _EXIT_REFUSED


class _PrefVectors:
    """Text/situation vectors of a dashboard preference write; None (pending for embed-pending) when the
    embedder fails, so the edit is never lost."""

    def __init__(self, home: Path) -> None:
        self.home, self.client, self.failed = home, None, False

    def embed(self, text: str):
        if not text:
            return None
        try:
            if self.client is None:
                self.client = EmbeddingClient(self.home)
            return list(self.client.embed_document(text))
        except (RuntimeError, OSError, ValueError) as exc:
            self.failed = True
            _EmbedFailureSink(self.home)("preference_pending", f"{type(exc).__name__}: {exc}")
            return None


def _run_pref_cmd(args: argparse.Namespace, store: MemoryStore) -> int:
    """pref-add / pref-update / pref-delete / pref-list (dashboard Preferences tab)."""
    try:
        if args.command == "pref-list":
            with store._lock:
                rows = pref_ops.list_preferences(store._conn, args.namespace)
            _emit_json({"preferences": rows})
            return 0
        if args.command == "pref-delete":
            _emit_json({"removed": store.remove_preference(args.namespace, args.id), "id": args.id})
            return 0
        vectors = _PrefVectors(Path(args.home).expanduser())
        if args.command == "pref-add":
            request = _load_json_payload(args.input, _PREF_ADD_KEYS, "pref-add")
            fields = pref_ops.clean_fields({"basis": pref_ops.BASIS_USER_SAID, **request})
            pref_ops.check_situation(fields["basis"], fields["situation"])
            embedding = vectors.embed(fields["text"])
            pref_id = store.add_preference(args.namespace, fields, embedding=embedding,
                                           situation_embedding=vectors.embed(fields["situation"]))
            if pref_id is None:
                _emit_json({"error": "duplicate", "message": "The same preference is already stored"})
                return _EXIT_REFUSED
            tidy = tidy_saved_preference(store, args.namespace, pref_id) if embedding is not None else {}
            _emit_json({"stored": True, "preference": store.preference(args.namespace, pref_id),
                        "deleted": tidy.get("deleted", []), "embedding_pending": vectors.failed})
            return 0
        request = _load_json_payload(args.input, _PREF_UPDATE_KEYS, "pref-update")
        pref_id = request.pop("id", None)
        if type(pref_id) is not int:
            raise ValueError("pref-update requires an integer id")
        expected = request.pop("expected_text", None)
        changes = pref_ops.clean_fields(request, partial=True)
        current = store.preference(args.namespace, pref_id)
        if current is None:
            _emit_json({"error": "not_found", "message": "Preference not found"})
            return _EXIT_REFUSED
        if "situation" in changes or "basis" in changes:
            pref_ops.check_situation(changes.get("basis", current["basis"]),
                                     changes.get("situation", current["situation"]))
        text = changes.get("text", current["text"])
        situation = changes.get("situation", current["situation"])
        status = store.change_preference(
            args.namespace, pref_id, changes, expected_text=expected,
            embedding=vectors.embed(text) if text != current["text"] else None,
            situation_embedding=vectors.embed(situation) if situation != current["situation"] else None)
        if status != "updated":
            _emit_json({"error": status, "message": f"Preference not updated: {status}",
                        "current_text": current["text"]})
            return _EXIT_REFUSED
        _emit_json({"updated": True, "preference": store.preference(args.namespace, pref_id),
                    "embedding_pending": vectors.failed})
        return 0
    except pref_ops.PreferenceError as exc:
        _emit_json(exc.as_dict())
        return _EXIT_REFUSED
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError, OSError, sqlite3.Error) as exc:
        _emit_json({"error": "invalid", "message": str(exc)})
        return _EXIT_REFUSED


def _run_link_cmd(args: argparse.Namespace, store: MemoryStore) -> int:
    """link-revert / memory-revisions / memory-restore."""
    try:
        if args.command == "link-revert":
            output = store.undo_supersede(args.link)
            _emit_json(output)
            return 0 if output["reverted"] else _EXIT_REFUSED
        if args.command == "memory-revisions":
            output = store.memory_history(args.id)
            _emit_json(output if output is not None else {"error": "not_found", "id": args.id})
            return 0 if output is not None else _EXIT_REFUSED
        current = store.memory_history(args.id)
        revision = store.revision_at(args.id, args.rev)
        if current is None or revision is None:
            _emit_json({"restored": False, "reason": "not_found"})
            return _EXIT_REFUSED
        # The embedding is computed before the short write transaction.
        embedder = EmbeddingClient(Path(args.home).expanduser())
        output = store.rollback_memory_revision(args.id, args.rev, expected_text=current["text"],
                                               embedding=embedder.embed_document(revision["text"]))
        _emit_json(output)
        return 0 if output["restored"] else _EXIT_REFUSED
    except (ValueError, RuntimeError, OSError, sqlite3.Error) as exc:
        _emit_json({"error": "invalid", "message": str(exc)})
        return _EXIT_REFUSED


class _DeferredEmbedder:
    """embed_document object that starts the inference client on first use (none when nothing is pending)."""

    def __init__(self, factory) -> None:
        self.factory, self.client = factory, None

    def embed_document(self, text: str, **kwargs):
        if self.client is None:
            self.client = self.factory()
        return self.client.embed_document(text, **kwargs)


class _EmbedFailureSink:
    """failure_log(operation, cause) of embed_pending: one line in <home>/neospike/logs/embedding-failures-*.jsonl."""

    def __init__(self, home: Path) -> None:
        self.directory = home / "neospike" / LOG_DIRECTORY

    def __call__(self, operation: str, cause: str) -> None:
        try:
            append_failure(self.directory, operation, cause)
        except OSError:
            pass


def _run_embed_pending_cmd(args: argparse.Namespace, store: MemoryStore) -> int:
    home = Path(args.home).expanduser()
    embedder = _DeferredEmbedder(functools.partial(EmbeddingClient, home))
    output = embed_pending(store, embedder, chunk_editor=EMBED_PENDING_EDITED_BY, failure_log=_EmbedFailureSink(home))
    # The save follow-up (note-absorb -> topic-assign) a turn did not run: memories saved without a
    # vector, an interrupted turn, dashboard edits. Incremental, so nothing new means nothing done.
    output["follow_up"] = topic_ops.after_save(store, args.namespace)
    _emit_json(output)
    return 0


def _move_topic_links(args: argparse.Namespace, store: MemoryStore) -> dict:
    """Hand edit of topic members; only mem:<id> keys can be members (note keys are skipped)."""
    keys = folder_ops.parse_item_keys(args.items)
    ids = [item_id for lane, item_id in keys if lane == "memory"]
    embedder = EmbeddingClient(Path(args.home).expanduser())
    result = topic_ops.move_members(store, args.namespace, ids, embedder, target_note=args.to_note,
                                      remove_from_note=args.remove_from_note, target_topic=args.to_topic,
                                      remove_from_topic=args.remove_from_topic)
    result["skipped_notes"] = [folder_ops.item_key(lane, item_id) for lane, item_id in keys if lane != "memory"]
    return result


def _run_pref_reconcile_cmd(args: argparse.Namespace, store: MemoryStore) -> int:
    """pref_reconcile.py runs inside Hermes (idle task with ctx.llm); from the command line it only reports that
    (exit 1), or {"disabled": true} when pref_reconcile_enabled is false."""
    from . import pref_reconcile
    output = pref_reconcile.run(store, Path(args.home).expanduser(), None)
    _emit_json(output)
    if "error" in output:
        sys.stderr.write(f"neospike: {output['error']}\n")
        return _EXIT_REFUSED
    return 0


def _run_topic_cmd(args: argparse.Namespace, store: MemoryStore) -> int:
    """Topics (order in topics.py). No LLM is called."""
    try:
        if args.command == "topic-assign":
            _emit_json(topic_ops.assign_all(store, args.namespace))
        elif args.command == "note-absorb":
            _emit_json(topic_ops.absorb(store, args.namespace))
        elif args.command == "topic-to-note":
            output = topic_ops.topic_to_note(store, args.namespace)
            _emit_json(output)
            return _EXIT_REFUSED if output["errors"] else 0
        elif args.command == "topic-discover":
            _emit_json(topic_ops.discover(store, args.namespace))
        else:
            _emit_json(_move_topic_links(args, store))
        return 0
    except NoteError as exc:
        _emit_json(exc.as_dict())
        return _EXIT_REFUSED
    except (ValueError, OSError) as exc:
        _emit_json({"error": "invalid", "message": f"{type(exc).__name__}: {exc}"})
        return _EXIT_REFUSED


def _run_conv_cmd(args: argparse.Namespace, store: MemoryStore) -> int:
    """Conversation family. No LLM is called here."""
    try:
        if args.command == "conv-read":
            with store._lock:
                _emit_json(conv_ops.read_message(store._conn, args.conversation, args.seq))
            return 0
        if args.command == "conv-import":
            log = conv_ops.ProgressLog(store.path.parent, "conv-import")
            output = conv_ops.import_state_db(args.state_db, store=store, limit=args.limit, until=args.until,
                                                         budget_seconds=args.budget_seconds, progress=log)
            log.finish(output)
            if args.index and "error" not in output:
                # Chunks and embeddings are built after (outside) the import transactions.
                embedder = EmbeddingClient(Path(args.home).expanduser())
                output["index"] = _index_conversations(store, embedder, conv_ops.INDEX_BUDGET_SECONDS)
            _emit_json(output)
            return _EXIT_REFUSED if "error" in output else 0
        embedder = EmbeddingClient(Path(args.home).expanduser())
        if args.command == "conv-index":
            _emit_json(_index_conversations(store, embedder, args.budget_seconds))
            return 0
        with store._lock:
            searchable = conv_ops.has_searchable(store._conn)
        results = (conv_ops.search(store, args.query, embedder.embed_query(args.query), limit=args.limit)
                   if searchable and args.query.strip() else [])
        _emit_json({"results": results})
        return 0
    except conv_ops.ConversationError as exc:
        _emit_json(exc.as_dict())
        return _EXIT_REFUSED
    except (ValueError, OSError, RuntimeError) as exc:
        _emit_json({"error": "invalid", "message": f"{type(exc).__name__}: {exc}"})
        return _EXIT_REFUSED


def _run_reembed_cmd(home: Path, budget_seconds: float | None) -> int:
    """Open without the embedding check (the DB still holds the old model's vectors) and re-embed.
    Exit 0 when complete; 3 when the budget stopped it (rerun to continue)."""
    from .reembed import reembed
    store = MemoryStore(canonical_database_path(home / "neospike", "memory"), dimensions=EMBEDDING_DIMENSIONS,
                        verify_embedding=False)
    try:
        output = reembed(store, EmbeddingClient(home), budget_seconds=budget_seconds)
    finally:
        store.close()
    print(json.dumps(output, ensure_ascii=False))
    return 0 if output["complete"] else 3


def _index_conversations(store: MemoryStore, embedder: "EmbeddingClient", budget_seconds: float | None) -> dict:
    """conv-index with its progress/result lines in <neospike dir>/logs; an embedder error is
    logged and propagates (vectors done so far are already saved)."""
    log = conv_ops.ProgressLog(store.path.parent, "conv-index")
    try:
        output = conv_ops.ConversationIndexer(
            store, embedder, embed_batch=conv_ops.EMBED_BATCH).index_pending(
            budget_seconds=budget_seconds, progress=log)
    except Exception as exc:
        log.finish({"error": f"{type(exc).__name__}: {exc}"})
        raise
    log.finish(output)
    return output


def _run_folder_cmd(args: argparse.Namespace, store: MemoryStore) -> int:
    """Folder writes. JSON out, exit 1 on invalid input. --actor auto keeps an auto origin."""
    try:
        with store._lock:
            connection = store._conn
            if args.command == "folder-list":
                _emit_json({"folders": folder_ops.list_folders(connection, include_deleted=args.all)})
            elif args.command == "folder-create":
                _emit_json({"folder": folder_ops.create_folder(
                    connection, args.name, color=args.color, actor=args.actor)})
            elif args.command == "folder-update":
                _emit_json({"folder": folder_ops.update_folder(
                    connection, args.id, name=args.name, color=args.color, actor=args.actor)})
            elif args.command == "folder-delete":
                _emit_json(folder_ops.delete_folder(connection, args.id, actor=args.actor))
            else:
                target = None if args.folder.strip().lower() == _NO_FOLDER_WORD else _parse_folder_id(args.folder)
                _emit_json(folder_ops.move_items(
                    connection, target, folder_ops.parse_item_keys(args.items), actor=args.actor))
        return 0
    except folder_ops.FolderError as exc:
        _emit_json(exc.as_dict())
        return _EXIT_REFUSED


def _parse_folder_id(text: str) -> int:
    if not text.strip().isdigit() or int(text) < 1:
        raise folder_ops.FolderError("--folder must be a folder id or 'none'")
    return int(text)


def _dry_run_conv_import(args: argparse.Namespace, db: Path) -> int:
    """conv-import --dry-run: the memory DB is opened mode=ro (never through MemoryStore, whose
    open rewrites the file), so the file bytes stay the same."""
    connection = None
    try:
        if db.is_file():
            connection = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
        output = conv_ops.import_state_db(args.state_db, read_connection=connection, limit=args.limit,
                                                     until=args.until, budget_seconds=args.budget_seconds)
        _emit_json(output)
        return _EXIT_REFUSED if "error" in output else 0
    except conv_ops.ConversationError as exc:
        _emit_json(exc.as_dict())
        return _EXIT_REFUSED
    except (ValueError, OSError, sqlite3.Error) as exc:
        _emit_json({"error": "invalid", "message": f"{type(exc).__name__}: {exc}"})
        return _EXIT_REFUSED
    finally:
        if connection is not None:
            connection.close()


def _run_map_cmd(home: Path) -> int:
    """map: the dashboard sphere map as JSON. The DB is opened mode=ro (never through
    MemoryStore); the layout is cached in <home>/neospike/map-cache/map-<hash>.json."""
    from . import sphere_map  # only this command pays for the map module
    try:
        config = NeospikeConfig.load(home)
    except (OSError, ValueError):
        config = NeospikeConfig()
    recent_days, stale_days = config.map_recent_days, config.map_stale_days
    db = home / "neospike" / "memory.db"
    if not db.is_file():
        _emit_json(sphere_map.empty_map(recent_days, stale_days))
        return 0
    connection = None
    try:
        connection = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True, timeout=SQLITE_BUSY_TIMEOUT_SECONDS)
        payload = sphere_map.build_map(connection, recent_window_days=recent_days, stale_window_days=stale_days,
                                       cache_dir=sphere_map.cache_directory(home))
    except ImportError as exc:
        _emit_json({"error": "unavailable", "message": f"The map needs numpy in the Hermes plugin Python: {exc}"})
        return _EXIT_REFUSED
    except (sqlite3.Error, OSError, ValueError) as exc:
        _emit_json({"error": "invalid", "message": f"The map could not be computed: {type(exc).__name__}: {exc}"})
        return _EXIT_REFUSED
    finally:
        if connection is not None:
            connection.close()
    _emit_json(payload)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neospike", description="Inspect and maintain Hermes Neospike")
    parser.add_argument("--home", default=_windows_default_home())
    parser.add_argument("--owner", dest="namespace", default=None,
                        help="Owner of the memory items (v21: --namespace). Default: the live provider owner "
                             "recorded in the DB, else default:local")
    cmds = parser.add_subparsers(dest="command", required=True)
    cmds.add_parser("memory-count", help="Database path and memory items per owner, as JSON")
    search_p = cmds.add_parser("memory-search", help="Hybrid search over memory items")
    search_p.add_argument("query")
    search_p.add_argument("--limit", type=int, default=5)
    search_p.add_argument("--include-history", action="store_true",
                        help="Also show superseded memories (half score, superseded_by)")
    cmds.add_parser("link-prune", help="Drop item_links/folder_items rows whose memory or note is gone "
                                      "(never deletes memories or notes; also revisions of gone "
                                      "items, topic links of gone memories, and next pointers to gone rows)")
    cmds.add_parser(EMBED_PENDING_VERB, help="Embed memories saved without an embedding, give the memory items "
                                               "the save-time duplicate handling, then the save follow-up "
                                               "(note-absorb, topic-assign)")
    cmds.add_parser(PREF_RECONCILE_VERB, help="Similar preference/constraint pairs judged by an LLM: runs inside "
                                              "Hermes as an idle task; from the command line it exits 1")
    archive_p = cmds.add_parser("memory-archive", help="Human archive: take one memory out of prefetch (direct search keeps it)")
    archive_p.add_argument("--id", type=int, required=True)
    unarchive_p = cmds.add_parser("memory-unarchive", help="Put an archived memory back into prefetch")
    unarchive_p.add_argument("--id", type=int, required=True)
    note_save_p = cmds.add_parser("note-save", help="Save one note from a JSON object on stdin (single writer path)")
    note_save_p.add_argument("--input", help="Read the JSON object from this file instead of stdin")
    note_read_p = cmds.add_parser("note-read", help="Print one note as JSON")
    note_target_p = note_read_p.add_mutually_exclusive_group(required=True)
    note_target_p.add_argument("--id", type=int)
    note_target_p.add_argument("--title")
    note_search_p = cmds.add_parser("note-search", help="Hybrid search over note chunks")
    note_search_p.add_argument("query")
    note_search_p.add_argument("--limit", type=int, default=5)
    note_search_p.add_argument("--project")
    note_delete_p = cmds.add_parser("note-delete", help="Move one note to the trash (human author, rev CAS)")
    note_delete_p.add_argument("--id", type=int, required=True)
    note_delete_p.add_argument("--expected-rev", type=int, required=True)
    note_restore_p = cmds.add_parser("note-restore", help="Save an earlier rev of one note as a new rev")
    note_restore_p.add_argument("--id", type=int, required=True)
    note_restore_p.add_argument("--rev", type=int, required=True)
    note_restore_p.add_argument("--expected-rev", type=int)
    cmds.add_parser("task-owners", help="Owned files of status:active TASK_ notes and the paths they share, as JSON")
    cmds.add_parser("task-next", help="Next TASK_ number (largest in notes + 1), as JSON")
    item_update_p = cmds.add_parser("memory-update", help="CAS-update one memory item from a JSON object on stdin")
    item_update_p.add_argument("--input", help="Read the JSON object from this file instead of stdin")
    item_delete_p = cmds.add_parser("memory-delete", help="Delete one memory item (the row's own scope authority)")
    item_delete_p.add_argument("--id", type=int, required=True)
    supersede_revert_p = cmds.add_parser("link-revert", help="Undo one supersedes link (the old memory is active again)")
    supersede_revert_p.add_argument("--link", type=int, required=True, help="item_links.id of the supersedes row")
    history_p = cmds.add_parser("memory-revisions", help="Current text and earlier revs of one memory, as JSON")
    history_p.add_argument("--id", type=int, required=True)
    memory_restore_p = cmds.add_parser("memory-restore", help="Bring back rev R of one memory (the current text becomes a rev)")
    memory_restore_p.add_argument("--id", type=int, required=True)
    memory_restore_p.add_argument("--rev", type=int, required=True)
    cmds.add_parser("topic-assign", help="Link new/changed live memories (cos >= 0.6, 3 closest); a memory without "
                                        "a topic linked to a topic member joins that topic")
    cmds.add_parser("topic-discover", help="Linked groups of more than 5 memories without a topic -> new topics")
    cmds.add_parser("note-absorb", help="Live memories with cos >= 0.8 to an automatic note chunk join that note "
                                       "(embedding copied, memory deleted)")
    cmds.add_parser("topic-to-note", help="Topics with more than 10 live members become automatic notes; those "
                                         "memories and the topic are deleted")
    topic_move_p = cmds.add_parser("topic-member-move", help="Put memories into a topic or take them out by hand "
                                                          "(kept by topic-assign/discover)")
    topic_move_p.add_argument("--items", required=True, help="mem:<id>,... (note keys are skipped)")
    topic_move_target_p = topic_move_p.add_mutually_exclusive_group(required=True)
    topic_move_target_p.add_argument("--to-note", type=int, help="Page note id of the topic to put them into")
    topic_move_target_p.add_argument("--remove-from-note", type=int, help="Page note id of the topic to take them out of")
    topic_move_target_p.add_argument("--to-topic", help="Topic name to put them into")
    topic_move_target_p.add_argument("--remove-from-topic", help="Topic name to take them out of")
    backup_p = cmds.add_parser("backup", help="Monthly backup: full VACUUM INTO snapshot, then incremental zips")
    backup_mode_p = backup_p.add_mutually_exclusive_group()
    backup_mode_p.add_argument("--full", action="store_true", help="Start a new chain with a full snapshot")
    backup_mode_p.add_argument("--list", action="store_true", help="List backups, sizes and chain state")
    restore_p = cmds.add_parser("backup-restore", help="Rebuild a NEW database file from the latest backup chain")
    restore_p.add_argument("--to", required=True, help="New database path; an existing file is refused")
    conv_index_p = cmds.add_parser("conv-index", help="Chunk and embed conversation messages not indexed yet")
    conv_index_p.add_argument("--budget-seconds", type=float, default=conv_ops.INDEX_BUDGET_SECONDS,
                            help="Stop after this many seconds; the rest is left for the next run (0 = no limit)")
    conv_import_p = cmds.add_parser("conv-import", help="Copy past Hermes sessions from state.db (read-only) as ended conversations")
    conv_import_p.add_argument("--state-db", required=True, help="Hermes state.db; opened with mode=ro")
    conv_import_p.add_argument("--dry-run", action="store_true", help="Count only; the memory database is not changed")
    conv_import_p.add_argument("--limit", type=int, help="At most this many sessions (oldest first)")
    conv_import_p.add_argument("--index", action="store_true", help="Run conv-index after the import (default off)")
    conv_import_p.add_argument("--until", type=float,
                             help="Boundary T (epoch seconds): only messages at or before T (default: now)")
    conv_import_p.add_argument("--budget-seconds", type=float, default=conv_ops.IMPORT_BUDGET_SECONDS,
                             help="Stop after this many seconds; the rest is left for the next run (0 = no limit)")
    conv_read_p = cmds.add_parser("conv-read", help="One message of a conversation, its chunks joined along the chunk chain")
    conv_read_p.add_argument("--conversation", type=int, required=True, help="conversations.id")
    conv_read_p.add_argument("--seq", type=int, required=True, help="Message seq")
    conv_search_p = cmds.add_parser("conv-search", help="Hybrid search over conversation chunks (adjacent hits merged)")
    conv_search_p.add_argument("query")
    conv_search_p.add_argument("--limit", type=int, default=5)
    folder_create_p = cmds.add_parser("folder-create", help="Create a folder (colour: next unused member colour)")
    folder_create_p.add_argument("--name", required=True)
    folder_create_p.add_argument("--color", help="#RRGGBB member colour")
    folder_update_p = cmds.add_parser("folder-update", help="Rename a folder or change its member colour")
    folder_update_p.add_argument("--id", type=int, required=True)
    folder_update_p.add_argument("--name")
    folder_update_p.add_argument("--color", help="#RRGGBB member colour")
    folder_move_p = cmds.add_parser("folder-move", help="Put items into one folder (an item leaves its old folder)")
    folder_move_p.add_argument("--folder", required=True, help="Folder id, or 'none' for unfiled")
    folder_move_p.add_argument("--items", required=True, help="Comma separated mem:<id> / note:<id>")
    folder_delete_p = cmds.add_parser("folder-delete", help="Move a folder to the trash; its items become unfiled")
    folder_delete_p.add_argument("--id", type=int, required=True)
    folder_list_p = cmds.add_parser("folder-list", help="Folders with member counts, as JSON")
    folder_list_p.add_argument("--all", action="store_true", help="Include trashed folders")
    for folder_p in (folder_create_p, folder_update_p, folder_move_p, folder_delete_p):
        folder_p.add_argument("--actor", choices=folder_ops.ACTORS, default="human",
                               help="auto: automation edit; human edits mark an auto folder human")
    cmds.add_parser("map", help="Dashboard sphere map as JSON (read-only; cached in neospike/map-cache)")
    reembed_parser_p = cmds.add_parser("reembed", help="Rewrite every stored text vector with the current model "
                                    "(bge-m3, 1024); resumable, commits every 32 rows; back up the DB first")
    reembed_parser_p.add_argument("--budget-seconds", type=float, default=None,
                                help="Stop cleanly after this many seconds; rerun to continue")
    register_eval_cli(cmds)
    schema_p = cmds.add_parser(SCHEMA_MIGRATE_VERB, help="Rename a v21 database to schema v22 in one "
                                                         "verified transaction (stop every neospike process first)")
    schema_mode_p = schema_p.add_mutually_exclusive_group(required=True)
    schema_mode_p.add_argument("--dry-run", action="store_true", help="Migrate an in-memory copy and report (read only)")
    schema_mode_p.add_argument("--apply", action="store_true",
                               help="Back up to memory.pre-v22.db (+ config/MEMORY_RULE copies), migrate, move config")
    schema_mode_p.add_argument("--restore", action="store_true",
                               help="Put memory.pre-v22.db and the .pre-v22 config copies back")
    pref_add_p = cmds.add_parser("pref-add", help="Add one preference/constraint from a JSON object on stdin "
                                              "(basis user_said unless given)")
    pref_add_p.add_argument("--input", help="JSON file instead of stdin")
    pref_update_p = cmds.add_parser("pref-update", help="Edit one preference/constraint in place from a JSON object "
                                                    "on stdin (id, expected_text, type/text/situation/...)")
    pref_update_p.add_argument("--input", help="JSON file instead of stdin")
    pref_delete_p = cmds.add_parser("pref-delete", help="Delete one preference/constraint")
    pref_delete_p.add_argument("--id", type=int, required=True)
    cmds.add_parser("pref-list", help="Preferences/constraints of the owner, as JSON")
    return parser


def _run_basic_cmd(args: argparse.Namespace, store: MemoryStore, db: Path) -> int:
    """Commands that need no JSON-in plumbing: backup, namespace, link-prune, search and index verbs."""
    project_dir = Path.cwd().resolve()
    # This standalone entrypoint has no runtime-bound session authority; env vars cannot grant access.
    session_id = ""
    verb = args.command
    if verb == "backup":
        folder = backup_tools.backup_directory(Path(args.home))
        if args.list:
            row = store._conn.execute("SELECT value FROM meta_entries WHERE key=?", (backup_tools.META_LAST_NAME,)).fetchone()
            result = backup_tools.list_backups(folder, last_recorded=None if row is None else row[0])
        else:
            result = backup_tools.run_backup(db, folder, full=args.full)
    elif verb == "memory-count":
        result = {"database": str(db), "owners": store.list_namespaces()}
    elif verb == "link-prune":
        result = store.prune_orphan_links()
    elif verb == "memory-archive":
        result = {"archived": store.archive(args.namespace, args.id), "id": args.id}
    elif verb == "memory-unarchive":
        result = {"unarchived": store.unarchive(args.namespace, args.id), "id": args.id}
    else:
        client = EmbeddingClient(Path(args.home).expanduser())
        roots = [Path(getattr(args, "root", project_dir)).expanduser().resolve()]
        service = NeospikeService(store=store, embedder=client, namespace=args.namespace, allowed_roots=roots)
        service.session_id = session_id
        if verb == "memory-search":
            scope = HISTORY_SHOW if args.include_history else HISTORY_OMIT
            hits = service.search(args.query, limit=args.limit, history=scope)
            result = {"results": [hit.as_dict() for hit in hits]}
        else:
            raise RuntimeError(f"Unsupported command: {verb}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


# (verb set, handler) pairs tried in order; every handler takes (args, store) and returns an exit code.
_STORE_DISPATCH = (
    (_NOTE_VERBS, _run_note_cmd),
    (_ITEM_VERBS, _run_item_cmd),
    (_PREF_VERBS, _run_pref_cmd),
    (_TOPIC_VERBS, _run_topic_cmd),
    (_CONV_VERBS, _run_conv_cmd),
    (_FOLDER_VERBS, _run_folder_cmd),
    (_LINK_VERBS, _run_link_cmd),
)


def forward_old_command(argv: list[str], stderr=None) -> list[str]:
    """argv with a v21 command (OLD_COMMANDS) replaced by its v22 command; a warning goes to stderr."""
    argv = list(argv)
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in _GLOBAL_VALUE_OPTIONS:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        break
    if index >= len(argv) or argv[index] not in OLD_COMMANDS:
        return argv
    old, new = argv[index], OLD_COMMANDS[argv[index]]
    rest = argv[index + 1:]
    if old in _POSITIONAL_ID_COMMANDS and rest and not rest[0].startswith("-"):
        rest = ["--id", *rest]
    (stderr or sys.stderr).write(f"neospike: '{old}' is renamed to '{new}'; the old name stops working in the "
                                 "next release\n")
    return [*argv[:index], new, *rest]


def _run_schema_cmd(args: argparse.Namespace, home: Path) -> int:
    """schema-migrate: the database is never opened through MemoryStore here."""
    from . import schema_v22
    db = home / "neospike" / "memory.db"
    try:
        if args.restore:
            result = schema_v22.restore(home, db)
        elif not db.is_file():
            raise schema_v22.SchemaError(f"No memory database at {db}")
        elif args.dry_run:
            result = schema_v22.dry_run(db, embedder=EmbeddingClient(home))
        else:
            # The re-split human/agent note parts are embedded now (a part whose embedding fails stays
            # embedding-pending for the daily embed-pending stage).
            result = schema_v22.apply(home, db, embedder=EmbeddingClient(home))
    except (schema_v22.SchemaError, sqlite3.Error, OSError) as exc:
        _emit_json({"error": "refused", "message": f"{type(exc).__name__}: {exc}"})
        return _EXIT_REFUSED
    _emit_json(result)
    return _EXIT_REFUSED if result.get("ok") is False else 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(forward_old_command(forward_old_options(sys.argv[1:] if argv is None else argv)))
    try:
        return _run_command(args)
    except EmbeddingUnavailable as exc:
        # Every command that embeds (search, save, re-embed, migrate, ...) ends the same way.
        _emit_json({"error": "invalid", "message": f"{type(exc).__name__}: {exc}"})
        return _EXIT_REFUSED


def _run_command(args: argparse.Namespace) -> int:
    home = Path(args.home).expanduser()
    args.namespace = pick_namespace(args.namespace, memory_db_path(Path(args.home)))
    verb = args.command

    if verb == "eval":
        return run_eval_cli(args)
    if verb == SCHEMA_MIGRATE_VERB:
        return _run_schema_cmd(args, home)
    if verb == "backup-restore":
        # Reads only the backup files; the live database is never opened.
        result = backup_tools.restore(backup_tools.backup_directory(Path(args.home)), Path(args.to))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if verb == "map":
        return _run_map_cmd(home)
    if verb == "reembed":
        return _run_reembed_cmd(home, args.budget_seconds)
    if verb == "conv-import" and args.dry_run:
        # Not canonical_database_path: it creates the folder and may migrate a legacy file. A legacy
        # .sqlite predates the conversation tables, so it holds no session to skip.
        return _dry_run_conv_import(args, home / "neospike" / "memory.db")

    db = canonical_database_path(home / "neospike", "memory")
    NeospikeConfig.load(args.home)  # a broken config must stop the command before the store opens
    store = MemoryStore(db, dimensions=EMBEDDING_DIMENSIONS)
    try:
        for verbs, handler in _STORE_DISPATCH:
            if verb in verbs:
                return handler(args, store)
        if verb == EMBED_PENDING_VERB:
            return _run_embed_pending_cmd(args, store)
        if verb == PREF_RECONCILE_VERB:
            return _run_pref_reconcile_cmd(args, store)
        return _run_basic_cmd(args, store, db)
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
