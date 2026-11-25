"""Daily tidy-up of similar preference/constraint pairs, judged by an LLM.

It runs inside Hermes as a plugin idle task (once a day, when Hermes has been idle), with the host-owned plugin
LLM (ctx.llm) routed through the auxiliary task auxiliary.neospike_pref_reconcile. No Hermes internals are imported.

Code picks the pairs: rows of the same owner whose text vectors have cos >= RememberTuning.CONTRADICTION_COSINE
and that the save-time duplicate rule (notes.tidy_saved_preference) does not already remove. An LLM then decides
per pair: keep both, update (rewrite one row, delete the other) or delete (delete one row). Decisions are applied
with the existing store wrappers (change_preference / remove_preference); a rewritten row loses its text vector
and the next embed-pending stage fills it. Pairs judged "keep" are remembered in meta_entries (with the text
hashes of both rows), so the same pair is not asked again until one of its rows changes.
"""
from __future__ import annotations

import json
import logging
import math
from array import array
from pathlib import Path
from typing import Any, Callable, Iterable

from . import memory_reconcile as memory_pairs
from . import vector_index
from . import preferences as pref_ops
from .config import EMBEDDING_DIMENSIONS, EXTRACTION_LANGUAGE_DEFAULT, LANGUAGE_NAMES, NeospikeConfig, read_enabled
from .database import canonical_database_path
from .notes import RememberTuning
from .store import MemoryStore, vector_present_sql

ACTION_KEEP, ACTION_UPDATE, ACTION_DELETE = "keep", "update", "delete"
ACTIONS = (ACTION_KEEP, ACTION_UPDATE, ACTION_DELETE)
KEPT_META_KEY = "pref_reconcile_kept"
LLM_TASK = "neospike_pref_reconcile"  # auxiliary.<task> in the Hermes config picks the model; unset: main model
LLM_TASK_DISPLAY_NAME = "Neospike preference tidy-up"
LLM_TASK_DESCRIPTION = "Daily check of similar stored preferences/constraints: keep both, merge or delete one."
LLM_PURPOSE = "neospike preference tidy-up"
IDLE_TASK_NAME = "neospike-pref-reconcile"
IDLE_INTERVAL_HOURS = 24.0  # Hermes runs it when idle (its default: 2 hours without activity), once a day
NO_LLM_MESSAGE = ("pref-reconcile needs the Hermes plugin LLM (ctx.llm) and runs inside Hermes as the idle "
                  "task 'neospike-pref-reconcile'; it cannot run from the command line.")
logger = logging.getLogger(__name__)


class ReconcileTuning:
    PAIRS_PER_CALL = 32
    MAX_CALLS = 4  # pairs beyond MAX_CALLS * PAIRS_PER_CALL wait for the next day (highest cos first)
    MAX_TOKENS = 4096
    TIMEOUT_SECONDS = 180.0


SKIP_BAD_JSON = "bad_json"
SKIP_UNKNOWN_PAIR = "unknown_pair"
SKIP_BAD_DECISION = "bad_decision"
SKIP_NO_ANSWER = "no_answer"
SKIP_PROTECTED = "user_said_protected"
SKIP_REFUSED = "write_refused"
SKIP_LLM_ERROR = "llm_error"
SKIP_CONSTRAINT_COMMON = "constraint_needs_situation"

SYSTEM_PROMPT = (
    "You tidy a user's stored preferences and constraints. Each input item is a pair of rows of the same user "
    "whose texts are similar. For every pair return one decision:\n"
    '- "keep": both rows stay (they apply to different situations, or say different things).\n'
    '- "update": merge the pair into one row: keep_id is the row that stays, text is its new full sentence; '
    "the other row is deleted.\n"
    '- "delete": keep_id is the row that stays unchanged; the other row is deleted.\n'
    "Rules: if both rows mean the same, keep one (the newer and more specific; use update when a merged sentence "
    "is clearer). If one reverses the other, keep the newer one (by updated_at). If their situation fields differ "
    "and both can be true, choose keep. Never delete a row with basis user_said because of a row with basis "
    "agent_inferred. When unsure, choose keep.\n"
    "Write every new text in {language}.\n"
    'Answer with a JSON array only, one object per pair: {{"pair":[id_a,id_b],"action":"keep|update|delete",'
    '"keep_id":id or null,"text":"new text" or null}}. No other words.'
)


def system_prompt(language: str) -> str:
    return SYSTEM_PROMPT.format(language=LANGUAGE_NAMES.get(language, LANGUAGE_NAMES[EXTRACTION_LANGUAGE_DEFAULT]))


class PluginLlmCall:
    """One chat completion through the host-owned plugin LLM (``ctx.llm`` of a Hermes PluginContext),
    routed through the auxiliary task this plugin registers."""

    def __init__(self, plugin_llm: Any) -> None:
        self._plugin_llm = plugin_llm

    def __call__(self, messages: list[dict]) -> str:
        result = self._plugin_llm.complete(messages, temperature=0, max_tokens=ReconcileTuning.MAX_TOKENS,
                                           timeout=ReconcileTuning.TIMEOUT_SECONDS, task=LLM_TASK,
                                           purpose=LLM_PURPOSE)
        text = getattr(result, "text", "")
        return text if isinstance(text, str) else ""


class IdleReconcile:
    """The idle task: one run for the home it was registered in. Hermes calls it with an IdleTaskRun."""

    def __init__(self, plugin_llm: Any, home: str | Path) -> None:
        self._llm = PluginLlmCall(plugin_llm)
        self._home = Path(home)

    def __call__(self, run_info: Any = None) -> str:
        if not read_enabled(self._home):
            logger.info("Neospike pref-reconcile skipped: Neospike is paused (enabled: false)")
            return "paused"
        store = MemoryStore(canonical_database_path(self._home / "neospike", "memory"),
                            dimensions=EMBEDDING_DIMENSIONS)
        try:
            result = run(store, self._home, self._llm, embedder=memory_pairs.LazyEmbedder(self._home))
        finally:
            store.close()
        if result.get("disabled"):
            logger.info("Neospike pref-reconcile skipped: pref_reconcile_enabled is false")
            return "disabled"
        summary = summary_line(result)
        errors = [*result["errors"], *result.get("memory", {}).get("errors", [])]
        if errors:
            raise RuntimeError(f"{summary}; LLM call failed: {errors[0]}")
        return summary


def summary_line(result: dict) -> str:
    line = (f"asked {result['asked']} of {result['candidates']} pairs in {result['calls']} calls: "
            f"keep {result[ACTION_KEEP]}, update {result[ACTION_UPDATE]}, delete {result[ACTION_DELETE]}, "
            f"skipped {result['skipped']}, deferred {result['deferred']}")
    return f"{line}; {memory_pairs.summary_part(result['memory'])}" if "memory" in result else line


def register_idle_task(ctx: Any, home: str | Path | None = None) -> bool:
    """Register the auxiliary LLM slot and the daily idle task on a plugin context that has ``llm``.

    The memory-provider load hands register() a collector without ``llm``; nothing is registered there and
    the general plugin load (plugin.yaml kind: standalone, listed in plugins.enabled) registers it.
    Returns True when the task was registered."""
    plugin_llm = getattr(ctx, "llm", None)
    if plugin_llm is None:
        logger.info("Neospike pref-reconcile not scheduled by this load: the plugin context has no llm")
        return False
    try:
        ctx.register_auxiliary_task(LLM_TASK, display_name=LLM_TASK_DISPLAY_NAME,
                                    description=LLM_TASK_DESCRIPTION)
        ctx.register_idle_task(IDLE_TASK_NAME, IdleReconcile(plugin_llm, home or _hermes_home()),
                               interval_hours=IDLE_INTERVAL_HOURS)
    except Exception as exc:  # a failed schedule must not cost the hooks registered by the same load
        logger.warning("Neospike pref-reconcile not scheduled: %s: %s", type(exc).__name__, exc)
        return False
    logger.info("Neospike pref-reconcile scheduled as idle task '%s' (every %g hours)",
                IDLE_TASK_NAME, IDLE_INTERVAL_HOURS)
    return True


def _hermes_home() -> Path:
    from hermes_constants import get_hermes_home
    return Path(get_hermes_home())


# -- candidate pairs (code only) ---------------------------------------------------------------------------

_ROW_COLUMNS = "id,owner,type,text,situation,basis,COALESCE(updated_at,created_at),text_hash,embedding"


def _row_dict(rec: tuple) -> dict:
    return {"id": int(rec[0]), "owner": rec[1], "type": rec[2], "text": rec[3], "situation": rec[4],
            "basis": rec[5], "updated_at": rec[6], "text_hash": rec[7]}


def _is_save_duplicate(similarity: float, left_text: str, right_text: str) -> bool:
    """The save-time duplicate rule of notes.tidy_saved_preference (that rule removes such pairs itself)."""
    gap = abs(len(str(left_text).split()) - len(str(right_text).split()))
    return similarity >= RememberTuning.DUPLICATE_COSINE and gap <= RememberTuning.DUPLICATE_WORD_GAP


def _owner_pairs(store: MemoryStore, records: list[tuple]) -> list[tuple[float, dict, dict]]:
    """Every pair a.id < b.id of one owner's rows at or above CONTRADICTION_COSINE that the save-time rule would
    not have removed: each row is a query against the owner's rows, only the later ones are kept."""
    pairs = []
    if len(records) < 2:
        return pairs
    ids = [int(rec[0]) for rec in records]
    rows = vector_index.similarities_many(store._conn, pref_ops.PREFERENCES_TABLE, ids,
                                          [array("f", rec[8]) for rec in records[:-1]],
                                          threshold=RememberTuning.CONTRADICTION_COSINE)
    for index, scores in enumerate(rows):
        for offset in range(index + 1, len(records)):
            similarity, other = scores[offset], records[offset]
            if (not math.isfinite(similarity) or similarity < RememberTuning.CONTRADICTION_COSINE
                    or _is_save_duplicate(similarity, records[index][3], other[3])):
                continue
            pairs.append((float(similarity), _row_dict(records[index]), _row_dict(other)))
    return pairs


def candidate_pairs(store: MemoryStore) -> list[tuple[float, dict, dict]]:
    """(cos, row a, row b) of every owner, a.id < b.id, highest cos first. Rows without a text vector are left out."""
    with store._lock:
        records = store._conn.execute(
            f"SELECT {_ROW_COLUMNS} FROM {pref_ops.PREFERENCES_TABLE} WHERE {vector_present_sql()} "
            "ORDER BY owner, id").fetchall()
    by_owner: dict[str, list[tuple]] = {}
    for rec in records:
        by_owner.setdefault(rec[1], []).append(rec)
    pairs = []
    for owner_records in by_owner.values():
        pairs.extend(_owner_pairs(store, owner_records))
    pairs.sort(key=_pair_order)
    return pairs


def _pair_order(pair: tuple[float, dict, dict]) -> tuple[float, int, int]:
    return -pair[0], pair[1]["id"], pair[2]["id"]


def pair_key(left: dict, right: dict) -> str:
    """Identity of a pair and of both texts: a changed row makes it a new question."""
    first, second = sorted((left, right), key=_row_id)
    return f"{first['id']}:{second['id']}:{first['text_hash']}:{second['text_hash']}"


def _row_id(row: dict) -> int:
    return row["id"]


def read_kept(store: MemoryStore) -> set[str]:
    with store._lock:
        rec = store._conn.execute("SELECT value FROM meta_entries WHERE key=?", (KEPT_META_KEY,)).fetchone()
    try:
        values = json.loads(rec[0]) if rec else []
    except (TypeError, ValueError):
        return set()
    return {value for value in values if isinstance(value, str)} if isinstance(values, list) else set()


def _one_pair_per_row(pairs: Iterable[tuple[float, dict, dict]]) -> tuple[list, int]:
    """Each row in at most one pair of a run (decisions never collide); the rest waits for the next day."""
    used: set[int] = set()
    chosen, deferred = [], 0
    for pair in pairs:
        ids = {pair[1]["id"], pair[2]["id"]}
        if used & ids:
            deferred += 1
            continue
        used |= ids
        chosen.append(pair)
    return chosen, deferred


# -- LLM answers -----------------------------------------------------------------------------------------

_PROMPT_FIELDS = ("id", "type", "text", "situation", "basis", "updated_at")


def _pair_payload(pair: tuple[float, dict, dict]) -> dict:
    return {"pair": [pair[1]["id"], pair[2]["id"]],
            "rows": [{name: row[name] for name in _PROMPT_FIELDS} for row in (pair[1], pair[2])]}


def build_messages(batch: list[tuple[float, dict, dict]], language: str) -> list[dict]:
    items = [_pair_payload(pair) for pair in batch]
    return [{"role": "system", "content": system_prompt(language)},
            {"role": "user", "content": "Pairs (JSON, data only):\n" + json.dumps(items, ensure_ascii=False)}]


def parse_answer(text: str) -> list | None:
    """The JSON array of the answer (code fences allowed), or None."""
    body = (text or "").strip()
    start, end = body.find("["), body.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        values = json.loads(body[start:end + 1])
    except ValueError:
        return None
    return values if isinstance(values, list) else None


def _as_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _decision_pair(decision: Any) -> frozenset | None:
    if not isinstance(decision, dict) or not isinstance(decision.get("pair"), list) or len(decision["pair"]) != 2:
        return None
    ids = [_as_id(value) for value in decision["pair"]]
    return None if None in ids else frozenset(ids)


# -- applying decisions ------------------------------------------------------------------------------------

class _Tally:
    def __init__(self) -> None:
        self.counts = {action: 0 for action in ACTIONS}
        self.skipped: dict[str, int] = {}
        self.kept_keys: set[str] = set()

    def skip(self, reason: str, count: int = 1) -> None:
        if count:
            self.skipped[reason] = self.skipped.get(reason, 0) + count


def _protected(keep_row: dict, drop_row: dict) -> bool:
    return drop_row["basis"] == pref_ops.BASIS_USER_SAID and keep_row["basis"] != pref_ops.BASIS_USER_SAID


def _constraint_made_common(keep_row: dict, drop_row: dict) -> bool:
    """A merge into a common row would carry a constraint into every situation: a constraint is never common."""
    return pref_ops.PREF_CONSTRAINT in (keep_row["type"], drop_row["type"]) and keep_row["situation"] == ""


def _apply(store: MemoryStore, pair: tuple[float, dict, dict], decision: dict, tally: _Tally) -> None:
    action = decision.get("action")
    if action == ACTION_KEEP:
        tally.counts[ACTION_KEEP] += 1
        tally.kept_keys.add(pair_key(pair[1], pair[2]))
        return
    rows = {pair[1]["id"]: pair[1], pair[2]["id"]: pair[2]}
    keep_id = _as_id(decision.get("keep_id"))
    if action not in (ACTION_UPDATE, ACTION_DELETE) or keep_id not in rows:
        tally.skip(SKIP_BAD_DECISION)
        return
    keep_row = rows[keep_id]
    drop_row = pair[2] if keep_id == pair[1]["id"] else pair[1]
    if _protected(keep_row, drop_row):
        tally.skip(SKIP_PROTECTED)
        return
    owner = keep_row["owner"]
    if action == ACTION_UPDATE:
        try:
            changes = pref_ops.clean_fields({"text": decision.get("text")}, partial=True)
        except pref_ops.PreferenceError:
            tally.skip(SKIP_BAD_DECISION)
            return
        if _constraint_made_common(keep_row, drop_row):
            tally.skip(SKIP_CONSTRAINT_COMMON)
            return
        if changes["text"] != keep_row["text"]:
            try:
                outcome = store.change_preference(owner, keep_id, changes, expected_text=keep_row["text"])
            except pref_ops.PreferenceError:
                tally.skip(SKIP_CONSTRAINT_COMMON)
                return
            if outcome == "duplicate" and changes["text"] == drop_row["text"]:
                # The merged text is the other row's text: keeping that row unchanged is the same result.
                keep_row, drop_row = drop_row, keep_row
                if _protected(keep_row, drop_row):
                    tally.skip(SKIP_PROTECTED)
                    return
                if _constraint_made_common(keep_row, drop_row):
                    tally.skip(SKIP_CONSTRAINT_COMMON)
                    return
            elif outcome != "updated":
                tally.skip(SKIP_REFUSED)
                return
        if not store.remove_preference(owner, drop_row["id"]):
            tally.skip(SKIP_REFUSED)
            return
        tally.counts[ACTION_UPDATE] += 1
        return
    if not store.remove_preference(owner, drop_row["id"]):
        tally.skip(SKIP_REFUSED)
        return
    tally.counts[ACTION_DELETE] += 1


def _judge_batch(store: MemoryStore, batch: list, llm: Callable[[list[dict]], str], language: str,
                 tally: _Tally, errors: list[str]) -> None:
    try:
        answer = llm(build_messages(batch, language))
    except Exception as exc:  # any provider failure: the batch is skipped, the stage reports it
        errors.append(f"{type(exc).__name__}: {exc}")
        tally.skip(SKIP_LLM_ERROR, len(batch))
        return
    decisions = parse_answer(answer)
    if decisions is None:
        tally.skip(SKIP_BAD_JSON, len(batch))
        return
    by_pair = {frozenset((pair[1]["id"], pair[2]["id"])): pair for pair in batch}
    answered: set[frozenset] = set()
    for decision in decisions:
        ids = _decision_pair(decision)
        if ids is None or ids not in by_pair or ids in answered:
            tally.skip(SKIP_UNKNOWN_PAIR)
            continue
        answered.add(ids)
        _apply(store, by_pair[ids], decision, tally)
    tally.skip(SKIP_NO_ANSWER, len(by_pair) - len(answered))


def reconcile(store: MemoryStore, llm: Callable[[list[dict]], str], *, language: str) -> dict:
    """One daily run. Returns counts: candidates, already_kept, asked, deferred, calls, keep/update/delete,
    skipped (total) and skip_reasons, errors (LLM call failures)."""
    kept_before = read_kept(store)
    pairs = candidate_pairs(store)
    live_keys = {pair_key(pair[1], pair[2]) for pair in pairs}
    fresh = [pair for pair in pairs if pair_key(pair[1], pair[2]) not in kept_before]
    chosen, deferred = _one_pair_per_row(fresh)
    limit = ReconcileTuning.MAX_CALLS * ReconcileTuning.PAIRS_PER_CALL
    deferred += max(0, len(chosen) - limit)
    chosen = chosen[:limit]
    tally, errors, calls = _Tally(), [], 0
    for start in range(0, len(chosen), ReconcileTuning.PAIRS_PER_CALL):
        calls += 1
        _judge_batch(store, chosen[start:start + ReconcileTuning.PAIRS_PER_CALL], llm, language, tally, errors)
    # Remembered keep pairs whose rows are gone or changed drop out (their key no longer matches a live pair).
    kept_after = (kept_before & live_keys) | tally.kept_keys
    store.write_meta(KEPT_META_KEY, json.dumps(sorted(kept_after)))
    return {"candidates": len(pairs), "already_kept": len(pairs) - len(fresh), "asked": len(chosen),
            "deferred": deferred, "calls": calls, **tally.counts, "skipped": sum(tally.skipped.values()),
            "skip_reasons": tally.skipped, "errors": errors, "language": language}


def run(store: MemoryStore, home: str | Path, llm: Callable[[list[dict]], str] | None, *,
        embedder: Any = None) -> dict:
    """One pref-reconcile run: settings from <home>/neospike/config.json. Without an llm: {"error": ...}.
    With an embedder, the waiting save-time memory pairs (memory_reconcile.judge) follow the preferences; their
    counts are under "memory"."""
    config = NeospikeConfig.load(home)
    if not config.pref_reconcile_enabled:
        return {"disabled": True}
    if llm is None:
        return {"error": NO_LLM_MESSAGE}
    result = reconcile(store, llm, language=config.extraction_language)
    if embedder is not None:
        result["memory"] = memory_pairs.judge(store, llm, embedder, language=config.extraction_language)
    return result
