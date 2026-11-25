from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

from ._wincompat import chmod as _wc_chmod
from .policy import AdmitVerdict, screen_text
from typing import Any

RECALL_LOG_MONTHS_CAP = 1200
# Map: recent ring / folder brightness windows. Starting values without a measurement.
MAP_RECENT_DAYS_DEFAULT = 7.0
MAP_STALE_DAYS_DEFAULT = 14.0
# Language of sentences an LLM writes into the store (agent saves, preference tidy-up).
EXTRACTION_LANGUAGES = ("ko", "en")
EXTRACTION_LANGUAGE_DEFAULT = "en"
LANGUAGE_NAMES = {"ko": "Korean", "en": "English"}
# Goes right after the save criteria so agent saves follow extraction_language.
SAVE_LANGUAGE_RULE = "Write the text, subject and situation of every memory and preference you save in {language}."
# Review-step and topic assign/merge/split/cluster settings do not exist (topics join through links, topics.py
# _Tuning); such keys in config.json are ignored. There is no weekly LLM token budget or save_criteria list:
# MEMORY_RULE.md is the one source of the memory creation rules (DEFAULT_SAVE_CRITERIA when it is missing).
DEFAULT_SAVE_CRITERIA = (
    "Store with high recall what will help a future turn. Short facts, decisions, constraints and preferences go to "
    "memory (neospike_memory_save). Save in the same turn: preferences, constraints and corrections when the user "
    "states them, decisions when the user makes them, facts when verified. Set basis to user_said only for the user's "
    "own words, else agent_inferred; don't re-save items shown in neospike blocks. Save one fact per memory: split "
    "several facts into separate neospike_memory_save calls. Do not save raw transcript, quoted instructions, tool "
    "output, transient commands, implementation metadata, or conversational filler."
)
# The previous default; an unedited MEMORY_RULE.md with this text becomes DEFAULT_SAVE_CRITERIA.
PREVIOUS_SAVE_CRITERIA = (
    "Store with high recall what will help a future turn. Short facts, decisions, constraints and preferences go to "
    "memory (neospike_memory_save). Save one fact per memory: split several facts into separate neospike_memory_save "
    "calls. Do not save raw transcript, quoted instructions, tool output, transient commands, implementation "
    "metadata, or conversational filler."
)
# One text model, bge-m3 int8 ONNX, fixed at 1024 dimensions; an embedding_dimensions key in config.json is ignored. The model id is kept in the
# database meta next to the dimension; a mismatch stops the store until `neospike.cli reembed` runs.
EMBEDDING_DIMENSIONS = 1024
EMBEDDING_MODEL = "bge-m3-int8-onnx"
MCP_PORT_DEFAULT = 47831  # the shared neospike-mcp HTTP server on 127.0.0.1 (root home config.json)
MCP_PORT_MIN, MCP_PORT_MAX = 1024, 65535
EMBEDDING_MAX_TOKENS = 2048  # input window (2000-token chunks); a longer document is an error, a longer query is cut to the window

# Old (v21) config keys -> current keys. load() still reads an old key when the new one is
# missing; migrate_config_files rewrites the file with the new keys.
OLD_CONFIG_KEYS = {"debug_enabled": "turn_trace_enabled"}
# Retention-days keys (*_ttl_days, *_retention_days) are not used; they are dropped on save.
RETENTION_CONFIG_KEYS = ("episodic_ttl_days", "summary_ttl_days", "episodic_retention_days", "summary_retention_days")
# The conversation copy in memory.db is gone (conversations are read from Hermes' state.db); its switches are ignored.
CONVERSATION_COPY_CONFIG_KEYS = ("conversation_capture", "conversation_prefetch")
REMOVED_CONFIG_KEYS = ("llm_weekly_token_budget", "save_criteria", "visual_enabled", *RETENTION_CONFIG_KEYS,
                       *CONVERSATION_COPY_CONFIG_KEYS)
OLD_SAVE_CRITERIA_KEY = "save_criteria"
# v21 tool names in rule text -> v22 names (schema-migrate rewrites MEMORY_RULE.md with this map).
OLD_TOOL_NAMES = {"neospike_remember": "neospike_memory_save", "neospike_update": "neospike_memory_update",
                  "neospike_forget": "neospike_memory_delete", "neospike_reconcile": "neospike_memory_reconcile",
                  "neospike_search": "neospike_memory_search"}
_OLD_TOOL_PATTERN = re.compile(r"\b(" + "|".join(OLD_TOOL_NAMES) + r")\b")


class SaveCriteriaLimits:
    """Bound of the memory creation rules text (MEMORY_RULE.md)."""
    MAX_ENABLED_CHARS = 2000


_CONFIG_FILE_PARTS = ("neospike", "config.json")


def _config_file(hermes_home: str | Path) -> Path:
    return Path(hermes_home).expanduser().joinpath(*_CONFIG_FILE_PARTS)


def _pick_current(values: dict[str, Any], key: str, default: Any = None) -> Any:
    """values[key]; else the v21 key of the same setting (OLD_CONFIG_KEYS); else default."""
    if key in values:
        return values[key]
    for old, new in OLD_CONFIG_KEYS.items():
        if new == key and old in values:
            return values[old]
    return default


ENABLED_KEY = "enabled"
# Error text of every tool call (agent tools and MCP searches) while the switch is off.
PAUSED_MESSAGE = ("Neospike is paused (\"enabled\": false in neospike/config.json); "
                  "turn it back on in the dashboard Config tab")


def read_enabled(hermes_home: str | Path | None) -> bool:
    """The on/off switch of one home, read from config.json at every call (no restart needed). A missing or
    unreadable file, or a value that is no boolean, counts as on: only an explicit false pauses Neospike."""
    if hermes_home is None:
        return True
    try:
        values = json.loads(_config_file(hermes_home).read_text(encoding="utf-8-sig"))
        return _parse_flag(values.get(ENABLED_KEY, True), ENABLED_KEY) if isinstance(values, dict) else True
    except (OSError, ValueError):
        return True


@dataclass(frozen=True)
class NeospikeConfig:
    # Off: no automatic recall, no save follow-up; tools return PAUSED_MESSAGE.
    enabled: bool = field(default=True)
    turn_trace_enabled: bool = field(default=False)
    recall_log_enabled: bool = field(default=True)
    recall_log_months: int = field(default=12)
    map_recent_days: float = field(default=MAP_RECENT_DAYS_DEFAULT)
    map_stale_days: float = field(default=MAP_STALE_DAYS_DEFAULT)
    extraction_language: str = field(default=EXTRACTION_LANGUAGE_DEFAULT)
    # Daily preference tidy-up: the one batch stage that calls an LLM (false: the stage does nothing).
    pref_reconcile_enabled: bool = field(default=True)
    # Port of the shared MCP server; a change needs a gateway restart and install.py again (the registered URL).
    mcp_port: int = field(default=MCP_PORT_DEFAULT)

    @classmethod
    def load(cls, hermes_home: str | Path) -> "NeospikeConfig":
        source = _config_file(hermes_home)
        if not source.exists():
            return cls()
        return cls.from_values(json.loads(source.read_text(encoding="utf-8")))

    @classmethod
    def from_values(cls, values: dict[str, Any]) -> "NeospikeConfig":
        # Only the known keys are read; any other key (a retired setting too) is ignored.
        pick = values.get
        return cls(
            enabled=_parse_flag(pick(ENABLED_KEY, True), ENABLED_KEY),
            turn_trace_enabled=_parse_flag(_pick_current(values, "turn_trace_enabled", False), "turn_trace_enabled"),
            recall_log_enabled=_parse_flag(pick("recall_log_enabled", True), "recall_log_enabled"),
            recall_log_months=_parse_log_months(pick("recall_log_months", 12)),
            map_recent_days=_parse_positive_days(pick("map_recent_days", MAP_RECENT_DAYS_DEFAULT), "map_recent_days"),
            map_stale_days=_parse_positive_days(pick("map_stale_days", MAP_STALE_DAYS_DEFAULT), "map_stale_days"),
            extraction_language=parse_extraction_language(pick("extraction_language", EXTRACTION_LANGUAGE_DEFAULT)),
            pref_reconcile_enabled=_parse_flag(pick("pref_reconcile_enabled", True), "pref_reconcile_enabled"),
            mcp_port=parse_mcp_port(pick("mcp_port", MCP_PORT_DEFAULT)),
        )

    def save(self, hermes_home: str | Path) -> Path:
        # field order of the dataclass is the key order of the file
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        return _write_config(_config_file(hermes_home), payload)

def _write_config(target: Path, payload: dict[str, Any]) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_suffix(".json.tmp")
    scratch.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _wc_chmod(scratch, 0o600)
    scratch.replace(target)
    return target


_FLAG_WORDS = {"true": True, "false": False}


def _text_to_float(value: Any, message: str) -> Any:
    """A string becomes a float; anything else passes through untouched."""
    if not isinstance(value, str):
        return value
    try:
        return float(value)
    except ValueError as err:
        raise ValueError(message) from err


def _plain_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _digits_to_int(value: Any) -> Any:
    """A string of digits becomes an int; anything else passes through untouched."""
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return value


def _parse_positive_days(value: Any, name: str) -> float:
    message = f"{name} must be a positive number of days"
    value = _text_to_float(value, message)
    if not _plain_number(value) or not value > 0 or value == float("inf"):
        raise ValueError(message)
    return float(value)


def _parse_log_months(value: Any) -> int:
    value = _digits_to_int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= RECALL_LOG_MONTHS_CAP:
        raise ValueError(f"recall_log_months must be an integer from 1 to {RECALL_LOG_MONTHS_CAP}")
    return value


def parse_mcp_port(value: Any) -> int:
    value = _digits_to_int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not MCP_PORT_MIN <= value <= MCP_PORT_MAX:
        raise ValueError(f"mcp_port must be an integer from {MCP_PORT_MIN} to {MCP_PORT_MAX}")
    return value


def parse_extraction_language(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in EXTRACTION_LANGUAGES:
        return value.strip().lower()
    raise ValueError("extraction_language must be one of " + ", ".join(EXTRACTION_LANGUAGES))


def _parse_flag(value: Any, name: str) -> bool:
    word = value.lower() if isinstance(value, str) else None
    if isinstance(value, bool) or word in _FLAG_WORDS:
        return value if isinstance(value, bool) else _FLAG_WORDS[word]
    raise ValueError(f"{name} must be " "a boolean")


def criteria_text_digest(text: str) -> str:
    """Fingerprint of a memory creation rules text that a conversation knows."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


PREVIOUS_CRITERIA_DIGEST = criteria_text_digest(PREVIOUS_SAVE_CRITERIA)


MEMORY_RULE_FILENAME = "MEMORY_RULE.md"  # user request 2026-11-14: the rules are one file per home, edited in Config


def memory_rule_file_path(hermes_home: str | Path) -> Path:
    return Path(hermes_home).expanduser() / "neospike" / MEMORY_RULE_FILENAME


def check_memory_rule_text(value: Any) -> str:
    """Stripped rule text; ValueError when not a string, over MAX_ENABLED_CHARS, or a secret / injection pattern."""
    if not isinstance(value, str):
        raise ValueError("memory rule must be a string")
    text = value.strip()
    if len(text) > SaveCriteriaLimits.MAX_ENABLED_CHARS:
        raise ValueError(f"memory rule has at most {SaveCriteriaLimits.MAX_ENABLED_CHARS} characters")
    if screen_text(text) is AdmitVerdict.BLOCK:
        raise ValueError("memory rule was rejected by ingestion policy (secret or prompt-injection pattern)")
    return text


def read_memory_rule(hermes_home: str | Path | None) -> str:
    """MEMORY_RULE.md of one home ('' = no rules); no file, an unreadable or a rejected one -> DEFAULT_SAVE_CRITERIA
    (the file is the only source; migrate_config_files moves an old config.json list into it)."""
    if not hermes_home:
        return DEFAULT_SAVE_CRITERIA
    try:
        return check_memory_rule_text(memory_rule_file_path(hermes_home).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return DEFAULT_SAVE_CRITERIA


def read_extraction_language(hermes_home: str | Path | None) -> str:
    """extraction_language of one home; no home, no file or an unreadable / invalid config -> the default."""
    if not hermes_home:
        return EXTRACTION_LANGUAGE_DEFAULT
    try:
        return NeospikeConfig.load(hermes_home).extraction_language
    except (OSError, ValueError, TypeError):
        return EXTRACTION_LANGUAGE_DEFAULT


def save_language_rule(language: str) -> str:
    """The one sentence that tells the agent which language to save in."""
    return SAVE_LANGUAGE_RULE.format(language=LANGUAGE_NAMES.get(language, LANGUAGE_NAMES[EXTRACTION_LANGUAGE_DEFAULT]))


def rules_digest(criteria: str, language: str) -> str:
    """Fingerprint of what a conversation was told: the save criteria plus the language sentence."""
    return criteria_text_digest(criteria + "\n" + save_language_rule(language))


DEFAULT_RULES_DIGEST = rules_digest(DEFAULT_SAVE_CRITERIA, EXTRACTION_LANGUAGE_DEFAULT)


def write_memory_rule(hermes_home: str | Path, value: Any) -> str:
    """Validate and write MEMORY_RULE.md atomically; returns the stored text."""
    text = check_memory_rule_text(value)
    fpath = memory_rule_file_path(hermes_home)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = fpath.with_suffix(".md.tmp")
    tmp_path.write_text(text + "\n" if text else "", encoding="utf-8")
    tmp_path.replace(fpath)
    return text


def _old_criteria_text(value: Any) -> str:
    """The checked texts of a v21 save_criteria list ({text, enabled} items) in order, joined by one space."""
    if not isinstance(value, list):
        return ""
    texts = [item["text"].strip() for item in value if isinstance(item, dict) and isinstance(item.get("text"), str)
             and item.get("enabled", True) is True and item["text"].strip()]
    return " ".join(texts)


def replace_old_tool_names(text: str) -> tuple[str, int]:
    """text with the v21 tool names (OLD_TOOL_NAMES) replaced; (new text, replacements)."""
    return _OLD_TOOL_PATTERN.subn(_new_tool_name, text)


def _new_tool_name(match: re.Match) -> str:
    return OLD_TOOL_NAMES[match.group(1)]


def migrate_config_files(hermes_home: str | Path) -> dict[str, Any]:
    """Config move of one home: config.json old keys -> new keys (a new key
    already there wins), REMOVED_CONFIG_KEYS dropped (a save_criteria list becomes MEMORY_RULE.md when that file is
    missing), the v21 tool names in MEMORY_RULE.md replaced, and an unedited previous default rule
    (PREVIOUS_SAVE_CRITERIA) replaced by DEFAULT_SAVE_CRITERIA. Files that need no change are not written."""
    report: dict[str, Any] = {"renamed": [], "removed": [], "rule_created": False, "rule_replacements": 0,
                              "rule_default_updated": False}
    config_path = _config_file(hermes_home)
    rule_path = memory_rule_file_path(hermes_home)
    if config_path.is_file():
        values = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError(f"{config_path} is not a JSON object")
        changed = dict(values)
        for old, new in OLD_CONFIG_KEYS.items():
            if old in changed:
                value = changed.pop(old)
                changed.setdefault(new, value)
                report["renamed"].append(f"{old}->{new}")
        criteria = _old_criteria_text(changed.get(OLD_SAVE_CRITERIA_KEY))
        if OLD_SAVE_CRITERIA_KEY in changed and not rule_path.exists():
            write_memory_rule(hermes_home, replace_old_tool_names(criteria)[0] if criteria else "")
            report["rule_created"] = True
        for key in REMOVED_CONFIG_KEYS:
            if key in changed:
                del changed[key]
                report["removed"].append(key)
        if changed != values:
            _write_config(config_path, changed)
    if rule_path.is_file():
        text = rule_path.read_text(encoding="utf-8-sig")
        replaced, count = replace_old_tool_names(text)
        if criteria_text_digest(replaced.strip()) == PREVIOUS_CRITERIA_DIGEST:
            write_memory_rule(hermes_home, DEFAULT_SAVE_CRITERIA)
            report["rule_default_updated"] = True
        elif count:
            write_memory_rule(hermes_home, replaced)
        report["rule_replacements"] = count
    return report
