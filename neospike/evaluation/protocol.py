"""Pre-registration seal and test ledger.

`seal` writes protocol.json: the question-set SHA-256 (CRLF -> LF first), split counts, lanes, grid, selection rule,
gate values, the neospike source hash and the database fingerprint. `check` lists what differs; the test command
refuses a set whose hash differs (code and database differences are reported, since a test run exists to score
changed code). Every test run appends one line to ledger.jsonl beside protocol.json; past
GateTuning.LEDGER_WARN_RUNS runs on one sealed set a new test set is advised (adaptive overfitting, Dwork 2015).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..eval_recall import SPLITS
from .core import GateTuning

PROTOCOL_VERSION = 1
LEDGER_NAME = "ledger.jsonl"
# Tables whose (row count, max id) fingerprint a database state.
FINGERPRINT_TABLES = ("items", "item_links", "conversation_chunks")
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def normalized_sha256(path: Path) -> str:
    """SHA-256 of the file bytes with CRLF folded to LF (a checkout on Windows hashes the same)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def code_sha256(root: Path = PACKAGE_ROOT) -> str:
    """Hash of every neospike *.py source (relative path + LF-normalised bytes), tests excluded.
    Works without git (some installs are zip copies)."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("tests/") or "__pycache__" in relative:
            continue
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def database_fingerprint(path: Path) -> dict[str, Any]:
    """{table: [rows, max id]} of FINGERPRINT_TABLES (absent tables: null), read-only."""
    if not Path(path).is_file():
        return {}
    reader = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        present = {row[0] for row in reader.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        result: dict[str, Any] = {}
        for table in FINGERPRINT_TABLES:
            if table not in present:
                result[table] = None
                continue
            rows, top = reader.execute(f"SELECT COUNT(*), COALESCE(MAX(id), 0) FROM {table}").fetchone()
            result[table] = [int(rows), int(top)]
        return result
    finally:
        reader.close()


def split_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Question count per split of raw set rows (fixture rows still name corpus keys)."""
    counts = {name: 0 for name in SPLITS}
    for row in rows:
        split = row.get("split")
        counts[split] = counts.get(split, 0) + 1
    return counts


def build_protocol(set_path: Path, splits: dict[str, int], *, lanes: list[str], grid: list[float],
                   threshold: float | None, database: Path | None, corpus: Path | None = None) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "sealed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "set": {"path": str(Path(set_path).resolve()), "sha256": normalized_sha256(set_path),
                "splits": dict(splits)},
        "metrics": ["recall", "mrr", "fpr", "irrelevant", "stale_injected"],
        "lanes": list(lanes),
        "grid": list(grid),
        "threshold": threshold,  # null: the product default at test time
        "rule": {"delta": GateTuning.SELECT_DELTA, "order": ["fpr", "irrelevant", "-threshold"],
                 "target_recall": GateTuning.TARGET_RECALL, "target_fpr": GateTuning.TARGET_FPR},
        "gate": {"bootstrap_resamples": GateTuning.BOOTSTRAP_RESAMPLES, "bootstrap_seed": GateTuning.BOOTSTRAP_SEED,
                 "ci_level": GateTuning.CI_LEVEL, "rule": "recall CI high < 0 or stale_injected grew"},
        "code_sha256": code_sha256(),
        "database": database_fingerprint(database) if database else {},
        "corpus_sha256": normalized_sha256(corpus) if corpus else None,  # synthetic fixture source only
    }


def write_protocol(path: Path, protocol: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8-sig")
    return path


def read_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(protocol, dict) or protocol.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"protocol.json version must be {PROTOCOL_VERSION}")
    return protocol


def check(protocol: dict[str, Any], set_path: Path, database: Path | None,
          corpus: Path | None = None) -> dict[str, Any]:
    """{'refused': [...], 'changed': [...]}: refused = the set (or fixture corpus) hash differs;
    changed = code or database."""
    refused, changed = [], []
    if normalized_sha256(set_path) != protocol["set"]["sha256"]:
        refused.append("set_sha256")
    if code_sha256() != protocol.get("code_sha256"):
        changed.append("code_sha256")
    if database is not None and protocol.get("database") and database_fingerprint(database) != protocol["database"]:
        changed.append("database")
    if corpus is not None and protocol.get("corpus_sha256") and normalized_sha256(corpus) != protocol["corpus_sha256"]:
        refused.append("corpus_sha256")
    return {"refused": refused, "changed": changed}


def ledger_path(protocol_path: Path) -> Path:
    return Path(protocol_path).parent / LEDGER_NAME


def append_ledger(protocol_path: Path, protocol: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """One line per test run; returns {'runs': runs on this sealed set, 'warn': past the limit}."""
    path = ledger_path(protocol_path)
    runs = 1 + sum(1 for entry in read_ledger(path) if entry.get("set_sha256") == protocol["set"]["sha256"])
    entry = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "set_sha256": protocol["set"]["sha256"],
             "code_sha256": code_sha256(), "run": runs, "summary": summary}
    with path.open("a", encoding="utf-8-sig") as handle:  # BOM only on a new file (TextIOWrapper skips it after 0)
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"runs": runs, "warn": runs > GateTuning.LEDGER_WARN_RUNS, "limit": GateTuning.LEDGER_WARN_RUNS}


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]
