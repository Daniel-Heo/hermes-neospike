"""`neospike eval` commands: run, sweep, seal, test.

Data source: --home (a live HERMES_HOME, snapshotted read-only) or --corpus with --set naming corpus keys
(the synthetic fixture; --lexicon defaults to lexicon.json beside the corpus). Output goes to
--out/<UTC time>-<command>-<split>/ (report.json, report.md, per_question.jsonl); the default --out is
RESULTS_ROOT/runs and the default protocol.json / ledger.jsonl live in RESULTS_ROOT: the `results` folder inside
this package (the repository while developing, the installed copy otherwise; git-ignored, excluded from the
wheel). Standard output carries aggregate numbers only.

Wiring: cli.py calls register_eval_cli(subparsers) and dispatches args.eval_command through run_eval_cli(args).
`python -m neospike.evaluation` runs main().
Exit codes: 0 pass, 1 regressed (test with --baseline), 2 refused (seal mismatch, sweep check mismatch).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .._wincompat import default_hermes_home
from ..eval_recall import SPLITS, load_question_set
from ..store import memory_db_path
from . import core, lanes, protocol
from .adapters import Prepared, RealSource, SyntheticSource, read_jsonl

EXIT_OK, EXIT_REGRESSED, EXIT_REFUSED = 0, 1, 2
COMMANDS = ("run", "sweep", "seal", "test")
LEXICON_NAME = "lexicon.json"
# Default outputs stay inside the package directory, never in a HERMES_HOME.
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
RUNS_NAME = "runs"
PROTOCOL_NAME = "protocol.json"
SPLIT_ALL = "all"


def _real_embedder(home: Path) -> Any:
    """bge-m3 client of the evaluated home (tests replace this function)."""
    from ..inference import EmbeddingClient
    return EmbeddingClient(home)


def _lane_k(lane: str) -> int:
    from ..mcp_server import DEFAULT_RESULTS
    from ..provider import RECALL_INJECT_LIMIT
    return RECALL_INJECT_LIMIT if lane == lanes.LANE_PREFETCH else DEFAULT_RESULTS


# -- parser ----------------------------------------------------------------------------------------------------

def _source_arguments(parser: argparse.ArgumentParser, *, set_required: bool = True) -> None:
    parser.add_argument("--set", dest="question_set", required=set_required, help="Question-set JSONL")
    parser.add_argument("--home", default=argparse.SUPPRESS, help="HERMES_HOME to read (snapshot, read-only)")
    parser.add_argument("--corpus", help="Synthetic fixture corpus JSONL (instead of a home)")
    parser.add_argument("--lexicon", help=f"Fixture embedder lexicon (default: {LEXICON_NAME} beside --corpus)")
    parser.add_argument("--namespace", default=None, help="Owner namespace (default: the live one of the home)")
    parser.add_argument("--gold-database", help="Older memory.db copy for gold bodies the home no longer has")
    parser.add_argument("--out", help="Output root (default: neospike/evaluation/results/runs)")


def register_eval_cli(subparsers: Any) -> argparse.ArgumentParser:
    """Add `eval {run,sweep,seal,test}` to an argparse subparsers object; returns the eval parser."""
    parser = subparsers.add_parser("eval", help="Search evaluation: run, sweep, seal, test (read-only on the home)")
    add_eval_commands(parser)
    return parser


def add_eval_commands(parser: argparse.ArgumentParser) -> None:
    """The run/sweep/seal/test sub-commands (dest eval_command) under parser."""
    verbs = parser.add_subparsers(dest="eval_command", required=True)
    run = verbs.add_parser("run", help="Score a question set on the lanes at the product thresholds")
    _source_arguments(run)
    run.add_argument("--split", choices=SPLITS)
    run.add_argument("--lanes", help=f"Comma list of {','.join(lanes.LANES)} (default prefetch,memory,note)")
    run.add_argument("--threshold", type=float, help="One cosine floor for every lane (default: product values)")
    sweep = verbs.add_parser("sweep", help="Score every grid threshold, propose a candidate (dev split)")
    _source_arguments(sweep)
    sweep.add_argument("--split", choices=SPLITS, default="dev")
    sweep.add_argument("--lanes")
    sweep.add_argument("--grid", default=None, help="start:stop:step (default 0.50:0.75:0.01)")
    seal = verbs.add_parser("seal", help="Write protocol.json (set hash, rule, code hash, DB fingerprint)")
    _source_arguments(seal)
    seal.add_argument("--protocol", help="protocol.json path (default neospike/evaluation/results/protocol.json)")
    seal.add_argument("--lanes")
    seal.add_argument("--grid", default=None)
    seal.add_argument("--threshold", type=float, help="Threshold the test split is scored at (default: product)")
    test = verbs.add_parser("test", help="Score the sealed test split; ledger +1; gate against --baseline")
    _source_arguments(test, set_required=False)
    test.add_argument("--protocol", help="protocol.json path (default neospike/evaluation/results/protocol.json)")
    test.add_argument("--baseline", help="report.json of an earlier test run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neospike-eval")
    parser.add_argument("--home", default=default_hermes_home(), help="HERMES_HOME")
    add_eval_commands(parser)
    return parser


# -- helpers ---------------------------------------------------------------------------------------------------

def _home(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "home", None) or default_hermes_home()).expanduser()


def _open_source(args: argparse.Namespace, set_path: Path) -> RealSource | SyntheticSource:
    if args.corpus:
        corpus = Path(args.corpus)
        lexicon = Path(args.lexicon) if args.lexicon else corpus.parent / LEXICON_NAME
        return SyntheticSource(corpus, set_path, lexicon)
    home = _home(args)
    return RealSource(home, load_question_set(set_path), embedder=_real_embedder(home), namespace=args.namespace,
                      gold_database=Path(args.gold_database) if args.gold_database else None)


def _selected(prepared: Prepared, split: str | None) -> list:
    return [question for question in prepared.questions if split is None or question.split == split]


def _out_directory(args: argparse.Namespace, command: str, split: str | None) -> Path:
    root = Path(args.out) if args.out else RESULTS_ROOT / RUNS_NAME
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = root / f"{stamp}-{command}-{split or SPLIT_ALL}"
    suffix = 1
    while target.exists():
        suffix += 1
        target = root / f"{stamp}-{command}-{split or SPLIT_ALL}-{suffix}"
    target.mkdir(parents=True)
    return target


def _points_report(swept: dict[str, dict[str, list]], splits: Sequence[str]) -> dict[str, Any]:
    """{lane: {'k', 'points': {threshold key: {split: split_report}}}}"""
    report: dict[str, Any] = {}
    for lane, points in swept.items():
        report[lane] = {"k": _lane_k(lane), "points": {
            key: {split: core.split_report(item for item in judged if item.question.split == split)
                  for split in splits if any(item.question.split == split for item in judged)}
            for key, judged in points.items()}}
    return report


def _paired_rows(swept: dict[str, dict[str, list]], key: str) -> dict[str, dict[str, dict[str, Any]]]:
    """{lane: {question id: {hit, false_positive, stale}}} at one threshold key (baseline material, no text)."""
    return {lane: {item.question.id: {"split": item.question.split, "hit": item.hit,
                                      "false_positive": item.false_positive, "stale": list(item.stale)}
                   for item in points[key]} for lane, points in swept.items() if key in points}


def _write_outputs(directory: Path, report: dict[str, Any], swept: dict[str, dict[str, list]]) -> None:
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8-sig")
    with (directory / "per_question.jsonl").open("w", encoding="utf-8-sig") as handle:
        for lane, points in swept.items():
            for key, judged in points.items():
                for item in judged:
                    row = {"lane": lane, "threshold": key, "question": item.question.question, **item.as_row()}
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (directory / "report.md").write_text(render_markdown(report), encoding="utf-8-sig")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# neospike eval {report['command']}", "", f"- source: {report['source']}",
             f"- set sha256: `{report['set_sha256']}`", f"- code sha256: `{report['code_sha256']}`", ""]
    for lane, entry in report["lanes"].items():
        lines += [f"## {lane} (k={entry['k']})", "",
                  "| threshold | split | n | Recall@k | MRR@k | FPR | irrelevant | stale |",
                  "|---|---|---|---|---|---|---|---|"]
        for key, splits in entry["points"].items():
            for split, summary in splits.items():
                lines.append(f"| {key} | {split} | {summary['count']} | {_fmt(summary['recall'])} | "
                             f"{_fmt(summary['mrr'])} | {_fmt(summary['fpr'])} | {_fmt(summary['irrelevant'])} | "
                             f"{summary['stale_injected']} |")
        selection = report.get("selection", {}).get(lane)
        if selection:
            lines += ["", (f"candidate (rule, user confirms): {_fmt(selection['candidate'])}; "
                           f"pareto: {', '.join(selection['pareto'])}")]
        lines.append("")
    if report.get("warnings"):
        lines += ["## warnings", "", *[f"- {warning}" for warning in report["warnings"]], ""]
    return "\n".join(lines)


def _aggregate_only(report: dict[str, Any]) -> dict[str, Any]:
    """Standard-output view: no per-question material."""
    return {name: value for name, value in report.items() if name not in ("paired",)}


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _base_report(command: str, source: Any, set_path: Path, prepared: Prepared, session: lanes.Session,
                 swept: dict[str, dict[str, list]], split: str | None) -> dict[str, Any]:
    splits = [split] if split else [name for name in SPLITS]
    gold_ids = {gold for question in prepared.questions for gold in question.gold}
    return {"command": command, "source": "synthetic" if isinstance(source, SyntheticSource) else "real",
            "set_sha256": protocol.normalized_sha256(set_path), "code_sha256": protocol.code_sha256(),
            "lanes": _points_report(swept, splits), "warnings": list(dict.fromkeys(session.warnings)),
            "gold_without_body": sorted(gold for gold in gold_ids if gold.startswith("mem:")
                                        and gold not in prepared.bodies)}


# -- commands --------------------------------------------------------------------------------------------------

def _run(args: argparse.Namespace) -> int:
    set_path = Path(args.question_set)
    source = _open_source(args, set_path)
    with source as prepared, lanes.Session(prepared.home, embedder=prepared.embedder, namespace=prepared.namespace,
                                           bodies=prepared.bodies) as session:
        questions = _selected(prepared, args.split)
        swept = lanes.collect(session, questions, lanes.parse_lanes(args.lanes), [args.threshold])
        report = _base_report("run", source, set_path, prepared, session, swept, args.split)
    key = lanes.threshold_key(args.threshold)
    report["paired"] = _paired_rows(swept, key)
    directory = _out_directory(args, "run", args.split)
    _write_outputs(directory, report, swept)
    _emit({"out": str(directory), **_aggregate_only(report)})
    return EXIT_OK


def _sweep(args: argparse.Namespace) -> int:
    set_path = Path(args.question_set)
    grid = lanes.parse_grid(args.grid) if args.grid else lanes.default_grid()
    source = _open_source(args, set_path)
    with source as prepared, lanes.Session(prepared.home, embedder=prepared.embedder, namespace=prepared.namespace,
                                           bodies=prepared.bodies) as session:
        questions = _selected(prepared, args.split)
        swept = lanes.collect(session, questions, lanes.parse_lanes(args.lanes), grid)
        check = lanes.verify_default(session, questions, swept)
        report = _base_report("sweep", source, set_path, prepared, session, swept, args.split)
    report["verification"] = check
    if any(entry["status"] == "mismatch" for entry in check.values()):
        _emit({"error": "sweep_check_mismatch", "verification": check})
        return EXIT_REFUSED
    selection = {}
    for lane, points in swept.items():
        rows = core.sweep_rows({float(key): core.summarize(judged) for key, judged in points.items()})
        chosen = core.select_threshold(rows)
        selection[lane] = {**chosen, "pareto": [lanes.threshold_key(row.threshold) for row in rows if row.pareto],
                           "targets_at_candidate": None if chosen["candidate"] is None else core.targets(
                               core.summarize(points[lanes.threshold_key(chosen["candidate"])]))}
    report["selection"] = selection
    directory = _out_directory(args, "sweep", args.split)
    _write_outputs(directory, report, swept)
    _emit({"out": str(directory), "verification": check, "selection": selection,
           "warnings": report["warnings"]})
    return EXIT_OK


def _protocol_path(args: argparse.Namespace) -> Path:
    return Path(args.protocol) if args.protocol else RESULTS_ROOT / PROTOCOL_NAME


def _database_for(args: argparse.Namespace) -> Path | None:
    if args.corpus:
        return None
    path = memory_db_path(_home(args))
    return path if path.is_file() else None


def _seal(args: argparse.Namespace) -> int:
    set_path = Path(args.question_set)
    rows = read_jsonl(set_path)
    selected = list(lanes.parse_lanes(args.lanes))
    grid = lanes.parse_grid(args.grid) if args.grid else lanes.default_grid()
    sealed = protocol.build_protocol(set_path, protocol.split_counts(rows), lanes=selected, grid=grid,
                                     threshold=args.threshold, database=_database_for(args),
                                     corpus=Path(args.corpus) if args.corpus else None)
    path = protocol.write_protocol(_protocol_path(args), sealed)
    _emit({"protocol": str(path), "set_sha256": sealed["set"]["sha256"], "splits": sealed["set"]["splits"]})
    return EXIT_OK


def _test(args: argparse.Namespace) -> int:
    protocol_path = _protocol_path(args)
    sealed = protocol.read_protocol(protocol_path)
    set_path = Path(args.question_set or sealed["set"]["path"])
    verdict = protocol.check(sealed, set_path, _database_for(args), Path(args.corpus) if args.corpus else None)
    if verdict["refused"]:
        _emit({"error": "sealed_set_changed", **verdict})
        return EXIT_REFUSED
    threshold = sealed.get("threshold")
    source = _open_source(args, set_path)
    with source as prepared, lanes.Session(prepared.home, embedder=prepared.embedder, namespace=prepared.namespace,
                                           bodies=prepared.bodies) as session:
        questions = _selected(prepared, "test")
        swept = lanes.collect(session, questions, sealed["lanes"], [threshold])
        report = _base_report("test", source, set_path, prepared, session, swept, "test")
    key = lanes.threshold_key(threshold)
    report["paired"] = _paired_rows(swept, key)
    report["changed_since_seal"] = verdict["changed"]
    report["targets"] = {lane: core.targets(entry["points"][key]["test"]) for lane, entry in report["lanes"].items()
                         if "test" in entry["points"][key]}
    regressed = False
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8-sig")).get("paired", {})
        report["gate"] = {lane: core.gate(rows, baseline[lane]) for lane, rows in report["paired"].items()
                          if lane in baseline}
        regressed = any(entry["regressed"] for entry in report["gate"].values())
        report["regressed"] = regressed
    summary = {lane: {name: entry["points"][key]["test"][name] for name in ("recall", "fpr", "stale_injected")}
               for lane, entry in report["lanes"].items() if "test" in entry["points"][key]}
    report["ledger"] = protocol.append_ledger(protocol_path, sealed, summary)
    directory = _out_directory(args, "test", "test")
    _write_outputs(directory, report, swept)
    _emit({"out": str(directory), **_aggregate_only(report)})
    return EXIT_REGRESSED if regressed else EXIT_OK


_HANDLERS = {"run": _run, "sweep": _sweep, "seal": _seal, "test": _test}


def run_eval_cli(args: argparse.Namespace) -> int:
    """Dispatch a parsed `eval <command>`; ValueError/OSError become a JSON error and exit 2."""
    try:
        return _HANDLERS[args.eval_command](args)
    except (ValueError, OSError) as exc:
        _emit({"error": "invalid", "message": str(exc)})
        return EXIT_REFUSED


def main(argv: Sequence[str] | None = None) -> int:
    return run_eval_cli(build_parser().parse_args(sys.argv[1:] if argv is None else list(argv)))
