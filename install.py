#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from neospike import hermes_runtime
from neospike._wincompat import shared_inference_home
from neospike.config import read_memory_rule, memory_rule_file_path, write_memory_rule
from neospike.cron.neospike_cron_common import target_homes
from neospike.cron.neospike_cron_common import yaml_scalar as _scalar_of
from neospike.store import memory_db_path, pick_namespace


# BAAI/bge-m3 int8 ONNX at a pinned revision; not gated (MIT).
MODEL_REPO = "Xenova/bge-m3"
MODEL_REVISION = "4de13258303883538bd53b696b452bf8099f0858"
MODEL_ONNX_NAME = "sentence_transformers_int8.onnx"  # embedder.ONNX_MODEL_FILE
MODEL_DOWNLOADS = ["onnx/" + MODEL_ONNX_NAME, "tokenizer.json"]
MODEL_FOLDER = "bge-m3-onnx"  # embedder.MODEL_FOLDER


STATUS_MARKERS = ("Plugin:    installed ✓", "Status:    available ✓")
PROBE_TIMEOUT_S = 120
# The gateway imports `hermes_bootstrap`, which swaps sys.path to the PM environment venv
# (installs/<id>/environments/<hash>/venv). Probes mirror the launcher's -c prologue so they see that sys.path.
# argv = [hermes-agent root, plugins dir]; argv is cleared before the import because bootstrap parses it as a command.
_BOOT_PREFIX = (
    "import os, sys; roots = sys.argv[1:3]; del sys.argv[1:]; "
    "[os.environ.pop(name, None) for name in ('PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV')]; "
    "sys.path.insert(0, roots[0]); import hermes_bootstrap; "
)
_LOAD_CHECK_SCRIPT = _BOOT_PREFIX + (
    "sys.path.insert(0, roots[1]); "
    "from neospike import NeospikeProvider; import neospike.provider; "
    "assert NeospikeProvider().is_available(), 'NeospikeProvider().is_available() is False'"
)
# Right after the PM sync only the imports are checked; is_available() also needs the model files,
# which --skip-models leaves out, so that assertion stays in --check (_LOAD_CHECK_SCRIPT).
_IMPORT_CHECK_SCRIPT = _BOOT_PREFIX + (
    "sys.path.insert(0, roots[1]); "
    "import numpy; import faiss; from neospike import NeospikeProvider; import neospike.provider"
)
_SITE_PROBE_SCRIPT = _BOOT_PREFIX + (
    "import json; from pathlib import Path; "
    "print(json.dumps([p for p in sys.path if Path(p).name in ('site-packages', 'dist-packages')]))"
)
# What the Hermes process itself imports from this package. Declared in neospike/plugin.yaml
# python_dependencies and installed by the Hermes package manager (tests tie it to plugin.yaml and pyproject).
# faiss-cpu: the HNSW candidates of vector_index.py, used in the Hermes process by every vector search of the
# database file (without it in the Hermes runtime every such search would fail).
PLUGIN_PIP_REQUIREMENTS = ("numpy>=2.0,<3", "faiss-cpu>=1.16,<2")
PLUGIN_REQS_KEY = "python_dependencies:"


# Cron scripts and jobs. Hermes runs cron scripts only from <home>/scripts (cron/scheduler_script.py
# _resolve_script_path); success delivers nothing (empty stdout / "local"), failures go to Bot Chat.
CRON_SRC_DIR = Path(__file__).resolve().parent / "neospike" / "cron"
CRON_FILES = ("neospike_cron_common.py", "neospike_daily.py")
CRON_DAILY_FILE = CRON_FILES[1]
DELIVER_OK = "local"
DELIVER_FAIL = "bot-chat"
# Claude Code MCP registration (user scope).
MCP_SERVER_NAME, MCP_REG_SCOPE = "neospike", "user"
DAILY_JOB_NAME, DAILY_JOB_WHEN = "neospike-daily", "every day at 4am"
DATA_FOLDER, PLUGIN_FOLDER, PROVIDER_ID = "neospike", "neospike", "neospike"

# Jobs and files of the retired weekly LLM review; removed on install (the daily job calls no LLM).
DROPPED_JOBS = ("neospike-weekly-review",)
DROPPED_FILES = ("neospike_weekly_monitor.py", "weekly_review_prompt.md", "weekly_review_prompt_kor.md")
MIGRATION_ZIP_NAME = "neospike_migration_backup_{stamp}.zip"
MIGRATION_STAMP_FMT = "%Y%m%dT%H%M%S"
TMP_ZIP_SUFFIX = ".partial"
# Old tool names in <home>/skills/**/*.md and the v22 names they become.
SKILLS_FOLDER, SKILL_GLOB = "skills", "*.md"
SKILL_REWRITES = (
    (re.compile(r"\bneospike_memory_remember\b"), "neospike_memory_save"),
    (re.compile(r"\bneospike_memory_forget\b"), "neospike_memory_delete"),
    # the MCP searches memory_search and note_search became the one mem_search
    # (\b keeps neospike_memory_search and the CLI note-search as they are).
    (re.compile(r"\b(?:memory|note)_search\b"), "mem_search"),
)
BOM_BYTES = b"\xef\xbb\xbf"


def _runner_or_default(runner=None):
    # Looked up per call so tests that patch subprocess.run take effect.
    return subprocess.run if runner is None else runner


def echo_and_run(command: list[str], **kwargs) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, **kwargs, check=True)


def default_home() -> str:
    configured = os.environ.get("HERMES_HOME")
    if configured:
        return configured
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        # Native Windows Hermes lives in %LOCALAPPDATA%\hermes, not ~/.hermes.
        return str(Path(os.environ["LOCALAPPDATA"]) / "hermes")
    return str(Path.home() / ".hermes")


def locate_launcher(home: Path) -> str | None:
    """The Hermes launcher: <home>/bin -> PATH -> ~/.local/bin (hermes_runtime.find_launcher)."""
    return hermes_runtime.find_launcher(home)


def find_runtime_python(hermes: str | None, home: Path, runner=None, cache: bool = False) -> tuple[str | None, str]:
    """(python Hermes really runs, how it was found); (None, why) when nothing exists. See hermes_runtime."""
    runner = _runner_or_default(runner)
    try:
        return hermes_runtime.runtime_python(home, launcher=hermes, runner=runner, cache=cache)
    except hermes_runtime.HermesRuntimeNotFound as exc:
        return None, str(exc)


def owning_venv_python(site_packages: Path) -> Path | None:
    """The venv interpreter owning *site_packages* (nearest parent with pyvenv.cfg), else None."""
    for parent in site_packages.parents:
        if (parent / "pyvenv.cfg").is_file():
            windows = (parent / "Scripts").is_dir()
            return parent / ("Scripts" if windows else "bin") / ("python.exe" if windows else "python")
    return None


def probe_runtime_env(python: str, home: Path, runner=None) -> tuple[Path | None, str]:
    """(venv python whose site-packages the bootstrapped Hermes process imports from, what was seen)."""
    runner = _runner_or_default(runner)
    command = [python, "-I", "-c", _SITE_PROBE_SCRIPT, str(hermes_runtime.hermes_agent_dir(home)), str(home / "plugins")]
    env = hermes_process_env(home)
    try:
        done = runner(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                      timeout=PROBE_TIMEOUT_S, env=env)
        lines = done.stdout.strip().splitlines() if done.returncode == 0 else []
        sites = json.loads(lines[-1]) if lines else None
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return None, f"environment probe failed: {exc}"
    if not isinstance(sites, list) or not sites:
        return None, f"environment probe (exit {done.returncode}) found no site-packages: {(done.stderr or '').strip()[-500:]}"
    for site in sites:
        venv = owning_venv_python(Path(site))
        if venv is not None:
            return venv, f"site-packages {site}"
    return None, f"no venv owns {sites}"


def audit_installation(home: Path, hermes: str | None, required: list[Path], runner=None) -> list[str]:
    """Failures of an existing installation (empty = healthy). Prints what it ran and saw."""
    runner = _runner_or_default(runner)
    failures = [f"Missing installation artifact: {path}" for path in required if not path.exists()]
    env = hermes_process_env(home)
    if not hermes:
        return failures + ["Hermes launcher not found (home/bin/hermes or hermes on PATH)."]
    print("+", hermes, "memory status")
    status = runner([hermes, "memory", "status"], capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=PROBE_TIMEOUT_S, env=env)
    print(status.stdout, end="")
    absent = [line for line in STATUS_MARKERS if line not in status.stdout]
    if status.returncode != 0 or absent:
        failures.append(f"`hermes memory status` (exit {status.returncode}) lacks: {', '.join(absent) or '-'}")
    python, source = find_runtime_python(hermes, home, runner)
    print(f"Hermes runtime python: {python} (from {source})")
    if python is None:
        return failures + [f"Hermes runtime python not found ({source})."]
    environment, seen = probe_runtime_env(python, home, runner)
    print(f"Hermes runtime environment: {environment} ({seen})")
    failure = provider_import_error(python, home, runner)
    return failures + ([failure] if failure else [])


def hermes_process_env(home: Path) -> dict:
    """Environment for Hermes commands: HERMES_HOME pins --home."""
    return {**os.environ, "HERMES_HOME": str(home), "PYTHONIOENCODING": "utf-8"}


def provider_import_error(python: str, home: Path, runner=None, check: str = _LOAD_CHECK_SCRIPT) -> str | None:
    """Run *check* (default _LOAD_CHECK_SCRIPT) in the bootstrapped runtime python; the failure text, or None."""
    runner = _runner_or_default(runner)
    command = [python, "-I", "-c", check, str(hermes_runtime.hermes_agent_dir(home)), str(home / "plugins")]
    try:
        loaded = runner(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                        timeout=PROBE_TIMEOUT_S, env=hermes_process_env(home))
    except (OSError, subprocess.SubprocessError) as exc:
        return f"neospike.provider check could not run in {python}: {exc}"
    if loaded.returncode != 0:
        return f"neospike.provider does not load in {python}:\n{(loaded.stderr or loaded.stdout).strip()}"
    return None


def enable_plugin_argv(hermes: str) -> list[str]:
    """The launcher syncs the PM environment with plugin.yaml python_dependencies before enabling."""
    return [hermes, "plugins", "enable", PLUGIN_FOLDER, "--no-allow-tool-override"]


def dependency_decl_errors(plugin_yaml: Path) -> list[str]:
    """The installed plugin.yaml must declare PLUGIN_PIP_REQUIREMENTS (text check, no YAML parser)."""
    try:
        lines = [line.strip() for line in plugin_yaml.read_text(encoding="utf-8-sig").splitlines()]
    except OSError:
        return []  # a missing plugin.yaml is already reported as a missing installation artifact
    missing = [item for item in PLUGIN_PIP_REQUIREMENTS if f'- "{item}"' not in lines]
    if PLUGIN_REQS_KEY in lines and not missing:
        return []
    return [f"{plugin_yaml} does not declare {PLUGIN_REQS_KEY} {', '.join(missing or PLUGIN_PIP_REQUIREMENTS)} "
            "(the Hermes package manager will not install it; rerun install.py)"]


def confirm_runtime_after_sync(hermes: str, home: Path, runner=None) -> str:
    """The PM sync may select a new environment generation; find it again and load the provider there.
    Returns the runtime python; SystemExit when the provider does not load. No pip fallback."""
    runner = _runner_or_default(runner)
    python, source = find_runtime_python(hermes, home, runner)
    if python is None:
        raise SystemExit(f"Could not find the runtime python after the Hermes PM sync ({source}). "
                         "Check the PM receipt with `hermes pm status`.")
    environment, seen = probe_runtime_env(python, home, runner)
    print(f"Hermes runtime environment after PM sync: {environment} ({seen})")
    failure = provider_import_error(python, home, runner, _IMPORT_CHECK_SCRIPT)
    if failure:
        raise SystemExit(
            "neospike provider import check failed after install.\n"
            f"Missing: {failure}\n"
            f"Runtime environment: {environment or seen}\n"
            f"Check the PM receipt with `hermes pm status`: plugin.yaml {PLUGIN_REQS_KEY} "
            f"{', '.join(PLUGIN_PIP_REQUIREMENTS)} must be installed in this environment.")
    return python


def neospike_cli_python(home: Path) -> str:
    """The python the Dashboard, the cron scripts and the MCP server run the CLI with (hermes_runtime.cli_python)."""
    try:
        return hermes_runtime.cli_python(home)
    except hermes_runtime.HermesRuntimeNotFound:
        return str(hermes_runtime.checkout_python(home))


def copy_cron_scripts(home: Path) -> Path:
    """Copy (overwrite) the cron scripts into <home>/scripts; the folder is created when missing."""
    target = home / "scripts"
    target.mkdir(parents=True, exist_ok=True)
    for name in CRON_FILES:
        shutil.copyfile(CRON_SRC_DIR / name, target / name)
    return target


def seed_memory_rule(home: Path) -> Path:
    """Create <home>/neospike/MEMORY_RULE.md when missing (old config.json rules or the default); never overwrite."""
    path = memory_rule_file_path(home)
    if not path.exists():
        write_memory_rule(home, read_memory_rule(home))
    return path


def read_cron_jobs(home: Path) -> dict[str, dict]:
    """Jobs of <home>/cron/jobs.json by lower-cased name (Hermes resolves names case-insensitively). Read only."""
    try:
        data = json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    jobs = data.get("jobs", []) if isinstance(data, dict) else []
    records = {str(job.get("name") or "").strip().lower(): job for job in jobs if isinstance(job, dict)}
    records.pop("", None)
    return records


def cron_name_set(home: Path) -> set[str]:
    return set(read_cron_jobs(home))


def planned_cron_jobs(home: Path) -> list[tuple[str, str, str | None, list[str]]]:
    """(name, schedule, prompt or None, options) of the Neospike cron jobs (daily only)."""
    deliver = ["--deliver", DELIVER_OK, "--failure-deliver", DELIVER_FAIL]
    return [
        (DAILY_JOB_NAME, DAILY_JOB_WHEN, None, ["--script", CRON_DAILY_FILE, "--no-agent", *deliver]),
    ]


def cron_argv(hermes: str, job: tuple[str, str, str | None, list[str]], existing: set[str]) -> list[str]:
    """`hermes cron edit` for a job that already exists by name, else `hermes cron create` (never a duplicate)."""
    name, schedule, prompt, options = job
    if name in existing:
        return [hermes, "cron", "edit", name, "--schedule", schedule, *options, *(["--prompt", prompt] if prompt else [])]
    return [hermes, "cron", "create", schedule, *([prompt] if prompt else []), "--name", name, *options]


def cron_argv_list(hermes: str, home: Path) -> list[list[str]]:
    existing = cron_name_set(home)
    return [cron_argv(hermes, job, existing) for job in planned_cron_jobs(home)]


def apply_cron_jobs(hermes: str, home: Path) -> None:
    """jobs.json is re-read before each job, so what an earlier command created is edited, not re-created."""
    env = hermes_process_env(home)
    for job in planned_cron_jobs(home):
        echo_and_run(cron_argv(hermes, job, cron_name_set(home)), env=env)


def cron_problems(home: Path) -> list[str]:
    """Failures: a job missing from jobs.json, or a script in <home>/scripts that differs from the repository."""
    failures = []
    try:
        names = cron_name_set(home)
    except (OSError, ValueError) as exc:
        return [f"cron jobs.json unreadable: {exc}"]
    failures += [f"Cron job not registered: {name}" for name in (DAILY_JOB_NAME,) if name not in names]
    for name in CRON_FILES:
        installed = home / "scripts" / name
        if not installed.is_file():
            failures.append(f"Cron script missing: {installed}")
        elif installed.read_bytes() != (CRON_SRC_DIR / name).read_bytes():
            failures.append(f"Cron script differs from the repository: {installed}")
    return failures


def flatten_model_file(text_target: Path) -> None:
    """hf download keeps the repo's onnx/ folder; the embedder reads the model next to tokenizer.json."""
    downloaded = text_target / MODEL_DOWNLOADS[0]
    if downloaded.is_file():
        os.replace(downloaded, text_target / MODEL_ONNX_NAME)


def mcp_server_entry(python: str, home: Path) -> dict:
    """The neospike MCP server entry, same shape as the hand-made registration."""
    namespace = pick_namespace(None, memory_db_path(home))
    return {
        "type": "stdio",
        "command": python,
        # -P: keep the launch folder off sys.path so a same-named package there is never imported.
        "args": ["-P", "-m", "neospike.mcp_server", "--home", str(home), "--owner", namespace],
        "env": {
            "HERMES_HOME": str(home),
            "HERMES_BGE_M3_DIR": str(shared_inference_home(home) / "models" / MODEL_FOLDER),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": str(home / "plugins"),
        },
    }


def mcp_argv_list(claude: str, python: str, home: Path) -> list[list[str]]:
    """remove the current entry (may fail when absent), then add-json; compact JSON so claude.cmd passes it
    through cmd.exe intact."""
    payload = json.dumps(mcp_server_entry(python, home), ensure_ascii=False, separators=(",", ":"))
    return [[claude, "mcp", "remove", "-s", MCP_REG_SCOPE, MCP_SERVER_NAME],
            [claude, "mcp", "add-json", "-s", MCP_REG_SCOPE, MCP_SERVER_NAME, payload]]


def is_profile(home: Path) -> bool:
    """A Hermes profile home (<root>/profiles/<name>) shares the root home's inference service."""
    return shared_inference_home(home) != Path(home).expanduser().resolve()


def skip_mcp_for_profile(home: Path) -> bool:
    """The user-scope Claude MCP entry is single and must point at the root home, so a profile install
    leaves it alone. Prints the one-line reason when skipping."""
    if not is_profile(home):
        return False
    print(f"Profile home {home}: Claude Code MCP server {MCP_SERVER_NAME} left pointing at {shared_inference_home(home)}.")
    return True


def skip_cron_for_profile(home: Path) -> bool:
    """The root home's cron serves every Neospike profile, so a profile install copies no cron script,
    and registers no job. Prints the one-line reason when skipping."""
    if not is_profile(home):
        return False
    print(f"Profile home {home}: cron job {DAILY_JOB_NAME} is run by the root home's cron "
          f"({shared_inference_home(home)}).")
    return True


def apply_mcp_registration(python: str, home: Path, runner=None) -> None:
    """Register or overwrite the neospike MCP server in Claude Code; skipped when `claude` is absent or *home* is
    a profile home."""
    runner = _runner_or_default(runner)
    if skip_mcp_for_profile(home):
        return
    claude = shutil.which("claude")
    if not claude:
        print(f"Claude Code (claude) not found; MCP server {MCP_SERVER_NAME} not registered.")
        return
    *removals, add = mcp_argv_list(claude, python, home)
    for remove in removals:
        print("+", subprocess.list2cmdline(remove))
        runner(remove, capture_output=True, text=True, encoding="utf-8", errors="replace",
               timeout=PROBE_TIMEOUT_S)
    print("+", subprocess.list2cmdline(add))
    runner(add, check=True, timeout=PROBE_TIMEOUT_S)


# Skill files that still name a renamed tool are backed up, then rewritten; a rerun is safe.
def apply_skill_renames(text: str) -> str:
    """*text* with every SKILL_REWRITES pattern applied."""
    for pattern, replacement in SKILL_REWRITES:
        text = pattern.sub(replacement, text)
    return text


def skills_with_old_names(home: Path) -> list[Path]:
    """<home>/skills/**/*.md that still hold an old name (files that are not UTF-8 are left alone)."""
    found = []
    root = home / SKILLS_FOLDER
    for path in sorted(root.rglob(SKILL_GLOB)) if root.is_dir() else []:
        try:
            text = path.read_bytes().decode("utf-8-sig")
        except (OSError, UnicodeDecodeError):
            continue
        if apply_skill_renames(text) != text:
            found.append(path)
    return found


def rewrite_skill_files(paths: list[Path]) -> None:
    """Rewrite *paths* with apply_skill_renames, keeping the BOM and the line endings."""
    for path in paths:
        raw = path.read_bytes()
        bom = raw.startswith(BOM_BYTES)
        path.write_bytes((BOM_BYTES if bom else b"") + apply_skill_renames(raw.decode("utf-8-sig")).encode("utf-8"))
        print(f"Renamed old names in skill: {path}")


def zip_skill_backup(home: Path, skills: list[Path], moment: float | None = None) -> Path | None:
    """*skills* in one zip at <home> before they are rewritten; None when there are none."""
    if not skills:
        return None
    stamp = time.strftime(MIGRATION_STAMP_FMT, time.localtime(time.time() if moment is None else moment))
    target = home / MIGRATION_ZIP_NAME.format(stamp=stamp)
    partial = target.with_name(target.name + TMP_ZIP_SUFFIX)
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in skills:
                archive.write(path, path.relative_to(home).as_posix())
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Migration backup: {target} ({len(skills)} files from {SKILLS_FOLDER})")
    return target


def migrate_skill_names(home: Path) -> Path | None:
    """Back up, then rewrite every skill file that still holds a renamed tool name."""
    skills = skills_with_old_names(home)
    backup = zip_skill_backup(home, skills)
    rewrite_skill_files(skills)
    return backup


def old_cron_remove_argv(hermes: str, home: Path) -> list[list[str]]:
    """`hermes cron remove <id>` for every DROPPED_JOBS job still in jobs.json."""
    records = read_cron_jobs(home)
    jobs = [records[name] for name in DROPPED_JOBS if name in records]
    return [[hermes, "cron", "remove", str(job.get("id") or job.get("name"))] for job in jobs]


def purge_old_crons(hermes: str, home: Path) -> None:
    """Step 5 (before the new jobs are registered): old jobs, then the old scripts in <home>/scripts."""
    env = hermes_process_env(home)
    for command in old_cron_remove_argv(hermes, home):
        echo_and_run(command, env=env)
    for name in DROPPED_FILES:
        script = home / "scripts" / name
        if script.is_file():
            script.unlink()
            print(f"Removed old cron script: {script}")


def _find_enabled_lines(lines: list[str]) -> tuple[int | None, list[int]]:
    """(index of `enabled:` under the top-level `plugins:` block, indices of its `- item` lines); comments skipped."""
    inside, indent, key, items = False, None, None, []
    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            if key is not None:
                break
            inside, indent = re.match(r"plugins:\s*(#.*)?$", line) is not None, None
            continue
        if not inside:
            continue
        width = len(line) - len(line.lstrip())
        indent = width if indent is None else indent
        if key is not None:
            if width >= indent and line.lstrip().startswith("- "):  # YAML allows `- ` at the key's own indent
                items.append(index)
                continue
            break
        if width == indent and re.match(r"\s*enabled:", line):
            key = index
    return key, items


def list_plugin_enabled(home: Path, dry_run: bool = False) -> bool:
    """Add PLUGIN_FOLDER to plugins.enabled in <home>/config.yaml (True = it was missing).
    The agent loads a memory provider through memory.provider alone (plugins_discovery.gate_manifest skips
    `exclusive` kinds), but the dashboard hides a user plugin's tab and API unless plugins.enabled lists it
    (hermes_cli/web_server_dashboard.py: "not in plugins.enabled"). Edited as text,
    so the BOM, comments and every other line stay."""
    path = home / "config.yaml"
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raw = b""
    bom = raw.startswith(BOM_BYTES)
    lines = raw.decode("utf-8-sig").splitlines(keepends=True)
    newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
    key, items = _find_enabled_lines(lines)
    flow = re.match(r"(\s*enabled:\s*\[)([^\]]*)(\].*)$", lines[key], re.S) if key is not None else None
    rest = _scalar_of(lines[key].split("enabled:", 1)[1]) if key is not None else ""
    if key is not None and not flow and rest.startswith(("[", "{")):
        raise SystemExit(f"plugins.enabled in {path} is not a one-line list; add {PLUGIN_FOLDER} with "
                         f"`hermes plugins enable {PLUGIN_FOLDER}`.")
    names = ([_scalar_of(name) for name in flow.group(2).split(",")] if flow
             else [_scalar_of(lines[index].lstrip()[2:]) for index in items])
    if PLUGIN_FOLDER in names:
        return False
    if dry_run:
        return True
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += newline
    if flow:
        inner = flow.group(2).strip()
        lines[key] = flow.group(1) + (inner + ", " if inner else "") + PLUGIN_FOLDER + flow.group(3)
    elif key is not None:
        if items:  # same indent and dash style as the last item
            last = lines[items[-1]]
            lines.insert(items[-1] + 1, last[:len(last) - len(last.lstrip())] + f"- {PLUGIN_FOLDER}{newline}")
        else:  # `enabled:` / `enabled: null` / `enabled: ~`
            indent = lines[key][:len(lines[key]) - len(lines[key].lstrip())]
            if rest:
                lines[key] = f"{indent}enabled:{newline}"
            lines.insert(key + 1, f"{indent}  - {PLUGIN_FOLDER}{newline}")
    else:
        block = next((index for index, line in enumerate(lines)
                      if re.match(r"plugins:\s*(\{\s*\})?\s*(#.*)?$", line)), None)
        if block is None:
            if any(re.match(r"plugins:", line) for line in lines):
                raise SystemExit(f"plugins in {path} is not a block mapping; add {PLUGIN_FOLDER} with "
                                 f"`hermes plugins enable {PLUGIN_FOLDER}`.")
            lines.append(f"plugins:{newline}")
            block = len(lines) - 1
        elif "{" in lines[block]:  # `plugins: {}` becomes a block
            lines[block] = f"plugins:{newline}"
        child = next((line for line in lines[block + 1:]
                      if line.strip() and not line.lstrip().startswith("#")), "")
        indent = child[:len(child) - len(child.lstrip())] or "  "
        lines[block + 1:block + 1] = [f"{indent}enabled:{newline}", f"{indent}  - {PLUGIN_FOLDER}{newline}"]
    path.write_bytes((BOM_BYTES if bom else b"") + "".join(lines).encode("utf-8"))
    print(f"Added {PLUGIN_FOLDER} to plugins.enabled in {path}")
    return True


def old_install_traces(home: Path) -> list[str]:
    """--check: what a dropped feature or a v22 tool rename left behind (empty = clean)."""
    try:
        names = cron_name_set(home)
    except (OSError, ValueError):
        names = set()
    traces = [f"Old cron job remains: {name}" for name in DROPPED_JOBS if name in names]
    traces += [f"Old cron script remains: {home / 'scripts' / name}" for name in DROPPED_FILES
               if (home / "scripts" / name).is_file()]
    traces += [f"Old names remain in skill file: {path}" for path in skills_with_old_names(home)]
    return traces


def migration_preview(hermes: str, home: Path) -> list[str]:
    """--dry-run lines for the dropped cron leftovers and the skill rewrites (empty when nothing is left)."""
    lines = [subprocess.list2cmdline(command) for command in old_cron_remove_argv(hermes, home)]
    lines += [f"remove {home / 'scripts' / name}" for name in DROPPED_FILES if (home / "scripts" / name).is_file()]
    lines += [f"rename old names in {path} (copy in the backup zip first)" for path in skills_with_old_names(home)]
    return lines



def _parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(description="Install Neospike without touching the Hermes checkout")
    cli.add_argument("--home", default=default_home())
    cli.add_argument("--skip-models", action="store_true",
                     help="Install code and dependencies without downloading model artifacts")
    cli.add_argument("--check", action="store_true", help="Only verify an existing installation")
    cli.add_argument("--dry-run", action="store_true",
                     help="Print the cron scripts and cron commands; change nothing")
    return cli.parse_args()


def _print_dry_run(hermes: str, home: Path) -> None:
    """--dry-run: every action as a '+' line, nothing executed."""
    for step in migration_preview(hermes, home):
        print("+", step)
    print("+", subprocess.list2cmdline(enable_plugin_argv(hermes)))
    for owner in target_homes(home):
        if not memory_rule_file_path(owner).exists():
            print("+ create", memory_rule_file_path(owner))
    if not skip_cron_for_profile(home):
        for name in CRON_FILES:
            print("+ copy", CRON_SRC_DIR / name, "->", home / "scripts" / name)
        for argv in cron_argv_list(hermes, home):
            print("+", subprocess.list2cmdline(argv))
    if not skip_mcp_for_profile(home):
        for argv in mcp_argv_list(shutil.which("claude") or "claude", neospike_cli_python(home), home):
            print("+", subprocess.list2cmdline(argv))


def _run_check(home: Path, hermes: str, required: list[Path], plugin_yaml: Path) -> int:
    """--check: collect every failure, then stop once."""
    problems = audit_installation(home, hermes, required)
    problems += dependency_decl_errors(plugin_yaml)
    if not skip_cron_for_profile(home):
        problems += cron_problems(home)
    problems += old_install_traces(home)
    if list_plugin_enabled(home, dry_run=True):
        problems.append(f"plugins.enabled does not list {PLUGIN_FOLDER} in {home / 'config.yaml'} "
                        "(the dashboard hides the Neospike tab)")
    if problems:
        raise SystemExit("Neospike check failed:\n" + "\n".join(problems))
    print("Neospike installation is healthy.")
    return 0


def _pick_cli_python(hermes: str, home: Path) -> tuple[Path, str | None]:
    """(CLI python, runtime python). The gateway imports from the PM environment venv, not the
    checkout venv."""
    runtime, source = find_runtime_python(hermes, home, subprocess.run)
    environment, seen = probe_runtime_env(runtime, home, subprocess.run) if runtime else (None, source)
    if environment is None:
        raise SystemExit(f"Hermes runtime environment not found ({seen}).")
    print(f"Hermes runtime environment: {environment} ({seen})")
    # HERMES_PYTHON / runtime.json / checkout venv; without one (macOS Desktop) the PM venv.
    try:
        chosen = Path(hermes_runtime.cli_python(home))
    except hermes_runtime.HermesRuntimeNotFound as exc:
        print(f"{exc}\nUsing the Hermes runtime environment for the Neospike CLI: {environment}")
        chosen = environment
    if not chosen.is_file():
        raise SystemExit(f"Neospike CLI python not found: {chosen}. Set HERMES_PYTHON to the Hermes venv python.")
    return chosen, runtime


def _swap_in_plugin(source_dir: Path, plugin_target: Path) -> None:
    """Copy the package next to the target, then move it into place; the old copy stays as neospike.previous."""
    staging = plugin_target.with_name("neospike.installing")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(source_dir, staging, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if plugin_target.exists():
        previous = plugin_target.with_name("neospike.previous")
        shutil.rmtree(previous, ignore_errors=True)
        plugin_target.replace(previous)
    staging.replace(plugin_target)


def _download_models(python: Path, model_dir: Path) -> None:
    hf_cli = python.parent / ("hf.exe" if os.name == "nt" else "hf")
    if not hf_cli.is_file():
        raise SystemExit("Hugging Face CLI was not installed into the Hermes venv")
    model_dir.mkdir(parents=True, exist_ok=True)
    fetch = [str(hf_cli), "download", MODEL_REPO, *MODEL_DOWNLOADS,
             "--revision", MODEL_REVISION, "--local-dir", str(model_dir)]
    try:
        echo_and_run(fetch)
        flatten_model_file(model_dir)
    except subprocess.CalledProcessError as exc:
        raise SystemExit("Model download failed; check the network and rerun install.py.") from exc


def main() -> int:
    args = _parse_args()
    home = Path(args.home).expanduser().resolve()
    hermes = locate_launcher(home)
    if not hermes:
        tried = "\n  ".join(hermes_runtime.launcher_candidates(home))
        raise SystemExit(f"Hermes launcher not found. Tried:\n  {tried}\nInstall Hermes Agent first or set HERMES_HOME.")

    plugin_target = home / "plugins" / "neospike"
    model_dir = shared_inference_home(home) / "models" / MODEL_FOLDER
    if args.dry_run:
        # The venv's own hermes script is bound to a build-time snapshot; cron goes through the user's launcher.
        _print_dry_run(hermes, home)
        return 0
    if args.check:
        needed = [plugin_target / "plugin.yaml", model_dir / MODEL_ONNX_NAME, model_dir / "tokenizer.json"]
        return _run_check(home, hermes, needed, plugin_target / "plugin.yaml")

    uv_exe = shutil.which("uv")
    if not uv_exe:
        raise SystemExit("uv is required because Hermes venvs may not include pip")
    repo_root = Path(__file__).resolve().parent
    migrate_skill_names(home)
    python, runtime = _pick_cli_python(hermes, home)
    echo_and_run([uv_exe, "pip", "install", "--python", str(python), str(repo_root)])
    hermes_runtime.write_cache(home, {hermes_runtime.CACHE_RUNTIME: runtime, hermes_runtime.CACHE_CLI: str(python)})

    plugin_target.parent.mkdir(exist_ok=True, parents=True)
    _swap_in_plugin(repo_root / "neospike", plugin_target)
    list_plugin_enabled(home)
    for owner in target_homes(home):  # the root and every profile (bot) that uses neospike
        seed_memory_rule(owner)
    # numpy comes from plugin.yaml python_dependencies through the Hermes PM, never a direct pip install.
    echo_and_run(enable_plugin_argv(hermes), env=hermes_process_env(home))
    if not args.skip_models:
        _download_models(python, model_dir)

    # imports only (not is_available), before memory.provider points at neospike.
    runtime = confirm_runtime_after_sync(hermes, home, subprocess.run)
    hermes_runtime.write_cache(home, {hermes_runtime.CACHE_RUNTIME: runtime})

    # HERMES_HOME pins the write to --home (without it Hermes picks its default home).
    echo_and_run([hermes, "config", "set", "memory.provider", PROVIDER_ID], env=hermes_process_env(home))
    purge_old_crons(hermes, home)
    if not skip_cron_for_profile(home):
        copy_cron_scripts(home)
        apply_cron_jobs(hermes, home)
    apply_mcp_registration(str(python), home)
    print("Installed. Restart the Hermes gateway (hermes gateway restart), then run: python install.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
