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
from neospike import inference
from neospike import runtime_check
from neospike._wincompat import shared_inference_home
from neospike.config import read_memory_rule, memory_rule_file_path, write_memory_rule
from neospike.cron.neospike_cron_common import configured_memory_provider, target_homes
from neospike.cron.neospike_cron_common import yaml_scalar as _scalar_of
from neospike.cron.neospike_cron_common import _yaml_data
from neospike.store import memory_db_path, pick_namespace


# BAAI/bge-m3 int8 ONNX at a pinned revision; not gated (MIT).
MODEL_REPO = "Xenova/bge-m3"
MODEL_REVISION = "4de13258303883538bd53b696b452bf8099f0858"
MODEL_ONNX_NAME = "sentence_transformers_int8.onnx"  # embedder.ONNX_MODEL_FILE
MODEL_DOWNLOADS = ["onnx/" + MODEL_ONNX_NAME, "tokenizer.json"]
MODEL_FOLDER = "bge-m3-onnx"  # embedder.MODEL_FOLDER


STATUS_MARKERS = ("Plugin:    installed ✓", "Status:    available ✓")
PROBE_TIMEOUT_S = 120
# The probes run neospike.runtime_check through the installed copy's hermes_launch.py, which starts Python the way
# the Hermes launcher does (hermes_bootstrap selects the PM environment venv, installs/<id>/environments/<hash>/venv),
# so they see the same sys.path as the daemon, the MCP server and the cron job.
SITE_CHECK = (runtime_check.CHECK_SITES,)
LOAD_CHECK = (runtime_check.CHECK_PROVIDER,)
# Right after the PM sync only the imports are checked; is_available() also needs the model files,
# which --skip-models leaves out, so that check stays in --check (LOAD_CHECK).
IMPORT_CHECK = (runtime_check.CHECK_IMPORTS, "numpy", "faiss", "neospike", "neospike.provider")
# What the Hermes process itself imports from this package. Declared in neospike/plugin.yaml
# python_dependencies and installed by the Hermes package manager (tests tie it to plugin.yaml and pyproject).
# faiss-cpu: the HNSW candidates of vector_index.py, used in the Hermes process by every vector search of the
# database file (without it in the Hermes runtime every such search would fail).
# tokenizers, onnxruntime (on Windows the onnxruntime-directml build of the same module): embedder.py; the
# embedding daemon runs in the same environment through hermes_launch.py
# (_wincompat.inference_command), and a catalog install does not run this script.
# huggingface-hub: the model download (`python -m huggingface_hub.cli.hf download`, here and in the Dashboard).
PLUGIN_PIP_REQUIREMENTS = ("numpy>=2.0,<3", "faiss-cpu>=1.16,<2", "tokenizers>=0.22,<1",
                           "onnxruntime-directml>=1.20,<2; sys_platform == 'win32'",
                           "onnxruntime>=1.20,<2; sys_platform != 'win32'", "huggingface-hub>=1.0,<2")
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
MCP_MODULE = "neospike.mcp_server"
HF_CLI_MODULE = "huggingface_hub.cli.hf"  # the `hf` console script's module
# An earlier install.py ran `uv pip install` of this package into the Hermes CLI venv (hermes-agent/venv);
# that copy is removed (this distribution only), since every Neospike process now runs the plugin folder.
OLD_DISTRIBUTION = "neospike"
OLD_DIST_GLOBS = ("Lib/site-packages/neospike-*.dist-info", "lib/python*/site-packages/neospike-*.dist-info")
# runtime.json keys an earlier install.py wrote and nothing reads any more; removed on install.
OLD_CACHE_KEYS = ("cli_python",)
DRY_RUN_PYTHON = "<hermes runtime python>"
DAILY_JOB_NAME, DAILY_JOB_WHEN = "neospike-daily", "every day at 12pm"
DATA_FOLDER, PLUGIN_FOLDER, PROVIDER_ID = "neospike", "neospike", "neospike"
# Hermes loads every folder in <home>/plugins as a plugin, and a second copy with the same manifest name
# shadows plugins/neospike. So the install copy and the previous plugin live in <home>/neospike/.install
# (same volume as plugins/, so Path.replace is a rename); one previous copy is kept, --uninstall removes it.
INSTALL_SCRATCH_FOLDER, STAGING_FOLDER, PREVIOUS_FOLDER = ".install", "staging", "previous"
# Folders an earlier install.py left in <home>/plugins: install and --uninstall remove them, --check fails on them.
PLUGIN_LEFTOVERS = ("neospike.previous", "neospike.installing")

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
# The preference/constraint cleanup model: Hermes auxiliary slot of pref_reconcile.LLM_TASK. "auto" follows the chat model.
AUXILIARY_SECTION = "auxiliary"
PREF_MODEL_TASK = "neospike_pref_reconcile"  # pref_reconcile.LLM_TASK (a test keeps them equal)
PREF_MODEL_DEFAULT = "auto"


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
    command = runtime_check_command(python, home, SITE_CHECK)
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


def runtime_check_command(python: str, home: Path, check: tuple[str, ...]) -> list[str]:
    """neospike.runtime_check *check* through the installed plugin's hermes_launch.py."""
    return runtime_target_command(python, home, [runtime_check.MODULE, *check])


def provider_import_error(python: str, home: Path, runner=None, check: tuple[str, ...] = LOAD_CHECK) -> str | None:
    """Run *check* (default LOAD_CHECK) in the bootstrapped runtime python; the failure text, or None."""
    runner = _runner_or_default(runner)
    command = runtime_check_command(python, home, check)
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
    failure = provider_import_error(python, home, runner, IMPORT_CHECK)
    if failure:
        raise SystemExit(
            "neospike provider import check failed after install.\n"
            f"Missing: {failure}\n"
            f"Runtime environment: {environment or seen}\n"
            f"Check the PM receipt with `hermes pm status`: plugin.yaml {PLUGIN_REQS_KEY} "
            f"{', '.join(PLUGIN_PIP_REQUIREMENTS)} must be installed in this environment.")
    return python


def old_cli_distributions(home: Path) -> list[Path]:
    """dist-info folders of the neospike package an earlier install.py put into the Hermes CLI venv."""
    venv = hermes_runtime.hermes_agent_dir(home) / "venv"
    return sorted(path for pattern in OLD_DIST_GLOBS for path in venv.glob(pattern))


def old_cli_uninstall_argv(uv: str, home: Path) -> list[str] | None:
    """`uv pip uninstall` of OLD_DISTRIBUTION alone from the Hermes CLI venv; None when nothing is there."""
    if not old_cli_distributions(home):
        return None
    return [uv, "pip", "uninstall", "--python", str(hermes_runtime.checkout_python(home)), OLD_DISTRIBUTION]


def remove_old_cli_install(home: Path) -> None:
    """Uninstall the old package copy from the Hermes CLI venv (no other package is touched)."""
    if not old_cli_distributions(home):
        return
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv is required to remove the old neospike package from the Hermes CLI venv "
                         f"({hermes_runtime.checkout_python(home)})")
    echo_and_run(old_cli_uninstall_argv(uv, home))


def installed_launch_script(home: Path) -> Path:
    """hermes_launch.py of the installed plugin copy (<home>/plugins/neospike)."""
    return home / "plugins" / PLUGIN_FOLDER / hermes_runtime.LAUNCH_SCRIPT


def runtime_target_command(python: str, home: Path, target: list[str]) -> list[str]:
    """*target* through the installed plugin's hermes_launch.py on the Hermes runtime *python*."""
    return hermes_runtime.launch_command(home, target, python=python, script=installed_launch_script(home))


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


def _top_level_block(text: str, name: str) -> list[str]:
    """Indented lines under the top-level `<name>:` key of a YAML text (no PyYAML needed)."""
    block, inside = [], False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            inside = re.match(re.escape(name) + r":\s*(#.*)?$", line) is not None
            continue
        if inside:
            block.append(line)
    return block


def pref_model_configured(home: Path) -> bool:
    """True when <home>/config.yaml already has auxiliary.<PREF_MODEL_TASK> (any value, set by the user or earlier)."""
    try:
        text = (home / "config.yaml").read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return False
    data = _yaml_data(text)
    if isinstance(data, dict):
        auxiliary = data.get(AUXILIARY_SECTION)
        return isinstance(auxiliary, dict) and PREF_MODEL_TASK in auxiliary
    child = re.compile(r"\s+['\"]?" + re.escape(PREF_MODEL_TASK) + r"['\"]?\s*:")
    return any(child.match(line) for line in _top_level_block(text, AUXILIARY_SECTION))


def pref_model_default_argv(hermes: str) -> list[str]:
    return [hermes, "config", "set", f"{AUXILIARY_SECTION}.{PREF_MODEL_TASK}.provider", PREF_MODEL_DEFAULT]


def homes_without_pref_model(home: Path) -> list[Path]:
    return [owner for owner in target_homes(home) if not pref_model_configured(owner)]


def apply_pref_model_default(hermes: str, home: Path) -> None:
    """auxiliary.<PREF_MODEL_TASK>.provider = auto (follow the chat model) in every home that has no such key yet;
    a value already there is never overwritten. HERMES_HOME pins each write to its home."""
    for owner in homes_without_pref_model(home):
        echo_and_run(pref_model_default_argv(hermes), env=hermes_process_env(owner))


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
    """The neospike MCP server entry, same shape as the hand-made registration. The Hermes runtime *python*
    starts the installed plugin's hermes_launch.py (-I: no launch folder, PYTHONPATH or user site on
    sys.path), which selects the current dependency environment at every start; no environment folder is
    written here."""
    namespace = pick_namespace(None, memory_db_path(home))
    command = runtime_target_command(python, home, [MCP_MODULE, "--home", str(home), "--owner", namespace])
    return {
        "type": "stdio",
        "command": command[0],
        "args": command[1:],
        "env": {
            "HERMES_HOME": str(home),
            "HERMES_BGE_M3_DIR": str(shared_inference_home(home) / "models" / MODEL_FOLDER),
            "PYTHONIOENCODING": "utf-8",
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
    traces += [f"Old neospike package remains in the Hermes CLI venv: {path} (rerun install.py)"
               for path in old_cli_distributions(home)]
    traces += [f"Old install folder remains: {path} (Hermes's plugin discovery takes it for the neospike plugin "
               "and skips the real one; rerun install.py)" for path in plugin_leftovers(home)]
    return traces


def install_scratch(home: Path) -> Path:
    """Where install.py stages the new plugin copy and keeps the previous one (outside <home>/plugins)."""
    return home / DATA_FOLDER / INSTALL_SCRATCH_FOLDER


def plugin_leftovers(home: Path) -> list[Path]:
    """PLUGIN_LEFTOVERS present in <home>/plugins."""
    plugins = home / "plugins"
    return [plugins / name for name in PLUGIN_LEFTOVERS if (plugins / name).exists() or (plugins / name).is_symlink()]


def old_cache_keys(home: Path) -> list[str]:
    """OLD_CACHE_KEYS still present in runtime.json."""
    cached = hermes_runtime.read_cache(home)
    return [key for key in OLD_CACHE_KEYS if key in cached]


def migration_preview(hermes: str, home: Path) -> list[str]:
    """--dry-run lines for the dropped cron leftovers and the skill rewrites (empty when nothing is left)."""
    lines = [subprocess.list2cmdline(command) for command in old_cron_remove_argv(hermes, home)]
    lines += [f"remove {home / 'scripts' / name}" for name in DROPPED_FILES if (home / "scripts" / name).is_file()]
    lines += [f"rename old names in {path} (copy in the backup zip first)" for path in skills_with_old_names(home)]
    uninstall = old_cli_uninstall_argv(shutil.which("uv") or "uv", home)
    lines += [subprocess.list2cmdline(uninstall)] if uninstall else []
    lines += [f"remove {key} from {hermes_runtime.cache_file(home)}" for key in old_cache_keys(home)]
    lines += [f"remove {path}" for path in plugin_leftovers(home)]
    return lines


# ---- --uninstall: what install.py made outside the plugin folder goes too; the user's data and the model stay.
# Every step skips what is already gone, so a rerun (or a run after a partial uninstall) is safe.
def _plan_or_run(command: list[str], dry_run: bool, env: dict | None = None) -> None:
    """Print *command* as a '+' line; run it (check=True) unless *dry_run*."""
    if dry_run:
        print("+", subprocess.list2cmdline(command))
        return
    echo_and_run(command, **({"env": env} if env is not None else {}))


def _plan_or_remove(path: Path, dry_run: bool) -> None:
    """'+ remove <path>'; the file or folder is deleted unless *dry_run*."""
    print("+ remove", path)
    if dry_run:
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def uninstall_cron(hermes: str, home: Path, dry_run: bool) -> None:
    """Step 1 (root home only; a profile install registers no job): `hermes cron remove` of the daily job, then
    the cron scripts in <home>/scripts."""
    if skip_cron_for_profile(home):
        return
    profiles = target_homes(home)[1:]
    if profiles:
        print(f"The daily job also served {', '.join(map(str, profiles))}; reinstall in the root home to keep it.")
    record = read_cron_jobs(home).get(DAILY_JOB_NAME)
    if record is None:
        print(f"Cron job {DAILY_JOB_NAME}: not registered (skipped)")
    else:
        _plan_or_run([hermes, "cron", "remove", str(record.get("id") or record.get("name"))], dry_run,
                     hermes_process_env(home))
    for name in CRON_FILES:
        script = home / "scripts" / name
        if script.is_file():
            _plan_or_remove(script, dry_run)


def mcp_points_at(claude: str, home: Path, runner=None) -> bool:
    """True when Claude Code has the neospike MCP server and its entry names *home* (`claude mcp get`, read only)."""
    runner = _runner_or_default(runner)
    try:
        shown = runner([claude, "mcp", "get", MCP_SERVER_NAME], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return False
    return shown.returncode == 0 and os.path.normcase(str(home)) in os.path.normcase(shown.stdout or "")


def uninstall_mcp(home: Path, dry_run: bool, runner=None) -> None:
    """Step 2: `claude mcp remove`, only when the user-scope entry points at *home* (never for a profile home)."""
    if skip_mcp_for_profile(home):
        return
    claude = shutil.which("claude")
    if not claude:
        print(f"Claude Code (claude) not found; MCP server {MCP_SERVER_NAME} left as it is.")
        return
    if not mcp_points_at(claude, home, runner):
        print(f"Claude Code MCP server {MCP_SERVER_NAME}: not registered for {home} (skipped)")
        return
    _plan_or_run([claude, "mcp", "remove", "-s", MCP_REG_SCOPE, MCP_SERVER_NAME], dry_run)


def uninstall_daemon(home: Path, dry_run: bool) -> None:
    """Step 3: ask the shared embedding service to stop, unless another home still uses Neospike."""
    shared = shared_inference_home(home)
    users = inference.daemon_users(home)
    if users:
        print(f"Embedding service of {shared} left running: still used by {', '.join(map(str, users))}")
        return
    if dry_run:
        print(f"+ ask the embedding service of {shared} to stop (when it runs)")
        return
    try:
        state, _users = inference.release_daemon(home)
    except OSError as exc:
        print(f"Embedding service of {shared} could not be asked to stop: {exc}")
        return
    print(f"Embedding service of {shared}: {'asked to stop' if state == inference.DAEMON_STOPPED else 'not running'}")


def unlist_plugin_enabled(home: Path, dry_run: bool = False) -> bool:
    """Remove PLUGIN_FOLDER from plugins.enabled in <home>/config.yaml (True = it was listed). Only for a home
    whose plugin folder is already gone (`hermes plugins remove` does it otherwise). Edited as text, like
    list_plugin_enabled."""
    path = home / "config.yaml"
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return False
    lines = raw.decode("utf-8-sig").splitlines(keepends=True)
    key, items = _find_enabled_lines(lines)
    if key is None:
        return False
    flow = re.match(r"(\s*enabled:\s*\[)([^\]]*)(\].*)$", lines[key], re.S)
    if flow:
        names = [part for part in flow.group(2).split(",") if part.strip()]
        kept = [part.strip() for part in names if _scalar_of(part) != PLUGIN_FOLDER]
        if len(kept) == len(names):
            return False
        lines[key] = flow.group(1) + ", ".join(kept) + flow.group(3)
    else:
        listed = [index for index in items if _scalar_of(lines[index].lstrip()[2:]) == PLUGIN_FOLDER]
        if not listed:
            return False
        for index in reversed(listed):
            del lines[index]
    if not dry_run:
        path.write_bytes((BOM_BYTES if raw.startswith(BOM_BYTES) else b"") + "".join(lines).encode("utf-8"))
    print(f"{'+ remove' if dry_run else 'Removed'} {PLUGIN_FOLDER} from plugins.enabled in {path}")
    return True


def uninstall_plugin(hermes: str, home: Path, dry_run: bool) -> None:
    """Step 4: `hermes plugins remove neospike` deletes <home>/plugins/neospike and clears plugins.enabled /
    plugins.disabled and a memory.provider naming it (hermes_cli/plugins_cmd.py _forget_plugin_config). With the
    folder already gone: `hermes memory off` when memory.provider is still neospike, and plugins.enabled edited.
    Then the install scratch (the previous plugin copy) and what an earlier install left in <home>/plugins."""
    env = hermes_process_env(home)
    target = home / "plugins" / PLUGIN_FOLDER
    if target.exists() or target.is_symlink():
        _plan_or_run([hermes, "plugins", "remove", PLUGIN_FOLDER], dry_run, env)
    else:
        print(f"Plugin folder {target}: not there (skipped)")
        if configured_memory_provider(home) == PROVIDER_ID:
            _plan_or_run([hermes, "memory", "off"], dry_run, env)
        unlist_plugin_enabled(home, dry_run)
    for leftover in plugin_leftovers(home):
        _plan_or_remove(leftover, dry_run)
    if install_scratch(home).exists():
        _plan_or_remove(install_scratch(home), dry_run)


def print_kept_data(home: Path) -> None:
    """Step 5: the user's data and the model stay; where they are and how to delete them."""
    data = home / DATA_FOLDER
    model = shared_inference_home(home) / "models" / MODEL_FOLDER
    print(f"Kept your Neospike data (memories, notes, conversations, backups): {data}")
    print(f"Kept the model{' (shared with the root home)' if is_profile(home) else ''}: {model}")
    print("To delete them for good, delete those folders yourself; the model folder is shared by every Neospike "
          "home, so delete it only when no home uses Neospike any more.")


def run_uninstall(hermes: str, home: Path, dry_run: bool) -> int:
    """--uninstall (with --dry-run: the plan only, nothing changed)."""
    print(f"Neospike uninstall of {home}{' (dry run: nothing is changed)' if dry_run else ''}")
    uninstall_cron(hermes, home, dry_run)
    uninstall_mcp(home, dry_run)
    uninstall_daemon(home, dry_run)
    uninstall_plugin(hermes, home, dry_run)
    print_kept_data(home)
    if not dry_run:
        print("Uninstalled. Restart the Hermes gateway (hermes gateway restart) so it stops loading Neospike.")
    return 0


def _parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(description="Install Neospike without touching the Hermes checkout")
    cli.add_argument("--home", default=default_home())
    cli.add_argument("--skip-models", action="store_true",
                     help="Install code and dependencies without downloading model artifacts")
    cli.add_argument("--check", action="store_true", help="Only verify an existing installation")
    cli.add_argument("--dry-run", action="store_true",
                     help="Print the cron scripts and cron commands; change nothing")
    cli.add_argument("--uninstall", action="store_true",
                     help="Remove Neospike from --home (cron job, Claude Code MCP entry, plugin folder); "
                          "your data and the model stay. With --dry-run: print the plan only")
    return cli.parse_args()


def _dry_run_python(home: Path) -> str:
    """The cached runtime python for the --dry-run lines (no launcher run), else a placeholder."""
    cached = hermes_runtime.read_cache(home).get(hermes_runtime.CACHE_RUNTIME)
    return cached if isinstance(cached, str) and cached else DRY_RUN_PYTHON


def _print_dry_run(hermes: str, home: Path) -> None:
    """--dry-run: every action as a '+' line, nothing executed."""
    for step in migration_preview(hermes, home):
        print("+", step)
    print("+", subprocess.list2cmdline(enable_plugin_argv(hermes)))
    print("+", subprocess.list2cmdline(runtime_check_command(_dry_run_python(home), home, IMPORT_CHECK)))
    for owner in target_homes(home):
        if not memory_rule_file_path(owner).exists():
            print("+ create", memory_rule_file_path(owner))
    for owner in homes_without_pref_model(home):
        print("+", f"HERMES_HOME={owner}", subprocess.list2cmdline(pref_model_default_argv(hermes)))
    if not skip_cron_for_profile(home):
        for name in CRON_FILES:
            print("+ copy", CRON_SRC_DIR / name, "->", home / "scripts" / name)
        for argv in cron_argv_list(hermes, home):
            print("+", subprocess.list2cmdline(argv))
    if not skip_mcp_for_profile(home):
        for argv in mcp_argv_list(shutil.which("claude") or "claude", _dry_run_python(home), home):
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


def _pick_runtime_python(hermes: str, home: Path) -> str:
    """The Hermes runtime python every Neospike process outside the gateway starts with (hermes_launch.py)."""
    runtime, source = find_runtime_python(hermes, home, subprocess.run)
    if runtime is None:
        raise SystemExit(f"Hermes runtime python not found ({source}).")
    print(f"Hermes runtime python: {runtime} ({source})")
    return runtime


def _remove_tree(path: Path) -> None:
    """Delete a folder (or a link to one) if it is there."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _swap_in_plugin(source_dir: Path, plugin_target: Path, scratch: Path) -> None:
    """Copy the package to scratch/staging, move the installed copy to scratch/previous (replacing the one
    kept there), then move staging into place; if that last move fails, the installed copy goes back.
    Folders an earlier install left in <home>/plugins are deleted first."""
    for name in PLUGIN_LEFTOVERS:
        leftover = plugin_target.with_name(name)
        if leftover.exists() or leftover.is_symlink():
            _remove_tree(leftover)
            print(f"Removed {leftover} (it shadowed the plugin in Hermes's plugin discovery)")
    staging, previous = scratch / STAGING_FOLDER, scratch / PREVIOUS_FOLDER
    scratch.mkdir(parents=True, exist_ok=True)
    _remove_tree(staging)
    shutil.copytree(source_dir, staging, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    moved_old = plugin_target.exists()
    if moved_old:
        _remove_tree(previous)
        plugin_target.replace(previous)
    try:
        staging.replace(plugin_target)
    except OSError:
        if moved_old:
            previous.replace(plugin_target)
        raise


def _download_models(python: str, home: Path, model_dir: Path) -> None:
    """`hf download` as a module of the PM environment (plugin.yaml huggingface-hub), so no hf executable
    path of an environment folder is needed."""
    model_dir.mkdir(parents=True, exist_ok=True)
    fetch = runtime_target_command(python, home, [HF_CLI_MODULE, "download", MODEL_REPO, *MODEL_DOWNLOADS,
                                                  "--revision", MODEL_REVISION, "--local-dir", str(model_dir)])
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
    if args.uninstall:
        return run_uninstall(hermes, home, args.dry_run)
    if args.dry_run:
        # The venv's own hermes script is bound to a build-time snapshot; cron goes through the user's launcher.
        _print_dry_run(hermes, home)
        return 0
    if args.check:
        needed = [plugin_target / "plugin.yaml", model_dir / MODEL_ONNX_NAME, model_dir / "tokenizer.json"]
        return _run_check(home, hermes, needed, plugin_target / "plugin.yaml")

    repo_root = Path(__file__).resolve().parent
    migrate_skill_names(home)
    runtime = _pick_runtime_python(hermes, home)
    hermes_runtime.write_cache(home, {hermes_runtime.CACHE_RUNTIME: runtime}, drop=OLD_CACHE_KEYS)
    remove_old_cli_install(home)

    plugin_target.parent.mkdir(exist_ok=True, parents=True)
    _swap_in_plugin(repo_root / "neospike", plugin_target, install_scratch(home))
    list_plugin_enabled(home)
    for owner in target_homes(home):  # the root and every profile (bot) that uses neospike
        seed_memory_rule(owner)
    # PLUGIN_PIP_REQUIREMENTS come from plugin.yaml python_dependencies through the Hermes PM, never a direct pip install.
    echo_and_run(enable_plugin_argv(hermes), env=hermes_process_env(home))
    if not args.skip_models:
        _download_models(runtime, home, model_dir)

    # imports only (not is_available), before memory.provider points at neospike.
    runtime = confirm_runtime_after_sync(hermes, home, subprocess.run)
    hermes_runtime.write_cache(home, {hermes_runtime.CACHE_RUNTIME: runtime})

    # HERMES_HOME pins the write to --home (without it Hermes picks its default home).
    echo_and_run([hermes, "config", "set", "memory.provider", PROVIDER_ID], env=hermes_process_env(home))
    apply_pref_model_default(hermes, home)
    purge_old_crons(hermes, home)
    if not skip_cron_for_profile(home):
        copy_cron_scripts(home)
        apply_cron_jobs(hermes, home)
    apply_mcp_registration(runtime, home)
    print("Installed. Restart the Hermes gateway (hermes gateway restart), then run: python install.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
