"""plugin.yaml python_dependencies covers every third-party module the plugin code imports.

A catalog install only installs python_dependencies into the Hermes runtime, so an import of an
undeclared package fails there. The scan reads every import statement (including ones inside functions).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from neospike.tests.test_install_check import REPO_ROOT

PACKAGE_DIR = REPO_ROOT / "neospike"
PLUGIN_YAML = PACKAGE_DIR / "plugin.yaml"
DEPENDENCIES_KEY = "python_dependencies:"
# Top-level import name -> distribution name declared in plugin.yaml.
DISTRIBUTION_OF = {
    "numpy": "numpy",
    "faiss": "faiss-cpu",
    "tokenizers": "tokenizers",
    "onnxruntime": "onnxruntime",
    "huggingface_hub": "huggingface-hub",  # run as a module (`hf download`), not imported
}
# Imported but not declared: provided by Hermes itself, or optional with an ImportError fallback.
NOT_DECLARED = {
    "agent": "Hermes agent package (host)",
    "hermes_constants": "Hermes agent package (host)",
    "fastapi": "Hermes dashboard (host)",
    "pydantic": "Hermes dashboard (host)",
    "yaml": "Hermes dependency; cron scripts fall back when absent",
    "mcp": "optional extra [mcp] for the Claude Code MCP server",
}
VERSION_MARKS = "<>=!~;[ "


def _plugin_sources() -> list[Path]:
    sources = [REPO_ROOT / "install.py"]
    sources += [path for path in PACKAGE_DIR.rglob("*.py") if "tests" not in path.relative_to(PACKAGE_DIR).parts]
    return sources


def _local_names() -> set[str]:
    """Modules of this repository (absolute imports between sibling scripts count as local)."""
    names = {path.stem for path in _plugin_sources()}
    names.update(path.name for path in PACKAGE_DIR.rglob("*") if path.is_dir())
    names.add(PACKAGE_DIR.name)
    return names


def _imported_top_names(source: Path) -> dict[str, int]:
    """Top-level module name -> first line, for absolute imports anywhere in the file."""
    found: dict[str, int] = {}
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8-sig"))):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        else:
            continue
        for module in modules:
            found.setdefault(module.split(".")[0], node.lineno)
    return found


def third_party_imports() -> dict[str, str]:
    """Third-party top-level module -> first 'file:line' that imports it."""
    skip = set(sys.stdlib_module_names) | _local_names()
    result: dict[str, str] = {}
    for source in _plugin_sources():
        for name, line in _imported_top_names(source).items():
            if name not in skip:
                result.setdefault(name, f"{source.relative_to(REPO_ROOT)}:{line}")
    return result


def declared_distributions(text: str) -> set[str]:
    """Distribution names in the python_dependencies list (text parse, same as install.py --check)."""
    names: set[str] = set()
    in_list = False
    for raw in text.splitlines():
        line = raw.strip()
        if line == DEPENDENCIES_KEY:
            in_list = True
            continue
        if in_list and line.startswith("- "):
            item = line[2:].strip().strip('"').strip("'")
            cut = min((item.index(mark) for mark in VERSION_MARKS if mark in item), default=len(item))
            names.add(item[:cut].lower())
        elif in_list and line:
            in_list = False
    return names


def undeclared_imports(text: str) -> list[str]:
    declared = declared_distributions(text)
    problems = []
    for name, where in sorted(third_party_imports().items()):
        if name in NOT_DECLARED:
            continue
        distribution = DISTRIBUTION_OF.get(name)
        if distribution is None:
            problems.append(f"{where}: {name} has no entry in DISTRIBUTION_OF or NOT_DECLARED")
        elif distribution not in declared:
            problems.append(f"{where}: {name} needs {distribution} in plugin.yaml {DEPENDENCIES_KEY}")
    return problems


def test_plugin_yaml_declares_every_third_party_import():
    assert undeclared_imports(PLUGIN_YAML.read_text(encoding="utf-8-sig")) == []


def test_scan_sees_the_inference_imports():
    found = third_party_imports()
    assert {"numpy", "faiss", "tokenizers", "onnxruntime"} <= set(found)


def test_a_missing_declaration_is_reported():
    text = PLUGIN_YAML.read_text(encoding="utf-8-sig")
    without = "\n".join(line for line in text.splitlines() if "onnxruntime" not in line)
    problems = undeclared_imports(without)
    assert len(problems) == 1 and "onnxruntime" in problems[0]
