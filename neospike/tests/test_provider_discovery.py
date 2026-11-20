"""Hermes 코어의 디렉터리 로더가 이 플러그인 폴더를 memory provider로 찾아내는지."""
from __future__ import annotations

from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]


def _find_plugin_dir() -> Path:
    """neospike 패키지 폴더(plugin.yaml이 있는 곳). stage 안에서도 메인 트리를 찾는다."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "neospike" / "plugin.yaml").is_file():
            return candidate / "neospike"
    return PLUGIN_DIR


def test_core_loader_recognises_and_loads_plugin_directory():
    memory = pytest.importorskip("plugins.memory", reason="Requires installed Hermes core")
    plugin_dir = _find_plugin_dir()

    assert memory._is_memory_provider_dir(plugin_dir)
    loaded = memory._load_provider_from_dir(plugin_dir)

    assert loaded is not None
    assert loaded.name == "neospike"


def test_plugin_directory_has_manifest_and_entry_module():
    plugin_dir = _find_plugin_dir()

    assert (plugin_dir / "plugin.yaml").is_file()
    assert (plugin_dir / "__init__.py").is_file()
