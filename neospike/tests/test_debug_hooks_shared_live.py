"""Two module copies of the package (as Hermes imports it twice) must share one tracked set."""
from __future__ import annotations

import gc
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_COPY_NAMES = ('_neospike_copy_memory_loader', '_neospike_copy_plugin_discovery')
_SESSION = 'sid-shared-live'
_MARK = '**Neospike: diagnostics**'


class _FakeProvider:
    """Only the attributes the ownership check and the output hook read."""

    def __init__(self, home: Path, session_id: str) -> None:
        self._home_root = home
        self._active_sid = session_id
        self._core = object()

    def rewrite_output(self, response_text, *, session_id=''):
        return f'{response_text}\n\n{_MARK}'


def _load_copy(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, _PACKAGE_DIR / '__init__.py', submodule_search_locations=[str(_PACKAGE_DIR)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return importlib.import_module(f'{name}.debug_hooks')


def _forget_copies() -> None:
    for key in list(sys.modules):
        if key.split('.')[0] in _COPY_NAMES:
            del sys.modules[key]


@pytest.fixture
def two_copies(tmp_path, monkeypatch):
    home = tmp_path.resolve()
    constants = types.ModuleType('hermes_constants')
    monkeypatch.setitem(sys.modules, 'hermes_constants', constants)
    constants.get_hermes_home = _HomeGetter(home)
    _forget_copies()
    try:
        yield home, _load_copy(_COPY_NAMES[0]), _load_copy(_COPY_NAMES[1])
    finally:
        _forget_copies()


class _HomeGetter:
    def __init__(self, home: Path) -> None:
        self._home = home

    def __call__(self) -> Path:
        return self._home


def test_copies_are_distinct_modules(two_copies):
    _, memory_copy, discovery_copy = two_copies

    assert memory_copy is not discovery_copy


def test_hook_of_one_copy_finds_provider_tracked_by_the_other(two_copies):
    home, memory_copy, discovery_copy = two_copies
    provider = _FakeProvider(home, _SESSION)

    memory_copy.track(provider)
    rewritten = discovery_copy.on_output('answer', session_id=_SESSION)

    assert rewritten is not None and rewritten.endswith(_MARK)


def test_shared_set_still_holds_providers_weakly(two_copies):
    home, memory_copy, discovery_copy = two_copies
    provider = _FakeProvider(home, _SESSION)
    memory_copy.track(provider)

    del provider
    gc.collect()

    assert discovery_copy.on_output('answer', session_id=_SESSION) is None
