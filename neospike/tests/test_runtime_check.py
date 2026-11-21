"""neospike.runtime_check: the checks the Dashboard and the provider run in the Hermes runtime."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from neospike import runtime_check as rc

MISSING = "neospike_missing_dep"
STDLIB_ONLY = {"importlib", "json", "os", "sys"}


class _Availability:
    def __init__(self, answer: bool):
        self.answer = answer

    def __call__(self) -> bool:  # a class attribute that is not a function is not bound: no self of the provider
        return self.answer


@pytest.mark.parametrize(("argv", "code"), [
    (["imports", "json", "sqlite3"], rc.EXIT_OK),
    (["imports", "json", MISSING], rc.EXIT_FAILED),
    (["present", "json"], rc.EXIT_OK),
    (["present", MISSING], rc.EXIT_FAILED),
    (["present", "json.no_such_part"], rc.EXIT_FAILED),
    ([], rc.EXIT_USAGE), (["imports"], rc.EXIT_USAGE), (["unknown"], rc.EXIT_USAGE), (["provider", "x"], rc.EXIT_USAGE),
    (["sites"], rc.EXIT_OK), (["sites", "x"], rc.EXIT_USAGE),
])
def test_exit_codes(argv, code):
    assert rc.main(argv) == code


def test_a_failed_import_names_the_module(capsys):
    assert rc.main(["imports", MISSING]) == rc.EXIT_FAILED
    assert MISSING in capsys.readouterr().err


@pytest.mark.parametrize(("available", "code"), [(True, rc.EXIT_OK), (False, rc.EXIT_FAILED)])
def test_provider_check_follows_is_available(monkeypatch, available, code):
    import neospike.provider as provider
    monkeypatch.setattr(provider.NeospikeProvider, "is_available", _Availability(available))
    assert rc.main(["provider"]) == code


def test_sites_prints_the_site_packages_of_sys_path_as_json(monkeypatch, capsys):
    site = str(Path("venv") / "Lib" / "site-packages")
    monkeypatch.setattr(rc.sys, "path", ["checkout", site, str(Path("plugins"))])
    assert rc.main(["sites"]) == rc.EXIT_OK
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == [site]


def test_module_imports_only_the_standard_library():
    tree = ast.parse(Path(rc.__file__).read_text(encoding="utf-8-sig"))
    top = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    top |= {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert top == STDLIB_ONLY
