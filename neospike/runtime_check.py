"""Checks that run in the Hermes runtime, started like every other Neospike module:

    <runtime python> -I hermes_launch.py <hermes checkout> neospike.runtime_check <check> [names...]

    imports <module>...   import each module (the Dashboard's dependency check)
    present <module>...   find each module without importing it (the provider's inference check)
    provider              NeospikeProvider().is_available() (the Dashboard's health check)
    sites                 print the site-packages folders on sys.path as a JSON list (install.py shows
                          which environment the bootstrap selected)

Exit 0 when the check passes, 1 when it fails (the reason on stderr), 2 on a usage error.
Stdlib only; nothing else is imported except what a check names.
"""
import importlib
import importlib.util
import json
import os
import sys

MODULE = "neospike.runtime_check"
CHECK_IMPORTS, CHECK_PRESENT, CHECK_PROVIDER, CHECK_SITES = "imports", "present", "provider", "sites"
SITE_FOLDERS = ("site-packages", "dist-packages")
EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2
USAGE = ("usage: neospike.runtime_check imports <module>... | present <module>... | provider | sites")


def check_imports(names: list[str]) -> int:
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # any import-time failure means the module is unusable here
            print(f"cannot import {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            return EXIT_FAILED
    return EXIT_OK


def check_present(names: list[str]) -> int:
    for name in names:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            print(f"module not found: {name}", file=sys.stderr)
            return EXIT_FAILED
    return EXIT_OK


def check_provider(names: list[str]) -> int:
    if names:
        print(USAGE, file=sys.stderr)
        return EXIT_USAGE
    from neospike import NeospikeProvider
    if NeospikeProvider().is_available():
        return EXIT_OK
    print("NeospikeProvider().is_available() is False", file=sys.stderr)
    return EXIT_FAILED


def check_sites(names: list[str]) -> int:
    if names:
        print(USAGE, file=sys.stderr)
        return EXIT_USAGE
    sites = [entry for entry in sys.path if os.path.basename(os.path.normpath(entry)) in SITE_FOLDERS]
    print(json.dumps(sites))
    return EXIT_OK


CHECKS = {CHECK_IMPORTS: check_imports, CHECK_PRESENT: check_present, CHECK_PROVIDER: check_provider,
          CHECK_SITES: check_sites}
NEEDS_NAMES = (CHECK_IMPORTS, CHECK_PRESENT)


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in CHECKS or (argv[0] in NEEDS_NAMES and len(argv) < 2):
        print(USAGE, file=sys.stderr)
        return EXIT_USAGE
    return CHECKS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
