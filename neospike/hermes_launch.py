"""Start a Neospike module in the Hermes runtime, the way the Hermes launcher starts its own.

    <runtime python> -I hermes_launch.py <hermes checkout> <module> [args...]

Only modules run (no code string): a check is a module of the package (neospike.runtime_check).

The steps follow the command `hermes --print-runtime-command` prints, in the same order: drop PYTHONHOME,
PYTHONPATH and VIRTUAL_ENV; put the checkout first on sys.path; pin the default home; import hermes_bootstrap,
which selects and leases the current dependency environment at every start (so no environment folder is
written into a registered command); settle HERMES_HOME. Then the folder that holds this plugin goes at the end
of sys.path, after Hermes and its dependencies so no plugin folder shadows them, and the module runs as
__main__. sys.argv is left as it is during the bootstrap import: when the bootstrap restarts the process with
another interpreter it re-runs this file with the same arguments.

Stdlib only; nothing of the package is imported before the paths are set.
"""
import os
import sys
import runpy

ENV_DROPPED = ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV")
PIN_DEFAULT_HOME_FLAG = "_hermes_pin_default_home"
BOOTSTRAP_MODULE, CONSTANTS_MODULE = "hermes_bootstrap", "hermes_constants"
HOME_VARIABLE = "HERMES_HOME"
USAGE = "usage: python -I hermes_launch.py <hermes checkout> <module> [args...]"


def script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def plugins_dir() -> str:
    """The folder that holds this plugin's package folder (<home>/plugins for an installed plugin)."""
    return os.path.dirname(script_dir())


def _same_path(left: str, right: str) -> bool:
    return os.path.normcase(os.path.abspath(left or os.curdir)) == os.path.normcase(os.path.abspath(right))


def prepare(checkout: str) -> None:
    """The launcher prologue, then the plugins folder at the end of sys.path."""
    for name in ENV_DROPPED:
        os.environ.pop(name, None)
    # Started without -I the package folder itself is sys.path[0]; its modules must not import as top level.
    sys.path[:] = [entry for entry in sys.path if not _same_path(entry, script_dir())]
    sys.path.insert(0, checkout)
    setattr(sys, PIN_DEFAULT_HOME_FLAG, True)
    __import__(BOOTSTRAP_MODULE)
    if not os.environ.get(HOME_VARIABLE):
        os.environ[HOME_VARIABLE] = str(__import__(CONSTANTS_MODULE).get_default_hermes_root())
    plugins = plugins_dir()
    if not any(_same_path(entry, plugins) for entry in sys.path):
        sys.path.append(plugins)


def run(target: list[str]) -> None:
    """*target* is `<module> [args...]`; sys.argv becomes what `python -m <module>` would set."""
    sys.argv = [sys.argv[0], *target[1:]]
    runpy.run_module(target[0], run_name="__main__", alter_sys=True)


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        raise SystemExit(USAGE)
    prepare(argv[0])
    run(argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
