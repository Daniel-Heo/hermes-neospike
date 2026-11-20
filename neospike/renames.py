"""v21 command-line option names, still accepted with a warning."""
from __future__ import annotations

import sys

OLD_OPTIONS = {"--namespace": "--owner"}
WARNING = "neospike: '{old}' is renamed to '{new}'; the old name stops working in the next release\n"


def forward_old_options(argv: list[str], stderr=None) -> list[str]:
    """argv with every v21 option (OLD_OPTIONS, also in the --name=value form) renamed; one warning per option."""
    result, warned = [], set()
    for token in argv:
        name, sep, value = token.partition("=")
        new = OLD_OPTIONS.get(name)
        if new is None:
            result.append(token)
            continue
        if name not in warned:
            (stderr or sys.stderr).write(WARNING.format(old=name, new=new))
            warned.add(name)
        result.append(new + sep + value if sep else new)
    return result
