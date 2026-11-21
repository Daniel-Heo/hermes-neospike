"""Background threads for the memory provider.

Inside Hermes every provider background job goes through agent.memory_provider.spawn_context_thread,
which runs the target under the spawner's contextvars (the active profile's HERMES_HOME override and
secret scope). Outside Hermes (the CLI, tests, older Hermes without that helper) a plain daemon
threading.Thread is used instead.
"""
from __future__ import annotations

import threading
from typing import Any, Callable


def background_thread(target: Callable[..., Any], name: str) -> threading.Thread:
    """An unstarted daemon thread running target; the caller starts it."""
    try:
        from agent.memory_provider import spawn_context_thread
    except ImportError:
        return threading.Thread(target=target, name=name, daemon=True)
    return spawn_context_thread(target, name=name, daemon=True)
