"""Wire live memory providers into Hermes' public lifecycle hooks.

register(ctx) registers the hooks through the public ``ctx.register_hook``; Hermes replaces
them when it runs register() again for the same plugin. Initialized providers are tracked
with weak references only: process-wide hooks must not keep finished agents or their memory
alive. Nothing in Hermes is replaced.

Hermes may import this package twice under different module names (once as the memory
provider, once by general plugin discovery) and keeps only one copy's hooks, while the
provider in use may come from the other copy. The tracked set therefore lives in one
process-wide holder module with a fixed name, shared by every copy.
"""
from __future__ import annotations

import sys
import threading
import types
import weakref
from pathlib import Path

# Bump the suffix if the holder's attributes ever change shape.
_SHARED_STATE_MODULE = '_neospike_live_providers_v1'


def _shared_state() -> types.ModuleType:
    """The process-wide holder of the lock and the weak provider set (created once)."""
    state = sys.modules.get(_SHARED_STATE_MODULE)
    if state is None:
        candidate = types.ModuleType(_SHARED_STATE_MODULE)
        candidate.lock = threading.RLock()
        candidate.live = weakref.WeakSet()
        # setdefault is atomic: concurrent first imports agree on one holder.
        state = sys.modules.setdefault(_SHARED_STATE_MODULE, candidate)
    return state


_STATE = _shared_state()
_LOCK = _STATE.lock
_LIVE = _STATE.live


def track(provider) -> None:
    """Make an initialized provider visible to the hooks (weakly)."""
    with _LOCK:
        _LIVE.add(provider)


def _owners_of(session_id):
    from hermes_constants import get_hermes_home
    home = Path(get_hermes_home()).resolve()
    with _LOCK:
        snapshot = list(_LIVE)
    return [p for p in snapshot
            if p._home_root == home and p._active_sid == session_id and p._core is not None]


def on_output(response_text, *, session_id='', **kwargs):
    owners = _owners_of(session_id)
    # With ambiguous ownership neither trace may be revealed.
    if len(owners) != 1:
        return None
    return owners[0].rewrite_output(response_text, session_id=session_id)


def on_end(*, session_id='', **kwargs):
    for owner in _owners_of(session_id):
        owner.drop_turn_trace(session_id)


HOOKS = (('transform_llm_output', on_output), ('on_session_end', on_end))


def register_hooks(ctx) -> None:
    """Register the hooks on a plugin context; contexts without hook support are skipped."""
    register_hook = getattr(ctx, 'register_hook', None)
    if register_hook is None:
        return
    for hook_name, callback in HOOKS:
        register_hook(hook_name, callback)
