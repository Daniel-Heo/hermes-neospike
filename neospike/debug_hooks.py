"""Wire live memory providers into Hermes' public lifecycle hooks.

Memory discovery ignores register_hook, so the hooks are registered after initialization
through a real PluginContext. Only weak references are kept: process-wide hooks must not
keep finished agents or their memory alive. Nothing in Hermes is replaced.
"""
from __future__ import annotations

import threading
import weakref
from pathlib import Path

_LOCK = threading.RLock()
_PROVIDERS_BY_MANAGER = weakref.WeakKeyDictionary()


def _owners_of(live, session_id):
    from hermes_constants import get_hermes_home
    home = Path(get_hermes_home()).resolve()
    with _LOCK:
        snapshot = list(live)
    return [p for p in snapshot
            if p._home_root == home and p._active_sid == session_id and p._core is not None]


def _make_hooks(live):
    def on_output(response_text, *, session_id='', **kwargs):
        owners = _owners_of(live, session_id)
        # With ambiguous ownership neither trace may be revealed.
        if len(owners) != 1:
            return None
        return owners[0].rewrite_output(response_text, session_id=session_id)

    def on_end(*, session_id='', **kwargs):
        for owner in _owners_of(live, session_id):
            owner.drop_turn_trace(session_id)

    return on_output, on_end


def hook_into_hermes(provider) -> None:
    from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager

    manager = get_plugin_manager()
    with _LOCK:
        live = _PROVIDERS_BY_MANAGER.get(manager)
        if live is None:
            live = _PROVIDERS_BY_MANAGER[manager] = weakref.WeakSet()
            on_output, on_end = _make_hooks(live)
            context = PluginContext(PluginManifest(name='neospike', kind='exclusive'), manager)
            context.register_hook('transform_llm_output', on_output)
            context.register_hook('on_session_end', on_end)
        live.add(provider)
