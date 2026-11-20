"""Neospike MemoryProvider. Import the Hermes provider only when requested.

The inference daemon and standalone storage/CLI do not require Hermes core.
"""
_LAZY_NAMES = ('NeospikeProvider', 'register')


def __getattr__(name):
    if name in _LAZY_NAMES:
        from . import provider
        return getattr(provider, name)
    raise AttributeError(name)
