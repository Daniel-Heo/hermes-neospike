"""Provider background threads: Hermes's spawn_context_thread when it exists, a plain daemon thread otherwise."""
from __future__ import annotations

import sys
import threading
import types

import pytest

from neospike import threads
from neospike.topics import SaveFollowUp


class SpawnLog:
    """Fake spawn_context_thread: records each request and returns an unstarted marked thread."""

    def __init__(self):
        self.requests: list[dict] = []

    def __call__(self, target, *, name, daemon=True, args=(), kwargs=None):
        self.requests.append({"target": target, "name": name, "daemon": daemon})
        made = threading.Thread(target=target, name=name, daemon=daemon)
        made.spawned_by_hermes = True
        return made


class Idle:
    def __call__(self):
        return None


def _fake_provider_module(monkeypatch, spawn=None) -> None:
    module = types.ModuleType("agent.memory_provider")
    if spawn is not None:
        module.spawn_context_thread = spawn
    monkeypatch.setitem(sys.modules, "agent.memory_provider", module)


def test_uses_spawn_context_thread_when_hermes_provides_it(monkeypatch):
    spawn = SpawnLog()
    _fake_provider_module(monkeypatch, spawn)
    target = Idle()
    made = threads.background_thread(target, "neospike-probe")
    assert getattr(made, "spawned_by_hermes", False)
    assert spawn.requests == [{"target": target, "name": "neospike-probe", "daemon": True}]
    assert not made.is_alive()  # returned unstarted


def test_falls_back_to_a_plain_daemon_thread_without_it(monkeypatch):
    _fake_provider_module(monkeypatch)
    made = threads.background_thread(Idle(), "neospike-probe")
    assert type(made) is threading.Thread and not hasattr(made, "spawned_by_hermes")
    assert made.name == "neospike-probe" and made.daemon and not made.is_alive()


@pytest.mark.parametrize(("owner", "name"), [
    (SaveFollowUp, "neospike-save-follow-up"),
])
def test_provider_workers_start_through_spawn_context_thread(monkeypatch, owner, name):
    spawn = SpawnLog()
    _fake_provider_module(monkeypatch, spawn)
    worker = owner.__new__(owner)
    worker._thread = None
    worker._stop = threading.Event()
    monkeypatch.setattr(worker, "_run", Idle(), raising=False)
    worker.start()
    worker._thread.join(5)
    assert [request["name"] for request in spawn.requests] == [name]
    assert spawn.requests[0]["daemon"] is True
    assert getattr(worker._thread, "spawned_by_hermes", False)
