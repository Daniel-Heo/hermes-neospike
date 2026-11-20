"""Real daemon launch/ownership behaviour. No model is ever loaded and the user's profile is never touched."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from neospike.tests.support_embed_service import (
    ON_WINDOWS,
    SLEEPER_CODE,
    SleepingPopen,
    cache_dir,
    clean_env,
    hard_kill,
    kill_and_wait,
    lock_down,
    make_private_dir,
    permission_snapshot,
    ping_pid,
    poll_until,
    profile_root,
    share_with_others,
    spawn_python,
    spawn_service,
    stop_daemon_of,
)

from neospike import _wincompat as wc
from neospike import inference
from neospike._wincompat import fcntl
from neospike.inference import INPUT_CHAR_CAP, EmbeddingClient
from neospike.tests.test_wincompat import pid_alive, read_line, wait_output

SHORT_BACKOFF = 1.0  # shrunk from inference._FAILURE_BACKOFF_SECONDS so the cooldown can really elapse
RACE_CLIENT = '''
import json, sys
from neospike.inference import INPUT_CHAR_CAP, EmbeddingClient
client = EmbeddingClient(sys.argv[1])
try:
    client.embed_query('x' * (INPUT_CHAR_CAP + 1))
except RuntimeError as exc:
    assert str(exc) == 'Invalid inference input', str(exc)
print(json.dumps(client.request('ping')))
'''
PENDING_CLIENT = '''
import json, subprocess, sys
from pathlib import Path
import neospike.inference as inference
from neospike.inference import EmbeddingClient
inference._FAILURE_BACKOFF_SECONDS = @BACKOFF@
real_popen = subprocess.Popen
kids = []
def never_ready(argv, **kwargs):
    kid = real_popen([sys.executable, '-c', @SLEEPER@], **kwargs)
    kids.append(kid)
    with (Path(sys.argv[1]) / 'launches').open('a') as out:
        out.write(str(kid.pid) + '\\n')
    return kid
subprocess.Popen = never_ready
try:
    try:
        EmbeddingClient(sys.argv[1], startup_timeout=.2).ensure_daemon()
    except (RuntimeError, TimeoutError) as exc:
        print(json.dumps({'error': str(exc)}), flush=True)
    else:
        raise AssertionError('a sleeping process can never become ready')
    sys.stdin.readline()
finally:
    for kid in kids:
        kid.kill()
        kid.wait(timeout=5)
'''.replace('@BACKOFF@', repr(SHORT_BACKOFF)).replace('@SLEEPER@', repr(SLEEPER_CODE))


@pytest.fixture
def two_homes(tmp_path, monkeypatch):
    with profile_root(tmp_path, monkeypatch) as root:
        homes = [root / 'one', root / 'two']
        yield homes
        for home in homes:
            stop_daemon_of(home)


@pytest.fixture
def never_ready(monkeypatch):
    """Replace the daemon command by a sleeper, recording every launch."""
    stand_in = SleepingPopen(subprocess.Popen)
    monkeypatch.setattr(inference.subprocess, 'Popen', stand_in)
    yield stand_in
    stand_in.reap()


def start_by_demand(home) -> int:
    """Trigger a launch with an over-long input (rejected before any model) and return the daemon pid."""
    client = EmbeddingClient(home)
    with pytest.raises(RuntimeError, match='Invalid inference input'):
        client.embed_query('y' * (INPUT_CHAR_CAP + 1))
    return ping_pid(client)


def daemon_is_down(home) -> bool:
    try:
        EmbeddingClient(home, timeout=.1).request('ping')
    except (OSError, ValueError):
        return True
    return False


class DaemonDown:
    def __init__(self, home):
        self.home = home

    def __call__(self):
        return daemon_is_down(self.home)


def test_two_processes_demanding_at_once_end_up_with_one_daemon(two_homes):
    racers = [spawn_python(RACE_CLIENT, str(two_homes[0])) for _ in range(2)]
    try:
        outputs = [racer.communicate(timeout=15) for racer in racers]
        assert [racer.returncode for racer in racers] == [0, 0], outputs
        pids = {json.loads(out)['pid'] for out, _ in outputs}
        assert len(pids) == 1
        assert ping_pid(EmbeddingClient(two_homes[0])) in pids
    finally:
        for racer in racers:
            kill_and_wait(racer)


def test_building_a_client_and_pinging_create_nothing(two_homes):
    client = EmbeddingClient(two_homes[0])
    with pytest.raises(FileNotFoundError):
        client.request('ping')
    assert not two_homes[0].exists()


def test_dead_daemon_comes_back_only_on_demand_and_other_profile_is_unaffected(two_homes):
    mine, theirs = start_by_demand(two_homes[0]), start_by_demand(two_homes[1])
    assert mine != theirs
    hard_kill(mine)
    assert poll_until(DaemonDown(two_homes[0]), 5)
    time.sleep(.1)
    assert daemon_is_down(two_homes[0])  # nothing resurrects it by itself
    assert ping_pid(EmbeddingClient(two_homes[1])) == theirs
    assert start_by_demand(two_homes[0]) not in {mine, theirs}


def test_unready_daemon_times_out_once_then_backoff_blocks_new_spawns(two_homes, never_ready):
    began = time.monotonic()
    with pytest.raises(TimeoutError):
        EmbeddingClient(two_homes[0], startup_timeout=.2).embed_query('sensitive text')
    assert time.monotonic() - began < 1.5
    for _ in range(3):
        with pytest.raises(RuntimeError, match='recently failed'):
            EmbeddingClient(two_homes[0], startup_timeout=.2).embed_query('sensitive text')
    assert len(never_ready.children) == 1
    folder = cache_dir(two_homes[0])
    for leaf in ('startup.lock', 'owner.lock', 'inference.log'):
        assert (folder / leaf).is_file()
        if not ON_WINDOWS:
            assert (folder / leaf).stat().st_mode & 0o777 == 0o600
    assert 'sensitive text' not in (folder / 'inference.log').read_text()


def test_backoff_is_measured_from_the_failure_not_from_the_spawn(two_homes, never_ready, monkeypatch):
    monkeypatch.setattr(inference, '_FAILURE_BACKOFF_SECONDS', SHORT_BACKOFF)
    patience = SHORT_BACKOFF + .1  # the first attempt itself outlasts the cooldown window
    with pytest.raises(TimeoutError):
        EmbeddingClient(two_homes[0], startup_timeout=patience).embed_query('probe')
    with pytest.raises(RuntimeError, match='recently failed'):
        EmbeddingClient(two_homes[0], startup_timeout=.1).embed_query('probe')
    assert len(never_ready.children) == 1


def test_launch_in_progress_is_not_restarted_by_later_or_concurrent_clients(two_homes):
    home = two_homes[0]
    clients = []

    def begin():
        proc = spawn_python(PENDING_CLIENT, str(home), stdin=subprocess.PIPE)
        clients.append(proc)
        return proc

    try:
        first = begin()
        assert 'deadline' in json.loads(read_line(first.stdout, 5))['error']
        launches = home / 'launches'
        initial = launches.read_text().splitlines()
        assert len(initial) == 1
        time.sleep(SHORT_BACKOFF + .2)  # the real cooldown runs out
        rivals = [begin() for _ in range(4)]
        outcomes = [rival.communicate('\n', timeout=5) for rival in rivals]
        assert all(rival.returncode == 0 for rival in rivals), outcomes
        assert launches.read_text().splitlines() == initial
        assert pid_alive(int(initial[0]))  # the pending launch was neither killed nor replaced
        first.communicate('\n', timeout=5)
        assert first.returncode == 0
        # its death frees the launch ownership: the next demand may start a successor
        successor = begin()
        out, err = successor.communicate('\n', timeout=5)
        assert successor.returncode == 0, (out, err)
        assert len(launches.read_text().splitlines()) == 2
    finally:
        for proc in clients:
            if proc.poll() is None:
                try:
                    proc.communicate('\n', timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            proc.wait(timeout=5)


def test_waiting_for_the_startup_lock_gives_up_on_time(two_homes):
    folder = make_private_dir(two_homes[0])
    lock_file = folder / 'startup.lock'
    lock_file.touch(mode=0o600)
    with lock_file.open('r+') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        began = time.monotonic()
        with pytest.raises(TimeoutError):
            EmbeddingClient(two_homes[0], startup_timeout=.15).embed_query('probe')
        assert time.monotonic() - began < 1
    assert not (folder / 'inference.log').exists()


def test_service_with_foreign_token_is_left_alone_and_missing_token_is_not_recreated(two_homes):
    home = two_homes[0]
    pid = start_by_demand(home)
    folder = cache_dir(home)
    token_file = folder / 'token'
    real_token = token_file.read_text()
    socket_inode = (folder / 'i.sock').stat().st_ino
    try:
        token_file.write_text('forged')
        with pytest.raises(RuntimeError, match='Unauthorized'):
            EmbeddingClient(home).embed_query('probe')
        assert token_file.read_text() == 'forged'
        assert (folder / 'i.sock').stat().st_ino == socket_inode
        token_file.unlink()
        with pytest.raises(PermissionError, match='credentials'):
            start_by_demand(home)
        assert not token_file.exists()
    finally:
        token_file.touch(mode=0o600)
        token_file.write_text(real_token)
    assert ping_pid(EmbeddingClient(home)) == pid


def test_held_owner_lock_without_a_socket_stops_a_second_launch(two_homes):
    folder = make_private_dir(two_homes[0])
    owner_file = folder / 'owner.lock'
    owner_file.touch(mode=0o600)
    with owner_file.open('r+') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(RuntimeError, match='owns profile'):
            EmbeddingClient(two_homes[0]).embed_query('probe')
    assert not (folder / 'inference.log').exists()


def test_loose_cache_dir_and_symlinked_lock_are_refused_without_repair(two_homes):
    folder = cache_dir(two_homes[0])
    folder.mkdir(parents=True, mode=0o755)
    share_with_others(folder)
    snapshot = permission_snapshot(folder)
    with pytest.raises(PermissionError, match='directory'):
        EmbeddingClient(two_homes[0]).embed_query('probe')
    assert permission_snapshot(folder) == snapshot
    lock_down(folder)
    bystander = two_homes[0] / 'bystander'
    bystander.write_text('leave me alone')
    (folder / 'startup.lock').symlink_to(bystander)
    with pytest.raises(OSError):
        EmbeddingClient(two_homes[0]).embed_query('probe')
    assert bystander.read_text() == 'leave me alone'
    assert not (folder / 'inference.log').exists()


@pytest.mark.parametrize('flaw', ['hardlink', 'symlink', 'loose-mode'])
def test_service_start_refuses_a_tampered_token_file_and_spares_its_target(two_homes, flaw):
    folder = make_private_dir(two_homes[0])
    target = two_homes[0] / 'target'
    target.write_text('contents stay put')
    target.chmod(0o600)
    token = folder / 'token'
    if flaw == 'loose-mode':
        share_with_others(target)
        target.rename(token)
        target = token
    elif flaw == 'symlink':
        token.symlink_to(target)
    else:
        os.link(target, token)
    stat_before = target.stat()
    perms_before = permission_snapshot(target)
    service = spawn_service(two_homes[0])
    try:
        wait_output(service, 5)
        assert target.read_text() == 'contents stay put'
        assert target.stat().st_mode == stat_before.st_mode
        assert permission_snapshot(target) == perms_before
        out, err = service.communicate(timeout=5)
        assert service.returncode != 0, (out, err)
    finally:
        kill_and_wait(service)


def test_client_refuses_a_world_readable_token_and_does_not_chmod_it(two_homes):
    start_by_demand(two_homes[0])
    token_file = cache_dir(two_homes[0]) / 'token'
    share_with_others(token_file)
    snapshot = permission_snapshot(token_file)
    try:
        with pytest.raises(PermissionError, match='token'):
            EmbeddingClient(two_homes[0]).embed_query('probe')
        assert permission_snapshot(token_file) == snapshot
    finally:
        lock_down(token_file)


# --- Windows: the launch guard is an inherited, non-shareable handle ----------------------------

WINDOWS_ONLY = pytest.mark.skipif(not ON_WINDOWS, reason='launch.lock is handed over as an inherited no-sharing handle')


def venv_python_exe() -> str:
    exe = Path(sys.prefix) / 'Scripts' / 'python.exe'
    if sys.prefix == sys.base_prefix or not exe.is_file():
        pytest.skip('Needs pytest running inside a Windows venv')
    return str(exe)


def another_process_can_open(path) -> bool:
    probe = 'import sys\nopen(sys.argv[1], "r+b").close()'
    return subprocess.run([sys.executable, '-c', probe, str(path)], capture_output=True, check=False).returncode == 0


class LaunchObserver:
    """Popen wrapper that notes, at spawn time, whether the launch guard is still held."""

    def __init__(self, real_popen, guard):
        self.real_popen, self.guard, self.sightings = real_popen, guard, []

    def __call__(self, argv, **kwargs):
        child = self.real_popen(argv, **kwargs)
        if '--launch-fd' in argv:  # the probe in another_process_can_open also goes through Popen
            held_open = another_process_can_open(self.guard)
            self.sightings.append((argv[0], kwargs['env'].get('__PYVENV_LAUNCHER__'), held_open))
        return child


@WINDOWS_ONLY
def test_windows_guard_is_held_during_start_and_free_after_ready(two_homes, monkeypatch):
    launcher = venv_python_exe()
    monkeypatch.setenv('HERMES_PYTHON', launcher)
    guard = cache_dir(two_homes[0]) / 'launch.lock'
    watcher = LaunchObserver(subprocess.Popen, guard)
    monkeypatch.setattr(inference.subprocess, 'Popen', watcher)
    EmbeddingClient(two_homes[0]).ensure_daemon()
    assert [opened for *_, opened in watcher.sightings] == [False]
    assert another_process_can_open(guard)
    # the daemon is the base interpreter, never the launcher that would pin the handle
    assert watcher.sightings == [(wc.detached_interpreter(launcher, {}), launcher, False)]


@WINDOWS_ONLY
def test_windows_only_the_starting_daemon_holds_the_guard_via_the_venv_launcher(two_homes):
    folder = cache_dir(two_homes[1])
    inference._ensure_owner_dir(folder)
    guard = folder / 'launch.lock'
    handle = wc.open_unshared(guard)
    env = clean_env()
    daemon = subprocess.Popen(
        [wc.detached_interpreter(venv_python_exe(), env), '-m', 'neospike.inference', '--home', str(two_homes[1]),
         *wc.launch_fd_args(handle)], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        **wc.detached_popen_kwargs(handle))
    try:
        os.close(handle)  # the starter lets go; only the booting daemon keeps the guard
        assert not another_process_can_open(guard)
        assert json.loads(read_line(daemon.stdout, 10))['pid'] == daemon.pid
        assert another_process_can_open(guard)
    finally:
        kill_and_wait(daemon)
        daemon.stdout.close()
