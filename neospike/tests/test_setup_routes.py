"""Dashboard backend: plugin loading, the protected /setup contract, probe sanitising, and the shipped UI bundle."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from neospike.dashboard import plugin_api as api

DASHBOARD_DIR = Path(api.__file__).resolve().parent
BUNDLE = DASHBOARD_DIR / 'dist' / 'index.js'
SOURCE = DASHBOARD_DIR / 'src' / 'index.js'
PROBE_FAIL_TEXT = 'Provider availability check failed; inspect local Dashboard logs'
HEALTHY_STATE = {'dependencies_installed': True, 'text_model_downloaded': True, 'active': True}
STUB_AGENT_PRELUDE = (
    "import importlib.util,sys,types;"
    "agent=types.ModuleType('agent');mp=types.ModuleType('agent.memory_provider');"
    "mp.MemoryProvider=type('MemoryProvider',(),{});agent.memory_provider=mp;"
    "sys.modules['agent']=agent;sys.modules['agent.memory_provider']=mp;"
)


class FakeCompleted:
    def __init__(self, returncode=1, stdout='', stderr=''):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class FakeRun:
    """Stand-in for subprocess.run: answers a canned result, or raises the given failure."""

    def __init__(self, result=None, failure=None):
        self.result, self.failure = result, failure

    def __call__(self, *args, **kwargs):
        if self.failure is not None:
            raise self.failure
        return self.result


class FixedAnswer:
    """Callable returning the same value whatever it is called with."""

    def __init__(self, value):
        self.value = value

    def __call__(self, *args, **kwargs):
        return self.value


class CommandLog:
    """Replaces api._run_logged: remembers each command instead of running it."""

    def __init__(self):
        self.commands: list[list[str]] = []

    def __call__(self, command, job):
        self.commands.append(command)


def await_job(http: TestClient, job_id: str) -> dict:
    for _ in range(100):
        state = http.get('/setup/progress', params={'job_id': job_id}).json()
        if state['state'] in {'complete', 'failed'}:
            return state
        time.sleep(0.01)
    raise AssertionError('setup job never reached a final state')


def hermes_home() -> Path:
    return Path(os.environ['HERMES_HOME'])


def js_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


@pytest.fixture()
def http(tmp_path, monkeypatch) -> TestClient:
    home = tmp_path / 'hermes'
    interpreter = home / 'hermes-agent' / 'venv' / 'bin' / 'python'
    interpreter.parent.mkdir(parents=True)
    interpreter.touch(mode=0o755)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_PYTHON', str(interpreter))
    with api._JOB_LOCK:
        api._JOB_TABLE.clear()
    application = FastAPI()
    application.include_router(api.router)
    return TestClient(application)


# --- loading ------------------------------------------------------------------------------------

def test_plugin_file_executes_as_a_free_standing_module():
    name = 'neo5_probe_dashboard_plugin'
    spec = importlib.util.spec_from_file_location(name, api.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        assert len(module.router.routes) > 0
    finally:
        del sys.modules[name]


def test_installed_copy_loads_from_an_unrelated_working_directory(tmp_path):
    copy = tmp_path / 'plugins' / 'neospike'
    shutil.copytree(DASHBOARD_DIR.parent, copy)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    entry = copy / 'dashboard' / 'plugin_api.py'
    program = (
        STUB_AGENT_PRELUDE
        + f"spec=importlib.util.spec_from_file_location('neo5_installed_probe',{str(entry)!r});"
        + "mod=importlib.util.module_from_spec(spec);sys.modules['neo5_installed_probe']=mod;"
        + "spec.loader.exec_module(mod);assert mod.router.routes"
    )
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    subprocess.run([sys.executable, '-c', program], cwd=elsewhere, env=env, check=True)


# --- status / config ----------------------------------------------------------------------------

def test_status_reports_database_and_disk_but_never_leaks_saved_settings(http, monkeypatch):
    folder = hermes_home() / 'neospike'
    folder.mkdir()
    (folder / 'config.json').write_text(
        json.dumps({'embedding_dimensions': 256, 'token': 'hunter2', 'user_id': 'person-77'}))
    # closing(): an open connection would block the legacy-database migration on Windows
    with closing(sqlite3.connect(folder / 'memory.sqlite')) as db:
        db.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, type TEXT NOT NULL DEFAULT 'memory')")
        db.execute('CREATE TABLE meta_entries(key TEXT PRIMARY KEY, value TEXT)')
        db.execute("INSERT INTO meta_entries VALUES('embedding_dimensions','256')")
        db.commit()
    monkeypatch.setattr(api.shutil, 'which', FixedAnswer(None))
    monkeypatch.setattr(api.subprocess, 'run', FakeRun(FakeCompleted()))
    reply = http.get('/setup/status')
    assert reply.status_code == 200
    payload = reply.json()
    assert payload['database']['healthy'] is True
    assert payload['disk']['free_bytes'] > 0
    assert payload['platform']['system']
    assert 'hunter2' not in reply.text and 'person-77' not in reply.text


def test_saving_config_discards_retired_keys_and_triggers_no_reindex(http):
    target = hermes_home() / 'neospike' / 'config.json'
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({'embedding_dimensions': 512, 'summary_retention_days': 3}), encoding='utf-8')
    reply = http.post('/setup/config', json={
        'embedding_dimensions': 768, 'episodic_retention_days': None,
        'summary_retention_days': 2.5, 'turn_trace_enabled': True})
    assert reply.status_code == 200
    assert reply.json()['reindex_performed'] is False
    stored = json.loads(target.read_text())
    assert 'embedding_dimensions' not in stored  # old clients may still send it; it is fixed in code now
    assert not {'episodic_retention_days', 'summary_retention_days'} & set(stored)
    assert stored['turn_trace_enabled'] is True
    if os.name != 'nt':  # no POSIX mode bits on Windows; the profile ACL protects the file
        assert target.stat().st_mode & 0o777 == 0o600


# --- setup actions ------------------------------------------------------------------------------

def test_model_download_runs_only_the_pinned_repository_files(http, monkeypatch, tmp_path):
    hf_stub = tmp_path / 'hf'
    hf_stub.touch()
    monkeypatch.setattr(api, '_hf_cli_path', FixedAnswer(hf_stub))
    log = CommandLog()
    monkeypatch.setattr(api, '_run_logged', log)
    reply = http.post('/setup/models', json={})
    assert reply.status_code == 202
    assert await_job(http, reply.json()['job_id'])['state'] == 'complete'
    assert {command[2] for command in log.commands} == {api.MODEL_REPO}
    first = log.commands[0]
    assert first[3:5] == list(api.MODEL_DOWNLOADS)
    assert first[first.index('--revision') + 1] == api.MODEL_REVISION


def test_activation_without_confirmation_is_refused(http):
    reply = http.post('/setup/activate', json={'confirm': False})
    assert reply.status_code == 400


def test_dependency_install_uses_the_managed_interpreter_and_pinned_packages(http, monkeypatch):
    log = CommandLog()
    monkeypatch.setattr(api, '_run_logged', log)
    reply = http.post('/setup/dependencies')
    assert reply.status_code == 202
    assert await_job(http, reply.json()['job_id'])['state'] == 'complete'
    command = log.commands[0]
    assert command[:4] == [str(api._cli_python_path()), '-m', 'pip', 'install']
    assert command[-len(api.PIP_PACKAGES):] == api.PIP_PACKAGES
    assert all(isinstance(part, str) for part in command)  # an argv list: no shell string


@pytest.mark.parametrize(('method', 'route', 'body'), [
    ('post', '/download', {'mode': 'text'}),
    ('put', '/config', {'embedding_dimensions': 256}),
    ('post', '/backfill', None),
    ('post', '/activate', None),
    ('post', '/setup/auth', {'token': 'hf_example-token'}),
])
def test_retired_mutating_routes_answer_404(http, method, route, body):
    send = getattr(http, method)
    reply = send(route) if body is None else send(route, json=body)
    assert reply.status_code == 404


def test_finished_jobs_are_pruned_to_twenty(http):
    with api._JOB_LOCK:
        for number in range(20):
            api._JOB_TABLE[str(number)] = {'id': str(number), 'kind': 'old', 'state': 'complete'}
    started = api._start_job('fresh', FixedAnswer(None))
    await_job(http, started['job_id'])
    with api._JOB_LOCK:
        assert len(api._JOB_TABLE) <= 20
        assert started['job_id'] in api._JOB_TABLE
        assert '0' not in api._JOB_TABLE  # the oldest finished entry went first


# --- health / probe sanitising ------------------------------------------------------------------

def test_health_check_hides_what_the_probe_process_printed(http, monkeypatch):
    leak = '/Users/private/path token=hf_should_never_escape'
    monkeypatch.setattr(api, '_collect_status', FixedAnswer(dict(HEALTHY_STATE)))
    monkeypatch.setattr(api.subprocess, 'run', FakeRun(FakeCompleted(1, leak, leak)))
    reply = http.post('/health')
    assert reply.status_code == 200
    assert reply.json()['healthy'] is False
    assert leak not in reply.text
    assert reply.json()['problems'] == [PROBE_FAIL_TEXT]


@pytest.mark.parametrize('failure', [subprocess.TimeoutExpired('python', 90), OSError('private path')],
                         ids=['timeout', 'oserror'])
def test_health_check_survives_probe_launch_errors_without_echoing_them(http, monkeypatch, failure):
    monkeypatch.setattr(api, '_collect_status', FixedAnswer(dict(HEALTHY_STATE)))
    monkeypatch.setattr(api.subprocess, 'run', FakeRun(failure=failure))
    reply = http.post('/health')
    assert reply.status_code == 200
    assert reply.json()['healthy'] is False
    assert 'private path' not in reply.text


@pytest.mark.parametrize('failure', [subprocess.TimeoutExpired('probe', 20), OSError('private probe path')],
                         ids=['timeout', 'oserror'])
def test_status_and_health_degrade_quietly_when_every_probe_fails(http, monkeypatch, tmp_path, failure):
    hf_stub = tmp_path / 'hf'
    hf_stub.touch()
    monkeypatch.setattr(api, '_hf_cli_path', FixedAnswer(hf_stub))
    monkeypatch.setattr(api.shutil, 'which', FixedAnswer('/managed/hermes'))
    __import__('platform').system()  # Windows shells out for this; fill the cache before run() starts failing
    monkeypatch.setattr(api.subprocess, 'run', FakeRun(failure=failure))
    status = http.get('/setup/status')
    health = http.post('/health')
    assert (status.status_code, health.status_code) == (200, 200)
    assert 'hf' not in status.json()
    assert health.json()['healthy'] is False
    assert 'private probe path' not in status.text + health.text


# --- shipped UI bundle --------------------------------------------------------------------------

def test_tab_bar_lists_the_expected_tabs_and_dropped_copy_is_gone():
    bundle = js_text(BUNDLE)
    start = bundle.index('const TAB_ITEMS = [')
    block = bundle[start:bundle.index('];', start)]
    assert re.findall(r'label: "tab\.([A-Za-z]+)"', block) == [
        'map', 'notes', 'docs', 'memory', 'preferences', 'conversations', 'config', 'setup']  # no Inbox tab; Docs right after Notes; Preferences after Memory
    for removed in ('"config.copy"', '"config.budget.copy"', '토픽 요약', 'inbox 검토',
                    '"config.none"', 'Background jobs call no LLM'):
        assert removed not in bundle, removed


def test_bundle_talks_to_the_setup_routes_and_drops_retired_features():
    bundle = js_text(BUNDLE)
    for route in ('/setup/status', '/setup/progress', '/setup/dependencies', '/setup/models',
                  '/setup/config', '/setup/activate'):
        assert route in bundle, route
    retired = ('/setup/backfill', 'Apply accepted memories', 'Sign in from a terminal', 'hf auth login',
               'terms_accepted', 'acceptedTerms', 'Gemma', 'embedding_dimensions', '"setup.dimensions"',
               '/setup/auth', 'hfToken', 'setup.signIn', 'setup.tokenLabel',
               'huggingface.co/settings/tokens', 'visual_enabled', 'SHOW_VISUAL_SETUP',
               'neospike-retention', 'retention_days', 'ChoiceButton')
    for gone in retired:
        assert gone not in bundle, gone
    for kept in ('turn_trace_enabled: debugOn', 'type: "radio"', 'name: "neospike-model-mode"'):
        assert kept in bundle, kept


def test_built_bundle_is_identical_to_its_source():
    assert js_text(BUNDLE) == js_text(SOURCE)


def test_manifest_points_at_assets_that_register_the_page():
    manifest = json.loads(js_text(DASHBOARD_DIR / 'manifest.json'))
    bundle = js_text(BUNDLE)
    assert manifest['name'] == 'neospike'
    assert manifest['tab']['path'] == '/neospike'
    assert manifest['entry'] == 'dist/index.js'
    assert manifest['css'] == 'dist/style.css'
    for key in ('entry', 'css'):
        assert (DASHBOARD_DIR / manifest[key]).is_file(), key
    assert f'register("{manifest["name"]}", Page)' in bundle
    assert f'const API = "/api/plugins/{manifest["name"]}"' in bundle
