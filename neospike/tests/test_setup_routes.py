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
    """`hf download` runs as the huggingface_hub module through hermes_launch.py on the runtime python."""
    log = CommandLog()
    monkeypatch.setattr(api, '_run_logged', log)
    reply = http.post('/setup/models', json={})
    assert reply.status_code == 202
    assert await_job(http, reply.json()['job_id'])['state'] == 'complete'
    assert {command[6] for command in log.commands} == {api.MODEL_REPO}
    first = log.commands[0]
    assert first[0] == os.environ['HERMES_PYTHON'] and first[1] == '-I'
    assert Path(first[2]).name == 'hermes_launch.py'
    assert first[4:6] == [api.HF_CLI_MODULE, 'download'] and first[7:9] == list(api.MODEL_DOWNLOADS)
    assert first[first.index('--revision') + 1] == api.MODEL_REVISION


def test_activation_without_confirmation_is_refused(http):
    reply = http.post('/setup/activate', json={'confirm': False})
    assert reply.status_code == 400


def test_dashboard_never_installs_packages_itself(http, monkeypatch):
    # Hermes's package manager installs plugin.yaml python_dependencies; the dashboard has no pip route.
    log = CommandLog()
    monkeypatch.setattr(api, '_run_logged', log)
    assert http.post('/setup/dependencies').status_code == 404
    assert log.commands == []
    assert not hasattr(api, 'PIP_PACKAGES')
    assert '"pip"' not in Path(api.__file__).read_text(encoding='utf-8-sig')


class ArgvLog:
    """Replaces api._try_run: remembers each argv and answers success."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, argv, timeout, **options):
        self.calls.append(argv)
        return FakeCompleted(returncode=0)


def test_dependency_check_imports_exactly_the_declared_python_dependencies(http, monkeypatch):
    from neospike.tests.test_plugin_dependencies import DISTRIBUTION_OF, PLUGIN_YAML, declared_distributions
    calls = ArgvLog()
    monkeypatch.setattr(api, '_try_run', calls)
    monkeypatch.setattr(api.shutil, 'which', FixedAnswer(None))
    assert http.get('/setup/status').json()['dependencies_installed'] is True
    # python -I hermes_launch.py checkout neospike.runtime_check imports <module>...
    probe = next(argv for argv in calls.calls if argv[4:6] == ['neospike.runtime_check', 'imports'])
    imported = set(probe[6:])
    declared = declared_distributions(PLUGIN_YAML.read_text(encoding='utf-8-sig'))
    assert {DISTRIBUTION_OF[name] for name in imported} == declared


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


def test_health_probe_runs_the_provider_check_module(http, monkeypatch):
    """The availability probe is a module of the package started through hermes_launch.py, not a code string."""
    calls = ArgvLog()
    monkeypatch.setattr(api, '_collect_status', FixedAnswer(dict(HEALTHY_STATE)))
    monkeypatch.setattr(api, '_try_run', calls)
    assert http.post('/health').json()['healthy'] is True
    assert [argv[4:] for argv in calls.calls] == [['neospike.runtime_check', 'provider']]
    assert Path(calls.calls[0][2]).name == 'hermes_launch.py'


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
    for route in ('/setup/status', '/setup/progress', '/setup/models',
                  '/setup/config', '/setup/activate', 'hermes gateway restart'):
        assert route in bundle, route
    retired = ('/setup/dependencies', 'setup.act.installDeps', 'setup.depsInstalled', '/setup/backfill', 'Apply accepted memories', 'Sign in from a terminal', 'hf auth login',
               'terms_accepted', 'acceptedTerms', 'Gemma', 'embedding_dimensions', '"setup.dimensions"',
               '/setup/auth', 'hfToken', 'setup.signIn', 'setup.tokenLabel',
               'huggingface.co/settings/tokens', 'visual_enabled', 'SHOW_VISUAL_SETUP',
               'neospike-retention', 'retention_days', 'ChoiceButton')
    for gone in retired:
        assert gone not in bundle, gone
    for kept in ('turn_trace_enabled: next', 'type: "radio"', 'name: "neospike-model-mode"'):
        assert kept in bundle, kept


def test_diagnostics_setting_lives_in_config_tab_not_setup():
    bundle = js_text(BUNDLE)
    setup = bundle[bundle.index('function InstallView('):bundle.index('function SettingsView(')]
    config = bundle[bundle.index('function SettingsView('):bundle.index('function enabledSavedText(')]
    for gone in ('turn_trace_enabled', 'setup.debug', 'debugOn', 'commitSettings', 'setup.step3'):
        assert gone not in setup, gone
    assert 'h(PrefModelView, null), h(DiagnosticsView, null));' in config  # last Config card
    card = bundle[bundle.index('function DiagnosticsView('):bundle.index('const EXTRACT_LANGS')]
    for kept in ('h(Badge, null, "4")', 't("config.diagnostics.title")', 't("setup.debug")', 't("setup.debugCopy")', '"/setup/config"'):
        assert kept in card, kept
    for retired in ('"setup.step3"', '"setup.act.saveConfig"'):
        assert retired not in bundle, retired


def test_setup_tab_keeps_model_apply_and_switch_cards():
    bundle = js_text(BUNDLE)
    setup = bundle[bundle.index('function InstallView('):bundle.index('function SettingsView(')]
    assert re.findall(r'h\(Badge, null, "(\d)"\), h\(CardTitle, null, t\("([\w.]+)"\)\)', setup) == [
        ('1', 'setup.step2'), ('2', 'setup.step5')]
    assert setup.count('h(EnabledSwitchView, null)') == 1  # the on/off card follows as step 3
    config = bundle[bundle.index('function SettingsView('):bundle.index('function enabledSavedText(')]
    assert 'EnabledSwitchView' not in config
    switch = bundle[bundle.index('function EnabledSwitchView('):bundle.index('function DiagnosticsView(')]
    assert 'h(Badge, null, "3"), h(CardTitle, null, t("config.enabled.title"))' in switch
    assert 't("setup.chk.deps")' in setup  # status row stays
    for retired in ('"setup.step1"', '"setup.step1Copy"', '"setup.depsEnable"', '"setup.depsSetup"',
                    'PLUGIN_ENABLE_COMMAND', 'MEMORY_SETUP_COMMAND', 'hermes plugins enable neospike', 'hermes memory setup'):
        assert retired not in bundle, retired
    assert '"setup.step5": "Apply to memory handling and verify"' in bundle
    assert '"setup.step5": "메모리 처리에 적용과 점검"' in bundle


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
