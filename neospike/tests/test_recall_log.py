from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from neospike.config import NeospikeConfig
from neospike.provider import NeospikeProvider
from neospike.recall_log import (REASON_FUSION_LIMIT, UNTRACKED_REASONS, RecallLog, build_entry,
                                  item_ref, query_hash)
from neospike.tests.test_remember_dedupe import no_remember_tidy  # noqa: F401 (fixture)

# These toy embedders give distinct memories one vector; post-save dedupe is not under test here.
pytestmark = pytest.mark.usefixtures("no_remember_tidy")

QUERY = 'Which secret-lantern codename did we pick for Python indexing?'
SESSION = 'recall-session'


class Embedder:
    dimensions = 3

    def embed_query(self, text, **_kwargs):
        return [1.0, 0.0, 0.0]

    embed_document = embed_query


def _provider(home: Path, **config) -> NeospikeProvider:
    NeospikeConfig(**config).save(home)
    (home / 'project').mkdir(exist_ok=True)
    provider = NeospikeProvider(embedder=Embedder())
    provider.initialize(SESSION, hermes_home=str(home), cwd=str(home / 'project'))
    for text in ('The project uses Python for indexing.', 'Indexing runs nightly on the build host.'):
        assert json.loads(provider.handle_tool_call('neospike_memory_save', {
            'text': text, 'scope': 'global', 'subject': 'indexing', 'durability': 'stable', 'kind': 'fact', 'basis': 'user_said'}))['stored']
    return provider


def _logs(home: Path) -> list[Path]:
    return sorted((home / 'neospike' / 'logs').glob('recall-*.jsonl'))


def test_prefetch_appends_one_line_without_query_text(tmp_path):
    provider = _provider(tmp_path)
    try:
        text = provider.prefetch(QUERY, session_id=SESSION)
    finally:
        provider.shutdown()
    assert 'Python' in text
    [path] = _logs(tmp_path)
    assert path.name == time.strftime('recall-%Y%m.jsonl')
    raw = path.read_bytes()
    lines = raw.decode('utf-8').splitlines()
    assert len(lines) == 1
    for fragment in (QUERY, 'secret-lantern', 'codename'):
        assert fragment.encode('utf-8') not in raw
    entry = json.loads(lines[0])
    assert entry['session'] == SESSION
    assert entry['query_hash'] == query_hash(QUERY) and len(entry['query_hash']) == 16
    assert [item['rank'] for item in entry['injected']] == [1, 2]
    assert {item['lane'] for item in entry['injected']} == {'memory'}
    assert all(isinstance(item['id'], int) and item['score'] > 0 for item in entry['injected'])
    assert entry['dropped'] == [] and entry['unavailable_lanes'] == []
    assert entry['untracked_reasons'] == list(UNTRACKED_REASONS)


def test_unwritable_log_directory_keeps_prefetch_result_and_warns(tmp_path):
    provider = _provider(tmp_path, turn_trace_enabled=True)
    try:
        (tmp_path / 'neospike' / 'logs').write_text('not a directory', encoding='utf-8')
        provider.on_turn_start(1, 'unused')
        blocked = provider.prefetch(QUERY, session_id=SESSION)
        key = provider._trace_key(SESSION)
        failures = provider._turn_log._open_turns[key].failures
        recall_log, provider._recall_sink = provider._recall_sink, None
        unlogged = provider.prefetch(QUERY, session_id=SESSION)
        provider._recall_sink = recall_log
    finally:
        provider.shutdown()
    assert blocked == unlogged and 'Python' in blocked
    assert any('Recall log not written' in line for line in failures)


def test_disabled_setting_writes_nothing(tmp_path):
    provider = _provider(tmp_path, recall_log_enabled=False)
    try:
        assert provider.prefetch(QUERY, session_id=SESSION)
    finally:
        provider.shutdown()
    assert not (tmp_path / 'neospike' / 'logs').exists()


class FixedClock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _month_file(directory: Path, now: float, months_back: int) -> Path:
    local = time.localtime(now)
    index = local.tm_year * 12 + local.tm_mon - 1 - months_back
    path = directory / f'recall-{index // 12:04d}{index % 12 + 1:02d}.jsonl'
    path.write_text('{}\n', encoding='utf-8')
    return path


def test_record_deletes_files_older_than_retention_and_keeps_current(tmp_path):
    now = time.mktime((2026, 11, 15, 12, 0, 0, 0, 0, -1))
    directory = tmp_path / 'logs'
    directory.mkdir()
    old = _month_file(directory, now, 13)
    edge = _month_file(directory, now, 12)
    current = _month_file(directory, now, 0)
    unrelated = directory / 'recall-notes.txt'
    unrelated.write_text('keep', encoding='utf-8')
    log = RecallLog(directory, months=12, clock=FixedClock(now))
    written = log.record(session_id='s', query='q', injected=[], overflow=[], unavailable_lanes=[])
    assert written == current
    assert not old.exists()
    assert edge.exists() and current.exists() and unrelated.exists()
    assert current.read_text(encoding='utf-8').count('\n') == 2


def test_entry_lists_fusion_overflow_with_reason_and_lane_ids():
    injected = [{'id': 4, 'kind': 'memory_item', 'score': 0.9}, {'id': 2, 'kind': 'note', 'score': 0.8}]
    overflow = [{'id': 7, 'kind': 'conversation', 'score': 0.4}]
    entry = build_entry(now=time.time(), session_id='s', query='q', injected=injected,
                        overflow=overflow, unavailable_lanes=['note'])
    assert entry['injected'] == [{'lane': 'memory', 'id': 4, 'score': 0.9, 'rank': 1},
                                 {'lane': 'note', 'id': 2, 'score': 0.8, 'rank': 2}]
    assert entry['dropped'] == [{'lane': 'conversation', 'id': 7, 'score': 0.4, 'rank': 3, 'reason': REASON_FUSION_LIMIT}]
    assert entry['unavailable_lanes'] == ['note']
    assert [item_ref(item) for item in injected + overflow] == ['mem:4', 'note:2', 'conv:7']


def test_config_round_trip_and_validation(tmp_path):
    assert NeospikeConfig.load(tmp_path).recall_log_enabled is True
    assert NeospikeConfig.load(tmp_path).recall_log_months == 12
    NeospikeConfig(recall_log_enabled=False, recall_log_months=3).save(tmp_path)
    loaded = NeospikeConfig.load(tmp_path)
    assert (loaded.recall_log_enabled, loaded.recall_log_months) == (False, 3)
    assert NeospikeConfig.from_values({'recall_log_months': '6'}).recall_log_months == 6
    for bad in (0, -1, True, '1.5', 1201):
        with pytest.raises(ValueError):
            NeospikeConfig.from_values({'recall_log_months': bad})
