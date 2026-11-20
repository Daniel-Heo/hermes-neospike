from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path

import pytest

from neospike import eval_recall
from neospike.service import NeospikeService
from neospike.store import MemoryStore

NAMESPACE = 'default:local'
AXES = ('python', 'rust', 'deploy', 'editor', 'database', 'weather', 'other')
PROJECT = 'D:\\eval-project'


class AxisEmbedder:
    """One axis per keyword: cosine is 1 for the same topic, 0 for unrelated ones."""
    dimensions = len(AXES)

    def embed_query(self, text, **_kwargs):
        folded = text.casefold()
        vector = [1.0 if axis in folded else 0.0 for axis in AXES[:-1]]
        vector.append(0.0 if any(vector) else 1.0)
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    embed_document = embed_query

    def take_warnings(self):
        return []


def _memory(store, embedder, text, *, importance=0.5, project=''):
    scope = 'project' if project else 'global'
    memory_id = store.insert(NAMESPACE, text, embedder.embed_document(text), source='test', kind='memory_item',
                             importance=importance, project=project,
                             metadata={'scope': scope, 'subject': 'eval', 'durability': 'stable', 'session_id': ''})
    assert memory_id is not None
    return f'mem:{memory_id}'


def _build_home(home: Path) -> dict[str, str]:
    embedder = AxisEmbedder()
    store = MemoryStore(home / 'neospike' / 'memory.db', dimensions=embedder.dimensions)
    try:
        ids = {
            'python': _memory(store, embedder, 'The project language is Python 3.12.'),
            'deploy_old': _memory(store, embedder, 'Deploy target is server alpha.', importance=0.1),
            'deploy_new': _memory(store, embedder, 'Deploy target changed to server beta.', importance=0.9),
            'editor': _memory(store, embedder, 'Editor theme is solarized dark.', project=PROJECT),
            'database': _memory(store, embedder, 'Database engine is postgres.'),
            'weather': _memory(store, embedder, 'Weather station feed is offline.'),
        }
        service = NeospikeService(store=store, embedder=embedder, namespace=NAMESPACE)
        note = service.save_note('Rust guide', 'Use cargo for every Rust build.', author='human')
        ids['rust_note'] = f"note:{note['id']}"
    finally:
        store.close()
    return ids


def _questions(ids: dict[str, str]) -> list[dict]:
    return [
        {'id': 'q01', 'split': 'test', 'type': 'normal', 'question': 'Which python version?', 'gold': [ids['python']]},
        {'id': 'q02', 'split': 'test', 'type': 'normal', 'question': 'rust or python build?',
         'gold': [ids['rust_note'], ids['database']]},
        {'id': 'q03', 'split': 'test', 'type': 'changed', 'question': 'Where is the deploy target?',
         'gold': [ids['deploy_new']], 'stale': [ids['deploy_old']]},
        {'id': 'q04', 'split': 'dev', 'type': 'changed', 'question': 'editor theme?', 'gold': [ids['editor']],
         'stale': [ids['database']], 'project': PROJECT},
        {'id': 'q05', 'split': 'test', 'type': 'no_answer', 'question': 'weather tomorrow?', 'gold': []},
    ]


def _write_set(path: Path, questions: list[dict]) -> Path:
    path.write_text(''.join(json.dumps(item) + '\n' for item in questions), encoding='utf-8-sig')
    return path


def _db_state(home: Path) -> tuple[str, list[tuple]]:
    path = home / 'neospike' / 'memory.db'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    connection = sqlite3.connect(f'{path.resolve().as_uri()}?mode=ro', uri=True)
    try:
        rows = connection.execute("SELECT id,access_count,accessed_at FROM items WHERE type='memory' ORDER BY id").fetchall()
    finally:
        connection.close()
    return digest, rows


@pytest.fixture
def eval_home(tmp_path):
    home = tmp_path / 'home'
    ids = _build_home(home)
    question_set = _write_set(tmp_path / 'recall_set.jsonl', _questions(ids))
    return home, ids, question_set


def test_metrics_match_hand_calculation(eval_home, tmp_path):
    home, ids, question_set = eval_home
    questions = eval_recall.load_question_set(question_set)
    output = eval_recall.evaluate(home, questions, embedder=AxisEmbedder(), work_directory=tmp_path)
    test, dev = output['splits']['test'], output['splits']['dev']
    per = {**test['per_question'], **dev['per_question']}
    # q01: only the Python memory; q02: memory wins the RRF tie, then the note.
    assert per['q01']['results'] == [ids['python']]
    assert per['q02']['results'] == [ids['python'], ids['rust_note']]
    assert per['q03']['results'] == [ids['deploy_new'], ids['deploy_old']]
    assert per['q04']['results'] == [ids['editor']]
    assert per['q05']['results'] == [ids['weather']]
    assert (per['q02']['recall_at_8'], per['q02']['reciprocal_rank']) == (0.5, 0.5)
    assert per['q03']['stale_hits'] == [ids['deploy_old']] and per['q04']['stale_hits'] == []
    assert test['count'] == 4 and dev['count'] == 1
    assert test['recall_at_8'] == pytest.approx((1 + 0.5 + 1) / 3)
    assert test['mrr'] == pytest.approx((1 + 0.5 + 1) / 3)
    assert (test['stale_injected'], test['false_injection']) == (1, 1)
    assert (dev['recall_at_8'], dev['mrr'], dev['stale_injected'], dev['false_injection']) == (1.0, 1.0, 0, 0)
    only_dev = eval_recall.evaluate(home, questions, embedder=AxisEmbedder(), split='dev', work_directory=tmp_path)
    assert list(only_dev['splits']) == ['dev']


def test_scoring_leaves_database_bytes_and_access_counts_unchanged(eval_home, tmp_path):
    home, _ids, question_set = eval_home
    before = _db_state(home)
    assert before[1] and all(count == 0 for _id, count, _at in before[1])
    eval_recall.evaluate(home, eval_recall.load_question_set(question_set), embedder=AxisEmbedder(),
                         work_directory=tmp_path)
    assert _db_state(home) == before
    assert not (home / 'neospike' / 'logs').exists()


def _scored(home, question_set, tmp_path, baseline=None) -> dict:
    """evaluate + compare_to_baseline (`eval test` gates on the baseline)."""
    output = eval_recall.evaluate(home, eval_recall.load_question_set(question_set), embedder=AxisEmbedder(),
                                  work_directory=tmp_path)
    if baseline is not None:
        output.update(eval_recall.compare_to_baseline(output, baseline))
    return output


def test_baseline_regression_sets_flag(eval_home, tmp_path):
    home, _ids, question_set = eval_home
    first = _scored(home, question_set, tmp_path)
    assert 'regressed' not in first
    same = _scored(home, question_set, tmp_path, baseline=first)
    assert (same['regressed'], same['reasons']) == (False, [])
    # Deliberately worse: a second weather memory leaks into the no_answer question.
    embedder = AxisEmbedder()
    store = MemoryStore(home / 'neospike' / 'memory.db', dimensions=embedder.dimensions)
    try:
        _memory(store, embedder, 'Weather radar is down for maintenance.')
    finally:
        store.close()
    worse = _scored(home, question_set, tmp_path, baseline=first)
    assert (worse['regressed'], worse['reasons']) == (True, ['false_injection'])
    assert worse['splits']['test']['false_injection'] == 2


def test_compare_flags_each_regression_kind():
    base = {'splits': {'test': {'recall_at_8': 0.8, 'mrr': 0.7, 'false_injection': 1, 'stale_injected': 1}}}
    better = {'splits': {'test': {'recall_at_8': 0.9, 'mrr': 0.1, 'false_injection': 0, 'stale_injected': 1}}}
    worse = {'splits': {'test': {'recall_at_8': 0.7, 'mrr': 0.9, 'false_injection': 2, 'stale_injected': 2}}}
    assert eval_recall.compare_to_baseline(better, base)['regressed'] is False
    assert eval_recall.compare_to_baseline(worse, base)['reasons'] == ['recall_at_8', 'false_injection', 'stale_injected']
    with pytest.raises(ValueError):
        eval_recall.compare_to_baseline({'splits': {'dev': {}}}, base)


@pytest.mark.parametrize('bad', [
    {'id': 'x', 'split': 'dev', 'type': 'normal', 'question': 'q', 'gold': []},
    {'id': 'x', 'split': 'dev', 'type': 'no_answer', 'question': 'q', 'gold': ['mem:1']},
    {'id': 'x', 'split': 'dev', 'type': 'normal', 'question': 'q', 'gold': ['mem:1'], 'stale': ['mem:2']},
    {'id': 'x', 'split': 'prod', 'type': 'normal', 'question': 'q', 'gold': ['mem:1']},
    {'id': 'x', 'split': 'dev', 'type': 'normal', 'question': 'q', 'gold': ['mem:1'], 'extra': 1},
])
def test_invalid_question_lines_are_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        eval_recall.load_question_set(_write_set(tmp_path / 'bad.jsonl', [bad]))


class _DuplicateChunkService:
    """search_notes stub: note 1 has two chunks in the hit list (score-descending, as notes.search)."""

    def search_notes(self, _query, *, limit=5):
        hits = [{'id': 1, 'kind': 'note', 'chunk_id': 10, 'score': 0.9}, {'id': 2, 'kind': 'note', 'chunk_id': 20, 'score': 0.8},
                {'id': 1, 'kind': 'note', 'chunk_id': 11, 'score': 0.7}]
        return hits[:limit]


def test_note_lane_keeps_one_slot_per_note_with_best_chunk():
    """Chunks of one note share one result slot; the best-scored chunk represents it."""
    from neospike.provider import NeospikeProvider
    provider = NeospikeProvider(embedder=AxisEmbedder())
    provider._core = _DuplicateChunkService()
    hits = provider._note_lane('rust', limit=8)
    assert [(hit['id'], hit['chunk_id']) for hit in hits] == [(1, 10), (2, 20)]


def test_body_rule_id_rule_and_block():
    """A gold memory whose body (whitespace normalised) is inside a result text is hit, even as a note
    chunk; the id rule is reported beside it; the injected block is scored with the body rule."""
    question = eval_recall.Question('q', 'dev', 'changed', 'deploy?', ('mem:9', 'mem:4'), stale=('mem:3',))
    bodies = {'mem:9': 'deploy on friday', 'mem:4': 'roll back with the tag'}
    refs = ['mem:5', 'note:1', 'mem:4']
    texts = ['unrelated', 'Rules:\n deploy   on friday\n\nmore', 'roll back with the tag']
    score = eval_recall.score_question(question, refs, texts, bodies, block='- [note:1] deploy on friday')
    assert (score.recall, score.reciprocal_rank) == (1.0, 0.5)
    assert (score.id_recall, score.id_reciprocal_rank) == (0.5, 1 / 3)
    assert score.block_recall == 0.5 and score.stale_hits == []
    missing_body = eval_recall.score_question(question, refs, texts, {'mem:9': 'deploy on friday'})
    assert missing_body.recall == 1.0  # mem:4 has no body: judged by id (hit at rank 3)
    plain = eval_recall.score_question(question, ['mem:5', 'note:1', 'mem:7'])
    assert (plain.recall, plain.reciprocal_rank) == (0.0, 0.0)
    nothing = eval_recall.score_question(eval_recall.Question('n', 'dev', 'no_answer', 'x?', ()), ['mem:1'], ['a'],
                                         {}, block='<neospike_recall>\nx\n- [id:1 memory_item a] a\n</neospike_recall>')
    assert (nothing.false_injection, nothing.false_block_lines) == (1, 1)
