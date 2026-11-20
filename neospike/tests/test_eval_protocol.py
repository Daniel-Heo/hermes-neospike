"""Search evaluation: scoring core (hand-calculated), fixture regression, seal/test/ledger, real adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from neospike import eval_recall
from neospike.eval_recall import Question
from neospike.evaluation import command, core, lanes, protocol
from neospike.evaluation.adapters import ConceptEmbedder, build_corpus, read_jsonl

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'eval'
CORPUS, QUESTIONS, LEXICON = FIXTURE / 'corpus.jsonl', FIXTURE / 'questions.jsonl', FIXTURE / 'lexicon.json'
SOURCE = ['--corpus', str(CORPUS)]


def _cli(capsys, *arguments) -> tuple[int, dict]:
    code = command.main(list(arguments))
    return code, json.loads(capsys.readouterr().out)


def _q(qid='q', kind='normal', gold=('mem:1',), stale=(), lang='', no_answer=''):
    return Question(qid, 'dev', kind, '?', tuple(gold), tuple(stale), '', lang, no_answer)


# -- core: hand calculations ----------------------------------------------------------------------------------

def test_judge_hit_rank_irrelevant_stale_and_false_positive():
    question = _q(kind='changed', gold=('mem:9', 'note:2'), stale=('mem:3',))
    judged = core.judge(question, ['mem:3', 'mem:5', 'note:2'], ['old', 'other', 'x'], {})
    assert (judged.hit, judged.reciprocal_rank, judged.irrelevant, judged.stale) == (1, 1 / 3, 2, ('mem:3',))
    # Body rule: the gold body inside a chunk text counts at that rank.
    by_body = core.judge(_q(gold=('mem:9',)), ['note:7', 'mem:9'], ['a  deploy on\nfriday', 'z'],
                         {'mem:9': 'deploy on friday'})
    assert (by_body.hit, by_body.reciprocal_rank, by_body.irrelevant) == (1, 1.0, 1)
    miss = core.judge(_q(), ['mem:2'], ['z'], {})
    assert (miss.hit, miss.reciprocal_rank, miss.irrelevant) == (0, 0.0, 1)
    empty = core.judge(_q(kind='no_answer', gold=()), [], [], {})
    leaked = core.judge(_q(kind='no_answer', gold=()), ['mem:4'], ['t'], {})
    assert (empty.false_positive, leaked.false_positive, leaked.hit) == (False, True, None)


def test_summary_and_breakdown_by_language_type_and_category():
    items = [core.judge(_q('a', lang='ko'), ['mem:1'], ['x']),
             core.judge(_q('b', lang='en'), ['mem:2', 'mem:1'], ['y', 'x']),
             core.judge(_q('c', kind='no_answer', gold=(), lang='ko', no_answer='near'), ['mem:5'], ['z']),
             core.judge(_q('d', kind='no_answer', gold=(), lang='en', no_answer='far'), [], [])]
    summary = core.summarize(items)
    assert (summary['recall'], summary['mrr'], summary['fpr'], summary['irrelevant']) == (1.0, 0.75, 0.5, 0.5)
    groups = core.breakdown(items)
    assert groups['lang']['ko']['fpr'] == 1.0 and groups['lang']['en']['mrr'] == 0.5
    assert groups['type']['no_answer:near']['fpr'] == 1.0 and groups['type']['no_answer:far']['fpr'] == 0.0
    assert groups['category']['none']['count'] == 4


def test_paired_bootstrap_and_gate():
    same = core.paired_bootstrap([1, 0, 1, 1], [1, 0, 1, 1])
    assert (same['diff'], same['low'], same['high']) == (0.0, 0.0, 0.0)
    worse = core.paired_bootstrap([0] * 20, [1] * 20)
    assert worse['diff'] == -1.0 and worse['high'] == -1.0
    mixed = core.paired_bootstrap([1, 0] * 10, [1] * 20)
    assert mixed['low'] < mixed['diff'] == -0.5 < mixed['high'] < 0
    assert core.paired_bootstrap([1, 0] * 10, [1] * 20) == mixed  # seeded: reproducible
    base = {f'q{i}': {'hit': 1, 'false_positive': None, 'stale': []} for i in range(20)}
    one_drop = {**base, 'q0': {'hit': 0, 'false_positive': None, 'stale': []}}
    verdict = core.gate(one_drop, base)
    assert verdict['regressed'] is False and verdict['recall_diff']['high'] == 0.0  # one of 20: noise, not a gate
    stale = {**base, 'q1': {'hit': 1, 'false_positive': None, 'stale': ['mem:3']}}
    assert core.gate(stale, base)['reasons'] == ['stale_injected']
    half = {qid: {'hit': index % 2, 'false_positive': None, 'stale': []} for index, qid in enumerate(base)}
    assert core.gate(half, base)['reasons'] == ['recall_ci_below_zero']


def test_threshold_rule_pareto_and_ties():
    rows = core.sweep_rows({
        0.50: {'recall': 0.95, 'fpr': 0.60, 'irrelevant': 2.0, 'mrr': 0.9},
        0.60: {'recall': 0.93, 'fpr': 0.20, 'irrelevant': 1.0, 'mrr': 0.9},
        0.65: {'recall': 0.91, 'fpr': 0.20, 'irrelevant': 1.0, 'mrr': 0.9},  # same as 0.60, lower recall
        0.70: {'recall': 0.85, 'fpr': 0.00, 'irrelevant': 0.5, 'mrr': 0.8},  # outside best - 0.05
    })
    assert [row.pareto for row in rows] == [True, True, False, True]
    chosen = core.select_threshold(rows)
    assert chosen['candidate'] == 0.65 and chosen['kept'] == [0.5, 0.6, 0.65]  # tie on (fpr, irrelevant): higher
    assert core.select_threshold(rows, delta=0.10)['candidate'] == 0.70
    assert core.targets({'recall': 0.8, 'fpr': 0.11})['fpr_met'] is False


def test_grid_lanes_and_keys():
    grid = lanes.default_grid()
    assert (len(grid), grid[0], grid[11], grid[-1]) == (26, 0.5, 0.61, 0.75)
    assert lanes.parse_grid('0.6:0.62:0.005') == [0.6, 0.605, 0.61, 0.615, 0.62]
    assert [lanes.threshold_key(value) for value in (None, 0.5, 0.61, 0.605)] == ['default', '0.5', '0.61', '0.605']
    assert lanes.parse_lanes('note,prefetch,note') == ('note', 'prefetch')
    for bad in ('0.7:0.6:0.01', 'x'):
        with pytest.raises(ValueError):
            lanes.parse_grid(bad)
    with pytest.raises(ValueError):
        lanes.parse_lanes('prefetch,web')


@pytest.mark.parametrize('extra,ok', [
    ({'lang': 'ko', 'category': 'cross_language'}, True),
    ({'lang': 'ru'}, False),
    ({'no_answer_kind': 'near'}, False),  # only on no_answer questions
])
def test_optional_question_fields(extra, ok):
    raw = {'id': 'x', 'split': 'dev', 'type': 'normal', 'question': 'q', 'gold': ['mem:1'], **extra}
    if ok:
        assert eval_recall.parse_question(raw, 1).category == 'cross_language'
    else:
        with pytest.raises(ValueError):
            eval_recall.parse_question(raw, 1)


def test_concept_embedder_cosines():
    """Fixture geometry the regression numbers below rest on (background b = 0.9)."""
    embedder = ConceptEmbedder.load(LEXICON)

    def cosine(left, right):
        return sum(a * b for a, b in zip(embedder.embed_query(left), embedder.embed_query(right)))
    assert cosine('편집기 테마', 'editor theme') == pytest.approx(1.0)  # cross-language
    assert cosine('server volcano', 'deploy server') == pytest.approx(1.81 / 2.81)  # one shared concept: 0.644
    assert cosine('meeting', 'weekly meeting schedule') == pytest.approx((1.81 / 3.81) ** 0.5)  # 0.689
    assert cosine('nothing here', 'deploy server') == pytest.approx((0.81 / 2.81) ** 0.5)  # background floor 0.537


# -- fixture regression ---------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def fixture_sweep(tmp_path_factory):
    out = tmp_path_factory.mktemp('sweep')
    code = command.main(['sweep', *SOURCE, '--set', str(QUESTIONS), '--out', str(out)])
    directory = next(out.iterdir())
    return code, json.loads((directory / 'report.json').read_text(encoding='utf-8-sig')), directory


def test_fixture_shape():
    rows = read_jsonl(QUESTIONS)
    assert 40 <= len(rows) <= 60
    assert sum(row['type'] == 'no_answer' for row in rows) / len(rows) >= 0.25
    assert {row['lang'] for row in rows} == {'ko', 'en', 'mixed'}
    assert {row.get('no_answer_kind') for row in rows if row['type'] == 'no_answer'} == {'near', 'far'}
    assert any(row['type'] == 'changed' for row in rows) and any(row.get('category') == 'cross_language'
                                                                 for row in rows)


def test_fixture_sweep_regression(fixture_sweep):
    code, report, _directory = fixture_sweep
    assert code == 0 and report['source'] == 'synthetic'
    assert {lane: entry['status'] for lane, entry in report['verification'].items()} == {
        'prefetch': 'ok', 'memory': 'ok', 'note': 'ok'}
    points = report['lanes']['prefetch']['points']
    assert len(points) == 26 and report['lanes']['prefetch']['k'] == 8
    at = {key: points[key]['dev'] for key in ('0.61', '0.68', '0.69', '0.75')}
    # 0.61: d17 (vocabulary mismatch, cosine 0.537) is the only miss; near no-answers leak (0.644, 0.7375).
    assert (at['0.61']['recall'], at['0.61']['fpr'], at['0.61']['stale_injected']) == (0.95, 0.5, 1)
    assert at['0.61']['by']['type']['no_answer:near']['fpr'] == 1.0
    assert at['0.61']['by']['type']['no_answer:far']['fpr'] == 0.0
    assert at['0.68']['recall'] == 0.95 and at['0.69']['recall'] == 0.9  # d16 cosine 0.689
    assert (at['0.75']['fpr'], at['0.75']['recall']) == (0.0, 0.9)
    assert set(at['0.61']['by']['lang']) == {'ko', 'en', 'mixed'}
    assert at['0.61']['by']['category']['cross_language']['recall'] == 1.0
    selection = report['selection']['prefetch']
    assert (selection['candidate'], selection['best_recall']) == (0.75, 0.95)
    assert '0.75' in selection['pareto'] and '0.61' not in selection['pareto']


def test_fixture_sweep_writes_three_outputs(fixture_sweep):
    _code, _report, directory = fixture_sweep
    assert sorted(path.name for path in directory.iterdir()) == ['per_question.jsonl', 'report.json', 'report.md']
    assert '| 0.61 | dev | 28 | 0.950 |' in (directory / 'report.md').read_text(encoding='utf-8-sig')


# -- seal, test, ledger, gate ---------------------------------------------------------------------------------

def test_seal_test_ledger_and_refusal(tmp_path, capsys):
    question_set = tmp_path / 'questions.jsonl'
    question_set.write_bytes(QUESTIONS.read_bytes())
    sealed = tmp_path / 'protocol' / 'protocol.json'
    code, seal = _cli(capsys, 'seal', *SOURCE, '--set', str(question_set), '--protocol', str(sealed))
    assert code == 0 and seal['splits'] == {'dev': 28, 'test': 24}
    stored = protocol.read_protocol(sealed)
    assert stored['set']['sha256'] == seal['set_sha256'] and stored['corpus_sha256']
    # CRLF checkouts hash the same.
    question_set.write_bytes(QUESTIONS.read_bytes().replace(b'\n', b'\r\n'))
    code, first = _cli(capsys, 'test', *SOURCE, '--protocol', str(sealed), '--out', str(tmp_path / 'runs'))
    assert code == 0 and first['ledger']['runs'] == 1 and first['changed_since_seal'] == []
    prefetch = first['lanes']['prefetch']['points']['default']['test']
    assert prefetch['count'] == 24 and prefetch['stale_injected'] == 1
    report_path = Path(first['out']) / 'report.json'
    code, again = _cli(capsys, 'test', *SOURCE, '--protocol', str(sealed), '--out', str(tmp_path / 'runs'),
                       '--baseline', str(report_path))
    assert (code, again['regressed'], again['ledger']['runs']) == (0, False, 2)
    assert again['gate']['prefetch']['recall_diff']['diff'] == 0.0
    # A baseline without the stale injection: the same run now regresses.
    baseline = json.loads(report_path.read_text(encoding='utf-8-sig'))
    for row in baseline['paired']['prefetch'].values():
        row['stale'] = []
    cleaner = tmp_path / 'cleaner.json'
    cleaner.write_text(json.dumps(baseline), encoding='utf-8')
    code, worse = _cli(capsys, 'test', *SOURCE, '--protocol', str(sealed), '--out', str(tmp_path / 'runs'),
                       '--baseline', str(cleaner))
    assert (code, worse['gate']['prefetch']['reasons']) == (1, ['stale_injected'])
    # An edited set is refused before anything runs; the ledger does not move.
    question_set.write_bytes(QUESTIONS.read_bytes().replace(b'Any news today?', b'Any news now?'))
    code, refused = _cli(capsys, 'test', *SOURCE, '--protocol', str(sealed), '--out', str(tmp_path / 'runs'))
    assert (code, refused['refused']) == (2, ['set_sha256'])
    assert len(protocol.read_ledger(protocol.ledger_path(sealed))) == 3


def test_ledger_warns_past_the_limit(tmp_path):
    path = tmp_path / 'protocol.json'
    sealed = {'set': {'sha256': 'abc'}}
    verdicts = [protocol.append_ledger(path, sealed, {}) for _ in range(core.GateTuning.LEDGER_WARN_RUNS + 1)]
    assert [item['warn'] for item in verdicts] == [False] * core.GateTuning.LEDGER_WARN_RUNS + [True]
    assert protocol.ledger_path(path).read_bytes().count(b'\xef\xbb\xbf') == 1  # one BOM, at the start


# -- real-data adapter ----------------------------------------------------------------------------------------

def _db_state(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_home_snapshot_is_read_only_and_scores_like_the_fixture(tmp_path, capsys, monkeypatch):
    embedder = ConceptEmbedder.load(LEXICON)
    home = tmp_path / 'live'
    refs = build_corpus(read_jsonl(CORPUS), home, embedder)
    rows = []
    for row in read_jsonl(QUESTIONS):
        rows.append({**row, 'gold': [refs[key] for key in row['gold']],
                     **({'stale': [refs[key] for key in row['stale']]} if 'stale' in row else {})})
    real_set = tmp_path / 'real.jsonl'
    real_set.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8-sig')
    database = home / 'neospike' / 'memory.db'
    before = _db_state(database)
    monkeypatch.setattr(command, '_real_embedder', _FixtureEmbedderFactory())
    code, real = _cli(capsys, 'run', '--home', str(home), '--set', str(real_set), '--split', 'dev',
                      '--out', str(tmp_path / 'runs'))
    code_synthetic, synthetic = _cli(capsys, 'run', *SOURCE, '--set', str(QUESTIONS), '--split', 'dev',
                                     '--out', str(tmp_path / 'runs'))
    assert (code, code_synthetic, real['source']) == (0, 0, 'real')
    for lane in ('prefetch', 'memory', 'note'):
        now = real['lanes'][lane]['points']['default']['dev']
        fixture = synthetic['lanes'][lane]['points']['default']['dev']
        assert {name: now[name] for name in ('recall', 'mrr', 'fpr', 'irrelevant')} == \
               {name: fixture[name] for name in ('recall', 'mrr', 'fpr', 'irrelevant')}
    assert _db_state(database) == before
    assert not (home / 'neospike' / 'eval').exists()  # --out given: nothing written into the home


class _FixtureEmbedderFactory:
    def __call__(self, _home):
        return ConceptEmbedder.load(LEXICON)


def test_register_eval_cli_adds_the_commands():
    parser = argparse.ArgumentParser()
    command.register_eval_cli(parser.add_subparsers(dest='command'))
    args = parser.parse_args(['eval', 'sweep', '--set', 's.jsonl', '--corpus', 'c.jsonl', '--grid', '0.6:0.7:0.05'])
    assert (args.command, args.eval_command, args.split, args.grid) == ('eval', 'sweep', 'dev', '0.6:0.7:0.05')
