"""Publication boundaries of the search evaluation.

Question text and per-question results stay out of git and out of every HERMES_HOME: by default every output goes
under command.RESULTS_ROOT/runs (inside the package, git-ignored); report.json and standard output carry ids and
aggregate numbers only.
"""
import json
from pathlib import Path

from neospike.evaluation import command
from neospike.evaluation.adapters import read_jsonl

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'eval'


def test_outputs_default_to_results_root_and_keep_question_text_out_of_reports(tmp_path, capsys, monkeypatch):
    home = tmp_path / 'hermes'
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(command, 'RESULTS_ROOT', tmp_path / 'results')  # the test never writes into the package
    code = command.main(['run', '--corpus', str(FIXTURE / 'corpus.jsonl'), '--set', str(FIXTURE / 'questions.jsonl'),
                         '--lanes', 'prefetch'])
    printed = capsys.readouterr().out
    assert code == 0
    directory = Path(json.loads(printed)['out'])
    assert directory.parent == tmp_path / 'results' / command.RUNS_NAME and not home.exists()
    texts = [row['question'] for row in read_jsonl(FIXTURE / 'questions.jsonl')]
    report = (directory / 'report.json').read_text(encoding='utf-8-sig')
    assert not [text for text in texts if text in report or text in printed]
    assert '"paired"' not in printed  # per-question rows are not printed
    assert all(text in (directory / 'per_question.jsonl').read_text(encoding='utf-8-sig') for text in texts)
