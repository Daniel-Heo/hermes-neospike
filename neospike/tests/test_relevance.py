"""Admission regressions use controlled cosines, not model-quality estimates."""
import math

import pytest

from neospike.provider import NeospikeProvider
from neospike.service import NeospikeService
from neospike.store import DEFAULT_SEMANTIC_CUTOFF, MemoryStore

FLOOR_MARGIN = 0.02  # cosine just below / above the default floor (follows the constant)


def vector(cosine):
    return [cosine, math.sqrt(1 - cosine * cosine)]


@pytest.fixture
def store(tmp_path):
    value = MemoryStore(tmp_path / 'memory.db', dimensions=2, semantic_threshold=.5)
    yield value
    value.close()


@pytest.mark.parametrize('text,cosine,kind,importance', [
    ('The telescope was repaired yesterday.', .30, 'episodic', .5),
    ('Telescope lens catalog', .30, 'durable', 1.0),
    ('An unrelated durable preference', .40, 'durable', 1.0),
    ('The telescope was repaired yesterday.', 0.0, 'durable', 1.0),
])
def test_lexical_and_metadata_bonuses_cannot_admit_weak_match(store, text, cosine, kind, importance):
    store.add('owner', text, vector(cosine), source='synthetic', kind=kind, importance=importance)
    assert store.search('owner', 'What is the telescope warranty?', [1., 0.]) == []
    assert store._conn.execute("SELECT access_count FROM items WHERE type='memory'").fetchone()[0] == 0


@pytest.mark.parametrize('query,text', [
    ('How often are backups removed?', 'Old snapshots get discarded every week.'),
    ('When do old copies get deleted?', 'Backups are discarded every week.'),
])
def test_semantic_paraphrases_need_no_shared_words(store, query, text):
    store.add('owner', text, vector(.75), source='synthetic', importance=0)
    store._conn.execute('UPDATE items SET created_at=0')
    assert [r.text for r in store.search('owner', query, [1., 0.])] == [text]


def test_gate_runs_before_top_k_and_access_updates(store):
    store.add('owner', 'The telescope warranty details are unavailable', vector(.40),
              source='weak', kind='durable', importance=1)
    store.add('owner', 'Coverage lasts for two years', vector(.75), source='relevant', importance=0)
    # The weak row's old hybrid score exceeds the strong semantic-only score.
    assert [r.origin for r in store.search('owner', 'What is the telescope warranty?', [1., 0.], limit=1)] == ['relevant']
    assert dict(store._conn.execute("SELECT origin,access_count FROM items WHERE type='memory'")) == {'weak': 0, 'relevant': 1}


def test_weak_text_is_gated_out_of_rrf(store):
    class Embedder:
        dimensions = 2

        def embed_query(self, query):
            return [1., 0.]

    store.add('owner', 'The telescope catalog', vector(.3), source='weak', kind='durable', importance=1)
    provider = NeospikeProvider(embedder=Embedder())
    provider._core = NeospikeService(store=store, embedder=Embedder(), namespace='owner')
    # The weak text is gated out (there is no image lane).
    assert provider._fused_search('What is the telescope warranty?', limit=1) == []


@pytest.mark.parametrize('embedding', [[0., 0.], [float('nan'), 1.], [float('inf'), 1.]])
def test_invalid_semantic_evidence_cannot_use_exact_lexical_match(store, embedding):
    store.add('owner', 'telescope warranty', embedding, source='invalid', kind='durable', importance=1)
    assert store.search('owner', 'telescope warranty', [1., 0.]) == []


@pytest.fixture
def default_store(tmp_path):
    value = MemoryStore(tmp_path / 'default.db', dimensions=2)
    yield value
    value.close()


def test_shared_word_below_cosine_floor_is_not_injected(default_store):
    # Cos just below the floor; the shared words "telescope warranty" add nothing.
    default_store.add('owner', 'telescope warranty', vector(DEFAULT_SEMANTIC_CUTOFF - FLOOR_MARGIN), source='overlap', kind='durable', importance=1)
    assert default_store.search('owner', 'telescope warranty', [1., 0.]) == []


def test_cosine_above_floor_is_injected_without_shared_words(default_store):
    default_store.add('owner', 'Coverage lasts two years', vector(DEFAULT_SEMANTIC_CUTOFF + FLOOR_MARGIN), source='semantic', importance=0)
    default_store._conn.execute('UPDATE items SET created_at=0')
    assert [r.origin for r in default_store.search('owner', 'What is the telescope warranty?', [1., 0.])] == ['semantic']
