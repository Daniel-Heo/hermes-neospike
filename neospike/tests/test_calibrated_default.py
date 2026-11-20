from neospike.mcp_server import SEMANTIC_THRESHOLD
from neospike.store import DEFAULT_SEMANTIC_CUTOFF, MemoryStore


def test_default_uses_cosine_only_floor(tmp_path):
    store = MemoryStore(tmp_path / 'memory.db', dimensions=2)
    try:
        assert store.semantic_threshold == DEFAULT_SEMANTIC_CUTOFF
        assert SEMANTIC_THRESHOLD == DEFAULT_SEMANTIC_CUTOFF  # mcp_server reads the MemoryStore default
    finally:
        store.close()
