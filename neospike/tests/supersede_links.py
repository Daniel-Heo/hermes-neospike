"""Test-only: a supersedes(new -> old) link between two existing memories.

The product writes these links only inside replace_item_with_new (neospike_memory_update mode=supersede);
tests that need an arbitrary chain or a refused cycle call the shared private step directly.
"""
import time

from neospike.store import CREATED_BY_HUMAN, ORIGIN_MIGRATION, MemoryStore


def link_supersede(store: MemoryStore, next_id: int, prev_id: int) -> dict:
    """supersedes(next_id -> prev_id) in its own write transaction; a refused link rolls back."""
    with store._lock:
        try:
            store._begin_write()
            verdict = store._insert_supersede_link(next_id, prev_id, created_by=CREATED_BY_HUMAN,
                                                   origin=ORIGIN_MIGRATION, session_id="", evidence="",
                                                   now=time.time())
            if verdict["linked"]:
                store._conn.commit()
            else:
                store._abort()
        except BaseException:
            store._abort()
            raise
    return verdict
