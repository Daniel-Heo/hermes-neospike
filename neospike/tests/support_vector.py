"""Test helper: the full in-process comparison a vector search must match (and stands in for it where a test
checks something else on a database the service cannot read, such as an old layout inside its migration)."""
from __future__ import annotations

import math

import pytest

from neospike import vector_index
from neospike.store import dot_similarities


def full_similarities(connection, table, ids, query, *, threshold=None, limit=None, column="embedding"):
    """Every allowed row compared on this connection (uncommitted rows included); NaN for an id with no row."""
    rows = dict(connection.execute(f"SELECT id,{column} FROM {table}").fetchall())
    sims = dot_similarities(query, [rows.get(int(rowid)) for rowid in ids])
    return [sim if int(rowid) in rows else math.nan for rowid, sim in zip(ids, sims)]


def full_similarities_many(connection, table, ids, queries, *, threshold=None, limit=None, column="embedding"):
    return [full_similarities(connection, table, ids, query, column=column) for query in queries]


def use_full_comparison(patch: pytest.MonkeyPatch) -> None:
    patch.setattr(vector_index, "similarities", full_similarities)
    patch.setattr(vector_index, "similarities_many", full_similarities_many)
