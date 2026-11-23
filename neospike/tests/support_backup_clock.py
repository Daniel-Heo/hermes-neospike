"""Backup clock at test scale: B = (now - 50 ms) rounded down to 1 ms, the clock read waits 60 ms.

The product rules are unchanged; only the tunables shrink. A backup treats rows stamped within
BOUNDARY_LAG_SECONDS before the last backup moment as changed, so a test that writes, backs up and backs up
again expecting no file needs a lag shorter than its own pace. The clock read waits longer than the lag
while the backup holds the write lock, so every write before the backup stamps below the lag window (by more
than the ~2 ms Python/SQLite clock skew) and every write after it stamps inside or above it.
"""
from __future__ import annotations

import time

import pytest

from neospike import backup as db_backup

TEST_LAG_SECONDS = 0.05
TEST_STEP_SECONDS = 0.001
TEST_SETTLE_SECONDS = 0.06  # > TEST_LAG_SECONDS + clock skew


def settled_now() -> float:
    """Backup clock for tests: wait past the lag (inside the backup's write lock), then read the wall clock."""
    time.sleep(TEST_SETTLE_SECONDS)
    return time.time()


@pytest.fixture
def fast_backup_boundary(monkeypatch) -> None:
    monkeypatch.setattr(db_backup, "BOUNDARY_LAG_SECONDS", TEST_LAG_SECONDS)
    monkeypatch.setattr(db_backup, "BOUNDARY_STEP_SECONDS", TEST_STEP_SECONDS)
    monkeypatch.setattr(db_backup, "_now", settled_now)
