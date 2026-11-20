"""An inactive row brings back the profile block. The
pattern lane is gone: no pattern-* command in the CLI and the daily cron has seven stages without a
paused set. No LLM, no CLI run."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

from neospike import cli
from neospike.provider import PROFILE_OPEN
from neospike.tests.test_live_context import (SID, add_row, deliver, home, make_state, provider,  # noqa: F401
                                              remember)

CRON_SCRIPT = Path(__file__).resolve().parents[1] / "cron" / "neospike_daily.py"
DAILY_STAGES = ["backup", "link-prune", "conv-import", "conv-index", "embed-pending", "pref-reconcile",
                "topic-discover",
                "topic-to-note"]
GONE_COMMANDS = ("pattern-ingest", "pattern-tastes", "pattern-predict", "pattern-compare", "pattern-research")
PROFILE_TEXT = "Prefers guitar examples."
QUESTION = "guitar?"


def load_daily():
    sys.path.insert(0, str(CRON_SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("daily_t465", CRON_SCRIPT)
        daily = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(daily)
    finally:
        sys.path.remove(str(CRON_SCRIPT.parent))
    return daily


# --- an inactive row brings back the profile block ---------------------------------------------------

def test_inactive_row_brings_back_profile(home, provider):
    make_state(home)
    remember(provider, PROFILE_TEXT, memory_kind="preference")
    first = provider.prefetch(QUESTION, session_id=SID)
    assert first.count(PROFILE_OPEN) == 1 and PROFILE_TEXT in first
    row = deliver(home, first, content=QUESTION)
    second = provider.prefetch(QUESTION, session_id=SID)
    assert PROFILE_OPEN not in second
    with closing(sqlite3.connect(home / "state.db")) as connection, connection:
        connection.execute("UPDATE messages SET active=0 WHERE id=?", (row,))
    add_row(home, QUESTION, None)
    third = provider.prefetch(QUESTION, session_id=SID)
    assert third.count(PROFILE_OPEN) == 1 and PROFILE_TEXT in third


# --- the pattern lane is gone ----------------------------------------------------------------------------

def test_cron_has_the_seven_stages_and_no_paused_set(tmp_path):
    daily = load_daily()
    assert [name for name, _args in daily.stages(tmp_path)] == DAILY_STAGES
    assert not hasattr(daily, "PAUSED_STAGES") and not hasattr(daily, "EXIT_PAUSED")


def test_cli_has_no_pattern_commands():
    parser = cli._build_parser()
    for command in GONE_COMMANDS:
        with pytest.raises(SystemExit):
            parser.parse_args([command])
    assert not hasattr(cli, "_PATTERN_VERBS")
