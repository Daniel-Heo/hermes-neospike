"""Correcting a memory must not reveal, or be blocked by, rows the caller cannot see."""
from __future__ import annotations

import pytest
from neospike.tests.support_correction import (
    CONFLICT,
    MSG_UPDATE_FAILED,
    NEW_RULE,
    OLD_RULE,
    S_MAIN,
    UNAVAILABLE,
    deny_commit,
    open_harness,
)

NEW_VEC = (0.0, 1.0, 0.0)


@pytest.fixture
def hx(tmp_path):
    with open_harness(tmp_path) as harness:
        yield harness


def _canonical_key(target):
    folded = " ".join(NEW_RULE.casefold().split())
    return f"memory_item\0{target}\0{_identity(target)}\0{folded}"


def _identity(target):
    return {"project": "@project", "session": S_MAIN}.get(target, "global" if target == "global" else "")


def plant_twin(h, target, boundary, hashing):
    """Insert a row holding NEW_RULE whose visibility to the caller is set by `boundary`."""
    scope, namespace, session = target, h.p._ns_label, S_MAIN
    project = h.p._project_dir if target == "project" else ""
    if boundary in ("stray-proj-col", "foreign-project"):
        project = "/unseen-project"
    elif boundary == "foreign-session":
        session = "beta"
    elif boundary == "bogus-scope":
        scope = "unknown"
    elif boundary == "foreign-ns":
        namespace = "elsewhere:local"
    if hashing == "canonical":
        ident = h.p._project_dir if target == "project" else _identity(target)
        key = f"memory_item\0{target}\0{ident}\0{' '.join(NEW_RULE.casefold().split())}"
    else:
        key = f"legacy-twin:{boundary}:{scope}"
    return h.seed(NEW_RULE, vec=NEW_VEC, namespace=namespace, project=project,
                  source=f"historical-session:{session}",
                  metadata={"scope": scope, "session_id": session, "subject": "older record"},
                  dedupe_key=key)


def _break_text_update(h, how):
    if how == "trigger":
        h.conn.execute("CREATE TRIGGER veto_text AFTER UPDATE OF text ON items "
                       "BEGIN SELECT RAISE(ABORT, 'private failure'); END")
    else:
        h.conn.set_authorizer(deny_commit)


def _assert_index_split(h, old_owner, new_owner):
    assert h.fts_ids("stamp") == old_owner
    assert h.fts_ids("revoked") == new_owner


HIDDEN_CASES = [
    ("global", "stray-proj-col"), ("session", "stray-proj-col"),
    ("session", "foreign-session"), ("project", "foreign-project"),
    ("global", "bogus-scope"), ("global", "foreign-ns"),
]


@pytest.mark.parametrize("hashing", ["legacy", "canonical"])
@pytest.mark.parametrize("target,boundary", HIDDEN_CASES)
def test_unseen_twin_is_ignored_and_left_untouched(hx, target, boundary, hashing):
    mine = hx.save(target)
    twin = plant_twin(hx, target, boundary, hashing)
    assert hx.store.fetch_item_for_update(hx.p._ns_label, twin["id"], project=hx.p._project_dir,
                                          session_id=S_MAIN) is None
    assert hx.correct(mine) == {"updated": True, "id": mine}
    mine_after, twin_after = hx.rows()
    assert mine_after["text"] == NEW_RULE
    assert twin_after == twin
    if boundary != "foreign-ns":
        assert mine_after["text_hash"] != twin["text_hash"]
    assert set(hx.fts_ids("revoked")) == {mine, twin["id"]}
    assert hx.fts_ids("stamp") == []
    assert not hx.conn.in_transaction


@pytest.mark.parametrize("hashing", ["legacy", "canonical"])
@pytest.mark.parametrize("target", ["global", "project", "session"])
def test_seen_twin_in_same_scope_still_blocks(hx, target, hashing):
    mine = hx.save(target)
    twin = plant_twin(hx, target, "visible", hashing)
    snapshot = hx.rows()
    assert hx.correct(mine) == CONFLICT
    assert hx.rows() == snapshot
    _assert_index_split(hx, [mine], [twin["id"]])


@pytest.mark.parametrize("boundary", ["stray-proj-col", "bogus-scope", "foreign-ns"])
def test_hidden_row_answers_like_a_missing_id(hx, boundary):
    hidden = plant_twin(hx, "global", boundary, "canonical")
    snapshot = hx.rows()
    real = hx.correct(hidden["id"], expected_text=NEW_RULE)
    absent = hx.correct(hidden["id"] + 999, expected_text=NEW_RULE)
    assert real == absent == UNAVAILABLE
    assert hx.rows() == snapshot


@pytest.mark.parametrize("breakage", ["trigger", "commit"])
def test_rollback_still_works_when_a_hidden_hash_twin_exists(hx, breakage):
    mine = hx.save("global")
    twin = plant_twin(hx, "global", "stray-proj-col", "canonical")
    snapshot = hx.rows()
    _break_text_update(hx, breakage)
    try:
        assert hx.correct(mine) == {"error": MSG_UPDATE_FAILED}
    finally:
        hx.conn.set_authorizer(None)
    assert hx.rows() == snapshot
    assert not hx.conn.in_transaction
    _assert_index_split(hx, [mine], [twin["id"]])


def test_non_memory_row_holding_the_hash_is_no_semantic_conflict(hx):
    mine = hx.save("global")
    other = plant_twin(hx, "global", "visible", "canonical")
    with hx.conn:
        hx.conn.execute("UPDATE items SET kind='document', origin='file:legacy' WHERE id=?",
                        (other["id"],))
    snapshot = hx.rows()
    assert hx.correct(mine) == {"updated": True, "id": mine}
    assert hx.rows()[1] == snapshot[1]


def test_hash_slot_taken_by_a_hidden_row_gets_a_fallback_and_repeat_corrections_work(hx):
    mine = hx.save("global")
    twin = plant_twin(hx, "global", "stray-proj-col", "canonical")
    assert hx.correct(mine) == {"updated": True, "id": mine}
    fallback = hx.rows()[0]["text_hash"]
    assert hx.correct(mine, expected_text=NEW_RULE, new_text=OLD_RULE) == {"updated": True, "id": mine}
    bystander = hx.seed("A stray legacy document that must stay as it is.", source="file:legacy",
                        kind="document")
    with hx.conn:
        hx.conn.execute("UPDATE items SET text_hash=? WHERE id=?", (fallback, bystander["id"]))
    frozen = hx.rows()
    assert hx.correct(mine) == {"updated": True, "id": mine}
    now = hx.rows()
    assert now[1:] == frozen[1:]
    assert now[1] == twin
    assert len({r["text_hash"] for r in now}) == 3


def test_unrelated_unique_violation_surfaces_only_a_generic_error(hx):
    mine = hx.save("global")
    snapshot = hx.rows()
    hx.conn.executescript("""
        CREATE TABLE side_ledger (token TEXT UNIQUE);
        INSERT INTO side_ledger VALUES ('private payload');
        CREATE TRIGGER dup_side_ledger AFTER UPDATE OF text ON items BEGIN
            INSERT INTO side_ledger VALUES ('private payload');
        END;
    """)
    result = hx.correct(mine)
    assert result == {"error": MSG_UPDATE_FAILED}
    assert "private" not in str(result)
    assert hx.rows() == snapshot
    assert not hx.conn.in_transaction
    _assert_index_split(hx, [mine], [])
