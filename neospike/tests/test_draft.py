"""Draft notes must cite a [[source]]; the draft skill file is well formed."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from neospike.notes import DRAFT_TAG, NoteError
from neospike.service import NeospikeService
from neospike.store import MemoryStore
from neospike.tests.test_notes import DIMENSIONS, BowEmbedder

SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "second-brain-draft" / "SKILL.md"
BODY = ("The release plan moves the dashboard launch to the second week of December "
        "because the sign-up review flow needs another round of testing.")


@pytest.fixture
def service(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.db", dimensions=DIMENSIONS)
    yield NeospikeService(store=store, embedder=BowEmbedder(), namespace="owner")
    store.close()


def test_draft_without_link_is_rejected(service) -> None:
    with pytest.raises(NoteError, match=r"source link"):
        service.notes.save("DRAFT: release plan", BODY, tags=[DRAFT_TAG], author="agent")
    saved = service.notes.save("DRAFT: release plan", BODY, tags=["plan"], author="agent")
    assert saved["id"] == 1  # the rejected draft left no row and no title behind


@pytest.mark.parametrize("link", ["[[Release notes]]", "[[mem:7]]"])
def test_draft_with_link_is_saved(service, link: str) -> None:
    saved = service.notes.save("DRAFT: release plan", f"{BODY} {link}", tags=[DRAFT_TAG], author="agent")
    assert service.notes.read(note_id=saved["id"])["tags"] == [DRAFT_TAG]


def test_non_draft_note_without_link_is_saved(service) -> None:
    saved = service.notes.save("Release plan", BODY, tags=["plan"], author="agent")
    assert saved["id"] >= 1


def test_skill_frontmatter_is_yaml_with_name_and_description() -> None:
    text = SKILL_PATH.read_text(encoding="utf-8-sig")
    assert text.startswith("---\n")
    front = yaml.safe_load(text.split("---\n", 2)[1])
    assert front["name"] == "second-brain-draft"
    assert isinstance(front["description"], str) and front["description"].strip()
