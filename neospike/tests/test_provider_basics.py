"""Provider lifecycle without implicit memory, backup declaration, config schema and retention keys."""
from __future__ import annotations

import json
from pathlib import Path

from neospike.tests.support_provider import ThreeTopicEmbedder, boot_provider

from neospike import NeospikeProvider
from neospike.config import NeospikeConfig
from neospike.store import MemoryStore

REMOVED_KEYS = ("episodic_retention_days", "summary_retention_days")


def test_ending_a_session_writes_no_memory_by_itself(tmp_path: Path) -> None:
    provider = boot_provider(tmp_path, "s1")

    provider.on_session_end(
        [
            {"role": "user", "content": "My guitar buzzes."},
            {"role": "assistant", "content": "Look at the neck relief."},
        ],
        session_id="s1",
    )

    assert provider._memory_db.count(provider._ns_label) == 0


def test_provider_reports_no_extra_backup_paths() -> None:
    assert NeospikeProvider(embedder=ThreeTopicEmbedder()).backup_paths() == []


def test_old_retention_keys_load_without_effect_and_leave_no_expiry_column(tmp_path: Path) -> None:
    """There is no retention setting; a config.json that still has the keys must keep loading."""
    folder = tmp_path / "neospike"
    folder.mkdir()
    (folder / "config.json").write_text(
        json.dumps({"episodic_retention_days": 90, "summary_ttl_days": 730}), encoding="utf-8"
    )

    loaded = NeospikeConfig.load(tmp_path)

    assert [hasattr(loaded, key) for key in REMOVED_KEYS] == [False, False]
    store = MemoryStore(tmp_path / "m.sqlite", dimensions=3)
    try:
        columns = [row[1] for row in store._conn.execute("PRAGMA table_info(items)")]
    finally:
        store.close()
    assert "expires_at" not in columns


def test_config_form_hides_removed_settings() -> None:
    form = {entry["key"] for entry in NeospikeProvider(embedder=ThreeTopicEmbedder()).get_config_schema()}

    assert form.isdisjoint({"embedding_dimensions", "visual_enabled", *REMOVED_KEYS})


def test_saving_removed_settings_is_harmless_and_not_persisted(tmp_path: Path) -> None:
    provider = NeospikeProvider(embedder=ThreeTopicEmbedder())

    provider.save_config(
        {"embedding_dimensions": "768", "episodic_retention_days": "30", "summary_retention_days": ""},
        str(tmp_path),
    )

    NeospikeConfig.load(tmp_path)
    written = json.loads((tmp_path / "neospike" / "config.json").read_text(encoding="utf-8"))
    assert "embedding_dimensions" not in written
    assert "episodic_retention_days" not in written
