"""Without the model files no inference daemon is started, and a daemon started anyway exits unready."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from neospike import embedder
from neospike import inference

MISSING_MODEL_FOLDER = "no-such-model"
DAEMON_EXIT_SECONDS = 30
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class _SpawnRecorded(Exception):
    pass


class _SpawnRecorder:
    """Stand-in for EmbeddingClient._spawn_daemon that records the launch instead of starting a process."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *_launch_args):
        self.calls += 1
        raise _SpawnRecorded()


def _missing_model_env(monkeypatch, tmp_path) -> Path:
    missing = tmp_path / MISSING_MODEL_FOLDER
    monkeypatch.setenv("HERMES_BGE_M3_DIR", str(missing))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("NEOSPIKE_INFERENCE_HOME", raising=False)
    return missing


def test_model_file_names_match_the_embedder():
    assert set(inference.MODEL_FILE_NAMES) == {embedder.TOKENIZER_JSON, embedder.ONNX_MODEL_FILE}
    assert inference.MODEL_FOLDER == embedder.MODEL_FOLDER


def test_client_without_model_files_does_not_start_a_daemon(tmp_path, monkeypatch):
    missing = _missing_model_env(monkeypatch, tmp_path)
    recorder = _SpawnRecorder()
    monkeypatch.setattr(inference.EmbeddingClient, "_spawn_daemon", recorder)
    client = inference.EmbeddingClient(tmp_path / "home", startup_timeout=2)
    with pytest.raises(inference.EmbeddingUnavailable) as caught:
        client.embed_query("hello")
    assert recorder.calls == 0
    assert "Embedding model file not found" in str(caught.value)
    assert str(missing) in str(caught.value)


def test_daemon_without_model_files_exits_without_ready(tmp_path, monkeypatch):
    _missing_model_env(monkeypatch, tmp_path)
    env = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    home = tmp_path / "home"
    child = subprocess.Popen([sys.executable, "-m", "neospike.inference", "--home", str(home)],
                             cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True)
    try:
        out, err = child.communicate(timeout=DAEMON_EXIT_SECONDS)
    except subprocess.TimeoutExpired:
        child.kill()
        child.communicate()
        pytest.fail("daemon without model files kept running")
    assert child.returncode != 0
    assert "ready" not in out
    assert "Embedding model file not found" in err
