"""An unavailable embedder ends a CLI command with a JSON error (exit 1), never a traceback."""
from __future__ import annotations

import contextlib
import json
import sys

import pytest

from neospike import cli
from neospike import inference

DAEMON_MESSAGE = "model could not be loaded"
MISSING_MODEL_FOLDER = "no-such-model"


class _UnavailableEmbedder:
    """EmbeddingClient stand-in whose every request fails as the daemon does without a model."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def embed_query(self, _text):
        raise inference.EmbeddingUnavailable(DAEMON_MESSAGE)

    def embed_document(self, _text, **_kwargs):
        raise inference.EmbeddingUnavailable(DAEMON_MESSAGE)


NOTE_REQUEST = {"title": "probe", "body": "hello body"}


@pytest.mark.parametrize("words", [("memory-search", "hello"), ("note-save", "--input", "{request}")])
def test_embedding_command_reports_unavailable_embedder_as_json(tmp_path, monkeypatch, capsys, words):
    request = tmp_path / "request.json"
    request.write_text(json.dumps(NOTE_REQUEST), encoding="utf-8")
    monkeypatch.setattr(cli, "EmbeddingClient", _UnavailableEmbedder)
    home = tmp_path / "home"
    home.mkdir()
    argv = [word.format(request=request) for word in words]
    monkeypatch.setattr(sys, "argv", ["neospike", "--home", str(home), *argv])
    assert cli.main() == 1
    output = json.loads(capsys.readouterr().out)
    assert output["error"] == "invalid"
    assert DAEMON_MESSAGE in output["message"]


def test_embedding_unavailable_is_a_runtime_error():
    assert issubclass(inference.EmbeddingUnavailable, RuntimeError)


class _NullConnection:
    def settimeout(self, _seconds) -> None:
        pass


@contextlib.contextmanager
def _null_socket(*_args, **_kwargs):
    yield _NullConnection()


def _error_frame(_conn):
    return {"error": DAEMON_MESSAGE}


def _token(_self):
    return "token"


def _ignore_frame(_conn, _payload) -> None:
    pass


def test_daemon_error_reply_raises_embedding_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(inference.EmbeddingClient, "_read_token", _token)
    monkeypatch.setattr(inference._wc, "connect_socket", _null_socket)
    monkeypatch.setattr(inference, "_write_frame", _ignore_frame)
    monkeypatch.setattr(inference, "_read_frame", _error_frame)
    client = inference.EmbeddingClient(tmp_path)
    with pytest.raises(inference.EmbeddingUnavailable, match=DAEMON_MESSAGE):
        client._exchange("query", "hello")


def test_missing_model_folder_is_named_in_the_error(tmp_path, monkeypatch):
    missing = tmp_path / MISSING_MODEL_FOLDER
    monkeypatch.setenv("HERMES_BGE_M3_DIR", str(missing))
    runner = inference.LocalModelRunner(tmp_path)
    with pytest.raises(FileNotFoundError) as caught:
        runner("query", "hello")
    assert str(missing) in str(caught.value)
    assert "model" in str(caught.value).lower()
