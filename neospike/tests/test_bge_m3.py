"""The bge-m3 int8 ONNX text embedder (runs only when the model folder is present)."""
from __future__ import annotations

import math

import pytest

from neospike.config import EMBEDDING_DIMENSIONS
from neospike.embedder import (
    ONNX_MODEL_FILE,
    TOKEN_WINDOW,
    TOKENIZER_JSON,
    resolve_model_dir,
)

pytestmark = pytest.mark.skipif(
    not ((resolve_model_dir() / ONNX_MODEL_FILE).is_file() and (resolve_model_dir() / TOKENIZER_JSON).is_file()),
    reason="bge-m3 model and tokenizer required (HERMES_BGE_M3_DIR)")

NORM_TOLERANCE = 1e-3


@pytest.fixture(scope="module")
def embedder():
    from neospike.embedder import OnnxDenseEmbedder
    return OnnxDenseEmbedder()


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def test_vectors_are_1024_and_unit_length(embedder) -> None:
    query = embedder.embed_query("임베딩 모델을 bge-m3로 바꾼다")
    document = embedder.embed_document("Switch the embedding model to BAAI bge-m3", title="ignored")
    for vector in (query, document):
        assert len(vector) == EMBEDDING_DIMENSIONS == 1024
        assert abs(math.sqrt(sum(value * value for value in vector)) - 1.0) < NORM_TOLERANCE


def test_same_meaning_across_korean_and_english_beats_unrelated(embedder) -> None:
    korean = embedder.embed_query("임베딩 모델을 bge-m3로 바꾼다")
    english = embedder.embed_document("Switch the embedding model to BAAI bge-m3")
    unrelated = embedder.embed_document("오늘 저녁 메뉴는 김치찌개다")
    assert _cosine(korean, english) > _cosine(korean, unrelated)


def test_query_and_document_share_one_space_without_prefix(embedder) -> None:
    text = "Neospike keeps memories on this device."
    assert _cosine(embedder.embed_query(text), embedder.embed_document(text)) > 1 - NORM_TOLERANCE


def test_over_window_document_is_an_error_not_a_cut(embedder) -> None:
    text = " ".join(["a"] * (TOKEN_WINDOW - 1))  # one token per word plus <s> and </s>
    assert embedder.count_tokens(text) == TOKEN_WINDOW + 1
    with pytest.raises(ValueError, match=f"window is {TOKEN_WINDOW}"):
        embedder.embed_document(text)


def test_over_window_query_keeps_its_head(embedder) -> None:
    head = " ".join(["a"] * (TOKEN_WINDOW - 2))  # TOKEN_WINDOW tokens with <s> and </s>
    long_query = head + " b b b"
    assert embedder.count_tokens(head) == TOKEN_WINDOW
    embedder.take_warnings()
    assert _cosine(embedder.embed_query(long_query), embedder.embed_query(head)) > 1 - NORM_TOLERANCE
    assert embedder.take_warnings() == [f"query cut to {TOKEN_WINDOW} tokens"]
