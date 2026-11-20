from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .config import EMBEDDING_DIMENSIONS, EMBEDDING_MAX_TOKENS

_client_cache: dict = {}
_client_cache_lock = threading.Lock()

# BAAI/bge-m3, community int8 ONNX export
# (Xenova/bge-m3 @ 4de13258303883538bd53b696b452bf8099f0858). Dense vector only.
ONNX_MODEL_FILE = "sentence_transformers_int8.onnx"
TOKENIZER_JSON = "tokenizer.json"
MODEL_FOLDER = "bge-m3-onnx"
TOKEN_WINDOW = EMBEDDING_MAX_TOKENS
# i5-12600KF, one 512-token input: 0 (default) 8.4 s, 2 -> 3.6 s, 4 -> 1.9 s, 6 -> 4.8 s, 8 -> 6.2 s.
ORT_INTRA_THREADS = 4
_OUTPUT_TENSOR = "sentence_embedding"


def resolve_model_dir() -> Path:
    override = os.environ.get("HERMES_BGE_M3_DIR")
    if override:
        return Path(override).expanduser()
    from ._wincompat import default_hermes_home, shared_inference_home
    return shared_inference_home(default_hermes_home()) / "models" / MODEL_FOLDER


class OnnxDenseEmbedder:
    """bge-m3 dense vectors (1024, L2-normalized) on onnxruntime CPU.

    Inputs run at their real length (dynamic axis, no padding, no prompt prefix).
    A document over TOKEN_WINDOW is an error, never a silent cut (callers chunk it). A query over the
    window keeps its first TOKEN_WINDOW tokens: a failed recall is worse than a search on the head.
    Why 2048: attention memory grows with the square of the length (8192 tokens held ~8.7 GB in the daemon).
    """

    sequence_length = TOKEN_WINDOW
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, *, intra_threads: int = ORT_INTRA_THREADS) -> None:
        model_dir = resolve_model_dir()
        self._tokenizer = Tokenizer.from_file(str(model_dir / TOKENIZER_JSON))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_threads
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_dir / ONNX_MODEL_FILE), sess_options=options,
                                             providers=["CPUExecutionProvider"])
        self._lock = threading.Lock()
        self._warnings: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text, cut=True)

    def embed_document(self, text: str, *, title: str = "none") -> list[float]:
        # bge-m3 takes no title/task prompt; title is accepted for the shared interface only.
        return self._vectorize(text)

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=True).ids)

    def take_warnings(self) -> list[str]:
        with self._lock:
            taken, self._warnings = self._warnings, []
        return taken

    def _vectorize(self, text: str, *, cut: bool = False) -> list[float]:
        ids = self._tokenizer.encode(text, add_special_tokens=True).ids
        if len(ids) > TOKEN_WINDOW:
            if not cut:
                raise ValueError(f"Embedding input has {len(ids)} tokens; the window is {TOKEN_WINDOW}. Chunk it first.")
            ids = ids[:TOKEN_WINDOW - 1] + ids[-1:]  # keep the closing </s>
            with self._lock:
                self._warnings.append(f"query cut to {TOKEN_WINDOW} tokens")
        feed = {"input_ids": np.asarray([ids], dtype=np.int64),
                "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}
        with self._lock:
            vector = self._session.run([_OUTPUT_TENSOR], feed)[0][0]
        if vector.shape[0] != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Model returned {vector.shape[0]} dimensions, expected {EMBEDDING_DIMENSIONS}")
        norm = float(np.linalg.norm(vector))
        if norm:
            vector = vector / norm
        return vector.astype(np.float32, copy=False).tolist()
