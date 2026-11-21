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

# Where the model runs. DirectML (Windows, any DirectX 12 GPU) is used when it works, otherwise the CPU.
DEVICE_CPU = "cpu"
DEVICE_DIRECTML = "directml"
DEVICE_AUTO = "auto"  # DirectML when it works, otherwise the CPU
DEVICE_CHOICES = (DEVICE_AUTO, DEVICE_CPU, DEVICE_DIRECTML)
_CPU_PROVIDER = "CPUExecutionProvider"
_DML_PROVIDER = "DmlExecutionProvider"
_DML_DEVICE_ID = 0  # the default adapter
# The first run on a device pays its setup (DirectML: about 1.2 s), so it is done once at load.
_PROBE_TEXT = "warm-up"


def resolve_model_dir() -> Path:
    override = os.environ.get("HERMES_BGE_M3_DIR")
    if override:
        return Path(override).expanduser()
    from ._wincompat import default_hermes_home, shared_inference_home
    return shared_inference_home(default_hermes_home()) / "models" / MODEL_FOLDER


def _feed(ids) -> dict:
    return {"input_ids": np.asarray([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64)}


class DeviceUnavailable(RuntimeError):
    """The requested device cannot run the model here."""


def _cpu_session(model_path: str, intra_threads: int):
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, sess_options=options, providers=[_CPU_PROVIDER])


def _directml_session(model_path: str, intra_threads: int):
    """DirectML needs memory patterns off and sequential execution; nodes it cannot run go to the CPU."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_threads
    options.inter_op_num_threads = 1
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(model_path, sess_options=options,
                                   providers=[(_DML_PROVIDER, {"device_id": _DML_DEVICE_ID}), _CPU_PROVIDER])
    if _DML_PROVIDER not in session.get_providers():
        raise DeviceUnavailable("the session did not take DirectML")
    return session


def directml_installed() -> bool:
    """True when this onnxruntime build carries DirectML (the Windows onnxruntime-directml package)."""
    return _DML_PROVIDER in ort.get_available_providers()


def open_session(model_path: str, intra_threads: int, device: str, probe_feed: dict):
    """(session, device, note) for the wanted device. auto: DirectML when the build has it and a session on
    it runs the probe, otherwise the CPU with the reason in note. cpu/directml: that device or an error."""
    if device not in DEVICE_CHOICES:
        raise ValueError(f"Unknown embedding device: {device}")
    note = ""
    if device != DEVICE_CPU:
        if directml_installed():
            try:
                session = _directml_session(model_path, intra_threads)
                session.run([_OUTPUT_TENSOR], probe_feed)
                return session, DEVICE_DIRECTML, note
            except Exception as error:  # any driver or session failure means "not usable here"
                note = f"DirectML failed ({type(error).__name__}: {str(error)[:200]})"
        else:
            note = "DirectML is not in this onnxruntime build"
        if device == DEVICE_DIRECTML:
            raise DeviceUnavailable(note)
    session = _cpu_session(model_path, intra_threads)
    session.run([_OUTPUT_TENSOR], probe_feed)
    return session, DEVICE_CPU, note


class OnnxDenseEmbedder:
    """bge-m3 dense vectors (1024, L2-normalized) on onnxruntime: DirectML when it works, otherwise the CPU.
    self.device is the device in use; self.device_note says why DirectML was not taken (empty when it was,
    or when the CPU was asked for). Vectors of the two devices differ slightly, so stored vectors and
    queries must come from one device.

    Inputs run at their real length (dynamic axis, no padding, no prompt prefix).
    A document over TOKEN_WINDOW is an error, never a silent cut (callers chunk it). A query over the
    window keeps its first TOKEN_WINDOW tokens: a failed recall is worse than a search on the head.
    Why 2048: attention memory grows with the square of the length (8192 tokens held ~8.7 GB in the daemon).
    """

    sequence_length = TOKEN_WINDOW
    dimensions = EMBEDDING_DIMENSIONS

    def __init__(self, *, intra_threads: int = ORT_INTRA_THREADS, device: str = DEVICE_AUTO) -> None:
        model_dir = resolve_model_dir()
        self._tokenizer = Tokenizer.from_file(str(model_dir / TOKENIZER_JSON))
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        probe_ids = self._tokenizer.encode(_PROBE_TEXT, add_special_tokens=True).ids
        self._session, self.device, self.device_note = open_session(
            str(model_dir / ONNX_MODEL_FILE), intra_threads, device, _feed(probe_ids))
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
        with self._lock:
            vector = self._session.run([_OUTPUT_TENSOR], _feed(ids))[0][0]
        if vector.shape[0] != EMBEDDING_DIMENSIONS:
            raise ValueError(f"Model returned {vector.shape[0]} dimensions, expected {EMBEDDING_DIMENSIONS}")
        norm = float(np.linalg.norm(vector))
        if norm:
            vector = vector / norm
        return vector.astype(np.float32, copy=False).tolist()
