"""Original CtxPipe context and statistical state encoders."""
from __future__ import annotations

import math
import hashlib
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
import torch
from baselines.common.llm_quantization import fp8_quantization_config


GTE_MODEL_PATH = "/mnt/sdb/dengjiale/dppbench/llm_ckpt/Qwen3-8B"
CONTEXT_CACHE_DIR = "/mnt/sdb/dengjiale/dppbench/outputs/CtxPipe/context_cache"
CONTEXT_DIM = 4096
MAX_CONTEXT_ROWS = 100
MAX_COLUMNS = 100
COLUMN_FEATURE_DIM = 33


class GTEEmbedder:
    """Average-pooling language-model encoder used by CtxPipe."""

    def __init__(
        self,
        model_path: str = GTE_MODEL_PATH,
        device: str = "cpu",
        *,
        cache_dir: str = CONTEXT_CACHE_DIR,
        task_name: str = "unknown",
        sample_seed: int = 42,
    ) -> None:
        self.model_path = str(model_path)
        self.device = torch.device(device)
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_name)) or "unknown"
        self.cache_dir = Path(cache_dir) / safe_task
        self.sample_seed = int(sample_seed)
        self._tokenizer = None
        self._model = None
        self._memory_cache: dict[str, torch.Tensor] = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        if not Path(self.model_path).is_dir():
            raise FileNotFoundError(f"CtxPipe language-model checkpoint not found: {self.model_path}")
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            device_map=str(self.device),
            quantization_config=fp8_quantization_config(self.device),
        ).eval()

    @staticmethod
    def _average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0,
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_text(self, text: str) -> torch.Tensor:
        self._load()
        inputs = self._tokenizer(
            [text], max_length=512, padding=True, truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            output = self._model(**inputs)
            embedding = self._average_pool(
                output.last_hidden_state, inputs["attention_mask"],
            ).cpu().squeeze(0)
        if embedding.numel() != CONTEXT_DIM:
            raise ValueError(
                f"CtxPipe encoder must produce {CONTEXT_DIM} dimensions, got {embedding.numel()}"
            )
        return embedding

    def _cache_key(self, csv_text: str) -> str:
        payload = (
            "ctxpipe-qwen3-8b-fp8-v1\0" + self.model_path + "\0" + csv_text
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _read_cache(self, key: str) -> torch.Tensor | None:
        cached = self._memory_cache.get(key)
        if cached is not None:
            return cached.clone()
        path = self.cache_dir / f"{key}.npy"
        if not path.is_file():
            return None
        try:
            array = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return None
        if array.shape != (CONTEXT_DIM,) or not np.issubdtype(array.dtype, np.floating):
            return None
        value = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())
        self._memory_cache[key] = value
        return value.clone()

    def _write_cache(self, key: str, embedding: torch.Tensor) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        array = embedding.detach().cpu().numpy().astype(np.float32, copy=False)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{key}.", suffix=".npy", dir=self.cache_dir,
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                np.save(handle, array, allow_pickle=False)
            os.replace(temporary, self.cache_dir / f"{key}.npy")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def embed_frame(self, frame: pd.DataFrame) -> torch.Tensor:
        sample = frame
        if len(sample.index) > MAX_CONTEXT_ROWS:
            sample = sample.sample(n=MAX_CONTEXT_ROWS, random_state=self.sample_seed)
        csv_text = sample.to_csv(index=False)
        key = self._cache_key(csv_text)
        cached = self._read_cache(key)
        if cached is not None:
            return cached
        embedding = self.embed_text(csv_text).to(dtype=torch.float32)
        self._write_cache(key, embedding)
        self._memory_cache[key] = embedding
        return embedding.clone()


_DTYPE_ID_MAP = {
    "uint8": 1, "uint16": 1, "uint32": 1, "uint64": 1,
    "int8": 1, "int16": 1, "int32": 1, "int64": 1, "int": 1,
    "float16": 2, "float32": 2, "float64": 2, "float": 2,
    "str": 3, "string": 3,
    "category": 4, "object": 4, "interval[float64]": 4,
    "bool": 5, "boolean": 5,
}


def _finite(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _frexp(value) -> tuple[float, float]:
    value = _finite(value)
    if value == 0.0:
        return 0.0, 0.0
    mantissa, exponent = np.frexp(value)
    return float(mantissa), float(exponent)


def _dtype_id(dtype) -> float:
    name = str(dtype).lower()
    if name in _DTYPE_ID_MAP:
        return float(_DTYPE_ID_MAP[name])
    if "int" in name:
        return 1.0
    if "float" in name:
        return 2.0
    if "bool" in name:
        return 5.0
    return 4.0


def data_feature(frame: pd.DataFrame) -> np.ndarray:
    """Reproduce upstream's 100 columns x 33 statistical state vector."""
    frame = pd.DataFrame(frame)
    features: list[list[float]] = []
    numeric_columns = set(frame.select_dtypes(include=np.number).columns)
    for column in list(frame.columns)[:MAX_COLUMNS]:
        series = frame[column]
        length = max(1, len(series))
        length_m, length_e = _frexp(len(series))
        unique_m, unique_e = _frexp(series.nunique())
        row = [
            _dtype_id(series.dtype), length_m, length_e,
            _finite(series.isnull().sum() / length),
            1.0 if column in numeric_columns else 2.0,
            unique_m, unique_e, _finite(series.nunique() / length),
        ]
        if column in numeric_columns:
            desc = series.describe()
            for key in ("mean", "std", "min", "25%", "50%", "75%", "max"):
                row.extend(_frexp(desc.get(key, 0.0)))
            row.extend(_frexp(series.median()))
            mode = series.mode()
            row.extend(_frexp(mode.iloc[0] if len(mode) else 0.0))
            counts = series.value_counts(dropna=True)
            row.append(_finite((counts.iloc[0] / length) if len(counts) else 0.0))
            row.extend(_frexp(series.sum()))
            row.extend(_frexp(series.skew()))
            row.extend(_frexp(series.kurt()))
        else:
            row.extend([0.0] * (COLUMN_FEATURE_DIM - len(row)))
        if len(row) != COLUMN_FEATURE_DIM:
            raise AssertionError(f"CtxPipe column feature width is {len(row)}, expected {COLUMN_FEATURE_DIM}")
        features.append(row)
    while len(features) < MAX_COLUMNS:
        features.append([0.0] * COLUMN_FEATURE_DIM)
    return np.asarray(features, dtype=np.float32).reshape(-1)


def feature_frame(frame: pd.DataFrame, target_col: str | None) -> pd.DataFrame:
    if "__split__" in frame.columns:
        frame = frame[frame["__split__"] != "std_test"]
    drop = [col for col in (target_col, "__split__") if col and col in frame.columns]
    return frame.drop(columns=drop, errors="ignore").reset_index(drop=True)


__all__ = [
    "COLUMN_FEATURE_DIM", "CONTEXT_CACHE_DIR", "CONTEXT_DIM", "GTEEmbedder", "GTE_MODEL_PATH",
    "MAX_COLUMNS", "data_feature", "feature_frame",
]
