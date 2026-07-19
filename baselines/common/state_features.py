"""Shared dataframe state features for pipeline-search baselines."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


MAX_COLUMNS = 100
COLUMN_FEATURE_DIM = 33

_DTYPE_ID_MAP = {
    "uint8": 1, "uint16": 1, "uint32": 1, "uint64": 1,
    "int8": 1, "int16": 1, "int32": 1, "int64": 1, "int": 1,
    "float16": 2, "float32": 2, "float64": 2, "float": 2,
    "str": 3, "string": 3, "category": 4, "object": 4,
    "interval[float64]": 4, "bool": 5, "boolean": 5,
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


def dataframe_state_features(frame: pd.DataFrame) -> np.ndarray:
    """Encode at most 100 columns with the shared 33-value profile."""
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
            raise AssertionError(
                f"column feature width is {len(row)}, expected {COLUMN_FEATURE_DIM}"
            )
        features.append(row)
    while len(features) < MAX_COLUMNS:
        features.append([0.0] * COLUMN_FEATURE_DIM)
    return np.asarray(features, dtype=np.float32).reshape(-1)


def primary_feature_frame(frame: pd.DataFrame, target_col: str | None) -> pd.DataFrame:
    """Remove frozen-test rows and bookkeeping/target columns from state input."""
    if "__split__" in frame.columns:
        frame = frame[frame["__split__"] != "std_test"]
    drop = [column for column in (target_col, "__split__") if column and column in frame.columns]
    return frame.drop(columns=drop, errors="ignore").reset_index(drop=True)


__all__ = [
    "COLUMN_FEATURE_DIM", "MAX_COLUMNS", "dataframe_state_features",
    "primary_feature_frame",
]
