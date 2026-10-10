"""Feature extraction compatible with HAIPipe's public 100 x 19 layout."""
from __future__ import annotations

import numpy as np
import pandas as pd


DTYPE_IDS = {
    "uint8": 1, "uint16": 1, "int16": 1, "int32": 1, "int64": 1,
    "int": 1, "float16": 2, "float32": 2, "float64": 2,
    "float": 2, "str": 3, "object": 4, "category": 4, "bool": 4,
}


def _finite(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def dataset_features(frame: pd.DataFrame, column_num: int = 100) -> np.ndarray:
    """Return the exact ordered 19 statistics used by upstream AIPipe."""
    rows: list[list[float]] = []
    frame = pd.DataFrame(frame)
    for column in frame.columns[:column_num]:
        series = frame[column]
        length = len(series)
        numeric = pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype)
        unique = series.nunique(dropna=True)
        base = [
            float(DTYPE_IDS.get(str(series.dtype), 4)),
            float(length),
            _finite(series.isna().sum() / max(1, length)),
            1.0 if numeric else 2.0,
            float(unique),
            _finite(unique / max(1, length)),
        ]
        if numeric:
            clean = pd.to_numeric(series, errors="coerce")
            desc = clean.describe()
            mode = clean.mode()
            mode_value = mode.iloc[0] if len(mode) else 0.0
            counts = clean.value_counts()
            mode_ratio = counts.iloc[0] / max(1, length) if len(counts) else 0.0
            extra = [
                desc.get("mean", 0), desc.get("std", 0), desc.get("min", 0),
                desc.get("25%", 0), desc.get("50%", 0), desc.get("75%", 0),
                desc.get("max", 0), clean.median(), mode_value, mode_ratio,
                clean.sum(), clean.skew(), clean.kurt(),
            ]
            base.extend(_finite(value) for value in extra)
        else:
            base.extend([0.0] * 13)
        rows.append(base)
    rows.extend([[0.0] * 19 for _ in range(column_num - len(rows))])
    return np.asarray(rows, dtype=np.float32).reshape(-1)


def feature_frame(frame, target_col: str | None = None) -> pd.DataFrame:
    """Profile predictors only, excluding frozen standard-test rows and markers."""
    frame = pd.DataFrame(frame)
    if "__split__" in frame.columns:
        frame = frame.loc[frame["__split__"] != "std_test"]
    return frame.drop(columns=[c for c in (target_col, "__split__") if c in frame.columns])


def frame_from_loaded_data(data, target_col: str | None = None) -> pd.DataFrame:
    for attr in ("train_df", "interaction_df"):
        frame = getattr(data, attr, None)
        if frame is not None:
            return feature_frame(frame, getattr(data, "target_col", None) or target_col)
    raise ValueError("HAIPipe could not find a training dataframe in the task loader")


__all__ = ["dataset_features", "feature_frame", "frame_from_loaded_data"]
