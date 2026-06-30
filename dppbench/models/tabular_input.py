import numpy as np
import pandas as pd


def convert_datetime_features(X, datetime_cols=None):
    """Convert DataFrame datetime columns to numeric Unix seconds.

    The output keeps the original column names so downstream feature alignment
    and importance reporting stay stable. Missing timestamps become 0.0.
    """
    if not isinstance(X, pd.DataFrame):
        return X, list(datetime_cols or [])

    out = X.copy()
    if datetime_cols is None:
        datetime_cols = [
            c for c in out.columns
            if pd.api.types.is_datetime64_any_dtype(out[c])
        ]
    else:
        datetime_cols = list(datetime_cols)

    for col in datetime_cols:
        if col not in out.columns:
            continue
        values = pd.to_datetime(out[col], errors="coerce")
        numeric = pd.Series(0.0, index=out.index, dtype=np.float64)
        mask = values.notna()
        if mask.any():
            numeric.loc[mask] = (
                values.loc[mask].astype("int64").astype(np.float64) / 1_000_000_000.0
            )
        out[col] = numeric

    return out, datetime_cols
