import pandas as pd
from ..base_op import TabularOp


class ExtractDateTimeFeature(TabularOp):
    """Extract calendar/time features from datetime columns."""

    DEFAULT_FEATURES = [
        "year", "month", "day", "dayofweek", "quarter",
        "is_weekend", "dayofyear",
    ]

    def __init__(self, cols, features=None, drop_original=False):
        super().__init__(name="ExtractDateTimeFeature")
        self.cols = cols if isinstance(cols, list) else [cols]
        self.features = features or self.DEFAULT_FEATURES
        self.drop_original = bool(drop_original)

    def get_op_description(self):
        description = """Operator name: ExtractDateTimeFeature

Function description:
Extract year/month/day/week/hour and related features
from datetime-like columns.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'ts': ['2024-01-02 03:04:00', '2024-02-05 10:00:00']})
>>> op = ExtractDateTimeFeature(cols=['ts'], features=['year', 'month', 'day', 'hour'])
>>> op.transform(df)
                    ts  ts_year  ts_month  ts_day  ts_hour
0  2024-01-02 03:04:00     2024         1       2        3
1  2024-02-05 10:00:00     2024         2       5       10

Example YAML:
  - op: ExtractDateTimeFeature
    target: both
    params:
      cols: [ts]
      features: [year, month, day, hour]
"""
        return description.strip()

    def transform(self, df):
        df = df.copy()
        for col in self.cols:
            if col not in df.columns:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                ts = pd.to_datetime(series, unit="s", errors="coerce")
            else:
                ts = pd.to_datetime(series, errors="coerce")
            if "year" in self.features:
                df[f"{col}_year"] = ts.dt.year.astype("Int32")
            if "month" in self.features:
                df[f"{col}_month"] = ts.dt.month.astype("Int8")
            if "day" in self.features:
                df[f"{col}_day"] = ts.dt.day.astype("Int8")
            if "dayofweek" in self.features:
                df[f"{col}_dayofweek"] = ts.dt.dayofweek.astype("Int8")
            if "day_of_week" in self.features:
                df[f"{col}_day_of_week"] = ts.dt.dayofweek.astype("Int8")
            if "quarter" in self.features:
                df[f"{col}_quarter"] = ts.dt.quarter.astype("Int8")
            if "is_weekend" in self.features:
                df[f"{col}_is_weekend"] = (ts.dt.dayofweek >= 5).astype("Int8")
            if "dayofyear" in self.features:
                df[f"{col}_dayofyear"] = ts.dt.dayofyear.astype("Int16")
            if "hour" in self.features:
                df[f"{col}_hour"] = ts.dt.hour.astype("Int8")
            if "hour_of_day" in self.features:
                df[f"{col}_hour_of_day"] = ts.dt.hour.astype("Int8")
            if "days_since_epoch" in self.features:
                epoch = pd.Timestamp("1970-01-01")
                df[f"{col}_days_since_epoch"] = (ts - epoch).dt.days.astype("Int32")
            if self.drop_original:
                df = df.drop(columns=[col])
        return df
