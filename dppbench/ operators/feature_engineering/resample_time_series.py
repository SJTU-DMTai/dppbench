import pandas as pd
from ..base_op import TabularOp


class ResampleTimeSeries(TabularOp):
    """Resample event data into fixed time granularity aggregates."""

    def __init__(self, time_col, freq, aggs=None, group_cols=None, count_col=None):
        super().__init__(name="ResampleTimeSeries")
        self.time_col = time_col
        self.freq = freq
        self.aggs = aggs or {}
        self.group_cols = group_cols or []
        self.count_col = count_col

    def get_op_description(self):
        description = """Operator name: ResampleTimeSeries

Function description:
Aggregate irregular event rows into fixed time buckets.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'ts': ['2024-01-01 00:10', '2024-01-01 00:40', '2024-01-01 01:05'], 'store': ['A', 'A', 'A'], 'sales': [2, 3, 4]})
>>> op = ResampleTimeSeries(time_col='ts', freq='1H', group_cols=['store'], aggs={'sales': 'sum'}, count_col='cnt')
>>> op.transform(df)
  store            _hour_ts  sales_sum  cnt           _hour_idx
0     A 2024-01-01 00:00:00          5    2          1704067200
1     A 2024-01-01 01:00:00          4    1          1704070800

Example YAML:
  - op: ResampleTimeSeries
    target: train
    params:
      time_col: ts
      freq: 1H
      group_cols: [store]
      aggs:
        sales: sum
      count_col: cnt
"""
        return description.strip()

    def _bucket_cols(self):
        freq = str(self.freq).lower()
        if "h" in freq:
            return "_hour_ts", "_hour_idx"
        return "_bucket_ts", "_bucket_idx"

    def _pandas_freq(self):
        freq = str(self.freq)
        return freq.replace("H", "h") if "H" in freq else freq

    def transform(self, df):
        if self.time_col not in df.columns:
            return df
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
        keys = [c for c in self.group_cols if c in df.columns]
        bucket_ts_col, bucket_idx_col = self._bucket_cols()
        df[bucket_ts_col] = df[self.time_col].dt.floor(self._pandas_freq())
        group_keys = keys + [bucket_ts_col]
        if self.aggs:
            out = df.groupby(group_keys, dropna=False).agg(self.aggs).reset_index()
            out.columns = [
                "_".join([str(x) for x in col if x]) if isinstance(col, tuple) else col
                for col in out.columns
            ]
            if self.count_col:
                counts = (
                    df.groupby(group_keys, dropna=False)
                    .size()
                    .reset_index(name=self.count_col)
                )
                out = out.merge(counts, on=group_keys, how="left")
        else:
            name = self.count_col or "count"
            out = df.groupby(group_keys, dropna=False).size().reset_index(name=name)
        out[bucket_idx_col] = out[bucket_ts_col].astype("int64") // 10 ** 9
        return out
