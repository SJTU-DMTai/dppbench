from ..base_op import TabularOp


class SortRows(TabularOp):
    """Sort rows by one or more keys."""

    KIND = "mergesort"
    NA_POSITION = "last"

    def __init__(self, by=None, ascending=True):
        super().__init__(name="SortRows")
        self.by = by if isinstance(by, list) or by is None else [by]
        self.ascending = ascending

    def get_op_description(self):
        description = """Operator name: SortRows

Function description:
Sort rows by one or more columns. This is useful before
time-series lag/rolling features or sequence construction.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
by : str or list[str] — Sort keys.
ascending : bool or list[bool] — Sort direction. Default True.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'user_id': [1, 1, 2], 'timestamp': [3, 1, 2]})
>>> op = SortRows(by=['user_id', 'timestamp'], ascending=True)
>>> op.transform(df)
   user_id  timestamp
0        1          1
1        1          3
2        2          2

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: SortRows
    prev:
    - s0
    params:
      by:
      - user_id
      - timestamp
      ascending: true
  train:
    prev:
    - o1
"""
        return description.strip()

    def transform(self, df):
        if not self.by:
            return df
        by = [c for c in self.by if c in df.columns]
        if not by:
            return df
        out = df.sort_values(
            by=by,
            ascending=self.ascending,
            na_position=self.NA_POSITION,
            kind=self.KIND,
        )
        return out.reset_index(drop=True)
