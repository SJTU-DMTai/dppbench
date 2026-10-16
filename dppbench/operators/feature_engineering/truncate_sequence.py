from ..base_op import RecOp


class TruncateSequence(RecOp):
    """Truncate sequence/list features to a fixed maximum length."""

    def __init__(self, cols, max_len=50, keep="last"):
        super().__init__(name="TruncateSequence")
        self.cols = cols if isinstance(cols, list) else [cols]
        self.max_len = int(max_len)
        if keep not in ("first", "last"):
            raise ValueError("keep must be 'first' or 'last'")
        self.keep = keep

    def get_op_description(self):
        description = """Operator name: TruncateSequence

Function description:
Truncate list-valued sequence features to a fixed length.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'user_id': [1, 2], 'item_seq': [[10, 11, 12, 13], [20]]})
>>> op = TruncateSequence(cols='item_seq', max_len=2, keep='last')
>>> op.transform(df)
   user_id  item_seq
0        1  [12, 13]
1        2      [20]

Example YAML:
dag:
  sources:
  - id: s0
    table: interaction
  ops:
  - id: o1
    op: TruncateSequence
    prev:
    - s0
    params:
      cols: item_seq
      max_len: 50
      keep: last
  train:
    prev:
    - o1
"""
        return description.strip()

    def _truncate(self, value):
        if not isinstance(value, (list, tuple)):
            return value
        return list(value[: self.max_len]) if self.keep == "first" else list(value[-self.max_len:])

    def transform(self, df):
        df = df.copy()
        for col in self.cols:
            if col in df.columns:
                df[col] = df[col].apply(self._truncate)
        return df
