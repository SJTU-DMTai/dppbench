import pandas as pd
from ..base_op import TabularOp


class ParseNumber(TabularOp):
    """Parse string numeric columns into numeric dtype."""

    ERRORS = "coerce"

    def __init__(self, cols, dtype="float64"):
        super().__init__(name="ParseNumber")
        self.cols = cols if isinstance(cols, list) else [cols]
        self.dtype = dtype

    def get_op_description(self):
        description = """Operator name: ParseNumber

Function description:
Convert string-valued numeric columns into numeric
dtype. Non-parseable values are coerced to NaN.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : str or list[str] — Source columns.
dtype : str — Output dtype, e.g. float64/int64/Int64. Default float64.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'amount': ['1200.50', 'bad']})
>>> op = ParseNumber(cols='amount')
>>> op.transform(df)
   amount
0  1200.5
1     NaN

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: ParseNumber
    prev:
    - s0
    params:
      cols:
      - amount
  train:
    prev:
    - o1
"""
        return description.strip()

    def transform(self, df):
        df = df.copy()
        for col in self.cols:
            if col not in df.columns:
                continue
            parsed = pd.to_numeric(df[col], errors=self.ERRORS)
            try:
                df[col] = parsed.astype(self.dtype)
            except Exception:
                df[col] = parsed
        return df
