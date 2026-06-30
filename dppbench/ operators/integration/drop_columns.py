from ..base_op import TabularOp


class DropColumns(TabularOp):
    """Drop columns by explicit names."""

    def __init__(self, cols):
        super().__init__(name="DropColumns")
        if cols is None:
            raise ValueError("cols must be a string or list of strings")
        self.cols = cols if isinstance(cols, list) else [cols]

    def get_op_description(self):
        description = """Operator name: DropColumns

Function description:
Drop columns by explicit names. Missing columns are ignored.
Cell values, dtypes, and row order of retained columns are preserved.

Input:
df : pd.DataFrame - Input table accepted by transform.

Parameters:
cols : str or list[str] - Columns to drop if present.

Output:
pd.DataFrame - Transformed table after dropping existing columns.

Example:
>>> df = pd.DataFrame({'id': [1, 2], 'amount': [10, 20], 'target': [0, 1]})
>>> op = DropColumns(cols=['id'])
>>> op.transform(df)
   amount  target
0      10       0
1      20       1

Example YAML:
  - op: DropColumns
    target: both
    params:
      cols: [id]
"""
        return description.strip()

    def transform(self, df):
        existing = [c for c in self.cols if c in df.columns]
        return df.drop(columns=existing)
