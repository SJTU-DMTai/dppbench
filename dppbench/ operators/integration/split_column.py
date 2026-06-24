from ..base_op import TabularOp


class SplitColumn(TabularOp):
    """Split one string column into multiple columns."""

    MAXSPLIT = -1

    def __init__(self, col, output_cols=None, sep=None, regex=False):
        super().__init__(name="SplitColumn")
        self.col = col
        self.output_cols = output_cols
        self.sep = sep
        self.regex = bool(regex)

    def get_op_description(self):
        description = """Operator name: SplitColumn

Function description:
Split a compound string column into multiple columns,
for example full_name -> first_name / last_name.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
col : str — Source column.
output_cols : list[str] or None — Names for expanded parts.
sep : str or None — Separator / regex pattern. None means whitespace.
regex : bool — Treat sep as regex. Default False.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'full_name': ['Alice Smith', 'Bob Lee']})
>>> op = SplitColumn(col='full_name', sep=' ', output_cols=['first', 'last'])
>>> op.transform(df)
     full_name  first   last
0  Alice Smith  Alice  Smith
1      Bob Lee    Bob    Lee

Example YAML:
  - op: SplitColumn
    target: both
    params:
      col: full_name
      sep: ' '
      output_cols: [first, last]
"""
        return description.strip()

    def transform(self, df):
        if self.col not in df.columns:
            return df
        df = df.copy()
        parts = df[self.col].astype("string").str.split(
            pat=self.sep,
            n=self.MAXSPLIT,
            expand=True,
            regex=self.regex,
        )
        if self.output_cols is None:
            output_cols = [f"{self.col}_{i}" for i in range(parts.shape[1])]
        else:
            output_cols = list(self.output_cols)
        for i, out_col in enumerate(output_cols):
            df[out_col] = parts.iloc[:, i] if i < parts.shape[1] else None
        return df
