from ..base_op import TabularOp


class SplitColumn(TabularOp):
    """Split one string column into multiple columns."""

    def __init__(self, col, output_cols=None, sep=None, regex=False, maxsplit=-1,
                 expand=True, drop_original=False):
        super().__init__(name="SplitColumn")
        self.col = col
        self.output_cols = output_cols
        self.sep = sep
        self.regex = bool(regex)
        self.maxsplit = int(maxsplit)
        self.expand = bool(expand)
        self.drop_original = bool(drop_original)

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
maxsplit : int — Maximum number of splits. Default -1.
expand : bool — Expand into columns. Default True.
drop_original : bool — Drop source column after split.

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
            n=self.maxsplit,
            expand=self.expand,
            regex=self.regex,
        )
        if not self.expand:
            df[self.output_cols[0] if self.output_cols else f"{self.col}_parts"] = parts
        else:
            if self.output_cols is None:
                output_cols = [f"{self.col}_{i}" for i in range(parts.shape[1])]
            else:
                output_cols = list(self.output_cols)
            for i, out_col in enumerate(output_cols):
                df[out_col] = parts.iloc[:, i] if i < parts.shape[1] else None
        if self.drop_original:
            df = df.drop(columns=[self.col])
        return df
