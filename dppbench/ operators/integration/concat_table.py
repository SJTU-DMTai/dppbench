import pandas as pd
from ..base_op import TabularOp


class ConcatTable(TabularOp):
    """Vertically (or horizontally) concatenate the main table with one or
    more same-schema auxiliary tables. Distinct from ``JoinTable`` which
    performs key-based horizontal merges with matching / aggregation.
    """

    def __init__(self, other_dfs, axis="vertical", dedup=False, fill_missing=True):
        super().__init__(name="ConcatTable")
        if isinstance(other_dfs, pd.DataFrame):
            self.other_dfs = [other_dfs]
        elif isinstance(other_dfs, list):
            self.other_dfs = other_dfs
        elif other_dfs is None:
            self.other_dfs = []
        else:
            raise ValueError("other_dfs must be a DataFrame or list of DataFrames")
        if axis not in ("vertical", "horizontal"):
            raise ValueError("axis must be 'vertical' or 'horizontal'")
        self.axis = axis
        self.dedup = bool(dedup)
        self.fill_missing = bool(fill_missing)

    def get_op_description(self):
        description = """Operator name: ConcatTable

Function description:
Stack the current table with one or more auxiliary
tables that share the same schema. Vertical = row-wise append (UNION),
horizontal = column-wise side-by-side. No key matching, no aggregation
(use ``JoinTable`` for that).

Input:
df : pd.DataFrame — Main table to be stacked with other_dfs.

Parameters:
other_dfs : DataFrame or list[DataFrame] — Tables to concat ($name in YAML).
axis : str — 'vertical' (default, append rows) or 'horizontal'.
dedup : bool — Drop exact duplicate rows after a vertical concat. Default False.
fill_missing : bool — Vertical: keep all columns and fill missing with NaN
(outer join). If False, use the column intersection (inner). Default True.

Output:
pd.DataFrame — Concatenated table.

Example:
>>> df = pd.DataFrame({'a': [1, 2], 'b': [10, 20]})
>>> other = pd.DataFrame({'a': [3, 4], 'b': [30, 40]})
>>> op = ConcatTable(other_dfs=other, axis='vertical')
>>> op.transform(df)
   a   b
0  1  10
1  2  20
2  3  30
3  4  40

Example YAML:
  - op: ConcatTable
    target: train
    params:
      other_dfs: [$year2, $year3, $year4]
      axis: vertical
"""
        return description.strip()

    def transform(self, df):
        all_frames = [df] + list(self.other_dfs)
        if self.axis == "vertical":
            join = "outer" if self.fill_missing else "inner"
            out = pd.concat(all_frames, axis=0, join=join, ignore_index=True)
            if self.dedup:
                out = out.drop_duplicates().reset_index(drop=True)
        else:
            out = pd.concat(all_frames, axis=1)
        return out
