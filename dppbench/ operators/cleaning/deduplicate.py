import pandas as pd
from ..base_op import RecOp, TabularOp


class Deduplicate(RecOp, TabularOp):
    APPLIES_TO_STD_TEST = False

    KEEP_OPTIONS = ("first", "last", False)

    def __init__(self, subset=None, keep="first"):
        super().__init__(name="Deduplicate")

        if keep not in self.KEEP_OPTIONS:
            raise ValueError(
                f"keep must be one of {self.KEEP_OPTIONS}, got '{keep}'"
            )

        self.subset = subset
        self.keep = keep

    def get_op_description(self):
        description = """Operator name: Deduplicate

Function description:
Remove duplicate rows from a pandas DataFrame. Duplicates are identified
based on all columns or a specified subset of columns. The 'keep' parameter controls which
occurrence of a duplicate row is retained.

Input:
df : pd.DataFrame — A DataFrame that may contain duplicate rows.

Parameters:
subset : list[str] or None — Column names to consider when identifying duplicates. If None, all columns are used. Default: None.
keep : str or False — Which duplicate to keep. 'first' keeps the first occurrence, 'last' keeps the last occurrence, False drops all duplicates. Default: 'first'.

Output:
pd.DataFrame — A DataFrame with duplicate rows removed. The index is reset.

Example:
>>> df = pd.DataFrame({'id': [1, 2, 2, 3], 'name': ['Alice', 'Bob', 'Bob', 'Carol']})
>>> op = Deduplicate(subset=['id', 'name'], keep='first')
>>> op.transform(df)
   id   name
0   1  Alice
1   2    Bob
2   3  Carol

Example YAML:
  - op: Deduplicate
    target: train
    params:
      subset: [user_id, item_id]
      keep: first
"""
        return description.strip()

    def transform(self, df):
        """
        Remove duplicate rows from the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with potential duplicate rows.

        Returns
        -------
        pd.DataFrame
            DataFrame with duplicates removed and index reset.
        """
        return df.drop_duplicates(subset=self.subset, keep=self.keep).reset_index(drop=True)
