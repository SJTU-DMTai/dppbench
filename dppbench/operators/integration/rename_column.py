from ..base_op import TabularOp


class RenameColumn(TabularOp):
    """Rename columns by an explicit mapping."""

    def __init__(self, rename_map):
        super().__init__(name="RenameColumn")
        if not isinstance(rename_map, dict):
            raise ValueError("rename_map must be dict {old_name: new_name}")
        self.rename_map = rename_map

    def get_op_description(self):
        description = """Operator name: RenameColumn

Function description:
Rename columns by an explicit mapping. Cell values,
dtypes, and row order are preserved.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
rename_map : dict[str, str] — Mapping {old_name: new_name}.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'TransactionAmt': [10, 20], 'isFraud': [0, 1]})
>>> op = RenameColumn(rename_map={'TransactionAmt': 'amount', 'isFraud': 'label'})
>>> op.transform(df)
   amount  label
0      10      0
1      20      1

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: RenameColumn
    prev:
    - s0
    params:
      rename_map:
        TransactionAmt: amount
        isFraud: label
  train:
    prev:
    - o1
"""
        return description.strip()

    def transform(self, df):
        existing = {k: v for k, v in self.rename_map.items() if k in df.columns}
        return df.rename(columns=existing)
