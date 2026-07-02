from ..base_op import TabularOp


class FilterSample(TabularOp):
    """Filter rows by query expression or NA-subset drop."""

    APPLIES_TO_STD_TEST = False

    def __init__(self, subset=None, query=None):
        super().__init__(name="FilterSample")
        self.subset = subset
        self.query = query

    def get_op_description(self):
        description = """Operator name: FilterSample

Function description:
Keep rows satisfying a pandas query expression. If no query is supplied,
drops rows with NA in the given subset of columns.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
subset : list[str] or None — Columns checked for NA when query is None.
query : str or None — pandas query expression.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'age': [20, None, 40], 'label': [1, 0, 1]})
>>> op = FilterSample(subset=['age'])
>>> op.transform(df)
    age  label
0  20.0      1
1  40.0      1

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: FilterSample
    prev:
    - s0
    params:
      subset:
      - age
  train:
    prev:
    - o1
"""
        return description.strip()

    def transform(self, df):
        if self.query:
            return df.query(self.query).reset_index(drop=True)
        return df.dropna(subset=self.subset).reset_index(drop=True)
