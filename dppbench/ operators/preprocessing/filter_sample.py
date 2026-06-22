from ..base_op import TabularOp


class FilterSample(TabularOp):
    """Filter rows by condition or by missing-value subset."""

    APPLIES_TO_STD_TEST = False

    def __init__(self, func=None, subset=None, query=None):
        super().__init__(name="FilterSample")
        self.func = func
        self.subset = subset
        self.query = query

    def get_op_description(self):
        description = """Operator name: FilterSample

Function description:
Keep rows satisfying a custom condition. If no func or
query is supplied, drops rows with NA in subset.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
func : callable or None — Row predicate returning True to keep.
subset : list[str] or None — Columns checked for NA.
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
  - op: FilterSample
    target: train
    params:
      subset: [age]
"""
        return description.strip()

    def transform(self, df):
        if self.query:
            return df.query(self.query).reset_index(drop=True)
        if self.func is not None:
            mask = df.apply(self.func, axis=1)
            return df[mask].reset_index(drop=True)
        return df.dropna(subset=self.subset).reset_index(drop=True)
