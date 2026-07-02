import pandas as pd
from ..base_op import TabularOp


class CustomFE(TabularOp):
    """Custom feature engineering operator."""

    def __init__(self, func=None):
        super().__init__(name="CustomFE")
        self.func = func

    def get_op_description(self):
        description = """Operator name: CustomFE

Function description:
Apply a user-provided feature engineering function to a pandas DataFrame. If no
function is provided, the operator returns a copy of the input DataFrame.

Input:
df : pd.DataFrame — Input table passed to the custom feature function.

Parameters:
func : callable or None — Function that receives a DataFrame and returns a DataFrame. Default: None.

Output:
pd.DataFrame — DataFrame returned by func, or an unchanged copy of df when func is None.

Example:
>>> df = pd.DataFrame({'a': [2, 4], 'b': [1, 2]})
>>> def add_ratio(df):
...     df = df.copy()
...     df['ratio'] = df['a'] / df['b']
...     return df
>>> op = CustomFE(func=add_ratio)
>>> op.transform(df)
   a  b  ratio
0  2  1    2.0
1  4  2    2.0

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: CustomFE
    prev:
    - s0
    params:
      func: null
  train:
    prev:
    - o1
"""
        return description.strip()

    def transform(self, df):
        df = df.copy()
        if self.func is not None:
            result = self.func(df)
            if not isinstance(result, pd.DataFrame):
                raise TypeError("CustomFE func must return a pandas DataFrame")
            return result
        return df
