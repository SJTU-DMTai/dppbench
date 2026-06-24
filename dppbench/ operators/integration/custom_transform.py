import pandas as pd
from ..base_op import TabularOp
from ..custom_op import CustomOp


class CustomTransform(TabularOp):
    """Run a sandboxed user-defined transformation function."""

    ENTRY = "pipeline"

    def __init__(self, code=None, func=None):
        super().__init__(name="CustomTransform")
        self.code = code
        self.func = func
        self._custom = CustomOp(code=code, entry=self.ENTRY) if code else None

    def get_op_description(self):
        description = """Operator name: CustomTransform

Function description:
Apply a user-defined transformation to a DataFrame.
Accepts either a callable ``func`` or sandboxed Python ``code`` defining
``def pipeline(df): return df``. Used for integration-stage custom logic.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
code : str or None — Sandboxed Python source defining a ``pipeline(df)`` entry function.
func : callable or None — Direct callable for programmatic use.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'amount': [10, 20]})
>>> code = 'def pipeline(df):
    df["amount2"] = df["amount"] * 2
    return df'
>>> op = CustomTransform(code=code)
>>> op.transform(df)
   amount  amount2
0      10       20
1      20       40

Example YAML:
  - op: CustomTransform
    target: both
    params:
      code: |
        def pipeline(df):
            df["amount2"] = df["amount"] * 2
            return df
"""
        return description.strip()

    def transform(self, df):
        if self.func is not None:
            result = self.func(df.copy())
            if not isinstance(result, pd.DataFrame):
                raise TypeError("CustomTransform func must return a pandas DataFrame")
            return result
        if self._custom is not None:
            return self._custom.transform(df)
        return df
