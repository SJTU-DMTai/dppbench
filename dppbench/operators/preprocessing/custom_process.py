import pandas as pd
from ..base_op import TabularOp
from ..custom_op import CustomOp


class CustomProcess(TabularOp):
    """Custom preprocessing operator plus common utility modes."""

    FIT_ON_TRAIN_ONLY = True

    def __init__(self, code=None, entry="pipeline", func=None, mode="code",
                 cols=None, threshold=0.8):
        super().__init__(name="CustomProcess")
        if mode not in ("code", "drop_high_null"):
            raise ValueError("mode must be code/drop_high_null")
        self.code = code
        self.entry = entry or "pipeline"
        self.func = func
        self.mode = mode
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.threshold = float(threshold)
        self.drop_cols_ = []
        self.fitted_ = False
        self._custom = CustomOp(code=code, entry=self.entry) if code else None

    def get_op_description(self):
        description = """Operator name: CustomProcess

Function description:
User-defined preprocessing. Built-in modes cover high-null column filtering.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
mode : str — code/drop_high_null.
threshold : float — Null ratio threshold for drop_high_null.
code, entry, func — Custom execution inputs.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: CustomProcess
    prev:
    - s0
    params:
      mode: drop_high_null
      threshold: 0.8
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
                raise TypeError("CustomProcess func must return a pandas DataFrame")
            return result
        if self.mode == "drop_high_null":
            if not self.fitted_:
                ratios = df.isna().mean()
                self.drop_cols_ = ratios[ratios > self.threshold].index.tolist()
                self.fitted_ = True
            existing = [c for c in self.drop_cols_ if c in df.columns]
            return df.drop(columns=existing)
        if self._custom is not None:
            return self._custom.transform(df)
        return df
