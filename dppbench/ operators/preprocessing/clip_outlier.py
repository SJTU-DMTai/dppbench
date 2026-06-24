import pandas as pd
from ..base_op import TabularOp


class ClipOutlier(TabularOp):
    """Winsorize numeric columns by quantile thresholds."""

    FIT_ON_TRAIN_ONLY = True

    def __init__(self, cols=None, lower_quantile=0.01, upper_quantile=0.99):
        super().__init__(name="ClipOutlier")
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self.bounds_ = {}
        self.fitted_ = False

    def get_op_description(self):
        description = """Operator name: ClipOutlier

Function description:
Cap extreme numeric values using quantile thresholds (winsorization).

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] or None — Numeric columns. None = all numeric.
lower_quantile : float — Lower quantile bound (default 0.01).
upper_quantile : float — Upper quantile bound (default 0.99).

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'amount': [1, 10, 999]})
>>> op = ClipOutlier(cols=['amount'], lower_quantile=0.0, upper_quantile=0.5)
>>> op.transform(df)
   amount
0     1.0
1    10.0
2    10.0

Example YAML:
  - op: ClipOutlier
    target: train
    params:
      cols: [amount]
      lower_quantile: 0.01
      upper_quantile: 0.99
"""
        return description.strip()

    def transform(self, df):
        df = df.copy()
        if not self.fitted_:
            cols = (
                df.select_dtypes(include=["number"]).columns.tolist()
                if self.cols is None else [c for c in self.cols if c in df.columns]
            )
            for col in cols:
                values = pd.to_numeric(df[col], errors="coerce")
                lo = values.quantile(self.lower_quantile)
                hi = values.quantile(self.upper_quantile)
                self.bounds_[col] = (lo, hi)
            self.fitted_ = True
        for col, (lo, hi) in self.bounds_.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)
        return df
