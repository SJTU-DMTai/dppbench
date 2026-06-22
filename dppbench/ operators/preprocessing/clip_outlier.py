import pandas as pd
from ..base_op import TabularOp


class ClipOutlier(TabularOp):
    """Winsorize numeric columns by quantile or explicit bounds."""

    FIT_ON_TRAIN_ONLY = True

    def __init__(self, cols=None, lower_quantile=0.01, upper_quantile=0.99,
                 lower=None, upper=None):
        super().__init__(name="ClipOutlier")
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower = lower
        self.upper = upper
        self.bounds_ = {}
        self.fitted_ = False

    def get_op_description(self):
        description = """Operator name: ClipOutlier

Function description:
Cap extreme numeric values using quantile thresholds
(winsorization) or explicit lower/upper bounds.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] or None — Numeric columns. None = all numeric.
lower_quantile, upper_quantile : float — Learned quantile bounds.
lower, upper : float or None — Explicit bounds overriding quantiles.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'amount': [1, 10, 999]})
>>> op = ClipOutlier(cols=['amount'], lower=0, upper=100)
>>> op.transform(df)
   amount
0       1
1      10
2     100

Example YAML:
  - op: ClipOutlier
    target: train
    params:
      cols: [amount]
      lower: 0
      upper: 100
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
                lo = self.lower if self.lower is not None else values.quantile(self.lower_quantile)
                hi = self.upper if self.upper is not None else values.quantile(self.upper_quantile)
                self.bounds_[col] = (lo, hi)
            self.fitted_ = True
        for col, (lo, hi) in self.bounds_.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)
        return df
