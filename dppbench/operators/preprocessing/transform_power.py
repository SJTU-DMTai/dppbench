import numpy as np
import pandas as pd
from ..base_op import TabularOp


class TransformPower(TabularOp):
    """Apply log/sqrt/quantile transformations."""

    FIT_ON_TRAIN_ONLY = True
    N_QUANTILES = 1000

    def __init__(self, cols=None, method="log", offset=1.0,
                 output_cols=None, output_distribution="uniform"):
        super().__init__(name="TransformPower")
        if method not in ("log", "sqrt", "quantile"):
            raise ValueError("method must be log/sqrt/quantile")
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.method = method
        self.offset = float(offset)
        self.output_cols = output_cols if (
            output_cols is None or isinstance(output_cols, list)
        ) else [output_cols]
        self.output_distribution = output_distribution
        self.cols_ = []
        self.transformer_ = None

    def get_op_description(self):
        description = """Operator name: TransformPower

Function description:
Transform numeric distributions via natural log, square root, or
sklearn QuantileTransformer.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] or None — Numeric columns. None = all numeric.
method : str — log/sqrt/quantile.
offset : float — Additive offset for log/sqrt to handle zeros (default 1.0).
output_cols : list[str] or None — Names for log/sqrt outputs (in-place if None).
output_distribution : str — uniform or normal for quantile mode.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'amount': [0, 3, 8]})
>>> op = TransformPower(cols=['amount'], method='log', offset=1.0)
>>> op.transform(df)
     amount
0  0.000000
1  1.386294
2  2.197225

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: TransformPower
    prev:
    - s0
    params:
      cols:
      - amount
      method: log
      offset: 1.0
  train:
    prev:
    - o1
"""
        return description.strip()

    def _target_col(self, i, col):
        return self.output_cols[i] if self.output_cols else col

    def _simple(self, df, cols):
        for i, col in enumerate(cols):
            values = pd.to_numeric(df[col], errors="coerce")
            if self.method == "sqrt":
                out = np.sqrt(np.maximum(values + self.offset, 0))
            else:
                out = np.log(values + self.offset)
            df[self._target_col(i, col)] = out
        return df

    def transform(self, df):
        df = df.copy()
        if self.transformer_ is None and not self.cols_:
            self.cols_ = (
                df.select_dtypes(include=[np.number]).columns.tolist()
                if self.cols is None else [c for c in self.cols if c in df.columns]
            )
        cols = [c for c in self.cols_ if c in df.columns]
        if not cols:
            return df
        if self.method in ("log", "sqrt"):
            return self._simple(df, cols)
        sub = df[cols].astype(float).fillna(0.0)
        try:
            from sklearn.preprocessing import QuantileTransformer
            if self.transformer_ is None:
                self.transformer_ = QuantileTransformer(
                    n_quantiles=min(self.N_QUANTILES, max(2, len(df))),
                    output_distribution=self.output_distribution,
                ).fit(sub)
            df[cols] = self.transformer_.transform(sub)
        except Exception as exc:
            print(f"  [TransformPower] sklearn quantile unavailable: {exc}")
        return df
