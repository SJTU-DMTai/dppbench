import numpy as np
import pandas as pd
from ..base_op import TabularOp


class TransformPower(TabularOp):
    """Apply log/sqrt/power/quantile transformations."""

    FIT_ON_TRAIN_ONLY = True

    def __init__(self, cols=None, method="yeo-johnson", base=None, offset=1.0,
                 output_cols=None, standardize=True,
                 output_distribution="uniform", n_quantiles=1000,
                 random_state=42):
        super().__init__(name="TransformPower")
        if method not in ("log", "sqrt", "box-cox", "yeo-johnson", "quantile"):
            raise ValueError("method must be log/sqrt/box-cox/yeo-johnson/quantile")
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.method = method
        self.base = base
        self.offset = offset
        self.output_cols = output_cols if (
            output_cols is None or isinstance(output_cols, list)
        ) else [output_cols]
        self.standardize = bool(standardize)
        self.output_distribution = output_distribution
        self.n_quantiles = int(n_quantiles)
        self.random_state = random_state
        self.cols_ = []
        self.transformer_ = None
        self.shift_ = None

    def get_op_description(self):
        description = """Operator name: TransformPower

Function description:
Transform numeric distributions via log, sqrt,
Box-Cox, Yeo-Johnson, or quantile transformation.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] or None — Numeric columns. None = all numeric.
method : str — log/sqrt/box-cox/yeo-johnson/quantile.
base, offset, output_cols — log/sqrt options.
standardize — sklearn PowerTransformer option.
output_distribution, n_quantiles — quantile options.

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
  - op: TransformPower
    target: both
    params:
      cols: [amount]
      method: log
      offset: 1.0
"""
        return description.strip()

    def _target_col(self, i, col):
        return self.output_cols[i] if self.output_cols else col

    def _simple(self, df, cols):
        for i, col in enumerate(cols):
            values = pd.to_numeric(df[col], errors="coerce")
            if self.method == "sqrt":
                out = np.sqrt(np.maximum(values + self.offset, 0))
            elif self.base is None:
                out = np.log(values + self.offset)
            elif self.base == 2:
                out = np.log2(values + self.offset)
            elif self.base == 10:
                out = np.log10(values + self.offset)
            else:
                out = np.log(values + self.offset) / np.log(self.base)
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
        if self.method == "box-cox":
            if self.shift_ is None:
                self.shift_ = sub.min(axis=0).clip(upper=0).abs() + 1e-6
            sub = sub + self.shift_
        try:
            if self.method == "quantile":
                from sklearn.preprocessing import QuantileTransformer
                if self.transformer_ is None:
                    self.transformer_ = QuantileTransformer(
                        n_quantiles=min(self.n_quantiles, max(2, len(df))),
                        output_distribution=self.output_distribution,
                        random_state=self.random_state,
                    ).fit(sub)
            else:
                from sklearn.preprocessing import PowerTransformer
                if self.transformer_ is None:
                    self.transformer_ = PowerTransformer(
                        method=self.method,
                        standardize=self.standardize,
                    ).fit(sub)
            df[cols] = self.transformer_.transform(sub)
        except Exception as exc:
            print(f"  [TransformPower] sklearn transform unavailable: {exc}")
        return df
