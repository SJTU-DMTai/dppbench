import numpy as np
import pandas as pd
from ..base_op import TabularOp


class ScaleFeature(TabularOp):
    """Scale numeric features with standard/minmax/maxabs/robust/L2 methods."""

    FIT_ON_TRAIN_ONLY = True

    QUANTILE_LO = 25.0
    QUANTILE_HI = 75.0
    EPS = 1e-6
    OUT_DTYPE = "float32"

    def __init__(self, cols=None, method="standard", pattern=None,
                 feature_range=(0.0, 1.0)):
        super().__init__(name="ScaleFeature")
        if method not in ("standard", "minmax", "maxabs", "robust", "l2"):
            raise ValueError("method must be standard/minmax/maxabs/robust/l2")
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.method = method
        self.pattern = pattern
        self.feature_range = tuple(feature_range)
        self.cols_ = []
        self.params_ = {}
        self.fitted_ = False

    def get_op_description(self):
        description = """Operator name: ScaleFeature

Function description:
Scale numeric columns via standard, minmax, maxabs,
robust, or row-wise L2 normalization.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] or None — Explicit numeric columns.
method : str — standard/minmax/maxabs/robust/l2.
pattern : str or None — Substring selector when cols is None.
feature_range : pair — Minmax output range (default (0.0, 1.0)).

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'amount': [10.0, 20.0, 30.0]})
>>> op = ScaleFeature(cols=['amount'], method='standard')
>>> op.transform(df)
     amount
0 -1.224745
1  0.000000
2  1.224745

Example YAML:
  - op: ScaleFeature
    target: train
    params:
      cols: [amount]
      method: standard
"""
        return description.strip()

    def _select_cols(self, df):
        if self.cols is not None:
            return [c for c in self.cols if c in df.columns]
        if self.pattern is not None:
            return [c for c in df.columns if self.pattern in c]
        return []

    def _fit_col(self, col, values):
        if self.method == "standard":
            self.params_[col] = (values.mean(), values.std())
        elif self.method == "minmax":
            self.params_[col] = (values.min(), values.max())
        elif self.method == "maxabs":
            self.params_[col] = values.abs().max()
        elif self.method == "robust":
            med = values.median()
            scale = values.quantile(self.QUANTILE_HI / 100.0) - values.quantile(self.QUANTILE_LO / 100.0)
            self.params_[col] = (med, scale if pd.notna(scale) and scale > 0 else 1.0)

    def transform(self, df):
        df = df.copy()
        if not self.fitted_:
            self.cols_ = self._select_cols(df)
        cols = [c for c in self.cols_ if c in df.columns]
        if not cols:
            return df
        if self.method == "l2":
            sub = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            norm = np.sqrt((sub ** 2).sum(axis=1)).replace(0, 1.0)
            df[cols] = sub.div(norm, axis=0).astype(self.OUT_DTYPE)
            self.fitted_ = True
            return df
        for col in cols:
            values = pd.to_numeric(df[col], errors="coerce")
            if not self.fitted_:
                self._fit_col(col, values)
            if self.method == "standard":
                mean, std = self.params_.get(col, (0.0, 1.0))
                std = std if pd.notna(std) and std >= self.EPS else 1.0
                out = (values - mean) / std
            elif self.method == "minmax":
                mn, mx = self.params_.get(col, (0.0, 0.0))
                lo, hi = self.feature_range
                out = pd.Series(lo, index=df.index) if mx == mn else (
                    (values - mn) / (mx - mn) * (hi - lo) + lo
                )
            elif self.method == "maxabs":
                mx = self.params_.get(col, 1.0)
                mx = mx if pd.notna(mx) and mx >= self.EPS else 1.0
                out = values / mx
            else:
                med, scale = self.params_.get(col, (0.0, 1.0))
                out = (values - med) / scale
            df[col] = out.astype(self.OUT_DTYPE)
        self.fitted_ = True
        return df
