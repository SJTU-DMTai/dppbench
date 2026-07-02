import numpy as np
import pandas as pd

from ..base_op import TabularOp


_VALID_ACTIONS = ("delete", "repair")
_VALID_REPAIR_METHODS = ("clip", "set_missing")


class HandleOutlier(TabularOp):
    """Detect numeric outliers and either delete or repair them."""

    FIT_ON_TRAIN_ONLY = True
    APPLIES_TO_STD_TEST = False

    def __init__(self, cols=None, method="iqr", threshold=3.0,
                 action="delete", repair_method="clip", flag_col=None):
        super().__init__(name="HandleOutlier")
        if method not in ("iqr", "zscore"):
            raise ValueError("method must be iqr/zscore")
        if action not in _VALID_ACTIONS:
            raise ValueError("action must be delete/repair")
        if repair_method not in _VALID_REPAIR_METHODS:
            raise ValueError("repair_method must be clip/set_missing")
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.method = method
        self.threshold = float(threshold)
        self.action = action
        self.repair_method = repair_method
        self.flag_col = flag_col
        self.cols_ = []
        self.bounds_ = {}
        self.fitted_ = False

    def _select_cols(self, df):
        if self.cols is None:
            return df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in self.cols if c in df.columns]

    def _fit_bounds(self, df):
        for col in self.cols_:
            values = pd.to_numeric(df[col], errors="coerce")
            if self.method == "zscore":
                mean, std = values.mean(), values.std()
                std = std if std and not pd.isna(std) else 1.0
                self.bounds_[col] = (mean - self.threshold * std, mean + self.threshold * std)
            else:
                q1, q3 = values.quantile(0.25), values.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0 or pd.isna(iqr):
                    self.bounds_[col] = (values.min(), values.max())
                else:
                    self.bounds_[col] = (q1 - self.threshold * iqr, q3 + self.threshold * iqr)

    def _fit(self, df):
        self.cols_ = self._select_cols(df)
        self._fit_bounds(df)
        self.fitted_ = True

    def _score(self, df):
        if not self.cols_:
            return pd.Series(False, index=df.index), {}
        per_col_mask = {}
        flag = pd.Series(False, index=df.index)
        for col in [c for c in self.cols_ if c in df.columns]:
            lo, hi = self.bounds_.get(col, (None, None))
            if lo is None or hi is None:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            col_flag = ((values < lo) | (values > hi)).fillna(False)
            flag = flag | col_flag
            per_col_mask[col] = col_flag
        return flag, per_col_mask

    def _repair(self, df, per_col_mask):
        for col, mask in per_col_mask.items():
            if col not in df.columns or not mask.any():
                continue
            if self.repair_method == "clip":
                lo, hi = self.bounds_.get(col, (None, None))
                if lo is not None or hi is not None:
                    df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=lo, upper=hi)
            elif self.repair_method == "set_missing":
                df.loc[mask, col] = np.nan
        return df

    def transform(self, df):
        df = df.copy()
        if not self.fitted_:
            self._fit(df)
        flag, per_col_mask = self._score(df)
        if self.action == "delete":
            return df.loc[~flag.values].reset_index(drop=True)
        if self.flag_col:
            df[self.flag_col] = flag.astype(int).values
        return self._repair(df, per_col_mask)

    def get_op_description(self):
        description = """Operator name: HandleOutlier

Function description:
Detect numeric outliers (IQR or z-score) and either delete the flagged rows
or repair the offending values via clip-to-bounds or set-to-missing.

Input:
df : pd.DataFrame — Table containing numeric feature columns.

Parameters:
cols : list[str] or None — Numeric columns to inspect. If None, use all numeric columns.
method : str — Detection method: "iqr" or "zscore".
threshold : float — IQR multiplier or z-score cutoff.
action : str — "delete" or "repair".
repair_method : str — "clip" or "set_missing" (only used when action='repair').
flag_col : str or None — Optional 0/1 marker column written when action='repair'.

Output:
pd.DataFrame — For action='delete', the original schema with outlier rows
removed. For action='repair', values are clipped or set to NaN.

Example:
>>> df = pd.DataFrame({"x": [1, 2, 100]})
>>> HandleOutlier(cols=["x"], method="iqr", threshold=1.5, action="delete").transform(df)
   x
0  1
1  2

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: HandleOutlier
    prev:
    - s0
    params:
      cols:
      - x
      method: iqr
      threshold: 3.0
      action: delete
  train:
    prev:
    - o1
"""
        return description.strip()
