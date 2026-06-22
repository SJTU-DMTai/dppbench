import numpy as np
import pandas as pd

from ..base_op import TabularOp


_VALID_ACTIONS = ("flag", "delete", "repair")
_VALID_REPAIR_METHODS = ("clip", "median", "set_missing", "winsorize")


class HandleOutlier(TabularOp):
    """Detect numeric outliers and either delete or repair them."""

    FIT_ON_TRAIN_ONLY = True
    APPLIES_TO_STD_TEST = False

    def __init__(self, cols=None, method="iqr", threshold=3.0,
                 action="delete", repair_method="clip",
                 flag_col="is_outlier", score_col=None,
                 winsorize_lower=0.01, winsorize_upper=0.99,
                 random_state=42):
        super().__init__(name="HandleOutlier")
        if method not in ("iqr", "zscore", "isolation_forest"):
            raise ValueError("method must be iqr/zscore/isolation_forest")
        if action not in _VALID_ACTIONS:
            raise ValueError("action must be flag/delete/repair")
        if repair_method not in _VALID_REPAIR_METHODS:
            raise ValueError("repair_method must be clip/median/set_missing/winsorize")
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.method = method
        self.threshold = float(threshold)
        self.action = action
        self.repair_method = repair_method
        self.flag_col = flag_col
        self.score_col = score_col
        self.winsorize_lower = float(winsorize_lower)
        self.winsorize_upper = float(winsorize_upper)
        self.random_state = random_state
        self.cols_ = []
        self.bounds_ = {}
        self.medians_ = {}
        self.winsor_bounds_ = {}
        self.detector_ = None
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
            self.medians_[col] = values.median()
            try:
                self.winsor_bounds_[col] = (
                    values.quantile(self.winsorize_lower),
                    values.quantile(self.winsorize_upper),
                )
            except Exception:
                self.winsor_bounds_[col] = (values.min(), values.max())

    def _fit(self, df):
        self.cols_ = self._select_cols(df)
        if self.method == "isolation_forest" and self.cols_:
            try:
                from sklearn.ensemble import IsolationForest
                x = df[self.cols_].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                self.detector_ = IsolationForest(
                    random_state=self.random_state,
                    contamination="auto",
                ).fit(x)
            except Exception as exc:
                print(f"  [HandleOutlier] IsolationForest unavailable, fallback to IQR: {exc}")
                self.method = "iqr"
        self._fit_bounds(df)
        self.fitted_ = True

    def _score(self, df):
        if not self.cols_:
            return pd.Series(False, index=df.index), pd.Series(0.0, index=df.index), {}
        per_col_mask = {}
        if self.method == "isolation_forest" and self.detector_ is not None:
            x = df[self.cols_].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            pred = self.detector_.predict(x)
            score = pd.Series(-self.detector_.score_samples(x), index=df.index)
            mask = pd.Series(pred == -1, index=df.index)
            for col in self.cols_:
                per_col_mask[col] = mask
            return mask, score, per_col_mask

        flag = pd.Series(False, index=df.index)
        score = pd.Series(0.0, index=df.index)
        for col in [c for c in self.cols_ if c in df.columns]:
            lo, hi = self.bounds_.get(col, (None, None))
            if lo is None or hi is None:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            col_flag = ((values < lo) | (values > hi)).fillna(False)
            scale = max(float(hi - lo), 1.0) if pd.notna(hi) and pd.notna(lo) else 1.0
            col_score = ((values - values.clip(lo, hi)).abs() / scale).fillna(0.0)
            flag = flag | col_flag
            score = pd.concat([score, col_score], axis=1).max(axis=1)
            per_col_mask[col] = col_flag
        return flag, score, per_col_mask

    def _repair(self, df, per_col_mask):
        for col, mask in per_col_mask.items():
            if col not in df.columns or not mask.any():
                continue
            if self.repair_method == "clip":
                lo, hi = self.bounds_.get(col, (None, None))
                if lo is not None or hi is not None:
                    df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=lo, upper=hi)
            elif self.repair_method == "winsorize":
                lo, hi = self.winsor_bounds_.get(col, (None, None))
                if lo is not None or hi is not None:
                    df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=lo, upper=hi)
            elif self.repair_method == "median":
                med = self.medians_.get(col)
                if med is not None and not pd.isna(med):
                    df.loc[mask, col] = med
            elif self.repair_method == "set_missing":
                df.loc[mask, col] = np.nan
        return df

    def transform(self, df):
        df = df.copy()
        if not self.fitted_:
            self._fit(df)
        flag, score, per_col_mask = self._score(df)
        if self.action == "delete":
            return df.loc[~flag.values].reset_index(drop=True)

        df[self.flag_col] = flag.astype(int).values
        if self.score_col:
            df[self.score_col] = score.values
        if self.action == "repair":
            df = self._repair(df, per_col_mask)
        return df

    def get_op_description(self):
        description = """Operator name: HandleOutlier

Function description:
Detect numeric outliers (IQR / z-score / IsolationForest) and either delete the
flagged rows or repair the offending values (clip, median, set_missing, winsorize).

Input:
df : pd.DataFrame — Table containing numeric feature columns.

Parameters:
cols : list[str] or None — Numeric columns to inspect. If None, use all numeric columns.
method : str — Detection method: "iqr", "zscore", or "isolation_forest".
threshold : float — IQR multiplier or z-score cutoff.
action : str — "delete", "repair", or "flag" (flag-only writes marker without changing rows).
repair_method : str — "clip", "median", "set_missing", or "winsorize".
flag_col : str — Name of the 0/1 marker column.
score_col : str or None — Optional continuous outlier score column.

Output:
pd.DataFrame — For action='delete', the original schema with outlier rows
removed. For action='flag' or action='repair', marker/score columns are added
and values may be repaired depending on action.

Example:
>>> df = pd.DataFrame({"x": [1, 2, 100]})
>>> HandleOutlier(cols=["x"], method="iqr", threshold=1.5, action="delete").transform(df)
   x
0  1
1  2

Example YAML:
  - op: HandleOutlier
    target: train
    params:
      cols: [x]
      method: iqr
      threshold: 3.0
      action: delete
"""
        return description.strip()
