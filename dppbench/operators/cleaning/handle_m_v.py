import numpy as np
import pandas as pd
from ..base_op import TabularOp


class HandleMV(TabularOp):
    """Handle missing values: delete rows, mark sentinels as NA, or impute."""

    FIT_ON_TRAIN_ONLY = True

    def __init__(self, cols=None, method="median", action="impute",
                 n_neighbors=5, max_iter=10, fill_value=None, na_values=None):
        super().__init__(name="HandleMV")
        if action not in ("delete", "impute", "mark"):
            raise ValueError("action must be delete/impute/mark")
        if action == "impute" and method not in (
            "median", "mean", "mode", "constant", "knn", "iterative"
        ):
            raise ValueError(
                "method must be median/mean/mode/constant/knn/iterative"
            )
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.method = method
        self.action = action
        self.n_neighbors = int(n_neighbors)
        self.max_iter = int(max_iter)
        if fill_value is None and method == "constant":
            self.fill_value = 0
        else:
            self.fill_value = fill_value
        if na_values is None:
            self.na_values = []
        elif isinstance(na_values, (list, tuple)):
            self.na_values = list(na_values)
        else:
            self.na_values = [na_values]
        self.cols_ = []
        self.fill_values_ = {}
        self.imputer_ = None
        self.fitted_ = False
        self.APPLIES_TO_STD_TEST = (action != "delete")

    def get_op_description(self):
        description = """Operator name: HandleMV

Function description:
Handle missing values in selected columns.

- action='delete' : drop rows that have NA in any of the selected columns.
- action='mark'   : replace sentinel values listed in ``na_values`` with NaN
  (useful for treating placeholders like "" / -1 / "N/A" as missing before
  downstream processing such as ID remapping). No rows are dropped and no
  imputation is performed.
- action='impute' : fill missing values. Simple methods use per-column
  median/mean/mode/constant; multivariate methods use sklearn KNNImputer or
  IterativeImputer when available. ``na_values`` sentinels are converted to
  NaN before imputation. ``fill_value`` specifies the constant used by
  method='constant' (default 0). After imputation, numeric columns that no
  longer contain NaN are safely cast back to integer dtype when possible.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] or None — Columns to handle. None = columns with missing
values for simple methods, or numeric columns for multivariate methods.
method : str — median/mean/mode/constant/knn/iterative (action='impute' only).
action : str — delete/impute/mark (default 'impute').
fill_value : any — Constant fill for method='constant' (default 0).
na_values : list or scalar — Sentinel values to treat as NaN before processing.
n_neighbors : int — KNN imputer neighbours (method='knn' only).
max_iter : int — Iterative imputer rounds (method='iterative' only).

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'age': [20, None, 40], 'city': ['A', '', 'A']})
>>> op = HandleMV(cols=['age'], method='constant', fill_value=0)
>>> op2 = HandleMV(cols=['city'], na_values=[""], action='mark')

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: HandleMV
    prev:
    - s0
    params:
      cols:
      - age
      - city
      action: impute
      method: mode
  train:
    prev:
    - o1
"""
        return description.strip()

    def _apply_na_values(self, df):
        if not self.na_values:
            return df
        for col in self.cols_:
            if col in df.columns:
                for sentinel in self.na_values:
                    mask = df[col] == sentinel
                    if mask.any():
                        df.loc[mask, col] = np.nan
        return df

    def _select_cols(self, df):
        if self.cols is not None:
            cols = [c for c in self.cols if c in df.columns]
        elif self.action == "mark":
            cols = [c for c in df.columns if df[c].isin(self.na_values).any()] if self.na_values else []
        elif self.method in ("knn", "iterative"):
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = [c for c in df.columns if df[c].isna().any()]
        if self.method in ("knn", "iterative"):
            cols = [c for c in cols if df[c].dtype.kind in ("i", "u", "f")]
        return cols

    def _fit_simple(self, df):
        self.fill_values_ = {}
        for col in self.cols_:
            if self.method == "median":
                if pd.api.types.is_numeric_dtype(df[col]):
                    value = df[col].median()
                else:
                    mode = df[col].mode()
                    value = mode.iloc[0] if len(mode) else 0
            elif self.method == "mean":
                if pd.api.types.is_numeric_dtype(df[col]):
                    value = df[col].mean()
                else:
                    mode = df[col].mode()
                    value = mode.iloc[0] if len(mode) else 0
            elif self.method == "mode":
                mode = df[col].mode()
                value = mode.iloc[0] if len(mode) else 0
            else:
                value = self.fill_value
            self.fill_values_[col] = value

    def _transform_simple(self, df):
        for col, value in self.fill_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)
        self._safe_cast_to_int(df)
        return df

    def _safe_cast_to_int(self, df):
        for col, value in self.fill_values_.items():
            if col not in df.columns:
                continue
            series = df[col]
            if series.isna().any():
                continue
            if not pd.api.types.is_float_dtype(series):
                continue
            if isinstance(value, (int, np.integer)) or (
                isinstance(value, float) and float(value).is_integer()
            ):
                try:
                    if np.isfinite(series).all() and (series == series.astype("int64")).all():
                        df[col] = series.astype("int64")
                except (TypeError, ValueError):
                    pass

    def _fit_multivariate(self, sub):
        if self.method == "knn":
            try:
                from sklearn.impute import KNNImputer
                self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors).fit(sub)
            except Exception as exc:
                print(f"  [HandleMV] KNN unavailable, fallback to median: {exc}")
                self.method = "median"
                self._fit_simple(pd.DataFrame(sub, columns=self.cols_))
        else:
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer
                from sklearn.linear_model import BayesianRidge

                self.imputer_ = IterativeImputer(
                    estimator=BayesianRidge(),
                    max_iter=self.max_iter,
                    random_state=42,
                ).fit(sub)
            except Exception as exc:
                print(f"  [HandleMV] Iterative unavailable, fallback to median: {exc}")
                self.method = "median"
                self._fit_simple(pd.DataFrame(sub, columns=self.cols_))

    def _delete_na_rows(self, df):
        if self.cols is not None:
            cols = [c for c in self.cols if c in df.columns]
        else:
            cols = [c for c in df.columns if df[c].isna().any()]
        if not cols:
            return df
        mask = df[cols].isna().any(axis=1)
        return df.loc[~mask].reset_index(drop=True)

    def transform(self, df):
        df = df.copy()

        if not self.fitted_:
            self.cols_ = self._select_cols(df)

        df = self._apply_na_values(df)

        if self.action == "mark":
            self.fitted_ = True
            return df

        if self.action == "delete":
            self.fitted_ = True
            return self._delete_na_rows(df)

        if not self.fitted_:
            if self.method in ("knn", "iterative"):
                sub = df[self.cols_].apply(pd.to_numeric, errors="coerce")
                self.cols_ = [c for c in self.cols_ if not sub[c].isna().all()]
                sub = sub[self.cols_] if self.cols_ else sub
                if self.cols_:
                    self._fit_multivariate(sub)
            else:
                self._fit_simple(df)
            self.fitted_ = True

        cols = [c for c in self.cols_ if c in df.columns]
        if not cols:
            return df
        if self.method in ("knn", "iterative") and self.imputer_ is not None:
            sub = df[cols].apply(pd.to_numeric, errors="coerce")
            filled = self.imputer_.transform(sub)
            for i, col in enumerate(cols):
                df[col] = filled[:, i]
            self._safe_cast_to_int(df)
            return df
        return self._transform_simple(df)
