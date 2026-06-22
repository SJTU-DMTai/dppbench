import numpy as np
import pandas as pd
from ..base_op import TabularOp


class HandleMV(TabularOp):
    """Handle missing values: delete rows or impute with simple/multivariate methods."""

    FIT_ON_TRAIN_ONLY = True

    def __init__(self, cols=None, method="median", action="impute", fill_value=None,
                 n_neighbors=5, weights="uniform", max_iter=10,
                 estimator="bayes_ridge", random_state=42,
                 sample_posterior=False):
        super().__init__(name="HandleMV")
        if action not in ("delete", "impute"):
            raise ValueError("action must be delete/impute")
        if method not in ("median", "mean", "mode", "constant", "knn", "iterative"):
            raise ValueError(
                "method must be median/mean/mode/constant/knn/iterative"
            )
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.method = method
        self.action = action
        self.fill_value = fill_value
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self.max_iter = int(max_iter)
        self.estimator = estimator
        self.random_state = random_state
        self.sample_posterior = bool(sample_posterior)
        self.cols_ = []
        self.fill_values_ = {}
        self.imputer_ = None
        self.fitted_ = False
        # Delete mode mutates training rows only; impute keeps train→test sharing.
        self.APPLIES_TO_STD_TEST = (action != "delete")

    def get_op_description(self):
        description = """Operator name: HandleMV

Function description:
Handle missing values in selected columns. With action='delete', drops rows
that have NA in any of the selected columns. With action='impute' (default),
fills missing values: simple methods use per-column median/mean/mode/constant;
multivariate methods use sklearn KNNImputer or IterativeImputer when available.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] or None — Columns to handle. None = columns with missing
values for simple methods, or numeric columns for multivariate methods.
method : str — median/mean/mode/constant/knn/iterative (only used when action='impute').
action : str — delete/impute (default 'impute').
fill_value : any — Used by method='constant'.
n_neighbors, weights : KNN imputer parameters.
max_iter, estimator, random_state, sample_posterior : iterative parameters.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'age': [20, None, 40], 'city': ['A', None, 'A']})
>>> op = HandleMV(cols=['age', 'city'], method='mode')
>>> op.transform(df)
    age city
0  20.0    A
1  20.0    A
2  40.0    A

Example YAML:
  - op: HandleMV
    target: both
    params:
      cols: [age, city]
      action: impute
      method: mode
"""
        return description.strip()

    def _select_cols(self, df):
        if self.cols is not None:
            cols = [c for c in self.cols if c in df.columns]
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
                value = 0 if self.fill_value is None else self.fill_value
            self.fill_values_[col] = value

    def _transform_simple(self, df):
        for col, value in self.fill_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(value)
        return df

    def _fit_multivariate(self, sub):
        if self.method == "knn":
            try:
                from sklearn.impute import KNNImputer
                self.imputer_ = KNNImputer(
                    n_neighbors=self.n_neighbors,
                    weights=self.weights,
                ).fit(sub)
            except Exception as exc:
                print(f"  [HandleMV] KNN unavailable, fallback to median: {exc}")
                self.method = "median"
                self._fit_simple(pd.DataFrame(sub, columns=self.cols_))
        else:
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer
                from sklearn.linear_model import BayesianRidge, Ridge
                from sklearn.tree import DecisionTreeRegressor
                from sklearn.neighbors import KNeighborsRegressor

                est_map = {
                    "bayes_ridge": BayesianRidge(),
                    "ridge": Ridge(),
                    "tree": DecisionTreeRegressor(random_state=self.random_state),
                    "knn": KNeighborsRegressor(n_neighbors=5),
                }
                self.imputer_ = IterativeImputer(
                    estimator=est_map.get(self.estimator, BayesianRidge()),
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    sample_posterior=self.sample_posterior,
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
        if self.action == "delete":
            return self._delete_na_rows(df)

        if not self.fitted_:
            self.cols_ = self._select_cols(df)
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
            return df
        return self._transform_simple(df)
