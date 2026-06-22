import numpy as np
import pandas as pd
from ..base_op import TabularOp


class SelectFeature(TabularOp):
    """Select or drop features by variance, univariate score, RFE, or model."""

    FIT_ON_TRAIN_ONLY = True

    def __init__(self, target_col=None, method="variance", k=100,
                 threshold=0.0, score_func="f_classif",
                 n_features_to_select=20, estimator="tree",
                 exclude_cols=None, random_state=42, step=1):
        super().__init__(name="SelectFeature")
        if method not in ("variance", "univariate", "rfe", "model"):
            raise ValueError("method must be variance/univariate/rfe/model")
        self.target_col = target_col
        self.method = method
        self.k = int(k)
        self.threshold = float(threshold)
        self.score_func = score_func
        self.n_features_to_select = int(n_features_to_select)
        self.estimator = estimator
        self.exclude_cols = exclude_cols or []
        self.random_state = random_state
        self.step = step
        self.keep_cols_ = None
        self.fitted_ = False

    def get_op_description(self):
        description = """Operator name: SelectFeature

Function description:
Select a feature subset using variance threshold,
univariate scores, RFE, or model importance.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'x1': [1, 1], 'x2': [3, 4], 'label': [0, 1]})
>>> op = SelectFeature(method='variance', threshold=0.0, target_col='label')
>>> op.transform(df)
   x2  label
0   3      0
1   4      1

Example YAML:
  - op: SelectFeature
    target: both
    params:
      method: variance
      threshold: 0.0
      target_col: label
"""
        return description.strip()

    def _candidate_cols(self, df):
        protected = set(self.exclude_cols)
        if self.target_col is not None:
            protected.add(self.target_col)
        return [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in protected
        ]

    def _fit_variance(self, df, cols):
        variances = df[cols].var(numeric_only=True, skipna=True)
        drop = {c for c in cols if variances.get(c, 0.0) <= self.threshold}
        return [c for c in df.columns if c not in drop]

    def _estimator(self, is_classif=True):
        if self.estimator in ("tree", "forest"):
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            cls = RandomForestClassifier if is_classif else RandomForestRegressor
            return cls(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        if is_classif:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=200, random_state=self.random_state)
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    def _fit_supervised(self, df, cols):
        if self.target_col not in df.columns or not cols:
            return df.columns.tolist()
        x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = df[self.target_col]
        is_classif = y.nunique(dropna=True) <= 20
        n_keep = min(max(1, self.n_features_to_select if self.method == "rfe" else self.k), len(cols))
        try:
            if self.method == "univariate":
                from sklearn.feature_selection import SelectKBest, f_classif, f_regression, chi2
                score_map = {"f_classif": f_classif, "f_regression": f_regression, "chi2": chi2}
                score = score_map.get(self.score_func, f_classif)
                x_use = x.clip(lower=0) if self.score_func == "chi2" else x
                selector = SelectKBest(score_func=score, k=n_keep).fit(x_use, y)
                selected = list(x.columns[selector.get_support()])
            elif self.method == "rfe":
                from sklearn.feature_selection import RFE
                selector = RFE(
                    self._estimator(is_classif=is_classif),
                    n_features_to_select=n_keep,
                    step=self.step,
                ).fit(x, y)
                selected = list(x.columns[selector.get_support()])
            else:
                model = self._estimator(is_classif=is_classif).fit(x, y)
                if hasattr(model, "feature_importances_"):
                    importance = model.feature_importances_
                else:
                    importance = np.abs(getattr(model, "coef_", np.zeros(len(cols)))).ravel()
                order = np.argsort(importance)[::-1][:n_keep]
                selected = [cols[i] for i in order]
        except Exception as exc:
            print(f"  [SelectFeature] {self.method} failed, keeping all: {exc}")
            selected = cols
        protected = [c for c in df.columns if c not in cols]
        return protected + selected

    def transform(self, df):
        df = df.copy()
        if not self.fitted_:
            candidates = self._candidate_cols(df)
            if self.method == "variance":
                self.keep_cols_ = self._fit_variance(df, candidates)
            else:
                self.keep_cols_ = self._fit_supervised(df, candidates)
            self.fitted_ = True
        keep = [c for c in self.keep_cols_ if c in df.columns]
        return df[keep]
