import numpy as np
import pandas as pd
from ..base_op import TabularOp


class DiscretizeFeature(TabularOp):
    """Discretize continuous columns into categorical bin ids."""

    FIT_ON_TRAIN_ONLY = True

    def __init__(self, cols=None, boundaries=None, n_bins=5, strategy="manual"):
        super().__init__(name="DiscretizeFeature")
        if strategy not in ("manual", "uniform", "quantile", "kmeans"):
            raise ValueError("strategy must be manual/uniform/quantile/kmeans")
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.boundaries = boundaries or {}
        self.n_bins = int(n_bins)
        self.strategy = strategy
        self.bin_edges_ = {}
        self.fitted_ = False
        target_cols = self.cols or list(self.boundaries.keys())
        self.output_col_types = {c: "categorical" for c in target_cols}

    def get_op_description(self):
        description = """Operator name: DiscretizeFeature

Function description:
Convert continuous columns into bin indices using
manual boundaries, uniform bins, quantile bins, or sklearn kmeans bins.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] or None — Columns to discretize.
boundaries : dict[str, list[float]] — Manual bin edges.
n_bins : int — Number of learned bins.
strategy : str — manual/uniform/quantile/kmeans.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'age': [18, 35, 70]})
>>> op = DiscretizeFeature(cols=['age'], boundaries={'age': [30, 60]}, strategy='manual')
>>> op.transform(df)
   age
0    0
1    1
2    2

Example YAML:
  - op: DiscretizeFeature
    target: train
    params:
      cols: [age]
      strategy: manual
      boundaries:
        age: [30, 60]
"""
        return description.strip()

    def _fit_edges(self, df, col):
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if self.strategy == "manual":
            return np.asarray(self.boundaries.get(col, []), dtype=float)
        if values.empty:
            return np.asarray([], dtype=float)
        if self.strategy == "uniform":
            return np.linspace(values.min(), values.max(), self.n_bins + 1)[1:-1]
        if self.strategy == "quantile":
            qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            return np.unique(values.quantile(qs).to_numpy(dtype=float))
        try:
            from sklearn.preprocessing import KBinsDiscretizer
            est = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode="ordinal",
                strategy="kmeans",
            ).fit(values.to_numpy().reshape(-1, 1))
            return est.bin_edges_[0][1:-1]
        except Exception as exc:
            print(f"  [DiscretizeFeature] kmeans unavailable, fallback quantile: {exc}")
            qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            return np.unique(values.quantile(qs).to_numpy(dtype=float))

    def transform(self, df):
        df = df.copy()
        cols = self.cols or list(self.boundaries.keys())
        cols = [c for c in cols if c in df.columns]
        if not self.fitted_:
            self.bin_edges_ = {c: self._fit_edges(df, c) for c in cols}
            self.fitted_ = True
        for col, bins in self.bin_edges_.items():
            if col not in df.columns:
                continue
            values = pd.to_numeric(df[col], errors="coerce")
            out = np.searchsorted(bins, values.fillna(0), side="right").astype(np.int64)
            out[pd.isna(values).to_numpy()] = -1
            df[col] = out
        return df.reset_index(drop=True)
