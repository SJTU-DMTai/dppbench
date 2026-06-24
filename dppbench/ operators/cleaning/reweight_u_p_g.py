import numpy as np
import pandas as pd

from ..base_op import TabularOp


class ReweightUPG(TabularOp):
    """Detect under-performing groups and up-weight their samples via sample_weight."""

    APPLIES_TO_STD_TEST = False

    def __init__(self, cluster_col=None, feature_cols=None, pred_probs=None,
                 label_col=None, threshold=0.9, n_bins=5):
        super().__init__(name="ReweightUPG")
        self.cluster_col = cluster_col
        self.feature_cols = feature_cols
        self.pred_probs = pred_probs
        self.label_col = label_col
        self.threshold = float(threshold)
        self.n_bins = int(n_bins)

    def _get_probs(self, df):
        if self.pred_probs is None:
            return None
        if isinstance(self.pred_probs, str) and self.pred_probs in df.columns:
            probs = np.asarray(df[self.pred_probs].tolist())
        elif isinstance(self.pred_probs, list) and all(c in df.columns for c in self.pred_probs):
            probs = df[self.pred_probs].to_numpy(dtype=float)
        else:
            probs = np.asarray(self.pred_probs)
        if probs.ndim == 1:
            probs = np.vstack([1.0 - probs, probs]).T
        return probs if len(probs) == len(df) else None

    def _groups(self, df):
        if self.cluster_col and self.cluster_col in df.columns:
            return df[self.cluster_col].astype("string")
        cols = (
            df.select_dtypes(include=[np.number]).columns.tolist()[:2]
            if self.feature_cols is None
            else [c for c in self.feature_cols if c in df.columns][:2]
        )
        if not cols:
            return pd.Series("all", index=df.index)
        parts = []
        for col in cols:
            values = pd.to_numeric(df[col], errors="coerce")
            try:
                bins = min(self.n_bins, max(1, values.nunique(dropna=True)))
                parts.append(pd.qcut(values, q=bins, duplicates="drop").astype("string"))
            except Exception:
                parts.append(values.fillna("missing").astype("string"))
        groups = parts[0]
        for part in parts[1:]:
            groups = groups + "|" + part
        return groups

    def _base_score(self, df):
        probs = self._get_probs(df)
        if probs is not None and self.label_col in df.columns:
            y = pd.to_numeric(df[self.label_col], errors="coerce").fillna(-1).astype(int)
            return (np.nanargmax(probs, axis=1) != y.to_numpy()).astype(float)
        if probs is not None:
            return 1.0 - np.nanmax(probs, axis=1)
        return np.zeros(len(df), dtype=float)

    def transform(self, df):
        df = df.copy()
        base_score = self._base_score(df)
        group_score = pd.Series(base_score, index=df.index).groupby(self._groups(df)).transform("mean")
        cutoff = group_score.quantile(self.threshold) if len(group_score) else np.inf
        marker = (group_score >= cutoff).to_numpy()
        weights = np.ones(len(df), dtype=float)
        weights[marker] = 2.0
        df["sample_weight"] = weights
        return df

    def get_op_description(self):
        description = """Operator name: ReweightUPG

Function description:
Detect underperforming subgroups by aggregating per-row correctness/uncertainty
over cluster or binned-feature groups, then up-weight their loss contribution
via sample_weight (1.0 default, 2.0 for flagged groups).

Input:
df : pd.DataFrame — Sample table with cluster/features and optional labels/predicted probabilities.

Parameters:
cluster_col : str or None — Existing subgroup/cluster column.
feature_cols : list[str] or None — Features used to derive fallback groups.
pred_probs : array-like, str, list[str], or None — Prediction probabilities.
label_col : str or None — Ground-truth label used for correctness scoring.
threshold : float — Quantile used to flag rows belonging to underperforming groups.
n_bins : int — Number of bins per feature for fallback group construction.

Output:
pd.DataFrame — Original table with sample_weight column.

Example:
>>> df = pd.DataFrame({"g": ["A", "A", "B"], "p": [0.9, 0.8, 0.2]})
>>> ReweightUPG(cluster_col="g", pred_probs="p", threshold=0.5).transform(df)[["sample_weight"]]
   sample_weight
0            1.0
1            1.0
2            2.0

Example YAML:
  - op: ReweightUPG
    target: train
    params:
      cluster_col: cluster
      pred_probs: pred_prob
      threshold: 0.9
"""
        return description.strip()
