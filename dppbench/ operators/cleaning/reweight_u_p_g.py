import numpy as np
import pandas as pd

from ..base_op import TabularOp


_VALID_WEIGHTING = ("upweight_marker", "score")
_VALID_COMBINE = ("overwrite", "multiply")


class ReweightUPG(TabularOp):
    """Detect under-performing groups and up-weight their samples via sample_weight."""

    APPLIES_TO_STD_TEST = False

    def __init__(self, cluster_col=None, feature_cols=None, pred_probs=None,
                 label_col=None, score_col="upg_score", flag_col="is_upg",
                 threshold=0.9, n_bins=5,
                 weighting="upweight_marker", weight_col="sample_weight",
                 marker_weight=2.0, default_weight=1.0,
                 min_weight=1.0, max_weight=2.0,
                 combine="overwrite"):
        super().__init__(name="ReweightUPG")
        if weighting not in _VALID_WEIGHTING:
            raise ValueError("weighting must be upweight_marker/score")
        if combine not in _VALID_COMBINE:
            raise ValueError("combine must be overwrite/multiply")
        self.cluster_col = cluster_col
        self.feature_cols = feature_cols
        self.pred_probs = pred_probs
        self.label_col = label_col
        self.score_col = score_col
        self.flag_col = flag_col
        self.threshold = float(threshold)
        self.n_bins = int(n_bins)
        self.weighting = weighting
        self.weight_col = weight_col
        self.marker_weight = float(marker_weight)
        self.default_weight = float(default_weight)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.combine = combine

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

    def _compute_weights(self, group_score, marker, n):
        if self.weighting == "upweight_marker":
            weights = np.full(n, self.default_weight, dtype=float)
            weights[marker] = self.marker_weight
            return weights
        score = np.asarray(group_score, dtype=float)
        lo, hi = float(np.nanmin(score)) if n else 0.0, float(np.nanmax(score)) if n else 0.0
        if hi > lo:
            risk = (score - lo) / (hi - lo)
        else:
            risk = np.zeros(n, dtype=float)
        return self.min_weight + risk * (self.max_weight - self.min_weight)

    def transform(self, df):
        df = df.copy()
        base_score = self._base_score(df)
        group_score = pd.Series(base_score, index=df.index).groupby(self._groups(df)).transform("mean")
        cutoff = group_score.quantile(self.threshold) if len(group_score) else np.inf
        marker = (group_score >= cutoff).to_numpy()
        df[self.score_col] = group_score.astype(float).values
        df[self.flag_col] = marker.astype(int)
        weights = self._compute_weights(group_score.to_numpy(), marker, len(df))
        if self.combine == "multiply" and self.weight_col in df.columns:
            base = pd.to_numeric(df[self.weight_col], errors="coerce").fillna(1.0).to_numpy()
            df[self.weight_col] = base * weights
        else:
            df[self.weight_col] = weights
        return df

    def get_op_description(self):
        description = """Operator name: ReweightUPG

Function description:
Detect underperforming subgroups (UPG, Cleanlab-style) by aggregating per-row
correctness/uncertainty over cluster or binned-feature groups, then up-weight
their loss contribution via sample_weight.

Input:
df : pd.DataFrame — Sample table with cluster/features and optional labels/predicted probabilities.

Parameters:
cluster_col : str or None — Existing subgroup/cluster column.
feature_cols : list[str] or None — Features used to derive fallback groups.
pred_probs : array-like, str, list[str], or None — Prediction probabilities.
label_col : str or None — Ground-truth label used for correctness scoring.
score_col : str — Continuous subgroup risk score.
flag_col : str — 0/1 low-performing group marker.
threshold : float — Quantile used to flag rows belonging to underperforming groups.
weighting : str — "upweight_marker" (default) or "score".
weight_col : str — Output weight column. Defaults to "sample_weight".
marker_weight : float — Weight for marked rows (default 2.0, up-weighting).
combine : str — "overwrite" or "multiply" with existing weights.

Output:
pd.DataFrame — Original table plus UPG score/marker columns and sample_weight.

Example:
>>> df = pd.DataFrame({"g": ["A", "A", "B"], "p": [0.9, 0.8, 0.2]})
>>> ReweightUPG(cluster_col="g", pred_probs="p", threshold=0.5).transform(df)[["is_upg", "sample_weight"]]
   is_upg  sample_weight
0       0            1.0
1       0            1.0
2       1            2.0

Example YAML:
  - op: ReweightUPG
    target: train
    params:
      cluster_col: cluster
      pred_probs: pred_prob
      marker_weight: 2.0
"""
        return description.strip()
