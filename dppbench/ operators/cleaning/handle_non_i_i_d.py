import numpy as np
import pandas as pd

from ..base_op import TabularOp


_VALID_ACTIONS = ("flag", "delete", "reweight")
_VALID_WEIGHTING = ("marker", "score")
_VALID_COMBINE = ("overwrite", "multiply")


class HandleNonIID(TabularOp):
    """Detect non-IID samples and either delete or reweight them."""

    APPLIES_TO_STD_TEST = False

    def __init__(self, feature_cols=None, pred_probs=None,
                 score_col="non_iid_score", flag_col="is_non_iid",
                 threshold=0.95,
                 action="reweight", weighting="marker",
                 weight_col="sample_weight", marker_weight=0.2,
                 default_weight=1.0, min_weight=0.2, max_weight=1.0,
                 combine="overwrite"):
        super().__init__(name="HandleNonIID")
        if action not in _VALID_ACTIONS:
            raise ValueError("action must be flag/delete/reweight")
        if weighting not in _VALID_WEIGHTING:
            raise ValueError("weighting must be marker/score")
        if combine not in _VALID_COMBINE:
            raise ValueError("combine must be overwrite/multiply")
        self.feature_cols = feature_cols
        self.pred_probs = pred_probs
        self.score_col = score_col
        self.flag_col = flag_col
        self.threshold = float(threshold)
        self.action = action
        self.weighting = weighting
        self.weight_col = weight_col
        self.marker_weight = float(marker_weight)
        self.default_weight = float(default_weight)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.combine = combine

    def _prob_score(self, df):
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
        return 1.0 - np.nanmax(probs, axis=1) if len(probs) == len(df) else None

    def _score(self, df):
        parts = []
        cols = (
            df.select_dtypes(include=[np.number]).columns.tolist()
            if self.feature_cols is None
            else [c for c in self.feature_cols if c in df.columns]
        )
        if cols:
            x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            sd = x.std(axis=0).replace(0, 1.0)
            parts.append(((x - x.mean(axis=0)) / sd).abs().mean(axis=1).to_numpy())
        prob_score = self._prob_score(df)
        if prob_score is not None:
            parts.append(prob_score)
        if not parts:
            return np.zeros(len(df), dtype=float)
        return np.mean(np.vstack(parts), axis=0)

    def _compute_weights(self, score, marker, n):
        if self.weighting == "marker":
            weights = np.full(n, self.default_weight, dtype=float)
            weights[marker] = self.marker_weight
            return weights
        score = np.asarray(score, dtype=float)
        lo, hi = float(np.nanmin(score)) if n else 0.0, float(np.nanmax(score)) if n else 0.0
        if hi > lo:
            risk = (score - lo) / (hi - lo)
        else:
            risk = np.zeros(n, dtype=float)
        return self.max_weight - risk * (self.max_weight - self.min_weight)

    def transform(self, df):
        df = df.copy()
        score = self._score(df)
        cutoff = np.nanquantile(score, self.threshold) if len(score) else np.inf
        marker = (score >= cutoff)
        df[self.score_col] = score
        df[self.flag_col] = marker.astype(int)
        if self.action == "delete":
            df = df.loc[~marker].reset_index(drop=True)
        elif self.action == "reweight":
            weights = self._compute_weights(score, marker, len(df))
            if self.combine == "multiply" and self.weight_col in df.columns:
                base = pd.to_numeric(df[self.weight_col], errors="coerce").fillna(1.0).to_numpy()
                df[self.weight_col] = base * weights
            else:
                df[self.weight_col] = weights
        return df

    def get_op_description(self):
        description = """Operator name: HandleNonIID

Function description:
Detect non-IID samples (Cleanlab-style: feature deviation + prediction
uncertainty) and either delete them or down-weight them via sample_weight.

Input:
df : pd.DataFrame — Ordered sample table with features and optional prediction probabilities.

Parameters:
feature_cols : list[str] or None — Feature columns used for distribution-deviation scoring.
pred_probs : array-like, str, list[str], or None — Prediction probability data or columns.
score_col : str — Continuous non-IID score column.
flag_col : str — 0/1 high-risk marker column.
threshold : float — Score quantile used to flag samples.
action : str — "delete" or "reweight" (default: reweight).
weighting : str — "marker" (constant marker_weight) or "score" (min-max mapped).
weight_col : str — Output weight column. Defaults to "sample_weight".
marker_weight : float — Weight assigned to flagged rows under weighting=marker.
combine : str — "overwrite" or "multiply" with existing weights.

Output:
pd.DataFrame — Original table plus non-IID score/marker columns, with rows
filtered or sample_weight column written.

Example:
>>> df = pd.DataFrame({"x": [0.0, 0.1, 9.0]})
>>> HandleNonIID(feature_cols=["x"], threshold=0.6, action="reweight").transform(df)[["sample_weight"]]
   sample_weight
0            1.0
1            1.0
2            0.2

Example YAML:
  - op: HandleNonIID
    target: train
    params:
      feature_cols: [x]
      threshold: 0.95
      action: reweight
"""
        return description.strip()
