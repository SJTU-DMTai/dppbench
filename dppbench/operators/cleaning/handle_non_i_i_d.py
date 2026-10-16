import numpy as np
import pandas as pd

from ..base_op import TabularOp


_VALID_ACTIONS = ("delete", "reweight")


class HandleNonIID(TabularOp):
    """Detect non-IID samples and either delete or down-weight them."""

    APPLIES_TO_STD_TEST = False

    def __init__(self, feature_cols=None, pred_probs=None,
                 threshold=0.95, action="reweight"):
        super().__init__(name="HandleNonIID")
        if action not in _VALID_ACTIONS:
            raise ValueError("action must be delete/reweight")
        self.feature_cols = feature_cols
        self.pred_probs = pred_probs
        self.threshold = float(threshold)
        self.action = action

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

    def transform(self, df):
        df = df.copy()
        score = self._score(df)
        cutoff = np.nanquantile(score, self.threshold) if len(score) else np.inf
        marker = (score >= cutoff)
        if self.action == "delete":
            return df.loc[~marker].reset_index(drop=True)
        weights = np.ones(len(df), dtype=float)
        weights[marker] = 0.2
        df["sample_weight"] = weights
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
threshold : float — Quantile cutoff used to flag samples.
action : str — "delete" or "reweight" (default).

Output:
pd.DataFrame — For action='delete', rows are removed. For action='reweight',
sample_weight column is written (1.0 default, 0.2 for flagged rows).

Example:
>>> df = pd.DataFrame({"x": [0.0, 0.1, 9.0]})
>>> HandleNonIID(feature_cols=["x"], threshold=0.6, action="reweight").transform(df)[["sample_weight"]]
   sample_weight
0            1.0
1            1.0
2            0.2

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: HandleNonIID
    prev:
    - s0
    params:
      feature_cols:
      - x
      threshold: 0.95
      action: reweight
  train:
    prev:
    - o1
"""
        return description.strip()
