import numpy as np
import pandas as pd
from ..base_op import TabularOp


class CorrectLabel(TabularOp):
    """Correct suspected label errors using predicted probabilities."""

    def __init__(self, label_col, pred_probs=None, strategy="flip",
                 confidence_threshold=0.9):
        super().__init__(name="CorrectLabel")
        if strategy not in ("flip", "flag"):
            raise ValueError("strategy must be 'flip' or 'flag'")
        self.label_col = label_col
        self.pred_probs = pred_probs
        self.strategy = strategy
        self.confidence_threshold = float(confidence_threshold)
        self.mask_col = f"{label_col}_corrected"

    def get_op_description(self):
        description = """Operator name: CorrectLabel

Function description:
Handle suspected label errors by comparing the current label with
high-confidence predicted probabilities. Can flip labels or only flag rows.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
label_col : str — Label column.
pred_probs : array-like or probability column(s).
strategy : str — flip or flag.
confidence_threshold : float — Minimum predicted class confidence.

Output:
pd.DataFrame — Original table plus a "<label_col>_corrected" 0/1 marker column.

Example:
>>> df = pd.DataFrame({'label': [0, 0, 1], 'p1': [0.95, 0.10, 0.92]})
>>> op = CorrectLabel(label_col='label', pred_probs='p1', strategy='flip', confidence_threshold=0.9)
>>> op.transform(df)
   label    p1  label_corrected
0      1  0.95                1
1      0  0.10                0
2      1  0.92                0

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: CorrectLabel
    prev:
    - s0
    params:
      label_col: label
      pred_probs: p1
      strategy: flip
      confidence_threshold: 0.9
  train:
    prev:
    - o1
"""
        return description.strip()

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

    def transform(self, df):
        if self.label_col not in df.columns:
            return df
        df = df.copy()
        probs = self._get_probs(df)
        if probs is None:
            df[self.mask_col] = 0
            return df
        predicted = np.nanargmax(probs, axis=1)
        confidence = np.nanmax(probs, axis=1)
        labels = pd.to_numeric(df[self.label_col], errors="coerce")
        mask = labels.notna() & (predicted != labels.astype(int)) & (
            confidence >= self.confidence_threshold
        )
        df[self.mask_col] = mask.astype(int).values
        if self.strategy == "flip":
            df.loc[mask, self.label_col] = predicted[mask.to_numpy()]
        return df
