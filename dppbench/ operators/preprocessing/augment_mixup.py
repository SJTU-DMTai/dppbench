import numpy as np
import pandas as pd
from ..base_op import TabularOp


class AugmentMixup(TabularOp):
    """Append Mixup synthetic samples for numeric columns."""

    APPLIES_TO_STD_TEST = False
    RANDOM_STATE = 42

    def __init__(self, label_col, cols=None, alpha=0.2, n_samples=None):
        super().__init__(name="AugmentMixup")
        self.label_col = label_col
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.alpha = float(alpha)
        self.n_samples = n_samples

    def get_op_description(self):
        description = """Operator name: AugmentMixup

Function description:
Generate Mixup rows by convex-combining pairs of rows and labels. Non-numeric
columns in synthetic rows are left missing.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
label_col : str — Label column.
cols : list[str] or None — Numeric columns to mix. None = all numeric.
alpha : float — Beta distribution parameter (default 0.2).
n_samples : int or None — Synthetic row count. None = len(df).

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'x': [0.0, 10.0], 'label': [0, 1]})
>>> op = AugmentMixup(label_col='label', cols=['x'], alpha=0.5, n_samples=1)
>>> op.transform(df)
       x  label
0    0.0    0.0
1   10.0    1.0
2    5.0    0.5

Example YAML:
  - op: AugmentMixup
    target: train
    params:
      label_col: label
      cols: [x1, x2]
      alpha: 0.2
      n_samples: 100
"""
        return description.strip()

    def transform(self, df):
        if self.label_col not in df.columns or len(df) < 2:
            return df
        rng = np.random.RandomState(self.RANDOM_STATE)
        cols = (
            df.select_dtypes(include=[np.number]).columns.tolist()
            if self.cols is None else [c for c in self.cols if c in df.columns]
        )
        cols = [c for c in cols if c != self.label_col]
        if not cols:
            return df
        n = int(self.n_samples or len(df))
        i = rng.randint(0, len(df), size=n)
        j = rng.randint(0, len(df), size=n)
        lam = rng.beta(self.alpha, self.alpha, size=n)
        synthetic = pd.DataFrame(columns=df.columns)
        x1 = df.iloc[i][cols].to_numpy(dtype=float)
        x2 = df.iloc[j][cols].to_numpy(dtype=float)
        synthetic[cols] = lam[:, None] * x1 + (1.0 - lam[:, None]) * x2
        y1 = pd.to_numeric(df.iloc[i][self.label_col], errors="coerce").to_numpy()
        y2 = pd.to_numeric(df.iloc[j][self.label_col], errors="coerce").to_numpy()
        synthetic[self.label_col] = lam * y1 + (1.0 - lam) * y2
        return pd.concat([df, synthetic], ignore_index=True)
