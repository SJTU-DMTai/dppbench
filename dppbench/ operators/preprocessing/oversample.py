import numpy as np
import pandas as pd
from ..base_op import TabularOp


class Oversample(TabularOp):
    """Increase minority classes by random duplication, SMOTE, or ADASYN."""

    APPLIES_TO_STD_TEST = False
    RANDOM_STATE = 42

    def __init__(self, target_col, method="random", n_neighbors=5):
        super().__init__(name="Oversample")
        if method not in ("random", "smote", "adasyn"):
            raise ValueError("method must be random/smote/adasyn")
        self.target_col = target_col
        self.method = method
        self.n_neighbors = int(n_neighbors)

    def get_op_description(self):
        description = """Operator name: Oversample

Function description:
Oversample minority labels with random duplication, SMOTE, or ADASYN. Falls
back to random oversampling if imblearn is unavailable or the selected
synthetic method cannot run.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
target_col : str — Label column.
method : str — random/smote/adasyn (default 'random').
n_neighbors : int — k_neighbors for SMOTE/ADASYN (default 5).

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'x': [1, 2, 3], 'label': [0, 0, 1]})
>>> op = Oversample(target_col='label', method='random')
>>> op.transform(df)
   x  label
0  1      0
1  2      0
2  3      1
3  3      1

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: Oversample
    prev:
    - s0
    params:
      target_col: label
      method: random
  train:
    prev:
    - o1
"""
        return description.strip()

    def _random(self, df):
        rng = np.random.RandomState(self.RANDOM_STATE)
        groups = list(df.groupby(self.target_col))
        if not groups:
            return df
        target_size = max(len(g) for _, g in groups)
        parts = []
        for _, group in groups:
            replace = len(group) < target_size
            idx = rng.choice(group.index, size=target_size, replace=replace)
            parts.append(df.loc[idx])
        return pd.concat(parts, axis=0).reset_index(drop=True)

    def _synthetic(self, df):
        feat_cols = [c for c in df.columns if c != self.target_col]
        x = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df[self.target_col]
        if self.method == "smote":
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(
                random_state=self.RANDOM_STATE,
                k_neighbors=self.n_neighbors,
            )
        else:
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(
                random_state=self.RANDOM_STATE,
                n_neighbors=self.n_neighbors,
            )
        xr, yr = sampler.fit_resample(x, y)
        out = pd.DataFrame(xr, columns=feat_cols)
        out[self.target_col] = yr
        return out[df.columns.tolist()].reset_index(drop=True)

    def transform(self, df):
        if self.target_col not in df.columns:
            return df
        if self.method == "random":
            return self._random(df)
        try:
            return self._synthetic(df)
        except Exception as exc:
            print(f"  [Oversample] {self.method} unavailable, fallback random: {exc}")
            return self._random(df)
