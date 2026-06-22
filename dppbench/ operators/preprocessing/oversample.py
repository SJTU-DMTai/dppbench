import numpy as np
import pandas as pd
from ..base_op import TabularOp


class Oversample(TabularOp):
    """Increase minority classes by random, SMOTE, SMOTE-NC, or ADASYN."""

    APPLIES_TO_STD_TEST = False

    def __init__(self, target_col, method="random", random_state=42,
                 sampling_strategy="auto", n_neighbors=5,
                 categorical_features=None):
        super().__init__(name="Oversample")
        if method not in ("random", "smote", "adasyn", "smote_nc"):
            raise ValueError("method must be random/smote/adasyn/smote_nc")
        self.target_col = target_col
        self.method = method
        self.random_state = int(random_state)
        self.sampling_strategy = sampling_strategy
        self.n_neighbors = int(n_neighbors)
        self.categorical_features = categorical_features

    def get_op_description(self):
        description = """Operator name: Oversample

Function description:
Oversample minority labels with random duplication,
SMOTE, SMOTE-NC, or ADASYN. Falls back to random oversampling if imblearn is
unavailable or the selected synthetic method cannot run.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'x': [1, 2, 3], 'label': [0, 0, 1]})
>>> op = Oversample(target_col='label', method='random', random_state=0)
>>> op.transform(df)
   x  label
0  1      0
1  2      0
2  3      1
3  3      1

Example YAML:
  - op: Oversample
    target: train
    params:
      target_col: label
      method: random
      random_state: 42
"""
        return description.strip()

    def _random(self, df):
        rng = np.random.RandomState(self.random_state)
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
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.n_neighbors,
            )
        elif self.method == "adasyn":
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy,
                n_neighbors=self.n_neighbors,
            )
        else:
            from imblearn.over_sampling import SMOTENC
            cat = self.categorical_features or []
            sampler = SMOTENC(
                categorical_features=cat,
                random_state=self.random_state,
                sampling_strategy=self.sampling_strategy,
                k_neighbors=self.n_neighbors,
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
