import numpy as np
import pandas as pd
from ..base_op import TabularOp


class Undersample(TabularOp):
    """Reduce majority classes by random, Tomek Links, or ENN."""

    APPLIES_TO_STD_TEST = False
    RANDOM_STATE = 42

    def __init__(self, target_col, method="random", n_neighbors=3):
        super().__init__(name="Undersample")
        if method not in ("random", "tomek", "enn"):
            raise ValueError("method must be random/tomek/enn")
        self.target_col = target_col
        self.method = method
        self.n_neighbors = int(n_neighbors)

    def get_op_description(self):
        description = """Operator name: Undersample

Function description:
Undersample imbalanced labels with random undersampling, Tomek Links, or
Edited Nearest Neighbours.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
target_col : str — Label column.
method : str — random/tomek/enn (default 'random').
n_neighbors : int — Neighbours for ENN (default 3).

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'x': [1, 2, 3], 'label': [0, 0, 1]})
>>> op = Undersample(target_col='label', method='random')
>>> op.transform(df)
   x  label
0  2      0
1  3      1

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: Undersample
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

    def _numeric_xy(self, df):
        y = df[self.target_col].values
        x = df.drop(columns=[self.target_col])
        cols = x.select_dtypes(include=[np.number]).columns.tolist()
        return x[cols].fillna(0.0).values, y

    def _random(self, df):
        rng = np.random.RandomState(self.RANDOM_STATE)
        groups = list(df.groupby(self.target_col))
        if not groups:
            return df
        target_size = min(len(g) for _, g in groups)
        parts = []
        for _, group in groups:
            idx = rng.choice(group.index, size=target_size, replace=False)
            parts.append(df.loc[idx])
        return pd.concat(parts, axis=0).reset_index(drop=True)

    def transform(self, df):
        if self.target_col not in df.columns:
            return df
        if self.method == "random":
            return self._random(df)
        try:
            if self.method == "tomek":
                from imblearn.under_sampling import TomekLinks
                sampler = TomekLinks()
            else:
                from imblearn.under_sampling import EditedNearestNeighbours
                sampler = EditedNearestNeighbours(n_neighbors=self.n_neighbors)
            x, y = self._numeric_xy(df)
            sampler.fit_resample(x, y)
            return df.iloc[sampler.sample_indices_].reset_index(drop=True)
        except Exception as exc:
            print(f"  [Undersample] {self.method} unavailable, fallback random: {exc}")
            return self._random(df)
