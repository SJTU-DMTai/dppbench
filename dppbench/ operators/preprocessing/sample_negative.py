import numpy as np
import pandas as pd
from ..base_op import RecOp


class SampleNegative(RecOp):
    """Sample negative items for recommendation interactions."""

    APPLIES_TO_STD_TEST = False

    def __init__(self, user_col="user_id", item_col="item_id",
                 target_col="rating", n_negatives=1, positive_label=1,
                 negative_label=0, seed=42):
        super().__init__(name="SampleNegative")
        if n_negatives < 0:
            raise ValueError("n_negatives must be >= 0")
        self.user_col = user_col
        self.item_col = item_col
        self.target_col = target_col
        self.n_negatives = int(n_negatives)
        self.positive_label = positive_label
        self.negative_label = negative_label
        self.seed = seed

    def get_op_description(self):
        description = """Operator name: SampleNegative

Function description:
For every positive interaction, sample items the user
has not interacted with and append them as negative rows.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'user_id': [1, 1, 2], 'item_id': [10, 11, 10], 'rating': [1, 1, 1]})
>>> op = SampleNegative(user_col='user_id', item_col='item_id', target_col='rating', n_negatives=1, seed=0)
>>> op.transform(df)
   user_id  item_id  rating
0        1       10       1
1        1       11       1
2        2       10       1
3        2       11       0

Example YAML:
  - op: SampleNegative
    target: train
    params:
      user_col: user_id
      item_col: item_id
      target_col: rating
      n_negatives: 1
"""
        return description.strip()

    def transform(self, df):
        if self.n_negatives == 0:
            return df.reset_index(drop=True)
        allowed = {self.positive_label, self.negative_label}
        if not set(df[self.target_col].unique()).issubset(allowed):
            raise ValueError(f"{self.target_col} must be binarized before SampleNegative")
        rng = np.random.default_rng(self.seed)
        all_items = set(df[self.item_col].unique())
        user_items = df.groupby(self.user_col)[self.item_col].apply(set).to_dict()
        pos_df = df[df[self.target_col] == self.positive_label]
        neg_df = df[df[self.target_col] != self.positive_label]
        rows = []
        for _, row in pos_df.iterrows():
            cands = list(all_items - user_items.get(row[self.user_col], set()))
            if not cands:
                continue
            sample = rng.choice(cands, size=min(self.n_negatives, len(cands)), replace=False)
            for item in sample:
                new_row = row.copy()
                new_row[self.item_col] = item
                new_row[self.target_col] = self.negative_label
                rows.append(new_row)
        sampled = pd.DataFrame(rows, columns=df.columns)
        return pd.concat([neg_df, pos_df, sampled], ignore_index=True)
