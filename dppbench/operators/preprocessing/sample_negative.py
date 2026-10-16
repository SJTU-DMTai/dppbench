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
dag:
  sources:
  - id: s0
    table: interaction
  ops:
  - id: o1
    op: SampleNegative
    prev:
    - s0
    params:
      user_col: user_id
      item_col: item_id
      target_col: rating
      n_negatives: 1
  train:
    prev:
    - o1
"""
        return description.strip()

    @staticmethod
    def _is_time_col(col):
        lowered = str(col).lower()
        return "time" in lowered or "date" in lowered

    def _item_feature_cols(self, df):
        return [
            col for col in df.columns
            if str(col).startswith("item_")
            and col != self.item_col
            and col != self.target_col
            and not str(col).endswith("_seq")
        ]

    def transform(self, df):
        if self.n_negatives == 0:
            return df.reset_index(drop=True)

        allowed = {self.positive_label, self.negative_label}
        unique_labels = set(df[self.target_col].dropna().unique().tolist())
        if not unique_labels.issubset(allowed):
            raise ValueError(
                f"{self.target_col} must be binarized before SampleNegative, "
                f"found values: {unique_labels}"
            )

        rng = np.random.default_rng(self.seed)
        item_pool = pd.unique(df[self.item_col].dropna())
        if len(item_pool) == 0:
            return df.reset_index(drop=True)
        item_pool = np.asarray(item_pool)
        item_feature_cols = self._item_feature_cols(df)

        # Identify columns that must be zeroed out on negative rows
        # (numeric interaction-level features; id/time/user/seq/item_feat kept).
        zero_cols = []
        for col in df.columns:
            if col in (self.user_col, self.item_col, self.target_col, "__split__"):
                continue
            if col in item_feature_cols:
                continue
            if str(col).startswith("user_"):
                continue
            if str(col).endswith("_seq"):
                continue
            if self._is_time_col(col):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                zero_cols.append(col)

        pos_mask = df[self.target_col].values == self.positive_label
        pos_df = df.loc[pos_mask].reset_index(drop=True)
        neg_df = df.loc[~pos_mask].reset_index(drop=True)

        if len(pos_df) == 0:
            return df.reset_index(drop=True)

        users = pos_df[self.user_col].values
        n_pos = len(pos_df)
        total_neg = n_pos * self.n_negatives

        # Reject observed user-item pairs using a sparse set. Sampling all rows
        # together avoids one RNG call per user on large, sparse datasets.
        observed_pairs = set(zip(df[self.user_col], df[self.item_col]))
        item_counts = df.groupby(self.user_col)[self.item_col].nunique()
        saturated_users = set(item_counts[item_counts >= len(item_pool)].index)
        eligible_pos_idx = np.flatnonzero(~np.isin(users, list(saturated_users)))
        out_pos_idx = np.repeat(eligible_pos_idx, self.n_negatives)
        sampled_users = users[out_pos_idx]
        neg_item_ids = rng.choice(item_pool, size=len(out_pos_idx), replace=True)
        conflicts = np.fromiter(
            ((user, item) in observed_pairs
             for user, item in zip(sampled_users, neg_item_ids)),
            dtype=bool, count=len(out_pos_idx),
        )
        while conflicts.any():
            neg_item_ids[conflicts] = rng.choice(
                item_pool, size=int(conflicts.sum()), replace=True,
            )
            conflict_idx = np.flatnonzero(conflicts)
            conflicts[conflict_idx] = np.fromiter(
                ((sampled_users[i], neg_item_ids[i]) in observed_pairs
                 for i in conflict_idx),
                dtype=bool, count=len(conflict_idx),
            )
        write_ptr = len(out_pos_idx)

        if write_ptr == 0:
            return pd.concat([neg_df, pos_df], ignore_index=True)

        neg_item_ids = neg_item_ids[:write_ptr]
        out_pos_idx = out_pos_idx[:write_ptr]

        # Build negative rows by taking positive rows as template, then overriding.
        sampled = pos_df.iloc[out_pos_idx].copy().reset_index(drop=True)
        sampled[self.item_col] = neg_item_ids
        sampled[self.target_col] = self.negative_label

        # Zero out numeric interaction features that are meaningless when
        # copied from a positive template.
        if zero_cols:
            zeros = {c: 0 for c in zero_cols}
            sampled = sampled.assign(**zeros)

        # Back-fill item-side features using the canonical item feature map.
        if item_feature_cols:
            item_feat_df = (
                df[[self.item_col] + item_feature_cols]
                .drop_duplicates(subset=[self.item_col], keep="first")
                .reset_index(drop=True)
            )
            sampled = sampled.drop(columns=item_feature_cols, errors="ignore")
            sampled = sampled.merge(item_feat_df, on=self.item_col, how="left")

        out = pd.concat([neg_df, pos_df, sampled], ignore_index=True)
        return out
