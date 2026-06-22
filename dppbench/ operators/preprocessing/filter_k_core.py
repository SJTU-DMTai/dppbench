from ..base_op import RecOp


class FilterKCore(RecOp):
    """Iteratively keep only users/items with at least k interactions."""

    APPLIES_TO_STD_TEST = False
    FILTER_STD_TEST_TO_TRAIN_DOMAIN = False

    def __init__(self, user_col="user_id", item_col="item_id", k=5, max_iter=100):
        super().__init__(name="FilterKCore")
        self.user_col = user_col
        self.item_col = item_col
        self.k = int(k)
        self.max_iter = int(max_iter)

    def get_op_description(self):
        description = """Operator name: FilterKCore

Function description:
Recommendation k-core filtering. Iteratively removes
users and items with fewer than k interactions until convergence.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'user_id': [1, 1, 2], 'item_id': [10, 11, 10]})
>>> op = FilterKCore(user_col='user_id', item_col='item_id', k=2)
>>> op.transform(df)
   user_id  item_id
0        1       10
1        1       11

Example YAML:
  - op: FilterKCore
    target: train
    params:
      user_col: user_id
      item_col: item_id
      k: 2
"""
        return description.strip()

    def transform(self, df):
        if self.k <= 1:
            return df
        df = df.copy()
        prev_len = -1
        if not self.user_col and not self.item_col:
            return df.reset_index(drop=True)
        for _ in range(self.max_iter):
            keep = None
            if self.user_col:
                user_counts = df[self.user_col].value_counts()
                users = set(user_counts[user_counts >= self.k].index)
                keep = df[self.user_col].isin(users)
            if self.item_col:
                item_counts = df[self.item_col].value_counts()
                items = set(item_counts[item_counts >= self.k].index)
                item_keep = df[self.item_col].isin(items)
                keep = item_keep if keep is None else (keep & item_keep)
            df = df[keep]
            if len(df) == prev_len:
                break
            prev_len = len(df)
        return df.reset_index(drop=True)
