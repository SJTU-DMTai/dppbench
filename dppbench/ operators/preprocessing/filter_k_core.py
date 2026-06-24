from ..base_op import RecOp


class FilterKCore(RecOp):
    """Iteratively keep only users/items with at least k interactions."""

    APPLIES_TO_STD_TEST = False
    FILTER_STD_TEST_TO_TRAIN_DOMAIN = False
    MAX_ITER = 100

    def __init__(self, user_col="user_id", item_col="item_id", k=5):
        super().__init__(name="FilterKCore")
        self.user_col = user_col
        self.item_col = item_col
        self.k = int(k)

    def get_op_description(self):
        description = """Operator name: FilterKCore

Function description:
Recommendation k-core filtering. Iteratively removes users and items with
fewer than k interactions until convergence (or 100 iterations).

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
user_col : str — User identifier column.
item_col : str — Item identifier column.
k : int — Minimum interactions per user/item.

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
        for _ in range(self.MAX_ITER):
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
