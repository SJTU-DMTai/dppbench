from ..base_op import TabularOp


class AggregateGroupFeature(TabularOp):
    """Create group-by aggregate features and merge them back."""

    def __init__(self, group_cols, agg_cols=None, agg_funcs=None):
        super().__init__(name="AggregateGroupFeature")
        self.group_cols = group_cols if isinstance(group_cols, list) else [group_cols]
        self.agg_cols = agg_cols
        self.agg_funcs = agg_funcs or ["mean", "sum", "count", "std"]

    def get_op_description(self):
        description = """Operator name: AggregateGroupFeature

Function description:
Compute group statistics such as mean/sum/count/std by
key and append the aggregated features to every row.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'user_id': [1, 1, 2], 'amount': [10, 20, 5]})
>>> op = AggregateGroupFeature(group_cols='user_id', agg_cols=['amount'], agg_funcs=['mean', 'count'])
>>> op.transform(df)
   user_id  amount  amount_mean  amount_count
0        1      10         15.0             2
1        1      20         15.0             2
2        2       5          5.0             1

Example YAML:
  - op: AggregateGroupFeature
    target: train
    params:
      group_cols: user_id
      agg_cols: [amount]
      agg_funcs: [mean, count]
"""
        return description.strip()

    def transform(self, df):
        keys = [c for c in self.group_cols if c in df.columns]
        if not keys:
            return df
        if self.agg_cols is None:
            cols = [
                c for c in df.columns
                if c not in keys and df[c].dtype.kind in ("i", "u", "f")
            ]
        else:
            cols = [c for c in self.agg_cols if c in df.columns and c not in keys]
        if not cols:
            return df
        grouped = df.groupby(keys)[cols].agg(self.agg_funcs)
        grouped.columns = [f"{col}_{func}" for col, func in grouped.columns]
        grouped = grouped.reset_index()
        return df.merge(grouped, on=keys, how="left")
