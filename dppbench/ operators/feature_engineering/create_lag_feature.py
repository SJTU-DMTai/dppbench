from ..base_op import TabularOp


class CreateLagFeature(TabularOp):
    """Create lagged time-series features."""

    def __init__(self, target_col, lags, group_cols=None, time_col=None):
        super().__init__(name="CreateLagFeature")
        self.target_col = target_col
        self.lags = lags if isinstance(lags, list) else [lags]
        self.group_cols = group_cols or []
        self.time_col = time_col

    def get_op_description(self):
        description = """Operator name: CreateLagFeature

Function description:
Add <target>_lag_k columns using shift(k), optionally
within groups and sorted by time.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'user_id': [1, 1, 1], 'day': [1, 2, 3], 'amount': [10, 20, 30]})
>>> op = CreateLagFeature(target_col='amount', lags=[1], group_cols=['user_id'], time_col='day')
>>> op.transform(df)
   user_id  day  amount  amount_lag_1
0        1    1      10           NaN
1        1    2      20          10.0
2        1    3      30          20.0

Example YAML:
  - op: CreateLagFeature
    target: train
    params:
      target_col: amount
      lags: [1, 7]
      group_cols: [user_id]
      time_col: day
"""
        return description.strip()

    def transform(self, df):
        if self.target_col not in df.columns:
            return df
        df = df.copy()
        sort_cols = [c for c in self.group_cols if c in df.columns]
        if self.time_col and self.time_col in df.columns:
            sort_cols.append(self.time_col)
        if sort_cols:
            df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        group_cols = [c for c in self.group_cols if c in df.columns]
        source = df.groupby(group_cols)[self.target_col] if group_cols else df[self.target_col]
        for lag in self.lags:
            df[f"{self.target_col}_lag_{lag}"] = source.shift(int(lag))
        return df
