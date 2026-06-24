from ..base_op import TabularOp


class CreateRollingFeature(TabularOp):
    """Create rolling aggregate features with one-step shift."""

    SUPPORTED_AGGS = {"mean", "std", "min", "max", "median", "sum"}
    MIN_PERIODS = 1

    def __init__(self, target_col, windows, aggs=None, group_cols=None, time_col=None):
        super().__init__(name="CreateRollingFeature")
        self.target_col = target_col
        self.windows = windows if isinstance(windows, list) else [windows]
        self.aggs = aggs or ["mean", "std"]
        unknown = set(self.aggs) - self.SUPPORTED_AGGS
        if unknown:
            raise ValueError(f"unsupported rolling aggs: {sorted(unknown)}")
        self.group_cols = group_cols or []
        self.time_col = time_col

    def get_op_description(self):
        description = """Operator name: CreateRollingFeature

Function description:
Add rolling statistics such as mean/std/sum over
historical values. Uses shift(1) to avoid leaking the current target value.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'user_id': [1, 1, 1], 'day': [1, 2, 3], 'amount': [10, 20, 30]})
>>> op = CreateRollingFeature(target_col='amount', windows=[2], aggs=['mean'], group_cols=['user_id'], time_col='day')
>>> op.transform(df)
   user_id  day  amount  amount_roll_mean_2
0        1    1      10                 NaN
1        1    2      20                10.0
2        1    3      30                15.0

Example YAML:
  - op: CreateRollingFeature
    target: train
    params:
      target_col: amount
      windows: [3, 7]
      aggs: [mean, sum]
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
        for window in self.windows:
            if group_cols:
                shifted = df.groupby(group_cols)[self.target_col].shift(1)
                roller = shifted.groupby([df[c] for c in group_cols]).rolling(
                    int(window), min_periods=self.MIN_PERIODS
                )
                result = roller.agg(self.aggs).reset_index(level=list(range(len(group_cols))), drop=True)
            else:
                roller = df[self.target_col].shift(1).rolling(
                    int(window), min_periods=self.MIN_PERIODS
                )
                result = roller.agg(self.aggs)
            if len(self.aggs) == 1:
                df[f"{self.target_col}_roll_{self.aggs[0]}_{window}"] = result
            else:
                for agg in self.aggs:
                    df[f"{self.target_col}_roll_{agg}_{window}"] = result[agg].values
        return df
