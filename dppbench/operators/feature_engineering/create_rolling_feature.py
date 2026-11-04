from ..base_op import TabularOp


class CreateRollingFeature(TabularOp):
    """Rolling aggregates over strictly earlier, explicitly available history."""

    SUPPORTED_AGGS = {"mean", "std", "min", "max", "median", "sum"}
    MIN_PERIODS = 1

    def __init__(self, target_col, windows, aggs=None, group_cols=None, time_col=None,
                 available_at_col=None, availability=None):
        super().__init__(name="CreateRollingFeature")
        self.target_col = target_col
        from dppbench.tabular_history import positive_counts
        self.windows = positive_counts(windows, 'windows')
        self.aggs = aggs or ["mean", "std"]
        unknown = set(self.aggs) - self.SUPPORTED_AGGS
        if unknown:
            raise ValueError(f"unsupported rolling aggs: {sorted(unknown)}")
        self.group_cols = group_cols
        self.time_col = time_col
        self.available_at_col = available_at_col
        self.availability = availability

    def get_op_description(self):
        description = """Operator name: CreateRollingFeature

Function description:
Add rolling statistics such as mean/std/sum over
historical values within explicit entities. Equal-time peers are excluded.
Availability must be declared by available_at_col or availability='event_time'.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'user_id': [1, 1, 1], 'day': [1, 2, 3], 'amount': [10, 20, 30]})
>>> op = CreateRollingFeature(target_col='amount', windows=[2], aggs=['mean'], group_cols=['user_id'], time_col='day', availability='event_time')
>>> op.transform(df)
   user_id  day  amount  amount_roll_mean_2
0        1    1      10                 NaN
1        1    2      20                10.0
2        1    3      30                15.0

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: CreateRollingFeature
    prev:
    - s0
    params:
      target_col: amount
      windows:
      - 3
      - 7
      aggs:
      - mean
      - sum
      group_cols:
      - user_id
      time_col: day
      availability: event_time
  train:
    prev:
    - o1
"""
        return description.strip()

    def transform(self, df):
        from dppbench.tabular_history import history_features
        return history_features(df, df, self)
