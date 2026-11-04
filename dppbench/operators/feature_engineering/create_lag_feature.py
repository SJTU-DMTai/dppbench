from ..base_op import TabularOp


class CreateLagFeature(TabularOp):
    """Observation lags within an explicit entity and strict available history."""

    def __init__(self, target_col, lags, group_cols=None, time_col=None,
                 available_at_col=None, availability=None):
        super().__init__(name="CreateLagFeature")
        self.target_col = target_col
        from dppbench.tabular_history import positive_counts
        self.lags = positive_counts(lags, 'lags')
        self.group_cols = group_cols
        self.time_col = time_col
        self.available_at_col = available_at_col
        self.availability = availability

    def get_op_description(self):
        description = """Operator name: CreateLagFeature

Function description:
Add <target>_lag_k columns using shift(k), optionally
within explicit entities and strictly earlier event times. Availability must
be declared by available_at_col or availability='event_time'. Dataset replay
uses training history for held-out queries; equal-time peers are excluded.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'user_id': [1, 1, 1], 'day': [1, 2, 3], 'amount': [10, 20, 30]})
>>> op = CreateLagFeature(target_col='amount', lags=[1], group_cols=['user_id'], time_col='day', availability='event_time')
>>> op.transform(df)
   user_id  day  amount  amount_lag_1
0        1    1      10           NaN
1        1    2      20          10.0
2        1    3      30          20.0

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: CreateLagFeature
    prev:
    - s0
    params:
      target_col: amount
      lags:
      - 1
      - 7
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
