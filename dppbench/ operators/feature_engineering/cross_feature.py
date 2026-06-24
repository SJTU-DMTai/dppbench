from ..base_op import TabularOp


class CrossFeature(TabularOp):
    """Create categorical cross features by concatenating columns."""

    SEPARATOR = "_"

    def __init__(self, cols, output_col=None):
        super().__init__(name="CrossFeature")
        self.cols = cols if isinstance(cols, list) else [cols]
        self.output_col = output_col

    def get_op_description(self):
        description = """Operator name: CrossFeature

Function description:
Create a categorical cross feature such as
user_id x item_id or city x category by joining column values row-wise.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'city': ['BJ', 'SH'], 'device': ['ios', 'android']})
>>> op = CrossFeature(cols=['city', 'device'], output_col='city_device')
>>> op.transform(df)
  city   device city_device
0   BJ      ios      BJ_ios
1   SH  android  SH_android

Example YAML:
  - op: CrossFeature
    target: both
    params:
      cols: [city, device]
      output_col: city_device
"""
        return description.strip()

    def transform(self, df):
        cols = [c for c in self.cols if c in df.columns]
        if len(cols) < 2:
            return df
        df = df.copy()
        out_col = self.output_col or self.SEPARATOR.join(cols)
        values = df[cols].where(df[cols].notna(), "")
        df[out_col] = values.astype(str).agg(self.SEPARATOR.join, axis=1)
        return df
