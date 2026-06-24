import pandas as pd
from ..base_op import TabularOp


class TargetEncode(TabularOp):
    """Target-mean encode categorical columns."""

    FIT_ON_TRAIN_ONLY = True

    def __init__(self, cols, target_col="rating", smoothing=1.0):
        super().__init__(name="TargetEncode")
        if isinstance(cols, str):
            cols = [cols]
        if smoothing < 0:
            raise ValueError("smoothing must be >= 0")
        self.cols = cols
        self.target_col = target_col
        self.smoothing = float(smoothing)
        self.encoding_map_ = {}
        self.global_mean_ = None
        self.fitted_ = False

    def get_op_description(self):
        description = """Operator name: TargetEncode

Function description:
Encode categorical values by smoothed target mean, useful for high-cardinality
categorical columns. Adds a "<col>_<target_col>" column per encoded column.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : str/list[str] — Categorical columns to encode.
target_col : str — Label/target column.
smoothing : float — Shrink category means toward the global mean.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'city': ['A', 'A', 'B'], 'label': [1, 0, 1]})
>>> op = TargetEncode(cols=['city'], target_col='label', smoothing=1.0)
>>> op.transform(df)
  city  label  city_label
0    A      1        0.50
1    A      0        0.50
2    B      1        0.75

Example YAML:
  - op: TargetEncode
    target: train
    params:
      cols: [city]
      target_col: label
      smoothing: 1.0
"""
        return description.strip()

    def transform(self, df):
        df = df.copy()
        if not self.fitted_:
            if self.target_col not in df.columns:
                return df
            self.global_mean_ = df[self.target_col].mean()
            for col in self.cols:
                if col not in df.columns:
                    continue
                stats = df.groupby(col)[self.target_col].agg(["mean", "count"])
                if self.smoothing > 0:
                    enc = (
                        (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_)
                        / (stats["count"] + self.smoothing)
                    )
                else:
                    enc = stats["mean"]
                self.encoding_map_[col] = enc.to_dict()
            self.fitted_ = True
        for col, enc in self.encoding_map_.items():
            if col not in df.columns:
                continue
            df[f"{col}_{self.target_col}"] = df[col].map(enc).fillna(self.global_mean_)
        return df.reset_index(drop=True)
