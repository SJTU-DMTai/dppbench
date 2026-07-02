import numpy as np
import pandas as pd
from ..base_op import TabularOp


class FrequencyEncode(TabularOp):
    """Add frequency-count features for high-cardinality categorical columns."""

    FIT_ON_TRAIN_ONLY = True

    def __init__(self, cols, suffix="_freq", drop_original=False):
        super().__init__(name="FrequencyEncode")
        self.cols = cols if isinstance(cols, list) else [cols]
        self.suffix = suffix
        self.drop_original = bool(drop_original)
        self.freq_maps_ = {}
        self.fitted_ = False
        self.output_col_types = {f"{col}{self.suffix}": "numeric" for col in self.cols}

    def get_op_description(self):
        description = """Operator name: FrequencyEncode

Function description:
Encode categorical columns by their empirical frequency in the training table.
For each source column, this operator appends a numeric ``<col><suffix>``
feature containing the count of that value in the fit data. Unseen values map
to 0.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : str/list[str] — Columns to frequency-encode.
suffix : str — Output column suffix, default ``_freq``.
drop_original : bool — Drop source columns after creating frequency features.

Output:
pd.DataFrame — Transformed table with frequency-count feature columns.

Example:
>>> df = pd.DataFrame({'user_id': ['u1', 'u1', 'u2'], 'tmp': [1, 2, 3]})
>>> op = FrequencyEncode(cols=['user_id'])
>>> op.transform(df)
  user_id  tmp  user_id_freq
0      u1    1             2
1      u1    2             2
2      u2    3             1

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: FrequencyEncode
    prev:
    - s0
    params:
      cols:
      - user_id
  train:
    prev:
    - o1
"""
        return description.strip()

    def transform(self, df):
        df = df.copy()
        if not self.fitted_:
            for col in self.cols:
                if col in df.columns:
                    self.freq_maps_[col] = df[col].value_counts()
            self.fitted_ = True
        for col, freq in self.freq_maps_.items():
            if col not in df.columns:
                continue
            df[f"{col}{self.suffix}"] = df[col].map(freq).fillna(0).astype(np.int32)
            if self.drop_original:
                df = df.drop(columns=[col])
        return df
