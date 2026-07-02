import pandas as pd
import numpy as np
from ..base_op import TabularOp


class LabelEncode(TabularOp):
    FIT_ON_TRAIN_ONLY = True

    def __init__(self, cols=None):
        super().__init__(name="LabelEncode")
        self.cols = cols
        self.mapping_ = {}
        self.cols_ = None
        self.fitted_ = False

    def get_op_description(self):
        description = """Operator name: LabelEncode

Function description:
Encode categorical (object/string) columns as integer codes using pd.factorize.
Suitable for tree-based models like LightGBM.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] or None — Columns to encode. If None, encodes all object-type columns.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'city': ['NY', 'LA', 'NY', 'SF']})
>>> op = LabelEncode(cols=['city'])
>>> op.transform(df)
   city
0     1
1     0
2     1
3     2

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: LabelEncode
    prev:
    - s0
    params:
      cols:
      - city
  train:
    prev:
    - o1
"""
        return description.strip()

    def transform(self, df):
        df = df.copy()
        if not self.fitted_:
            if self.cols is None:
                cols = [c for c in df.columns if df[c].dtype == object]
            else:
                cols = [c for c in self.cols if c in df.columns]
            self.cols_ = cols
            self.mapping_ = {}
            for col in cols:
                uniques = pd.Index(pd.unique(df[col].dropna()))
                self.mapping_[col] = {v: i for i, v in enumerate(uniques)}
            self.fitted_ = True
        else:
            cols = [c for c in (self.cols_ or []) if c in df.columns]

        for col in cols:
            df[col] = df[col].map(self.mapping_.get(col, {})).fillna(-1).astype(np.int64)

        return df
