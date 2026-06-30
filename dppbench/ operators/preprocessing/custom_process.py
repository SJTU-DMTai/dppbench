import numpy as np
import pandas as pd
from ..base_op import TabularOp
from ..custom_op import CustomOp


class CustomProcess(TabularOp):
    """Custom preprocessing operator plus common utility modes."""

    FIT_ON_TRAIN_ONLY = True

    def __init__(self, code=None, entry="pipeline", func=None, mode="code",
                 cols=None, threshold=0.8):
        super().__init__(name="CustomProcess")
        if mode not in ("code", "drop_high_null", "frequency_encode"):
            raise ValueError("mode must be code/drop_high_null/frequency_encode")
        self.code = code
        self.entry = entry or "pipeline"
        self.func = func
        self.mode = mode
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.threshold = float(threshold)
        self.freq_maps_ = {}
        self.drop_cols_ = []
        self.fitted_ = False
        self._custom = CustomOp(code=code, entry=self.entry) if code else None

    def get_op_description(self):
        description = """Operator name: CustomProcess

Function description:
User-defined preprocessing. Built-in modes cover high-null column filtering
and frequency encoding.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
mode : str — code/drop_high_null/frequency_encode.
cols : list[str] or None — Columns for frequency_encode.
threshold : float — Null ratio threshold for drop_high_null.
code, entry, func — Custom execution inputs.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'user_id': ['u1', 'u1', 'u2'], 'tmp': [1, 2, 3]})
>>> op = CustomProcess(mode='frequency_encode', cols=['user_id'])
>>> op.transform(df)
  user_id  tmp  user_id_freq
0      u1    1             2
1      u1    2             2
2      u2    3             1

Example YAML:
  - op: CustomProcess
    target: train
    params:
      mode: frequency_encode
      cols: [user_id]
"""
        return description.strip()

    def transform(self, df):
        df = df.copy()
        if self.func is not None:
            result = self.func(df)
            if not isinstance(result, pd.DataFrame):
                raise TypeError("CustomProcess func must return a pandas DataFrame")
            return result
        if self.mode == "drop_high_null":
            if not self.fitted_:
                ratios = df.isna().mean()
                self.drop_cols_ = ratios[ratios > self.threshold].index.tolist()
                self.fitted_ = True
            existing = [c for c in self.drop_cols_ if c in df.columns]
            return df.drop(columns=existing)
        if self.mode == "frequency_encode":
            if not self.fitted_:
                for col in self.cols or []:
                    if col in df.columns:
                        self.freq_maps_[col] = df[col].value_counts()
                self.fitted_ = True
            for col, freq in self.freq_maps_.items():
                if col in df.columns:
                    df[f"{col}_freq"] = df[col].map(freq).fillna(0).astype(np.int32)
            return df
        if self._custom is not None:
            return self._custom.transform(df)
        return df
