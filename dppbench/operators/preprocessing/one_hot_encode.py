import pandas as pd
import numpy as np
from ..base_op import BaseOp


class OneHotEncode(BaseOp):
    """One-hot encode categorical columns into dummy 0/1 indicator columns."""

    FIT_ON_TRAIN_ONLY = True

    def __init__(self, cols, max_cardinality=None):
        super().__init__(name="OneHotEncode")
        self.op_type = "basic op"
        self.cols = cols if isinstance(cols, list) else [cols]
        self.max_cardinality = max_cardinality
        self.cols_to_encode_ = []
        self.output_columns_ = None
        self.fitted_ = False
        self.output_col_types = {}
        self.drop_col_types = []

    def get_op_description(self):
        description = """Operator name: OneHotEncode

Function description:
One-hot encode categorical columns; produces a 0/1 column per unique value.
For high-cardinality cols, set max_cardinality to skip columns whose nunique
exceeds it (with a warning).

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] — Columns to encode.
max_cardinality : int or None — Skip cols whose nunique > this. Default None.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'education': ['HS', 'BS', 'HS']})
>>> op = OneHotEncode(cols=['education'])
>>> op.transform(df)
   education_BS  education_HS
0             0             1
1             1             0
2             0             1

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: OneHotEncode
    prev:
    - s0
    params:
      cols:
      - education
  train:
    prev:
    - o1
"""
        return description.strip()

    def transform(self, df):
        if not self.fitted_:
            cols_to_encode = []
            for c in self.cols:
                if c not in df.columns:
                    print(f"  [OneHotEncode] skip '{c}' (column not found in df)")
                    continue
                n_unique = df[c].nunique(dropna=True)
                if self.max_cardinality is not None and n_unique > self.max_cardinality:
                    print(f"  [OneHotEncode] skip '{c}' (nunique={n_unique} > {self.max_cardinality})")
                    continue
                cols_to_encode.append(c)
            seen = set()
            cols_to_encode = [c for c in cols_to_encode if not (c in seen or seen.add(c))]
            self.cols_to_encode_ = cols_to_encode
            self.fitted_ = True
        else:
            cols_to_encode = [c for c in self.cols_to_encode_ if c in df.columns]
        if not cols_to_encode:
            if self.fitted_ and self.output_columns_ is not None:
                return self._align_output(df)
            return df.copy(deep=False)

        df_for_dummies = df.copy(deep=False)
        for c in cols_to_encode:
            df_for_dummies[c] = df_for_dummies[c].astype("category")
        from dppbench.memory import require_memory_headroom
        width = sum(len(df_for_dummies[c].cat.categories) for c in cols_to_encode)
        require_memory_headroom(len(df) * width * 3, "OneHotEncode")
        encoded = pd.get_dummies(
            df_for_dummies,
            columns=cols_to_encode,
            dtype="int8",
        )
        if self.output_columns_ is None:
            self.output_columns_ = encoded.columns.tolist()
            retained_columns = set(df.columns) - set(cols_to_encode)
            self.output_col_types = {
                col: "numeric" for col in encoded.columns
                if col not in retained_columns
            }
            self.drop_col_types = list(cols_to_encode)
        else:
            encoded = self._align_output(encoded)
        return encoded

    def _align_output(self, encoded):
        encoded = encoded.reindex(columns=self.output_columns_, fill_value=np.int8(0))
        # reindex creates absent dummy columns as int64. Without this cast,
        # concatenating holdout partitions promotes training's int8 columns
        # to int64 as well (8x storage for otherwise identical 0/1 values).
        for col in self.output_col_types:
            if encoded[col].dtype != np.dtype("int8"):
                encoded[col] = encoded[col].astype("int8")
        return encoded
