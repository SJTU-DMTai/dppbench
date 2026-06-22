import pandas as pd
import numpy as np
from ..base_op import BaseOp


class OrdinalEncode(BaseOp):
    """Encode categorical columns into integers following an explicit ordering.

    Distinct from:
      - LabelEncode : assigns integers in alphabetical / first-seen order.
      - FrequencyEncode : encodes by frequency.
      - OneHotEncode : produces dummy 0/1 indicator cols per value.
      - TargetEncoding : encodes by target mean.
    OrdinalEncode lets you pass an ordered ``ordering`` list per column to
    enforce the desired rank (e.g. junior < classic < gold).
    """
    FIT_ON_TRAIN_ONLY = True


    def __init__(self, cols, ordering=None, unknown_value=-1):
        super().__init__(name="OrdinalEncode")
        self.op_type = "basic op"
        self.cols = cols if isinstance(cols, list) else [cols]
        # ordering: dict[col -> list[value]] OR list[value] (applied to all cols)
        self.ordering = ordering or {}
        self.unknown_value = unknown_value
        self.mapping_ = {}
        self.fitted_ = False

    def get_op_description(self):
        description = """Operator name: OrdinalEncode

Function description:
Map each value in the specified categorical columns to
an integer following an explicit user-supplied ordering. Values that are not
listed are encoded with ``unknown_value`` (default -1).

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] — Columns to encode.
ordering : dict[str -> list] OR list — Per-column ordering, or one shared
list applied to every column.
unknown_value : int — Value for unseen categories (default -1).

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'type': ['classic', 'gold', 'junior']})
>>> op = OrdinalEncode(cols=['type'], ordering={'type': ['junior', 'classic', 'gold']})
>>> op.transform(df)
   type
0     1
1     2
2     0

Example YAML:
  - op: OrdinalEncode
    target: train
    params:
      cols: [type]
      ordering:
        type: [junior, classic, gold]
"""
        return description.strip()

    def transform(self, df):
        df = df.copy()
        if not self.fitted_:
            self.mapping_ = {}
            for c in self.cols:
                if c not in df.columns:
                    continue
                if isinstance(self.ordering, dict):
                    order = self.ordering.get(c)
                else:
                    order = self.ordering
                if not order:
                    # Learn default ordering from the training slice only.
                    order = [v for v in df[c].dropna().unique().tolist()]
                self.mapping_[c] = {v: i for i, v in enumerate(order)}
            self.fitted_ = True

        for c, mapping in self.mapping_.items():
            if c not in df.columns:
                continue
            df[c] = df[c].map(mapping).fillna(self.unknown_value).astype(int)
            print(f"  [OrdinalEncode] '{c}' encoded with {len(mapping)} levels")
        return df
