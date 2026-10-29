import pandas as pd
import numpy as np
from ..base_op import RecOp


class CreateSequence(RecOp):
    USES_TRAIN_HISTORY_FOR_STD_TEST = True

    def __init__(self, user_col="user_id", item_col="item_id",
                 time_col="timestamp", seq_col="item_id_seq", max_len=50):
        super().__init__(name="CreateSequence")

        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")

        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.seq_col = seq_col
        self.max_len = max_len
        self.output_col_types = {seq_col: "categorical_list"}

    def get_op_description(self):
        description = """Operator name: CreateSequence

Function description:
Construct a sequence feature for each row representing the user's historical
interactions prior to the current row. The sequence is sorted by time (most recent last), and truncated
to max_len by keeping the most recent interactions. Each element in the sequence can be the item ID
(default) or a list of item feature values specified via feature_cols.

Input:
df : pd.DataFrame — Interaction DataFrame containing user_col, item_col, and time_col.

Parameters:
user_col : str — Column name for user identifiers. Default: 'user_id'.
item_col : str — Column name for item identifiers. Default: 'item_id'.
time_col : str — Column name for timestamps used to determine interaction order. Default: 'timestamp'.
seq_col : str — Name of the output sequence column. Default: 'item_id_seq'.
max_len : int — Maximum sequence length. If exceeded, keep the most recent interactions. Default: 50.
feature_cols : list[str] or None — Columns to include as each element in the sequence. If None, each
element is the item ID. If provided, each element is a dict of {col: value} pairs. Default: None.

Output:
pd.DataFrame — Original DataFrame with an additional sequence column. Each entry is a list representing
the user's historical interactions prior to the current row.

Example:
>>> df = pd.DataFrame({'user_id': [1, 1, 1, 2, 2], 'item_id': [101, 102, 103, 201, 202], 'timestamp': [1, 2, 3, 1, 2]})
>>> op = CreateSequence(user_col='user_id', item_col='item_id', time_col='timestamp', max_len=2)
>>> op.transform(df)
   user_id  item_id  timestamp item_id_seq
0        1      101          1          []
1        1      102          2       [101]
2        1      103          3  [101, 102]
3        2      201          1          []
4        2      202          2       [201]

Example YAML:
dag:
  sources:
  - id: s0
    table: interaction
  ops:
  - id: o1
    op: CreateSequence
    prev:
    - s0
    params:
      user_col: user_id
      item_col: item_id
      time_col: timestamp
      seq_col: item_id_seq
      max_len: 50
  train:
    prev:
    - o1
"""
        return description.strip()

    def transform(self, df):
        train = df.loc[df["__split__"].eq("train")] if "__split__" in df else df
        return self.transform_with_history(df, train)

    def transform_with_history(self, df, train):
        """Query a separate, label-free training context without sampling it away."""
        if "__split__" in train and not train["__split__"].eq("train").all():
            raise ValueError('CreateSequence reference history must contain training events only')
        # Events at the same timestamp are not assumed to have an observable order.
        import bisect
        df = df.copy()
        sequences = [[] for _ in range(len(df))]
        positions = {}
        for pos, uid in enumerate(df[self.user_col]):
            positions.setdefault(uid, []).append(pos)
        for uid, history in train.groupby(self.user_col, sort=False, dropna=False):
            history = history.sort_values(self.time_col, kind="stable")
            if history[self.time_col].isna().any():
                raise ValueError("CreateSequence requires non-null training timestamps")
            times = history[self.time_col].tolist()
            items = history[self.item_col].tolist()
            for pos in positions.get(uid, []):
                when = df[self.time_col].iloc[pos]
                if pd.isna(when):
                    continue
                stop = bisect.bisect_left(times, when)
                sequences[pos] = items[max(0, stop-self.max_len):stop]
        df[self.seq_col] = sequences
        return df.reset_index(drop=True)
