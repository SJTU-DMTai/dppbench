import pandas as pd
import numpy as np
from ..base_op import RecOp


class CreateSequence(RecOp):
    USES_TRAIN_HISTORY_FOR_STD_TEST = True

    def __init__(self, user_col="user_id", item_col="item_id",
                 time_col="timestamp", seq_col="item_id_seq",
                 max_len=50, feature_cols=None):
        super().__init__(name="CreateSequence")

        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")

        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.seq_col = seq_col
        self.max_len = max_len
        self.feature_cols = feature_cols
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
  - op: CreateSequence
    target: train
    params:
      user_col: user_id
      item_col: item_id
      time_col: timestamp
      seq_col: item_id_seq
      max_len: 50
"""
        return description.strip()

    def transform(self, df):
        df = df.copy()
        has_split = "__split__" in df.columns
        if has_split:
            df["__dppbench_seq_order__"] = np.arange(len(df))
            # std-test candidates should be scored against training history
            # only; they must not leak held-out positives into other candidates.
            df["__dppbench_seq_rank__"] = (
                df["__split__"].astype(str).eq("std_test").astype(int)
            )
            sort_cols = [self.user_col]
            if self.time_col in df.columns:
                sort_cols.append(self.time_col)
            sort_cols.extend(["__dppbench_seq_rank__", "__dppbench_seq_order__"])
            df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
        else:
            df = df.sort_values([self.user_col, self.time_col]).reset_index(drop=True)

        users = df[self.user_col].values

        if self.feature_cols is not None:
            elements = df[self.feature_cols].to_dict("records")
        else:
            elements = df[self.item_col].values

        sequences = [None] * len(df)
        user_hist = {}

        for i in range(len(df)):
            uid = users[i]
            hist = user_hist.get(uid)
            if hist is None:
                sequences[i] = []
                user_hist[uid] = []
            else:
                sequences[i] = hist[-self.max_len:]
            if not (has_split and df.at[i, "__split__"] == "std_test"):
                user_hist[uid].append(elements[i])

        df[self.seq_col] = sequences
        if has_split:
            df = (
                df.sort_values("__dppbench_seq_order__")
                .drop(columns=["__dppbench_seq_order__", "__dppbench_seq_rank__"])
                .reset_index(drop=True)
            )
        return df.reset_index(drop=True)
