import hashlib
import pandas as pd
from ..base_op import BaseOp


class HashEncode(BaseOp):
    """Hash high-cardinality categorical columns into ``n_buckets`` integer ids.

    Distinct from:
      - LabelEncode    : 1-to-1 dictionary; cardinality unchanged.
      - FrequencyEncode: encodes by frequency, not identity.
      - OneHotEncode   : explodes columns, unsuitable for very high cardinality.
      - HashEncode     : fixed-width hashed bucket id; many-to-one collisions.
    """

    def __init__(self, cols, n_buckets=1024, prefix="h_", drop_original=False):
        super().__init__(name="HashEncode")
        self.op_type = "basic op"
        self.cols = cols if isinstance(cols, list) else [cols]
        self.n_buckets = int(n_buckets)
        self.prefix = prefix
        self.drop_original = bool(drop_original)

    def get_op_description(self):
        description = """Operator name: HashEncode

Function description:
Replace each value of a high-cardinality column with
``hash(value) % n_buckets``, written into ``<prefix><col>``.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] — Columns to hash-encode.
n_buckets : int — Number of hash buckets (output integer range [0, n_buckets)).
prefix : str — Output column prefix (default ``h_``).
drop_original : bool — Drop the source column after hashing.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'card1': ['user_42', 'user_7', 'user_42']})
>>> op = HashEncode(cols=['card1'], n_buckets=8)
>>> op.transform(df)
     card1  h_card1
0  user_42        5
1   user_7        2
2  user_42        5

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: HashEncode
    prev:
    - s0
    params:
      cols:
      - card1
      n_buckets: 4096
      prefix: h_
  train:
    prev:
    - o1
"""
        return description.strip()

    @staticmethod
    def _hash_to_bucket(value, n_buckets):
        if pd.isna(value):
            return 0
        h = hashlib.md5(str(value).encode("utf-8")).digest()
        return int.from_bytes(h[:8], "big") % n_buckets

    def transform(self, df):
        df = df.copy()
        for c in self.cols:
            if c not in df.columns:
                continue
            out = df[c].apply(lambda v: self._hash_to_bucket(v, self.n_buckets))
            df[f"{self.prefix}{c}"] = out.astype("int32")
            if self.drop_original:
                df = df.drop(columns=[c])
            print(f"  [HashEncode] '{c}' -> '{self.prefix}{c}' ({self.n_buckets} buckets)")
        return df
