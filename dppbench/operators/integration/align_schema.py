import pandas as pd
from ..base_op import TabularOp


class AlignSchema(TabularOp):
    """Align column names and dtypes to a target schema."""

    def __init__(self, column_mapping=None, dtype_mapping=None, required_cols=None,
                 fill_value=None):
        super().__init__(name="AlignSchema")
        self.column_mapping = column_mapping or {}
        self.dtype_mapping = dtype_mapping or {}
        self.required_cols = required_cols or []
        self.fill_value = fill_value
        if not isinstance(self.column_mapping, dict):
            raise ValueError("column_mapping must be a dict {source: target}")
        if not isinstance(self.dtype_mapping, dict):
            raise ValueError("dtype_mapping must be a dict {col: dtype}")

    def get_op_description(self):
        description = """Operator name: AlignSchema

Function description:
Align field names, types, and required columns across
multiple tables, e.g. mapping uid -> user_id before concatenation or join.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
column_mapping : dict[str, str] — Source-to-target column name mapping.
dtype_mapping : dict[str, str] — Target dtype per column.
required_cols : list[str] — Columns that must exist after alignment.
fill_value : any — Value used for missing required columns. Default None.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'uid': [1, 2], 'amount': ['10.5', '20.0']})
>>> op = AlignSchema(column_mapping={'uid': 'user_id'}, dtype_mapping={'user_id': 'string', 'amount': 'float'})
>>> op.transform(df)
  user_id  amount
0       1    10.5
1       2    20.0

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: AlignSchema
    prev:
    - s0
    params:
      column_mapping:
        uid: user_id
      dtype_mapping:
        user_id: string
        amount: float
  train:
    prev:
    - o1
"""
        return description.strip()

    def transform(self, df):
        df = df.copy()
        existing_mapping = {
            src: dst for src, dst in self.column_mapping.items()
            if src in df.columns
        }
        if existing_mapping:
            df = df.rename(columns=existing_mapping)

        for col in self.required_cols:
            if col not in df.columns:
                df[col] = self.fill_value

        for col, dtype in self.dtype_mapping.items():
            if col not in df.columns:
                continue
            try:
                if str(dtype).startswith("datetime"):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif dtype in ("int", "int32", "int64"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                        "Int64" if dtype == "int" else dtype
                    )
                elif dtype in ("float", "float32", "float64"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(
                        "float64" if dtype == "float" else dtype
                    )
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as exc:
                print(f"  [AlignSchema] failed to cast {col} -> {dtype}: {exc}")
        return df
