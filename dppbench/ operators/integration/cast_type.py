import pandas as pd
from ..base_op import TabularOp


class CastType(TabularOp):
    """Coerce specified columns to target dtypes."""

    DTYPE_ALIASES = {
        "int": "Int64", "int32": "Int32", "int64": "Int64",
        "float": "float64", "float32": "float32", "float64": "float64",
        "str": "string", "string": "string",
        "bool": "boolean",
        "category": "category",
        "datetime": "datetime64[ns]",
    }

    def __init__(self, col_dtypes):
        super().__init__(name="CastType")
        if not isinstance(col_dtypes, dict):
            raise ValueError("col_dtypes must be a dict {col_name: dtype}")
        self.col_dtypes = col_dtypes

    def get_op_description(self):
        description = """Operator name: CastType

Function description:
Cast columns to specified dtypes. Failed numeric or
datetime coercions become missing values where pandas supports it.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
col_dtypes : dict[str, str] — Mapping {column_name: target_dtype}. Supported:
int/int32/int64/float/float32/float64/str/string/bool/category/datetime.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'age': ['20', '35'], 'is_vip': ['true', 'false']})
>>> op = CastType(col_dtypes={'age': 'int32', 'is_vip': 'bool'})
>>> op.transform(df)
   age  is_vip
0   20    True
1   35   False

Example YAML:
  - op: CastType
    target: both
    params:
      col_dtypes:
        age: int32
        is_vip: bool
"""
        return description.strip()

    def transform(self, df):
        df = df.copy()
        for col, dtype in self.col_dtypes.items():
            if col not in df.columns:
                continue
            target = self.DTYPE_ALIASES.get(dtype, dtype)
            try:
                if str(target).startswith("datetime"):
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                elif target in ("Int32", "Int64", "int32", "int64"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(target)
                elif str(target).startswith("float"):
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype(target)
                else:
                    df[col] = df[col].astype(target)
            except Exception as exc:
                print(f"  [CastType] failed to cast {col} -> {target}: {exc}")
        return df
