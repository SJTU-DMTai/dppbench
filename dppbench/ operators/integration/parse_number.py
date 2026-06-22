import pandas as pd
from ..base_op import TabularOp


class ParseNumber(TabularOp):
    """Parse string numeric columns into numeric dtype."""

    def __init__(self, cols, output_cols=None, dtype="float64", errors="coerce",
                 remove_chars=None, thousands=None, decimal="."):
        super().__init__(name="ParseNumber")
        self.cols = cols if isinstance(cols, list) else [cols]
        self.output_cols = output_cols
        if self.output_cols is not None:
            self.output_cols = (
                output_cols if isinstance(output_cols, list) else [output_cols]
            )
            if len(self.output_cols) != len(self.cols):
                raise ValueError("output_cols length must match cols length")
        if errors not in ("coerce", "raise", "ignore"):
            raise ValueError("errors must be 'coerce', 'raise' or 'ignore'")
        self.dtype = dtype
        self.errors = errors
        self.remove_chars = remove_chars or []
        self.thousands = thousands
        self.decimal = decimal

    def get_op_description(self):
        description = """Operator name: ParseNumber

Function description:
Convert string-valued numeric columns into numeric
dtype, optionally stripping currency symbols, thousands separators, or custom
characters first.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : str or list[str] — Source columns.
output_cols : str/list[str] or None — If given, write parsed values to these
columns; otherwise overwrite source columns.
dtype : str — Output dtype, e.g. float64/int64/Int64.
errors : str — pandas coercion mode: coerce/raise/ignore.
remove_chars : list[str] — Literal characters to remove before parsing.
thousands : str or None — Thousands separator to remove.
decimal : str — Decimal separator. Default '.'.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'amount': ['$1,200.50', 'bad']})
>>> op = ParseNumber(cols='amount', remove_chars=['$', ','])
>>> op.transform(df)
   amount
0  1200.5
1     NaN

Example YAML:
  - op: ParseNumber
    target: both
    params:
      cols: [amount]
      remove_chars: ['$', ',']
"""
        return description.strip()

    def _clean(self, ser):
        out = ser.astype("object")
        mask = out.notna()
        text = out.where(~mask, out.astype(str))
        chars = list(self.remove_chars)
        if self.thousands is not None:
            chars.append(self.thousands)
        for ch in chars:
            text = text.where(~mask, text.astype(str).str.replace(ch, "", regex=False))
        if self.decimal != ".":
            text = text.where(~mask, text.astype(str).str.replace(self.decimal, ".", regex=False))
        return text

    def transform(self, df):
        df = df.copy()
        for i, col in enumerate(self.cols):
            if col not in df.columns:
                continue
            target_col = self.output_cols[i] if self.output_cols else col
            parsed = pd.to_numeric(self._clean(df[col]), errors=self.errors)
            try:
                df[target_col] = parsed.astype(self.dtype)
            except Exception:
                df[target_col] = parsed
        return df
