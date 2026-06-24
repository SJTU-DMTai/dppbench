import pandas as pd
from datetime import date
from ..base_op import TabularOp


class ParseDate(TabularOp):
    """Parse date-like columns from strings or YYMMDD integers."""

    EPOCH = date(1970, 1, 1)
    ERRORS = "coerce"

    def __init__(self, cols, mode="string", target_format=None,
                 out_features=None, drop_original=False):
        super().__init__(name="ParseDate")
        self.cols = cols if isinstance(cols, list) else [cols]
        if mode not in ("string", "int_yymmdd", "berka_birth"):
            raise ValueError("mode must be 'string', 'int_yymmdd', or 'berka_birth'")
        self.mode = mode
        self.target_format = target_format
        self.out_features_explicit = out_features is not None
        self.out_features = out_features or [
            "year", "month", "day", "days_since_epoch"
        ]
        self.drop_original = bool(drop_original)

    def get_op_description(self):
        description = """Operator name: ParseDate

Function description:
Parse date columns. ``mode='string'`` converts
heterogeneous date strings to pandas datetime or target strftime format.
``mode='int_yymmdd'`` parses six-digit YYMMDD values. ``mode='berka_birth'``
also decodes Berka birth_number gender by subtracting 50 from female months.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : str or list[str] — Date columns.
mode : str — 'string' / 'int_yymmdd' / 'berka_birth'.
target_format : str or None — strftime format for string mode.
out_features : list[str] — Integer-date derived columns: year/month/day/
days_since_epoch/day_of_week.
drop_original : bool — Drop source column after parsing.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'date': ['2024-01-02', 'bad']})
>>> op = ParseDate(cols='date', mode='string', target_format='%Y-%m-%d')
>>> op.transform(df)
        date  date_year  date_month  date_day
0 2024-01-02     2024.0         1.0       2.0
1        NaT        NaN         NaN       NaN

Example YAML:
  - op: ParseDate
    target: both
    params:
      cols: [date]
      mode: string
      target_format: '%Y-%m-%d'
      drop_original: false
"""
        return description.strip()

    @classmethod
    def _to_components(cls, value, berka_birth=False):
        if value is None or pd.isna(value):
            return None
        try:
            iv = int(value)
        except (TypeError, ValueError):
            return None
        if iv <= 0:
            return None
        s = str(iv).zfill(6)
        try:
            yy = int(s[:2])
            mm = int(s[2:4])
            dd = int(s[4:6])
        except ValueError:
            return None

        gender = None
        if berka_birth:
            if mm > 50:
                mm -= 50
                gender = "F"
            else:
                gender = "M"

        year = 1900 + yy if yy >= 50 else 2000 + yy
        if berka_birth:
            year = 1900 + yy
        try:
            d = date(year, mm, dd)
        except ValueError:
            return None
        return d, gender

    def _add_date_features(self, df, col, dates):
        parsed = pd.to_datetime(dates, errors=self.ERRORS)
        if "year" in self.out_features:
            df[f"{col}_year"] = parsed.dt.year
        if "month" in self.out_features:
            df[f"{col}_month"] = parsed.dt.month
        if "day" in self.out_features:
            df[f"{col}_day"] = parsed.dt.day
        if "days_since_epoch" in self.out_features:
            df[f"{col}_days_since_epoch"] = (
                parsed - pd.Timestamp(self.EPOCH)
            ).dt.days
        if "day_of_week" in self.out_features or "dayofweek" in self.out_features:
            df[f"{col}_day_of_week"] = parsed.dt.weekday

    def _parse_string(self, df, col):
        try:
            parsed = pd.to_datetime(df[col], errors=self.ERRORS, format="mixed")
        except ValueError:
            parsed = pd.to_datetime(df[col], errors=self.ERRORS)
        if self.target_format is not None:
            df[col] = parsed.dt.strftime(self.target_format)
        else:
            df[col] = parsed
        return parsed

    def _parse_int_date(self, df, col):
        berka = self.mode == "berka_birth"
        parsed = df[col].apply(lambda v: self._to_components(v, berka_birth=berka))
        dates = parsed.apply(lambda x: x[0] if x is not None else None)
        genders = parsed.apply(lambda x: x[1] if x is not None else None)

        self._add_date_features(df, col, dates)
        if berka:
            df[f"{col}_gender"] = genders

    def transform(self, df):
        df = df.copy()
        for col in self.cols:
            if col not in df.columns:
                continue
            if self.mode == "string":
                parsed = self._parse_string(df, col)
                if self.out_features_explicit:
                    self._add_date_features(df, col, parsed)
            else:
                self._parse_int_date(df, col)
            if self.drop_original:
                df = df.drop(columns=[col])
        return df
