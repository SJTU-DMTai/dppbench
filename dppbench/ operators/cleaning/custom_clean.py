import numpy as np
import pandas as pd
from ..base_op import TabularOp
from ..custom_op import CustomOp


class CustomClean(TabularOp):
    """Custom cleaning operator with common map/replace helpers."""

    SUPPORTED_OPS = ("eq", "ne", "lt", "le", "gt", "ge", "in")

    def __init__(self, code=None, entry="pipeline", func=None,
                 mode="code", cols=None, rules=None, replace_with=None,
                 pattern=None, replacement="", regex=True,
                 fillna_replacement=None):
        super().__init__(name="CustomClean")
        if mode not in ("code", "map_values", "replace_text"):
            raise ValueError("mode must be code/map_values/replace_text")
        self.mode = mode
        self.code = code
        self.entry = entry or "pipeline"
        self.func = func
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.rules = rules or []
        self.replace_with = np.nan if replace_with is None else replace_with
        self.pattern = pattern
        self.replacement = replacement
        self.regex = bool(regex)
        self.fillna_replacement = fillna_replacement
        self._custom = CustomOp(code=code, entry=self.entry) if code else None

    def get_op_description(self):
        description = """Operator name: CustomClean

Function description:
User-defined cleaning. In addition to sandboxed code or
callable func, supports built-in modes that replace old MapValues and
ReplaceText behavior.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
mode : str — code/map_values/replace_text.
code, entry, func — Custom execution inputs.
cols, rules, replace_with — map_values mode.
pattern, replacement, regex, fillna_replacement — replace_text mode.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'age': [25, -1, 40], 'note': ['ok', 'bad id', None]})
>>> op = CustomClean(mode='map_values', rules=[{'col': 'age', 'lt': 0, 'replace_with': np.nan}])
>>> op.transform(df)
    age    note
0  25.0      ok
1   NaN  bad id
2  40.0    None

Example YAML:
  - op: CustomClean
    target: both
    params:
      mode: map_values
      rules:
        - col: age
          lt: 0
          replace_with: null
"""
        return description.strip()

    def _build_mask(self, series, rule):
        for op in self.SUPPORTED_OPS:
            if op in rule:
                target = rule[op]
                if op == "eq":
                    return series == target
                if op == "ne":
                    return series != target
                if op == "lt":
                    return series < target
                if op == "le":
                    return series <= target
                if op == "gt":
                    return series > target
                if op == "ge":
                    return series >= target
                return series.isin(target if isinstance(target, list) else [target])
        return pd.Series(False, index=series.index)

    def _map_values(self, df):
        if not self.rules:
            return df
        cols_to_rules = {}
        for rule in self.rules:
            rule_col = rule.get("col")
            targets = [rule_col] if rule_col is not None else (self.cols or [])
            for col in targets:
                cols_to_rules.setdefault(col, []).append(rule)
        for col, rules in cols_to_rules.items():
            if col not in df.columns:
                continue
            series = df[col]
            for rule in rules:
                rep = rule.get("replace_with", self.replace_with)
                mask = self._build_mask(series, rule)
                series = series.where(~mask.fillna(False), rep)
            df[col] = series
        return df

    def _replace_text(self, df):
        if self.pattern is None:
            return df
        for col in self.cols or []:
            if col not in df.columns:
                continue
            ser = df[col].astype("object")
            mask = ser.notna()
            ser = ser.where(
                ~mask,
                ser.astype(str).str.replace(
                    self.pattern, self.replacement, regex=self.regex
                ),
            )
            if self.fillna_replacement is not None:
                ser = ser.fillna(self.fillna_replacement)
            df[col] = ser
        return df

    def transform(self, df):
        df = df.copy()
        if self.func is not None:
            result = self.func(df)
            if not isinstance(result, pd.DataFrame):
                raise TypeError("CustomClean func must return a pandas DataFrame")
            return result
        if self.mode == "map_values":
            return self._map_values(df)
        if self.mode == "replace_text":
            return self._replace_text(df)
        if self._custom is not None:
            return self._custom.transform(df)
        return df
