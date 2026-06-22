import pandas as pd

from ..base_op import TabularOp


_VALID_ACTIONS = ("flag", "repair", "delete")
_VALID_REPAIR_METHODS = ("set_missing", "fill_constant", "clip", "median", "mode")


class HandleError(TabularOp):
    """Detect rule-violating values and either delete or repair them."""

    APPLIES_TO_STD_TEST = False

    def __init__(self, cols=None, rule="numeric", flag_col=None,
                 min_value=None, max_value=None, pattern=None,
                 allowed_values=None, rules=None, mask_col=None,
                 action="delete", repair_method="set_missing", fill_value=None):
        super().__init__(name="HandleError")
        if rule not in ("numeric", "positive", "in_range", "regex", "not_in", "custom_mask_col"):
            raise ValueError("rule must be numeric/positive/in_range/regex/not_in/custom_mask_col")
        if action not in _VALID_ACTIONS:
            raise ValueError("action must be flag/repair/delete")
        if repair_method not in _VALID_REPAIR_METHODS:
            raise ValueError("repair_method must be set_missing/fill_constant/clip/median/mode")
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.rule = rule
        self.flag_col = flag_col
        self.min_value = min_value
        self.max_value = max_value
        self.pattern = pattern
        self.allowed_values = list(allowed_values) if allowed_values is not None else None
        self.rules = rules or []
        self.mask_col = mask_col
        self.action = action
        self.repair_method = repair_method
        self.fill_value = fill_value

    @staticmethod
    def _maybe_datetime(value):
        if isinstance(value, str):
            try:
                return pd.to_datetime(value)
            except (ValueError, TypeError):
                return value
        return value

    def _iter_rule_specs(self):
        if self.rules:
            yield from self.rules
            return
        for col in self.cols or []:
            yield {
                "col": col,
                "rule": self.rule,
                "min": self.min_value,
                "max": self.max_value,
                "pattern": self.pattern,
                "allowed_values": self.allowed_values,
                "mask_col": self.mask_col,
                "flag_col": self.flag_col,
            }

    def _mask_for_spec(self, df, spec):
        col = spec.get("col")
        if spec.get("rule") == "custom_mask_col" or spec.get("mask_col"):
            mask_col = spec.get("mask_col", self.mask_col)
            if mask_col in df.columns:
                return df[mask_col].astype(bool), col
            return pd.Series(False, index=df.index), col
        if col not in df.columns:
            return pd.Series(False, index=df.index), col

        rule = spec.get("rule")
        if rule is None:
            has_lower = spec.get("min", self.min_value) is not None
            has_upper = spec.get("max", self.max_value) is not None
            rule = "in_range" if (has_lower or has_upper) else self.rule
        ser = df[col]
        if rule == "numeric":
            return pd.to_numeric(ser, errors="coerce").isna() & ser.notna(), col
        if rule == "positive":
            return pd.to_numeric(ser, errors="coerce").fillna(0) <= 0, col
        if rule == "regex":
            pattern = spec.get("pattern", self.pattern)
            return ~ser.astype(str).str.match(pattern, na=False), col
        if rule == "not_in":
            allowed = spec.get("allowed_values", self.allowed_values)
            return ~ser.isin(set(allowed or [])), col

        lower = self._maybe_datetime(spec.get("min", self.min_value))
        upper = self._maybe_datetime(spec.get("max", self.max_value))
        original_missing = ser.isna()
        if isinstance(lower, pd.Timestamp) or isinstance(upper, pd.Timestamp):
            values = pd.to_datetime(ser, errors="coerce")
            parse_failed = values.isna() & ~original_missing
        else:
            values = pd.to_numeric(ser, errors="coerce")
            parse_failed = values.isna() & ~original_missing
        mask = parse_failed.copy()
        if lower is not None:
            mask = mask | ((values < lower) & ~original_missing)
        if upper is not None:
            mask = mask | ((values > upper) & ~original_missing)
        return mask.fillna(False), col

    def _repair_values(self, df, mask, col, spec):
        if col is None or col not in df.columns or not mask.any():
            return

        method = spec.get("repair_method", self.repair_method)
        if method not in _VALID_REPAIR_METHODS:
            raise ValueError("repair_method must be set_missing/fill_constant/clip/median/mode")
        if method == "set_missing":
            df.loc[mask, col] = pd.NA
            return
        if method == "fill_constant":
            df.loc[mask, col] = spec.get("fill_value", self.fill_value)
            return
        if method == "clip":
            lower = self._maybe_datetime(spec.get("min", self.min_value))
            upper = self._maybe_datetime(spec.get("max", self.max_value))
            if lower is not None or upper is not None:
                df[col] = df[col].clip(lower=lower, upper=upper)
            return

        valid = df.loc[~mask, col]
        if valid.empty:
            return
        if method == "median":
            numeric = pd.to_numeric(valid, errors="coerce").dropna()
            if not numeric.empty:
                df.loc[mask, col] = numeric.median()
            return
        if method == "mode":
            mode = valid.dropna().mode()
            if not mode.empty:
                df.loc[mask, col] = mode.iloc[0]

    def transform(self, df):
        df = df.copy()
        marker_mask = pd.Series(False, index=df.index)
        delete_mask = pd.Series(False, index=df.index)
        for spec in self._iter_rule_specs():
            mask, col = self._mask_for_spec(df, spec)
            mask = mask.reindex(df.index, fill_value=False)
            action = spec.get("action", self.action)
            if action not in _VALID_ACTIONS:
                raise ValueError("action must be flag/repair/delete")

            if action == "delete":
                delete_mask = delete_mask | mask
                continue

            marker_mask = marker_mask | mask
            out_col = spec.get("flag_col") or self.flag_col or f"{col}_is_error"
            df[out_col] = mask.astype(int).values
            if action == "repair":
                self._repair_values(df, mask, col, spec)
        if self.flag_col is None and self.rules and marker_mask.any():
            df["is_error"] = marker_mask.astype(int).values
        if delete_mask.any():
            df = df.loc[~delete_mask].reset_index(drop=True)
        return df

    def get_op_description(self):
        description = """Operator name: HandleError

Function description:
Detect values violating rules or constraints and either delete the offending
rows or repair them (set_missing, fill_constant, clip, median, mode).

Input:
df : pd.DataFrame — Table containing columns to validate.

Parameters:
cols : list[str] or None — Columns to validate for a single rule.
rule : str — numeric, positive, in_range, regex, not_in, or custom_mask_col.
rules : list[dict] — Optional per-column rule specs with col/min/max/pattern/allowed_values.
flag_col : str or None — Shared output marker name. Defaults to "<col>_is_error".
action : str — flag, repair, or delete. Defaults to delete.
repair_method : str — set_missing, fill_constant, clip, median, or mode.
fill_value : object — Constant value when repair_method=fill_constant.

Output:
pd.DataFrame — For action='delete', rows violating rules are removed. For
action='flag' or action='repair', error marker columns are added and values may
be repaired depending on action.

Example:
>>> df = pd.DataFrame({"x": ["1", "bad"]})
>>> HandleError(cols=["x"], rule="numeric", action="delete").transform(df)
   x
0  1

Example YAML:
  - op: HandleError
    target: train
    params:
      cols: [x]
      rule: numeric
      action: delete
"""
        return description.strip()
