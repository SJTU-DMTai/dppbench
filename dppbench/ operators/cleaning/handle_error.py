import pandas as pd

from ..base_op import TabularOp


_VALID_ACTIONS = ("repair", "delete")
_VALID_REPAIR_METHODS = ("set_missing", "clip")


class HandleError(TabularOp):
    """Detect rule-violating values and either delete or repair them."""

    APPLIES_TO_STD_TEST = False

    def __init__(self, cols=None, rule=None, min_value=None, max_value=None,
                 allowed_values=None, rules=None,
                 action="delete", repair_method="set_missing"):
        super().__init__(name="HandleError")
        if rule is not None and rule not in ("numeric", "positive", "in_range", "not_in"):
            raise ValueError("rule must be numeric/positive/in_range/not_in")
        if action not in _VALID_ACTIONS:
            raise ValueError("action must be repair/delete")
        if repair_method not in _VALID_REPAIR_METHODS:
            raise ValueError("repair_method must be set_missing/clip")
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.rule = rule
        self.min_value = min_value
        self.max_value = max_value
        self.allowed_values = list(allowed_values) if allowed_values is not None else None
        self.rules = rules or []
        self.action = action
        self.repair_method = repair_method

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
                "allowed_values": self.allowed_values,
            }

    def _mask_for_spec(self, df, spec):
        col = spec.get("col")
        if col not in df.columns:
            return pd.Series(False, index=df.index), col

        rule = spec.get("rule")
        if rule is None:
            has_lower = spec.get("min") is not None
            has_upper = spec.get("max") is not None
            rule = "in_range" if (has_lower or has_upper) else self.rule
        ser = df[col]
        if rule == "numeric":
            return pd.to_numeric(ser, errors="coerce").isna() & ser.notna(), col
        if rule == "positive":
            return pd.to_numeric(ser, errors="coerce").fillna(0) <= 0, col
        if rule == "not_in":
            allowed = spec.get("allowed_values", self.allowed_values)
            return ~ser.isin(set(allowed or [])), col

        lower = self._maybe_datetime(spec.get("min"))
        upper = self._maybe_datetime(spec.get("max"))
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
            raise ValueError("repair_method must be set_missing/clip")
        if method == "set_missing":
            df.loc[mask, col] = pd.NA
            return
        if method == "clip":
            lower = self._maybe_datetime(spec.get("min", self.min_value))
            upper = self._maybe_datetime(spec.get("max", self.max_value))
            if lower is not None or upper is not None:
                df[col] = df[col].clip(lower=lower, upper=upper)

    def transform(self, df):
        df = df.copy()
        delete_mask = pd.Series(False, index=df.index)
        for spec in self._iter_rule_specs():
            mask, col = self._mask_for_spec(df, spec)
            mask = mask.reindex(df.index, fill_value=False)
            action = spec.get("action", self.action)
            if action not in _VALID_ACTIONS:
                raise ValueError("action must be repair/delete")
            if action == "delete":
                delete_mask = delete_mask | mask
            else:
                self._repair_values(df, mask, col, spec)
        if delete_mask.any():
            df = df.loc[~delete_mask].reset_index(drop=True)
        return df

    def get_op_description(self):
        description = """Operator name: HandleError

Function description:
Detect values violating rules or constraints and either delete the offending
rows or repair them by setting to NaN or clipping to bounds.

Input:
df : pd.DataFrame — Table containing columns to validate.

Parameters:
cols : list[str] or None — Columns to validate when using single-rule API.
rule : str or None — numeric, positive, in_range, or not_in.
min_value, max_value : Lower/upper bounds for in_range / clip repair.
allowed_values : list — Whitelist for not_in rule.
rules : list[dict] — Per-column rule specs each containing
    col, rule, min, max, allowed_values, action, repair_method (overrides).
action : str — repair or delete (default delete).
repair_method : str — set_missing or clip.

Output:
pd.DataFrame — For action='delete', rows violating rules are removed. For
action='repair', offending values are repaired in place.

Example:
>>> df = pd.DataFrame({"x": ["1", "bad"]})
>>> HandleError(cols=["x"], rule="numeric", action="delete").transform(df)
   x
0  1

Example YAML:
  - op: HandleError
    target: train
    params:
      action: delete
      rules:
        - col: x
          min: 0
          max: 100
"""
        return description.strip()
