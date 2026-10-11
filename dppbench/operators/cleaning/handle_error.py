import re

import numpy as np
import pandas as pd

from ..base_op import TabularOp


_VALID_ACTIONS = ("repair", "delete")
_VALID_REPAIR_METHODS = ("set_missing", "clip", "fill_constant", "median", "mode")


class HandleError(TabularOp):
    """Detect rule-violating values and either delete or repair them."""

    APPLIES_TO_STD_TEST = False
    FIT_ON_TRAIN_ONLY = True

    def __init__(self, cols=None, rule=None, min_value=None, max_value=None,
                 allowed_values=None, rules=None,
                 action="delete", repair_method="set_missing", pattern=None,
                 fill_value=None):
        super().__init__(name="HandleError")
        if rule is not None and rule not in ("numeric", "positive", "in_range", "not_in", "regex"):
            raise ValueError("rule must be numeric/positive/in_range/not_in/regex")
        if action not in _VALID_ACTIONS:
            raise ValueError("action must be repair/delete")
        if repair_method not in _VALID_REPAIR_METHODS:
            raise ValueError(f"repair_method must be one of {_VALID_REPAIR_METHODS}")
        self.cols = cols if (cols is None or isinstance(cols, list)) else [cols]
        self.rule = rule
        self.min_value = min_value
        self.max_value = max_value
        self.allowed_values = list(allowed_values) if allowed_values is not None else None
        self.rules = rules or []
        self.action = action
        self.repair_method = repair_method
        self.pattern = pattern
        self.fill_value = fill_value
        self.fill_values_ = {}
        self.fitted_ = False
        self.specs_ = list(self._iter_rule_specs())
        if not self.specs_:
            raise ValueError("HandleError requires cols or nonempty rules")
        for spec in self.specs_:
            self._validate_spec(spec)

    @staticmethod
    def _maybe_datetime(value):
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                try:
                    return pd.Timestamp(value)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid numeric/date bound: {value!r}") from exc
        return value

    def _iter_rule_specs(self):
        raw_specs = self.rules or [{"col": col} for col in self.cols or []]
        for raw in raw_specs:
            spec = {
                "rule": self.rule,
                "min": self.min_value,
                "max": self.max_value,
                "allowed_values": self.allowed_values,
                "pattern": self.pattern,
                "fill_value": self.fill_value,
                "action": self.action,
                "repair_method": self.repair_method,
            }
            spec.update(raw)
            if spec["rule"] is None and (spec["min"] is not None or spec["max"] is not None):
                spec["rule"] = "in_range"
            yield spec

    def _validate_spec(self, spec):
        rule = spec["rule"]
        if not spec.get("col"):
            raise ValueError("Each HandleError rule requires col")
        if rule not in ("numeric", "positive", "in_range", "not_in", "regex"):
            raise ValueError(f"Unknown HandleError rule: {rule!r}")
        if spec["action"] not in _VALID_ACTIONS or spec["repair_method"] not in _VALID_REPAIR_METHODS:
            raise ValueError("Invalid HandleError action or repair_method")
        if rule == "regex":
            if not isinstance(spec["pattern"], str) or not spec["pattern"]:
                raise ValueError("regex requires a nonempty pattern")
            try:
                re.compile(spec["pattern"])
            except re.error as exc:
                raise ValueError(f"Invalid regex pattern: {exc}") from exc
        if rule == "not_in" and not spec["allowed_values"]:
            raise ValueError("not_in requires nonempty allowed_values")
        if rule == "in_range":
            lo, hi = self._maybe_datetime(spec["min"]), self._maybe_datetime(spec["max"])
            if lo is None and hi is None:
                raise ValueError("in_range requires min_value/max_value (or rules min/max)")
            if lo is not None and hi is not None:
                try:
                    ordered = lo <= hi
                except TypeError as exc:
                    raise ValueError("Range bounds must have compatible types") from exc
                if not ordered:
                    raise ValueError("Range lower bound must not exceed upper bound")
            for bound in (lo, hi):
                if bound is not None and (pd.isna(bound) or
                        (not isinstance(bound, pd.Timestamp) and not np.isfinite(bound))):
                    raise ValueError("Range bounds must be finite numbers or valid dates")
        if spec["action"] != "repair":
            return
        method = spec["repair_method"]
        if method == "clip" and rule != "in_range":
            raise ValueError("clip repair requires an in_range rule with explicit bounds")
        if method == "median" and rule not in ("numeric", "positive", "in_range"):
            raise ValueError("median repair requires a numeric rule")
        if method == "median" and any(isinstance(self._maybe_datetime(spec[k]), pd.Timestamp) for k in ("min", "max")):
            raise ValueError("median repair does not support date bounds; use mode")
        if method == "fill_constant":
            value = spec["fill_value"]
            if not pd.api.types.is_scalar(value) or pd.isna(value):
                raise ValueError("fill_constant requires a nonmissing scalar fill_value")
            probe = pd.DataFrame({spec["col"]: [value]})
            if self._mask_for_spec(probe, spec)[0].any():
                raise ValueError("fill_value must satisfy its validation rule")

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
        if rule == "regex":
            # Missing values belong to HandleMV, not format validation.
            return ser.notna() & ~ser.astype("string").str.fullmatch(spec["pattern"], na=False), col

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

    def _repair_values(self, df, mask, col, spec, index):
        if col is None or col not in df.columns or not mask.any():
            return
        method = spec.get("repair_method", self.repair_method)
        if method not in _VALID_REPAIR_METHODS:
            raise ValueError(f"repair_method must be one of {_VALID_REPAIR_METHODS}")
        if method == "set_missing":
            df.loc[mask, col] = pd.NA
            return
        if method == "clip":
            lower = self._maybe_datetime(spec.get("min", self.min_value))
            upper = self._maybe_datetime(spec.get("max", self.max_value))
            dates = isinstance(lower, pd.Timestamp) or isinstance(upper, pd.Timestamp)
            values = pd.to_datetime(df[col], errors="coerce") if dates else pd.to_numeric(df[col], errors="coerce")
            if (values.isna() & mask).any():
                raise ValueError(f"clip cannot repair unparseable values in {col!r}; use set_missing or another repair")
            repaired = values.clip(lower=lower, upper=upper)
            self._assign(df, col, mask, repaired.loc[mask])
            return
        value = spec["fill_value"] if method == "fill_constant" else self.fill_values_.get(index)
        if value is None or pd.isna(value):
            raise ValueError(f"No valid training values for {method} repair of {col!r}")
        self._assign(df, col, mask, value)

    @staticmethod
    def _assign(df, col, mask, value):
        # Nullable integers, StringDtype and categoricals may reject otherwise
        # valid replacements (fractional medians or a new constant category).
        try:
            df.loc[mask, col] = value
        except (TypeError, ValueError):
            df[col] = df[col].astype(object)
            df.loc[mask, col] = value

    def transform(self, df):
        df = df.copy()
        delete_mask = pd.Series(False, index=df.index)
        for index, spec in enumerate(self.specs_):
            mask, col = self._mask_for_spec(df, spec)
            mask = mask.reindex(df.index, fill_value=False)
            action = spec.get("action", self.action)
            if action not in _VALID_ACTIONS:
                raise ValueError("action must be repair/delete")
            if action == "delete":
                delete_mask = delete_mask | mask
            else:
                method = spec["repair_method"]
                if not self.fitted_ and method in ("median", "mode") and col in df:
                    valid = df.loc[~mask, col].dropna()
                    if method == "median":
                        valid = pd.to_numeric(valid, errors="coerce")
                        valid = valid[np.isfinite(valid)]
                        value = valid.median() if len(valid) else None
                    else:
                        modes = valid.mode()
                        value = modes.iloc[0] if len(modes) else None
                    self.fill_values_[index] = value
                self._repair_values(df, mask, col, spec, index)
        self.fitted_ = True
        if delete_mask.any():
            df = df.loc[~delete_mask].reset_index(drop=True)
        return df

    def get_op_description(self):
        description = """Operator name: HandleError

Function description:
Detect values violating rules or constraints and either delete the offending
rows or repair them by set_missing, clip, fill_constant, median, or mode.
Median/mode are fitted on valid nonmissing training values, then reused.
No valid training values: raise if repair is needed. Held-out rows are untouched.

Input:
df : pd.DataFrame — Table containing columns to validate.

Parameters:
cols : list[str] or None — Columns to validate when using single-rule API.
rule : str or None — numeric, positive, in_range, not_in, or regex.
pattern : str — Required for regex; full-string match; existing NA is ignored.
fill_value : scalar — Required for fill_constant; must satisfy the rule.
min_value, max_value : Lower/upper bounds for in_range / clip repair.
allowed_values : list — Whitelist for not_in rule.
rules : list[dict] — Per-column rule specs each containing
    col, rule, min, max, allowed_values, pattern, fill_value, action,
    repair_method (overrides). Bounds without rule imply in_range.
action : str — repair or delete (default delete).
repair_method : str — set_missing/clip/fill_constant/median/mode.
clip requires in_range and numeric/date bounds; unparseable values raise.
median supports numeric rules only; mode also supports categorical/date rules.

Output:
pd.DataFrame — For action='delete', rows violating rules are removed. For
action='repair', offending values are repaired in place.

Example:
>>> df = pd.DataFrame({"x": ["1", "bad"]})
>>> HandleError(cols=["x"], rule="numeric", action="delete").transform(df)
   x
0  1

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: HandleError
    prev:
    - s0
    params:
      action: delete
      rules:
      - col: x
        min: 0
        max: 100
  train:
    prev:
    - o1
"""
        return description.strip()
