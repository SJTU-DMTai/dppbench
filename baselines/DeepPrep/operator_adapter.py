"""Explicit mappings from released DeepPrep syntax to DPPBench operators.

Only mappings with a direct shared-operator equivalent are accepted. There is
no fuzzy substitution; unsupported original operators fail in the sandbox.
"""
from __future__ import annotations


ORIGINAL_OPERATOR_ALIASES = {
    "DropNA": "HandleMV",
    "MissingValueImputation": "HandleMV",
    "DropColumn": "DropColumns",
    "Sort": "SortRows",
    "StandardizeDatetime": "ParseDate",
    "Join": "JoinTable",
    "Append": "ConcatTable",
}


def _without_table(params: dict) -> dict:
    params = dict(params)
    params.pop("table", None)
    return params


def adapt_operator_instance(op_name: str, params: dict) -> tuple[str, dict]:
    """Return one exact DPPBench operator instance or raise ``ValueError``."""
    p = _without_table(params)
    if op_name == "DropNA":
        how = p.pop("how", "any")
        if how != "any":
            raise ValueError("DropNA(how='all') has no exact DPPBench equivalent")
        subset = p.pop("subset", None)
        if p:
            raise ValueError(f"unsupported DropNA parameters: {sorted(p)}")
        return "HandleMV", {"cols": subset, "method": "median", "action": "delete"}
    if op_name == "MissingValueImputation":
        column = p.pop("column", None)
        mode = p.pop("mode", "median")
        if p:
            raise ValueError(f"unsupported MissingValueImputation parameters: {sorted(p)}")
        return "HandleMV", {"cols": [column], "method": mode, "action": "impute"}
    if op_name == "DropColumn":
        return "DropColumns", {"cols": p.pop("columns")} if set(p) == {"columns"} else p
    if op_name == "Sort":
        return "SortRows", p
    if op_name == "StandardizeDatetime":
        column = p.pop("column", None)
        target_format = p.pop("format", None)
        if p:
            raise ValueError(f"unsupported StandardizeDatetime parameters: {sorted(p)}")
        return "ParseDate", {"cols": [column], "mode": "string", "target_format": target_format}
    if op_name == "CastType" and "column" in p and "dtype" in p:
        return "CastType", {"col_dtypes": {p.pop("column"): p.pop("dtype")}}
    if op_name == "Join":
        right = p.pop("right", None)
        p.pop("left", None)
        on = p.pop("on", None)
        how = p.pop("how", "left")
        if p or not right or not on:
            raise ValueError("Join requires left, right, on, and how")
        return "JoinTable", {
            "method": "key", "aux_df": f"${right}_df", "key_col": on, "how": how,
        }
    if op_name == "Append":
        other = p.pop("other", None)
        if p or not other:
            raise ValueError("Append requires an existing other table")
        return "ConcatTable", {"other_dfs": [f"${other}_df"], "axis": "vertical"}
    return ORIGINAL_OPERATOR_ALIASES.get(op_name, op_name), p


__all__ = ["ORIGINAL_OPERATOR_ALIASES", "adapt_operator_instance"]
