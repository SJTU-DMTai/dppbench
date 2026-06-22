import inspect
import numpy as np
import pandas as pd
from ..base_op import BaseOp


def _builtin_mean(df, cols, **_):
    return df[cols].mean(axis=1)


def _builtin_sum(df, cols, **_):
    return df[cols].sum(axis=1)


def _builtin_std(df, cols, **_):
    return df[cols].std(axis=1)


def _builtin_min(df, cols, **_):
    return df[cols].min(axis=1)


def _builtin_max(df, cols, **_):
    return df[cols].max(axis=1)


def _builtin_median(df, cols, **_):
    return df[cols].median(axis=1)


def _builtin_product(df, cols, **_):
    return df[cols].prod(axis=1)


def _builtin_diff(df, cols, **_):
    if len(cols) < 2:
        raise ValueError("CreateFeature[diff] requires at least 2 source_cols")
    return df[cols[0]] - df[cols[1]]


def _builtin_ratio(df, cols, **_):
    if len(cols) < 2:
        raise ValueError("CreateFeature[ratio] requires at least 2 source_cols")
    denom = df[cols[1]].replace(0, np.nan)
    return df[cols[0]] / denom


def _builtin_inc_ratio(df, cols, offset=1, **_):
    if len(cols) < 2:
        raise ValueError("CreateFeature[inc_ratio] requires at least 2 source_cols")
    return df[cols[0]] / (offset + df[cols[1]])


def _builtin_concat(df, cols, sep="_", **_):
    return df[cols].astype(str).agg(sep.join, axis=1)


def _builtin_identity(df, cols, **_):
    return df[cols[0]].copy()


BUILTIN_METHODS = {
    "mean": _builtin_mean,
    "sum": _builtin_sum,
    "std": _builtin_std,
    "min": _builtin_min,
    "max": _builtin_max,
    "median": _builtin_median,
    "product": _builtin_product,
    "diff": _builtin_diff,
    "ratio": _builtin_ratio,
    "inc_ratio": _builtin_inc_ratio,
    "concat": _builtin_concat,
    "identity": _builtin_identity,
}


class CreateFeature(BaseOp):
    """Create one new column from `source_cols` via a built-in or user-defined
    algorithm. Replaces the old DeriveFeatures + CreateFeature operators.
    """

    def __init__(
        self,
        source_cols,
        output_col,
        method="mean",
        method_kwargs=None,
        col_type="numeric",
        # Backwards-compatible aliases:
        input_cols=None,
        func=None,
    ):
        super().__init__(name="CreateFeature")
        self.op_type = "tabular op"

        if source_cols is None and input_cols is not None:
            source_cols = input_cols
        if isinstance(source_cols, str):
            source_cols = [source_cols]
        if not source_cols:
            raise ValueError("CreateFeature: source_cols must be a non-empty list")

        if method == "mean" and func is not None:
            method = func

        self.source_cols = list(source_cols)
        self.output_col = output_col
        self.method = method
        self.method_kwargs = dict(method_kwargs or {})
        self.col_type = col_type
        self.output_col_types = {output_col: col_type}

    def get_op_description(self):
        description = """Operator name: CreateFeature

Function description:
Create one new column from `source_cols` by applying a built-in or user-supplied algorithm. Combines the old DeriveFeatures (templated arithmetic / aggregation) and CreateFeature (UDF) into a single op.

Input:
df : pd.DataFrame - DataFrame containing source_cols.

Parameters:
source_cols : list[str] - Source column names (>=1).
output_col  : str       - Name of the new column.
method      : str | callable, default 'mean'
Built-in str values: mean, sum, std, min, max, median, product, diff, ratio, inc_ratio, concat, identity.
Callable: either f(row: pd.Series, **kwargs) -> scalar (row UDF)
or f(df, source_cols, **kwargs) -> pd.Series (vectorized UDF);
the form is auto-detected via the first parameter name.
method_kwargs : dict, optional - extra kwargs for the method (e.g.
offset for inc_ratio, sep for concat).
col_type    : str, default 'numeric' - Output dtype hint.

Output:
pd.DataFrame - Original DataFrame with an additional output_col.

Example:
>>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
>>> op = CreateFeature(source_cols=['a', 'b'], output_col='ab_mean', method='mean')
>>> op.transform(df)
   a   b  ab_mean
0  1  10      5.5
1  2  20     11.0
2  3  30     16.5

Example YAML:
  - op: CreateFeature
    target: train
    params:
      source_cols: [LIMIT_BAL, AGE]
      output_col: LIMIT_PER_AGE
      method: ratio
"""
        return description.strip()

    @classmethod
    def from_features(cls, features):
        """Build a list of CreateFeature ops from a DeriveFeatures-style spec.

        ``features`` is a list of dicts with keys ``name``, ``op`` and either
        ``cols`` or ``pattern``. ``offset`` (for inc_ratio) is forwarded.
        ``pattern`` is resolved against an optional ``columns`` argument.
        """
        ops = []
        for feat in features:
            ops.append(
                cls(
                    source_cols=feat.get("cols") or [],
                    output_col=feat["name"],
                    method=feat["op"],
                    method_kwargs={
                        k: v for k, v in feat.items()
                        if k in ("offset", "sep")
                    },
                )
            )
        return ops

    def _resolve_callable(self, df, source_cols):
        method = self.method
        try:
            sig = inspect.signature(method)
            params = list(sig.parameters.keys())
            first = params[0] if params else ""
        except (TypeError, ValueError):
            first = ""
        if first in ("df", "frame", "data"):
            return method(df, source_cols, **self.method_kwargs)
        # Row-wise UDF.
        if len(source_cols) == 1:
            return df[source_cols[0]].apply(
                lambda x: method(x, **self.method_kwargs)
            )
        return df[source_cols].apply(
            lambda row: method(row, **self.method_kwargs), axis=1
        )

    def transform(self, df):
        df = df.copy()
        existing = [c for c in self.source_cols if c in df.columns]
        if not existing:
            return df
        if callable(self.method):
            df[self.output_col] = self._resolve_callable(df, existing)
            return df
        if isinstance(self.method, str):
            fn = BUILTIN_METHODS.get(self.method)
            if fn is None:
                raise ValueError(
                    f"CreateFeature: unknown built-in method '{self.method}'. "
                    f"Supported: {list(BUILTIN_METHODS.keys())}"
                )
            df[self.output_col] = fn(df, existing, **self.method_kwargs)
            return df
        raise TypeError(
            f"CreateFeature: method must be str or callable, got {type(self.method).__name__}"
        )
