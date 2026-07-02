"""CustomOp -- LLM-friendly catch-all operator that executes a sandboxed
``def pipeline(df) -> df`` user function.

Used by the SPIO baseline: SPIO generates Python code per stage, and each
generated code snippet is wrapped as a ``CustomOp`` step in the pipeline so
the dppbench evaluator can execute it just like any other operator.

Code spec (strictly enforced; LLM prompts surface this via
``get_op_description``):
  * The ``code`` string must define a function with the name given by
    ``entry`` (default ``"pipeline"``) whose signature is
    ``def pipeline(df: pd.DataFrame) -> pd.DataFrame``.
  * The function takes exactly one DataFrame and returns a DataFrame.
  * No ``import``, ``__import__``, ``eval``, ``exec``, ``open`` or other
    side-effecting names — those nodes are rejected at AST level.
  * Only the whitelisted globals are exposed (``pd``, ``np``, ``re``,
    ``math``, a few sklearn submodules, plus a small set of safe builtins).
  * On any error during compile / exec / call, the operator prints the
    error, records it on ``self.last_error`` and returns the *original*
    DataFrame unchanged (parity with ``ExecCode``'s style).
"""
from __future__ import annotations

import ast
import math
import re

import numpy as np
import pandas as pd

try:  # sklearn is an optional dependency for some test environments
    from sklearn import preprocessing as _sk_preprocessing
    from sklearn import impute as _sk_impute
except Exception:  # pragma: no cover - keep operator usable without sklearn
    _sk_preprocessing = None
    _sk_impute = None

from .base_op import BaseOp


# AST nodes that are unconditionally forbidden in the user-supplied code.
_FORBIDDEN_NODES: tuple = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
)

# Names that must never appear (function calls, attribute access, etc.).
_FORBIDDEN_NAMES = {
    "__import__", "eval", "exec", "compile", "open", "input",
    "globals", "locals", "vars", "exit", "quit",
    "getattr", "setattr", "delattr",
}

_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "round": round, "len": len,
    "sum": sum, "any": any, "all": all,
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "range": range, "enumerate": enumerate, "zip": zip,
    "sorted": sorted, "reversed": reversed,
    "map": map, "filter": filter,
    "isinstance": isinstance, "type": type,
    "print": lambda *a, **kw: None,  # no-op; the spec forbids printing
}


def _build_safe_globals() -> dict:
    g = {
        "__builtins__": dict(_SAFE_BUILTINS),
        "np": np,
        "pd": pd,
        "re": re,
        "math": math,
    }
    if _sk_preprocessing is not None:
        g["preprocessing"] = _sk_preprocessing
    if _sk_impute is not None:
        g["impute"] = _sk_impute
    return g


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise ValueError(
                f"CustomOp: forbidden node {type(node).__name__} in code"
            )
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(
                f"CustomOp: forbidden name '{node.id}' in code"
            )
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            # Block dunder attribute access (e.g. ``df.__class__.__bases__``).
            raise ValueError(
                f"CustomOp: forbidden dunder attribute '{node.attr}'"
            )


class CustomOp(BaseOp):
    """Generic catch-all operator that runs sandboxed ``def pipeline(df)`` code.

    Parameters
    ----------
    code : str
        A complete Python source string. It must define a top-level function
        whose name matches ``entry`` and whose signature accepts a single
        ``pd.DataFrame`` and returns a ``pd.DataFrame``.
    entry : str, default ``"pipeline"``
        Name of the entry function to call inside ``code``.
    """

    def __init__(self, code: str, entry: str = "pipeline"):
        super().__init__(name="CustomOp")
        self.op_type = "basic op"
        self.code = code or ""
        self.entry = entry or "pipeline"
        self._compiled = None
        self.last_error: str | None = None

    # ------------------------------------------------------------------
    def get_op_description(self):
        return """Operator name: CustomOp

Function description: Sandboxed execution of an LLM-authored Python function
on a single ``pd.DataFrame``. The function is passed the current target
DataFrame (``train`` / ``test`` / ``interaction`` / etc.) and must return
the new DataFrame. Used as the catch-all stage executor by the SPIO baseline.

Code specification (STRICT — must be followed exactly):
  - The ``code`` string MUST define a function whose name equals ``entry``
    (default: ``pipeline``) with signature
        def pipeline(df: pd.DataFrame) -> pd.DataFrame:
  - The function takes exactly ONE argument (a DataFrame) and returns
    exactly ONE DataFrame. No tuples, no extra outputs, no side effects.
  - The function must NOT print, write files, open network sockets, modify
    globals, or rely on any state outside ``df``.
  - The code must NOT contain ``import`` / ``from ... import`` statements.
    All required modules are pre-injected into the sandbox globals.
  - Available names inside the code: ``pd`` (pandas), ``np`` (numpy),
    ``re``, ``math``, ``preprocessing`` (sklearn.preprocessing) and
    ``impute`` (sklearn.impute), plus a small subset of safe builtins.
  - Forbidden names / nodes: ``import``, ``__import__``, ``eval``, ``exec``,
    ``open``, ``compile``, ``globals``, ``locals``, ``getattr`` / ``setattr``,
    dunder attribute access, ``Import`` / ``ImportFrom`` / ``Global`` /
    ``Nonlocal`` AST nodes.

Parameters:
code  : str — Python source defining the entry function.
entry : str — Name of the entry function (default ``pipeline``).

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: CustomOp
    prev:
    - s0
    params:
      code: |
        def pipeline(df):
            df = df.copy()
            df['amount_log'] = np.log1p(df['amount'].fillna(0))
            return df
      entry: pipeline
  train:
    prev:
    - o1
"""

    # ------------------------------------------------------------------
    def _compile(self) -> None:
        """Parse + validate + compile the user code.

        Raises ``ValueError`` if the AST contains forbidden constructs.
        """
        if self._compiled is not None:
            return
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("CustomOp: empty 'code'")
        tree = ast.parse(self.code, mode="exec")
        _validate_ast(tree)
        self._compiled = compile(tree, filename="<CustomOp>", mode="exec")

    # ------------------------------------------------------------------
    def transform(self, df):
        try:
            self._compile()
            globals_safe = _build_safe_globals()
            locals_: dict = {}
            exec(self._compiled, globals_safe, locals_)  # noqa: S102 (sandboxed)
            fn = locals_.get(self.entry) or globals_safe.get(self.entry)
            if fn is None or not callable(fn):
                raise ValueError(
                    f"CustomOp: entry function '{self.entry}' not defined"
                )
            result = fn(df.copy())
            if not isinstance(result, pd.DataFrame):
                raise TypeError(
                    f"CustomOp: '{self.entry}' must return a pandas DataFrame, "
                    f"got {type(result).__name__}"
                )
            self.last_error = None
            print(f"  [CustomOp] '{self.entry}' executed -> shape={tuple(result.shape)}")
            return result
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self.last_error = err
            print(f"  [CustomOp] execution failed: {err}; returning df unchanged")
            return df


__all__ = ["CustomOp"]
