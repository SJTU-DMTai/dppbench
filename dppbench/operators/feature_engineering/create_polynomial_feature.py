import pandas as pd
from ..base_op import TabularOp


class CreatePolynomialFeature(TabularOp):
    """Generate polynomial and interaction features."""

    INCLUDE_BIAS = False

    def __init__(self, cols, degree=2, interaction_only=False):
        super().__init__(name="CreatePolynomialFeature")
        self.cols = cols if isinstance(cols, list) else [cols]
        self.degree = int(degree)
        self.interaction_only = bool(interaction_only)

    def get_op_description(self):
        description = """Operator name: CreatePolynomialFeature

Function description:
Generate polynomial and interaction features from numeric columns using sklearn PolynomialFeatures.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] — Numeric columns to combine.
degree : int — Polynomial degree (default 2).
interaction_only : bool — If True, only interaction terms (no a^2, a^3, ...). Default False.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'x': [2, 3], 'y': [10, 20]})
>>> op = CreatePolynomialFeature(cols=['x'], degree=2)
>>> op.transform(df)
   x   y  poly_xp2
0  2  10       4.0
1  3  20       9.0

Example YAML:
dag:
  sources:
  - id: s0
    table: main
  ops:
  - id: o1
    op: CreatePolynomialFeature
    prev:
    - s0
    params:
      cols:
      - x
      - y
      degree: 2
      interaction_only: true
  train:
    prev:
    - o1
"""
        return description.strip()

    def transform(self, df):
        try:
            from sklearn.preprocessing import PolynomialFeatures
        except Exception as exc:
            print(f"  [CreatePolynomialFeature] sklearn unavailable: {exc}")
            return df
        cols = [c for c in self.cols if c in df.columns]
        if not cols:
            return df
        df = df.copy()
        sub = df[cols].astype(float).fillna(0.0).values
        pf = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.INCLUDE_BIAS,
        )
        out = pf.fit_transform(sub)
        names = pf.get_feature_names_out(cols)
        for j, name in enumerate(names):
            if name in cols or name == "1":
                continue
            new_name = "poly_" + name.replace(" ", "_").replace("^", "p")
            df[new_name] = out[:, j]
        return df
