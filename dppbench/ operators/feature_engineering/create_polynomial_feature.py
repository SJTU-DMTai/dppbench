import pandas as pd
from ..base_op import TabularOp


class CreatePolynomialFeature(TabularOp):
    """Generate polynomial and interaction features."""

    def __init__(self, cols, degree=2, interaction_only=False,
                 include_bias=False, max_features=None):
        super().__init__(name="CreatePolynomialFeature")
        self.cols = cols if isinstance(cols, list) else [cols]
        self.degree = int(degree)
        self.interaction_only = bool(interaction_only)
        self.include_bias = bool(include_bias)
        self.max_features = max_features

    def get_op_description(self):
        description = """Operator name: CreatePolynomialFeature

Function description:
Generate polynomial and interaction features from
numeric columns using sklearn PolynomialFeatures.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
See __init__ signature for supported parameters and defaults.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'x': [2, 3], 'y': [10, 20]})
>>> op = CreatePolynomialFeature(cols=['x'], degree=2, include_bias=False)
>>> op.transform(df)
   x   y  x^2
0  2  10  4.0
1  3  20  9.0

Example YAML:
  - op: CreatePolynomialFeature
    target: train
    params:
      cols: [x]
      degree: 2
      include_bias: false
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
            include_bias=self.include_bias,
        )
        out = pf.fit_transform(sub)
        names = pf.get_feature_names_out(cols)
        new_cols = []
        for j, name in enumerate(names):
            if name in cols or name == "1":
                continue
            new_cols.append(("poly_" + name.replace(" ", "_").replace("^", "p"), j))
        if self.max_features is not None:
            new_cols = new_cols[: self.max_features]
        for new_name, j in new_cols:
            df[new_name] = out[:, j]
        return df
