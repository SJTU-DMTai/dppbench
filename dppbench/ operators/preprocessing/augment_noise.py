import numpy as np
import pandas as pd
from ..base_op import BaseOp


class AugmentNoise(BaseOp):
    """Light-weight numeric data augmentation: append ``n_copies`` perturbed
    duplicates of every row, where each duplicate has Gaussian noise
    (or uniform jitter) added to the chosen numeric columns. Intended for
    training-set augmentation; do NOT apply to test/val.
    """

    APPLIES_TO_STD_TEST = False

    def __init__(self, cols, noise_type="gaussian", noise_scale=0.01,
                 n_copies=1, random_state=42):
        super().__init__(name="AugmentNoise")
        self.op_type = "basic op"
        self.cols = cols if isinstance(cols, list) else [cols]
        if noise_type not in ("gaussian", "jitter"):
            raise ValueError("noise_type must be 'gaussian' or 'jitter'")
        self.noise_type = noise_type
        self.noise_scale = float(noise_scale)
        self.n_copies = int(n_copies)
        self.random_state = int(random_state)

    def get_op_description(self):
        description = """Operator name: AugmentNoise

Function description:
Append n_copies noisy duplicates of each row. The
selected numeric columns receive additive Gaussian noise (sigma =
column_std * noise_scale) or uniform jitter in [-noise_scale, +noise_scale]
times the column std.

Input:
df : pd.DataFrame — Input table accepted by transform; required columns are listed in Parameters.

Parameters:
cols : list[str] — Numeric columns to perturb.
noise_type : 'gaussian' (default) or 'jitter'.
noise_scale : float — Multiplier on per-column std. Default 0.01.
n_copies : int — Number of perturbed copies to append. Default 1.
random_state : int — RNG seed.

Output:
pd.DataFrame — Transformed table after applying the operator.

Example:
>>> df = pd.DataFrame({'temp': [10.0, 20.0]})
>>> op = AugmentNoise(cols=['temp'], noise_scale=0.0, n_copies=1, random_state=0)
>>> op.transform(df)
   temp
0  10.0
1  20.0
2  10.0
3  20.0

Example YAML:
  - op: AugmentNoise
    target: train
    params:
      cols: [temp]
      noise_scale: 0.05
      n_copies: 1
      random_state: 42
"""
        return description.strip()

    def transform(self, df):
        df = df.copy()
        cols = [c for c in self.cols if c in df.columns]
        if not cols or self.n_copies <= 0:
            return df
        rng = np.random.RandomState(self.random_state)
        stds = {c: float(pd.to_numeric(df[c], errors="coerce").std() or 1.0)
                for c in cols}
        copies = [df]
        for k in range(self.n_copies):
            cp = df.copy()
            for c in cols:
                v = pd.to_numeric(cp[c], errors="coerce").fillna(0).values
                scale = stds[c] * self.noise_scale
                if self.noise_type == "gaussian":
                    noise = rng.normal(0.0, scale, size=v.shape)
                else:  # jitter
                    noise = rng.uniform(-scale, scale, size=v.shape)
                cp[c] = v + noise
            copies.append(cp)
        return pd.concat(copies, axis=0, ignore_index=True)
