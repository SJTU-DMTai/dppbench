"""Lightweight context encoder for CtxPipe.

The original CtxPipe paper uses a GTE-large embedding (~600 MB) to embed the
dataset's schema/column names into a vector. To keep dependencies lean, this
implementation uses a fixed 32-dimensional vector built from cheap
schema/statistics features. The downstream RL agent still sees a context
signal that varies across datasets, which is what the "context-aware" component
is supposed to provide.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from baselines.common.pipeline import DataContext


CONTEXT_DIM = 32


class ContextEncoder:
    """Encode a (DataContext, executor) pair into a fixed-length numpy vector."""

    dim: int = CONTEXT_DIM

    def encode(self, ctx: "DataContext", executor) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)

        # 0..1 : task_type one-hot (tabular, rec)
        if ctx.task_type == "tabular":
            v[0] = 1.0
        else:
            v[1] = 1.0

        # 2..5 : column counts (log-scaled)
        v[2] = math.log1p(len(ctx.numeric_cols))
        v[3] = math.log1p(len(ctx.categorical_cols))
        v[4] = math.log1p(len(ctx.list_cols))
        v[5] = math.log1p(len(ctx.text_cols))

        # 6..7 : data size (log10 rows / cols), 8 : null ratio
        try:
            data = executor._load_data()
            if ctx.task_type == "tabular":
                df = data.train_df
            else:
                df = data.interaction_df
            n_rows = int(df.shape[0])
            n_cols = int(df.shape[1])
            v[6] = math.log10(max(n_rows, 1))
            v[7] = math.log10(max(n_cols, 1))
            try:
                v[8] = float(df.isnull().mean().mean())
            except Exception:
                v[8] = 0.0
        except Exception:
            df = None

        # 9..11 : has_user_df / has_item_df / has_aux
        v[9] = 1.0 if ctx.has_user_df else 0.0
        v[10] = 1.0 if ctx.has_item_df else 0.0
        v[11] = 1.0 if ctx.aux_dfs else 0.0

        # 12..14 : target distribution (mean / std / pos_ratio)
        try:
            if df is not None and ctx.target_col and ctx.target_col in df.columns:
                col = df[ctx.target_col]
                if col.dtype.kind in ("i", "u", "f", "b"):
                    v[12] = float(col.mean())
                    v[13] = float(col.std() if len(col) > 1 else 0.0)
                    pos_ratio = float((col > 0).mean())
                    v[14] = pos_ratio
        except Exception:
            pass

        # 15..17 : has_target / has_time / has_id
        v[15] = 1.0 if ctx.target_col else 0.0
        v[16] = 1.0 if ctx.time_col else 0.0
        v[17] = 1.0 if ctx.id_col else 0.0

        # 18..21 : sentinel rule count, aux_df count (log), is_rec_seq, is_rec_ctr
        v[18] = math.log1p(len(ctx.sentinel_rules))
        v[19] = math.log1p(len(ctx.aux_dfs))
        v[20] = 1.0 if (ctx.task_type == "rec" and ctx.time_col) else 0.0
        v[21] = 1.0 if (ctx.task_type == "rec" and not ctx.time_col) else 0.0

        # 22..31 : reserved / padding (kept zero)
        # We deliberately keep these zero so future extensions can fill them in
        # without changing the state dimensionality.

        # Replace any NaN/Inf with 0 to keep the network well-behaved.
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return v
