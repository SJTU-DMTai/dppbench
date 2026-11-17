"""Legacy numeric recommendation proxy inputs on the current fixed partitions."""
import numpy as np
import pandas as pd
import torch


def recommendation_inputs(data, ctx, max_features, device):
    # Keep the current train-fitted ID mapping. Never randomly repartition here.
    data._remap_ids()
    frame = data.interaction_df
    tr = np.flatnonzero(frame['__split__'].eq('train').to_numpy())
    va = np.flatnonzero(frame['__split__'].eq('val').to_numpy())
    if not len(tr) or not len(va):
        raise ValueError('DiffPrep requires nonempty prepared recommendation train/val')
    if not frame['__split__'].isin(['train', 'val']).all():
        raise ValueError('Recommendation proxy must exclude held-out test rows')
    labels = pd.to_numeric(frame[ctx.target_col], errors='raise')
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError('Frozen recommendation labels must be binary and non-null')
    protected = {ctx.target_col, ctx.user_col, ctx.item_col, ctx.time_col, '__split__'}
    names = [c for c in ctx.numeric_cols if c not in protected and c in frame]
    if max_features:
        names = names[:max_features]
    if names:
        values = frame[names].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
    else:
        names = ['__constant__']
        values = np.zeros((len(frame), 1), dtype=np.float32)
    missing = np.isnan(values).astype(np.float32)
    values = np.nan_to_num(values, nan=0., posinf=0., neginf=0.)
    ref = values[tr]
    scale = ref.std(axis=0)
    values = np.clip((values - ref.mean(axis=0)) / np.where(scale < 1e-6, 1., scale), -5., 5.)
    tensor = lambda x, dtype: torch.as_tensor(x, dtype=dtype, device=device)
    users = tensor(frame[ctx.user_col].to_numpy(), torch.long)
    items = tensor(frame[ctx.item_col].to_numpy(), torch.long)
    return (tensor(values, torch.float32), tensor(missing, torch.float32), users, items,
            tensor(labels.to_numpy(dtype=np.float32), torch.float32), names, tr, va,
            max(len(data.category_maps_.get(ctx.user_col, {})) + 1, 2),
            max(len(data.category_maps_.get(ctx.item_col, {})) + 1, 2))
