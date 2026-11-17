"""Frozen categorical branches for the first daghie layer."""
import numpy as np
import pandas as pd
import torch

from dppbench.operators.preprocessing.target_encode import TargetEncode
from dppbench.operators.feature_engineering.frequency_encode import FrequencyEncode


ENCODERS = ('TargetEncoder', 'FrequencyEncoder')


def fit_category_state(frame, labels, specs, names):
    """Reuse public fit semantics and serialize only scalar lookup tables."""
    columns = [spec['column'] for spec in specs if spec['kind'] == 'category']
    if not columns or not names:
        return {}
    train = frame[columns].astype('string').copy()
    state = {}
    for name in names:
        if name == 'TargetEncoder':
            target = '__dagpipe_fit_target__'
            while target in train:
                target += '_'
            y = pd.to_numeric(labels, errors='raise').to_numpy(dtype=float)
            if len(y) != len(train) or not np.isfinite(y).all():
                raise ValueError('DAGPipe TargetEncoder requires aligned finite training labels')
            op = TargetEncode(cols=columns, target_col=target)
            op.transform(train.assign(**{target: y}))
            maps, fallback = op.encoding_map_, float(op.global_mean_)
        elif name == 'FrequencyEncoder':
            op = FrequencyEncode(cols=columns)
            op.transform(train)
            maps, fallback = op.freq_maps_, 0.0
        else:
            raise ValueError(f'Unknown categorical branch {name}')
        state[name] = {
            col: [fallback] + [float(maps[col].get(value, fallback))
                               for value in sorted(specs[i]['mapping'], key=specs[i]['mapping'].get)]
            for i, col in enumerate(spec['column'] for spec in specs) if col in columns
        }
    return state


class FrozenCategoryBranch(torch.nn.Module):
    """Replace categorical channels within this branch, retaining its width."""

    def __init__(self, specs, tables):
        super().__init__()
        self.columns = [i for i, spec in enumerate(specs) if spec['kind'] == 'category']
        if not self.columns or set(tables) != {specs[i]['column'] for i in self.columns}:
            raise ValueError('Invalid DAGPipe categorical branch schema')
        for i in self.columns:
            spec = specs[i]
            values = tables[spec['column']]
            if len(values) != len(spec['mapping']) + 1 or not np.isfinite(values).all():
                raise ValueError('Invalid DAGPipe categorical lookup table')
            self.register_buffer(f'lookup_{i}', torch.tensor(values, dtype=torch.float64), persistent=False)
        self.register_buffer('means', torch.tensor([specs[i]['mean'] for i in self.columns], dtype=torch.float64), persistent=False)
        self.register_buffer('scales', torch.tensor([specs[i]['scale'] for i in self.columns], dtype=torch.float64), persistent=False)

    def forward(self, x, source_input=None):
        # InputEncoder uses contiguous ordinal IDs; restore them before routing.
        source = x if source_input is None else source_input
        codes = (source[:, self.columns] * self.scales + self.means).round().long() + 1
        out = x.clone()
        for j, i in enumerate(self.columns):
            lookup = getattr(self, f'lookup_{i}')
            index = codes[:, j]
            index = torch.where((index >= 0) & (index < len(lookup)), index, 0)
            out[:, i] = lookup[index]
        return out
