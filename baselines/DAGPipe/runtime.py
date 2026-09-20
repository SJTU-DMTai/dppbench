"""Train-only input conversion and inductive execution of the original network.

Each query is evaluated with the same fixed training reference rows. Original
batch-statistic operators remain unchanged; other query rows never enter its
context. This is an explicit deployment protocol, not a paper discretization.
"""
from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from baselines.common.budget import get_active_budget
from .vendor import operations as ops
from .vendor.dagpipe_hie import HierarchicalDAGPipe

LAYERS = [
    ['MeanImputer', 'MedianImputer', 'KNNImputer'],
    ['TargetEncoder', 'FrequencyEncoder'],
    ['StandardScaler', 'RobustScaler', 'MinMaxScaler', 'QuantileScaler', 'LogScaler', 'PolynomialTransform'],
    ['StandardScaler', 'RobustScaler', 'MinMaxScaler', 'QuantileScaler', 'LogScaler', 'PolynomialTransform'],
]


def limited_layers(max_nodes):
    """Round-robin prefixes preserve original per-layer order and coverage."""
    if max_nodes < 1:
        raise ValueError('DagPipe needs at least one operation node')
    selected = [[] for _ in LAYERS]
    for pos in range(max(map(len, LAYERS))):
        for i, layer in enumerate(LAYERS):
            if pos < len(layer) and sum(map(len, selected)) < max_nodes:
                selected[i].append(layer[pos])
    return [layer for layer in selected if layer]


def make_model(layers, n_features, device='cpu', dtype=torch.float64):
    if not layers or any(not layer for layer in layers):
        raise ValueError('DagPipe has empty operation layers')
    allowed = {name for layer in LAYERS for name in layer}
    if any(name not in allowed for layer in layers for name in layer):
        raise ValueError('Unknown DagPipe operation in checkpoint')
    return HierarchicalDAGPipe(n_features, [
        [getattr(ops, name)() for name in layer] for layer in layers
    ]).to(device=device, dtype=dtype)


def check_resources():
    budget = get_active_budget()
    if budget is not None:
        budget.check_resources()


class InputEncoder:
    """Fit numeric fill/scale and category maps on search training rows only.

    The single public DagPipeInputEncoding step includes median imputation,
    ordinal coding (unknown=-1), and z-scoring. This prevents the vendor's
    routing from destroying its missing-value sentinel before imputation.
    """
    def __init__(self, state=None):
        self.state = copy.deepcopy(state) if state is not None else None

    @staticmethod
    def values(series, spec):
        if spec['kind'] == 'numeric':
            return pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64, na_value=np.nan)
        if spec['kind'] == 'datetime':
            dt = pd.to_datetime(series, errors='coerce')
            return np.where(dt.notna(), dt.astype('int64').to_numpy()/1e9, np.nan)
        return series.astype('string').map(spec['mapping']).fillna(-1).to_numpy(dtype=np.float64)

    def fit(self, frame):
        if frame.empty or not len(frame.columns):
            raise ValueError('DagPipe requires nonempty training features')
        self.state = []
        for col in frame.columns:
            s = frame[col]
            kind = ('datetime' if pd.api.types.is_datetime64_any_dtype(s) else
                    'numeric' if pd.api.types.is_numeric_dtype(s) else 'category')
            spec = {'column': str(col), 'kind': kind}
            if kind == 'category':
                spec['mapping'] = {v: i for i, v in enumerate(sorted(s.dropna().astype(str).unique()))}
            v = self.values(s, spec)
            finite = v[np.isfinite(v)]
            fill = float(np.median(finite)) if len(finite) else 0.0
            v = np.where(np.isfinite(v), v, fill)
            spec.update(fill=fill, mean=float(v.mean()), scale=max(float(v.std()), 1e-6))
            self.state.append(spec)
        return self

    def transform(self, frame):
        if self.state is None:
            raise RuntimeError('DagPipe input encoder is not fitted')
        columns = []
        for spec in self.state:
            col = spec['column']
            if col not in frame:
                raise ValueError(f'DagPipe missing feature column {col!r}')
            v = self.values(frame[col], spec)
            v = np.where(np.isfinite(v), v, spec['fill'])
            columns.append((v-spec['mean'])/spec['scale'])
        out = np.column_stack(columns).astype(np.float64)
        if not np.isfinite(out).all():
            raise ValueError('DagPipe input conversion produced non-finite values')
        return out


class FrozenTransform:
    def __init__(self, payload, device='cpu'):
        self.encoder = InputEncoder(payload['encoder'])
        dtype = getattr(torch, payload.get('dtype', 'float64'))
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('DagPipe supports float32/float64 checkpoints only')
        self.model = make_model(payload['layers'], len(payload['encoder']), device, dtype)
        state = {k: torch.tensor(v, dtype=dtype, device=device)
                 for k, v in payload['state_dict'].items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.reference = torch.tensor(payload['reference'], dtype=dtype, device=device)
        if self.reference.ndim != 2 or len(self.reference) < 2:
            raise ValueError('DagPipe checkpoint requires at least two training reference rows')
        if self.reference.shape[1] != len(payload['encoder']) or not torch.isfinite(self.reference).all():
            raise ValueError('DagPipe checkpoint has invalid training reference shape/values')
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError('DagPipe checkpoint has non-finite parameters')
        self.cache = OrderedDict()
        self.cache_entries = int(payload.get('cache_entries', 2))

    def transform(self, frame):
        check_resources()
        x = self.encoder.transform(frame)
        key = (x.shape, hashlib.sha256(x.tobytes()).hexdigest())
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key].copy()
        out = np.empty_like(x)
        with torch.no_grad():
            for i, row in enumerate(x):
                if i % 32 == 0:
                    check_resources()
                query = torch.as_tensor(row, device=self.reference.device, dtype=self.reference.dtype).unsqueeze(0)
                # A fixed training context plus this query, never other test rows.
                value = self.model(torch.cat([self.reference, query], dim=0))[-1]
                if not torch.isfinite(value).all():
                    raise ValueError(f'DagPipe non-finite transform at query row {i}')
                out[i] = value.cpu().numpy()
        if self.cache_entries:
            self.cache[key] = out.copy()
            while len(self.cache) > self.cache_entries:
                self.cache.popitem(last=False)
        check_resources()
        return out
