"""Train-only input conversion and inductive execution of the original network.

All stages fit operator statistics only on the same fixed training reference.
Query rows never contribute to statistics, including during search. This is an explicit deployment protocol, not a paper discretization.
"""
from __future__ import annotations

import copy
import hashlib
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from baselines.common.budget import get_active_budget
from .vendor import operations as ops
from .categorical import ENCODERS, FrozenCategoryBranch
from .vendor.dagpipe_hie import HierarchicalDAGPipe

# Numeric defaults also define the original v2 checkpoint vocabulary.
CURRENT_LAYERS = [
    ['Identity', 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'PolynomialTransform'],
    ['Identity', 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'PolynomialTransform'],
]
# Released daghie candidate order, before task applicability and node budgets.
LAYERS = [
    ['MeanImputer', 'MedianImputer', 'KNNImputer'],
    ['TargetEncoder', 'FrequencyEncoder'],
    ['StandardScaler', 'RobustScaler', 'MinMaxScaler', 'QuantileScaler', 'LogScaler', 'PolynomialTransform'],
    ['StandardScaler', 'RobustScaler', 'MinMaxScaler', 'QuantileScaler', 'LogScaler', 'PolynomialTransform'],
]


def reference_impute(op, values, reference_rows):
    """Use only observed training-reference values as imputation donors."""
    missing = values.isnan()
    if not missing.any():
        return values
    ref = values[:reference_rows]
    fill = torch.nanmean(ref, dim=0).nan_to_num(nan=0.0)
    if isinstance(op, ops.MedianImputer):
        fill = torch.nanquantile(ref, .5, dim=0).nan_to_num(nan=0.0)
    if not isinstance(op, ops.KNNImputer):
        return torch.where(missing, fill, values)
    out = torch.where(missing, fill, values)
    for column in missing.any(dim=0).nonzero().flatten().tolist():
        rows = missing[:, column]
        donors = ref[~ref[:, column].isnan()]
        if not len(donors):
            continue
        delta = values[rows, None, :] - donors[None, :, :]
        overlap = (~delta.isnan()).sum(dim=-1)
        distance = (delta.nan_to_num().square().sum(dim=-1) / overlap.clamp_min(1))
        distance = distance.masked_fill(overlap == 0, float('inf'))
        nearest, indices = distance.topk(min(5, len(donors)), largest=False)
        valid = nearest.isfinite()
        counts = valid.sum(dim=1)
        estimate = (donors[indices, column] * valid).sum(dim=1) / counts.clamp_min(1)
        out[rows, column] = torch.where(counts > 0, estimate, fill[column])
    return out


def reference_operation(op, x, reference_rows, scale_protocol=None, *,
                        unrouted_input=None, routing_weight=None, source_input=None, observe=None):
    """Fit statistics on the training reference prefix, never on query rows."""
    from .protocol import TypedOperation
    if isinstance(op, TypedOperation):
        return op.execute(x, reference_rows, source_input, unrouted_input, routing_weight)
    if isinstance(op, FrozenCategoryBranch):
        if unrouted_input is None or routing_weight is None:
            raise ValueError('FrozenCategoryBranch requires unrouted input and routing weights')
        encoded = op(unrouted_input, source_input)
        if observe is not None:
            observe('operator_output_before_routing', encoded)
        return encoded * routing_weight
    if isinstance(op, (ops.MeanImputer, ops.MedianImputer, ops.KNNImputer)):
        if source_input is None or routing_weight is None:
            raise ValueError('DAGPipe imputers require the source values and routing weights')
        filled = reference_impute(op, source_input, reference_rows)
        if observe is not None:
            observe('operator_output_before_routing', filled)
        return filled * routing_weight
    ref = x[:reference_rows]
    if isinstance(op, torch.nn.Identity):
        return x
    if isinstance(op, ops.PolynomialTransform):
        return op(x)
    if isinstance(op, ops.LogScaler):
        return (x - ref.min(0, keepdim=True).values + 1).clamp_min(1e-8).log()
    if isinstance(op, ops.QuantileScaler):
        sorted_ref = ref.sort(dim=0).values.T.contiguous()
        return (torch.searchsorted(sorted_ref, x.T.contiguous()).T / reference_rows).to(x.dtype)
    if isinstance(op, ops.StandardScaler):
        center, scale = ref.mean(0, keepdim=True), ref.std(0, keepdim=True)
    elif isinstance(op, ops.RobustScaler):
        center = (torch.quantile(ref, .5, dim=0, keepdim=True) if scale_protocol == 'public_v1'
                  else ref.median(0, keepdim=True).values)
        q75 = torch.quantile(ref, .75, dim=0, keepdim=True)
        q25 = torch.quantile(ref, .25, dim=0, keepdim=True)
        scale = q75 - q25
    elif isinstance(op, ops.MinMaxScaler):
        center = ref.min(0, keepdim=True).values
        scale = ref.max(0, keepdim=True).values - center
    else:
        raise ValueError(f'Unsupported inductive DAGPipe operator: {type(op).__name__}')
    raw_scale = scale
    details = {}
    if observe is not None:
        if isinstance(op, ops.RobustScaler):
            details.update(q25=q25, q75=q75)
        elif isinstance(op, ops.MinMaxScaler):
            details.update(reference_min=center, reference_max=ref.max(0, keepdim=True).values)
    if scale_protocol == "public_v1":
        if isinstance(op, ops.StandardScaler):
            from dppbench.operators.preprocessing.scale_feature import ScaleFeature
            usable = torch.isfinite(scale) & (scale >= ScaleFeature.EPS)
        else:
            usable = torch.isfinite(scale) & (scale > 0)
        denominator = torch.where(usable, scale, torch.ones_like(scale))
        if observe is not None:
            observe('scaler_statistics', None, center=center, raw_scale=raw_scale, denominator=denominator,
                    constant_output=(~usable if isinstance(op, ops.MinMaxScaler) else torch.zeros_like(usable)),
                    **details)
        out = (x - center) / denominator
        if isinstance(op, ops.MinMaxScaler):
            out = torch.where(usable, out, torch.zeros_like(out))
        return out
    # A nearly degenerate IQR/range may exceed the absolute epsilon yet
    # amplify rare values by millions. Use training spread as a relative
    # floor; never estimate this safeguard from validation/query rows.
    if scale_protocol != 'absolute_epsilon_v2':
        spread = ref.std(0, keepdim=True, unbiased=False)
        floor = torch.maximum(spread * 1e-6, torch.full_like(spread, 1e-8))
        scale = torch.where(scale.abs() <= floor, spread, scale)
    scale = torch.where(scale.abs() <= 1e-8, torch.ones_like(scale), scale)
    if observe is not None:
        observe('scaler_statistics', None, center=center, raw_scale=raw_scale, denominator=scale, **details)
    return (x-center)/scale


def reference_forward(model, queries, reference):
    if getattr(model, 'frozen_statistics', False):
        return model(queries, reference_rows=0)
    return model(torch.cat([reference, queries], dim=0), reference_rows=len(reference))[len(reference):]


def limited_layers(max_nodes, categorical=False, operator_space='current', *, encoder_bypasses=False):
    """Round-robin prefixes preserve original per-layer order and coverage."""
    if max_nodes < 1:
        raise ValueError('DAGPipe needs at least one operation node')
    if operator_space == 'legacy_daghie':
        layers = [list(layer) for layer in LAYERS]
        if not categorical:
            if encoder_bypasses:
                layers[1] = ['Identity'] * len(layers[1])
            else:
                del layers[1]
    elif operator_space == 'current':
        layers = [list(layer) for layer in CURRENT_LAYERS]
    else:
        raise ValueError(f'Unknown DAGPipe operator space {operator_space!r}')
    if categorical and operator_space == 'current':
        layers[0][:0] = ENCODERS
    selected = [[] for _ in layers]
    for pos in range(max(map(len, layers))):
        for i, layer in enumerate(layers):
            if pos < len(layer) and sum(map(len, selected)) < max_nodes:
                selected[i].append(layer[pos])
    return [layer for layer in selected if layer]


def make_model(layers, n_features, device='cpu', dtype=torch.float64, *,
               specs=None, category_state=None, scale_protocol=None, operator_space='current'):
    if not layers or any(not layer for layer in layers):
        raise ValueError('DAGPipe has empty operation layers')
    if operator_space == 'typed_public_v4':
        from .protocol import make_typed_model
        if not specs or len(specs) != n_features:
            raise ValueError('Typed DAG requires a matching feature schema')
        return make_typed_model(layers, specs, category_state or {}, device, dtype)
    if operator_space not in ('current', 'legacy_daghie'):
        raise ValueError(f'Unknown DAGPipe operator space {operator_space!r}')
    allowed = {name for layer in CURRENT_LAYERS for name in layer}
    if operator_space == 'legacy_daghie':
        allowed.update(name for layer in LAYERS for name in layer if name not in ENCODERS)
    category_state = category_state or {}
    operation_layers = []
    for index, layer in enumerate(layers):
        operations = []
        for name in layer:
            if name in ENCODERS and index == (1 if operator_space == 'legacy_daghie' else 0) and name in category_state and specs:
                operations.append(FrozenCategoryBranch(specs, category_state[name]))
            elif name in allowed:
                operations.append(torch.nn.Identity() if name == 'Identity' else getattr(ops, name)())
            else:
                raise ValueError(f'Unsupported or unfitted DAGPipe node: {name}')
        operation_layers.append(operations)
    model = HierarchicalDAGPipe(n_features, operation_layers)
    model.operator_space = operator_space
    model.scale_protocol = scale_protocol
    return model.to(device=device, dtype=dtype)


def check_resources():
    budget = get_active_budget()
    if budget is not None:
        budget.check_resources()


class InputEncoder:
    """Fit numeric fill/scale and category maps on search training rows only.

    The single public DAGPipeInputEncoding step includes median imputation,
    ordinal coding (unknown=-1), and z-scoring. This prevents the vendor's
    routing from destroying its missing-value sentinel before imputation.
    """
    def __init__(self, state=None, preserve_missing=False):
        self.state = copy.deepcopy(state) if state is not None else None
        self.preserve_missing = preserve_missing

    @staticmethod
    def values(series, spec):
        if spec['kind'] == 'numeric':
            return pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64, na_value=np.nan)
        if spec['kind'] == 'datetime':
            dt = pd.to_datetime(series, errors='coerce')
            return np.where(dt.notna(), dt.astype('int64').to_numpy()/1e9, np.nan)
        return series.astype('string').map(spec['mapping']).fillna(-1).to_numpy(dtype=np.float64)

    def fit(self, frame, categorical_columns=()):
        if frame.empty or not len(frame.columns):
            raise ValueError('DAGPipe requires nonempty training features')
        self.state = []
        for col in frame.columns:
            s = frame[col]
            kind = ('category' if col in categorical_columns else
                    'datetime' if pd.api.types.is_datetime64_any_dtype(s) else
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
            raise RuntimeError('DAGPipe input encoder is not fitted')
        columns = []
        for spec in self.state:
            col = spec['column']
            if col not in frame:
                raise ValueError(f'DAGPipe missing feature column {col!r}')
            v = self.values(frame[col], spec)
            fill = np.nan if self.preserve_missing else spec['fill']
            v = np.where(np.isfinite(v), v, fill)
            columns.append((v-spec['mean'])/spec['scale'])
        out = np.column_stack(columns).astype(np.float64)
        if np.isinf(out).any() or (not self.preserve_missing and not np.isfinite(out).all()):
            raise ValueError('DAGPipe input conversion produced non-finite values')
        return out


class FrozenTransform:
    def __init__(self, payload, device='cpu'):
        space = payload.get('operator_space', 'current')
        self.encoder = InputEncoder(payload['encoder'], preserve_missing=space in ('legacy_daghie', 'typed_public_v4'))
        dtype = getattr(torch, payload.get('dtype', 'float64'))
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('DAGPipe supports float32/float64 checkpoints only')
        self.model = make_model(payload['layers'], len(payload['encoder']), device, dtype,
                                specs=payload['encoder'], category_state=payload.get('category_state'),
                                scale_protocol=payload.get('scale_protocol'), operator_space=space)
        self.typed_protocol = space == 'typed_public_v4'
        state = {k: torch.tensor(v, dtype=dtype, device=device)
                 for k, v in payload['state_dict'].items()}
        self.model.load_state_dict(state, strict=True)
        if space == 'typed_public_v4':
            from .protocol import install_statistics
            install_statistics(self.model, payload['frozen_nodes'])
        self.model.eval()
        self.reference = torch.tensor(payload['reference'], dtype=dtype, device='cpu' if self.typed_protocol else device)
        if self.reference.ndim != 2 or len(self.reference) < 2:
            raise ValueError('DAGPipe checkpoint requires at least two training reference rows')
        invalid = self.reference.isinf().any() or (space not in ('legacy_daghie', 'typed_public_v4') and self.reference.isnan().any())
        if self.reference.shape[1] != len(payload['encoder']) or invalid:
            raise ValueError('DAGPipe checkpoint has invalid training reference shape/values')
        if self.typed_protocol:
            # Audit rows remain in the YAML payload. Query execution needs only
            # frozen node statistics, so do not retain a training matrix on GPU.
            self.reference = torch.empty((0, len(payload['encoder'])), dtype=dtype, device=device)
        if any(not torch.isfinite(p).all() for p in self.model.parameters()):
            raise ValueError('DAGPipe checkpoint has non-finite parameters')
        self.cache = OrderedDict()
        self.cache_entries = int(payload.get('cache_entries', 2))
        self.cache_max_bytes = int(float(payload.get('cache_max_mb', 512))*1024**2)
        self.cache_bytes = 0
        self.batch_size = int(payload.get('transform_batch_size', 1024))
        if self.cache_entries < 0 or self.cache_max_bytes < 0 or self.batch_size < 1:
            raise ValueError('Invalid DAGPipe transform cache/batch size')
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'seconds': 0.0,
                      'encoding_seconds': 0.0, 'hash_seconds': 0.0,
                      'model_transform_wall_seconds': 0.0, 'cache_copy_seconds': 0.0,
                      'cached_bytes': 0, 'peak_cached_bytes': 0, 'oversize_skips': 0}

    def transform(self, frame):
        started = time.perf_counter()
        check_resources()
        encoding_started = time.perf_counter()
        if self.typed_protocol:
            from .protocol import validate_raw_features
            validate_raw_features(frame, self.encoder.state)
        x = self.encoder.transform(frame)
        encoding_seconds = time.perf_counter()-encoding_started
        hash_started = time.perf_counter()
        # Hash the contiguous buffer directly; tobytes() would allocate another
        # full matrix even on a hit. Schema is fixed by this frozen encoder.
        key = (x.shape, x.dtype.str, hashlib.sha256(memoryview(x).cast('B') if x.size else b'').hexdigest())
        hash_seconds = time.perf_counter()-hash_started
        self.stats['encoding_seconds'] += encoding_seconds
        self.stats['hash_seconds'] += hash_seconds
        if key in self.cache:
            self.cache.move_to_end(key)
            self.stats['hits'] += 1
            copy_started = time.perf_counter()
            result = self.cache[key].copy()
            self.stats['cache_copy_seconds'] += time.perf_counter()-copy_started
            self.stats['seconds'] += time.perf_counter()-started
            print(f'[DAGPipe transform] cache=hit rows={len(x)} encode_s={encoding_seconds:.6f} hash_s={hash_seconds:.6f}')
            check_resources()
            return result
        self.stats['misses'] += 1
        out = np.empty_like(x)
        transform_started = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                check_resources()
                query = torch.as_tensor(x[start:start+self.batch_size], device=self.reference.device, dtype=self.reference.dtype)
                value = reference_forward(self.model, query, self.reference)
                if not torch.isfinite(value).all():
                    raise ValueError(f'DAGPipe non-finite transform at query row {start}')
                out[start:start+len(query)] = value.cpu().numpy()
        transform_seconds = time.perf_counter()-transform_started
        self.stats['output_abs_max'] = float(np.max(np.abs(out))) if out.size else 0.0
        self.stats['model_transform_wall_seconds'] += transform_seconds
        if self.cache_entries and self.cache_max_bytes and out.nbytes <= self.cache_max_bytes:
            # Evict BEFORE making a copy, keeping retained cache bytes bounded.
            while self.cache and (len(self.cache) >= self.cache_entries or
                                  self.cache_bytes+out.nbytes > self.cache_max_bytes):
                _, removed = self.cache.popitem(last=False)
                self.cache_bytes -= removed.nbytes
                del removed
                self.stats['evictions'] += 1
            copy_started = time.perf_counter()
            self.cache[key] = out.copy()
            self.stats['cache_copy_seconds'] += time.perf_counter()-copy_started
            self.cache_bytes += out.nbytes
            self.stats['cached_bytes'] = self.cache_bytes
            self.stats['peak_cached_bytes'] = max(self.stats['peak_cached_bytes'], self.cache_bytes)
        elif self.cache_entries and out.nbytes > self.cache_max_bytes:
            self.stats['oversize_skips'] += 1
        check_resources()
        elapsed = time.perf_counter()-started
        self.stats['seconds'] += elapsed
        print(f'[DAGPipe transform] cache=miss rows={len(x)} encode_s={encoding_seconds:.6f} '
              f'hash_s={hash_seconds:.6f} transform_wall_s={transform_seconds:.6f} '
              f'cached_bytes={self.cache_bytes} total_wall_s={elapsed:.6f} '
              f'output_abs_max={self.stats["output_abs_max"]:.6g}')
        return out
