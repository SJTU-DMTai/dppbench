"""Typed, train-only protocol around the vendor weighted DAG (not paper-exact)."""
import hashlib
from itertools import islice
import numpy as np
import pandas as pd
import torch
from .categorical import fit_category_state, FrozenCategoryBranch

REFERENCE_ROWS = 16
SPACE = 'typed_public_v4'
CANDIDATES = {
    'MeanImputer': ('impute', True, 'Public HandleMV mean on numeric training values'),
    'MedianImputer': ('impute', True, 'Public HandleMV median on numeric training values'),
    'KNNImputer': ('impute', False, 'Public sklearn nan-euclidean KNN is not the vendor soft KNN; no verified differentiable adapter'),
    'TargetEncoder': ('encode', True, 'Public smoothed means in feature-hash cross-fit folds; batch tasks only'),
    'FrequencyEncoder': ('encode', True, 'Public training-only category frequency, unknown/missing=0'),
    'Identity': ('numeric', True, 'Real pass-through node; consumes one step'),
    'StandardScaler': ('numeric', True, 'Public sample std and EPS rule'),
    'RobustScaler': ('numeric', True, 'Public interpolated quartiles, positive IQR retained'),
    'MinMaxScaler': ('numeric', True, 'Public default [0,1], constant columns output 0'),
    'QuantileScaler': ('numeric', False, 'Vendor batch ranks are not a verified public quantile transform'),
    'LogScaler': ('numeric', False, 'Vendor train-min offset/clamp is not a verified public log protocol'),
    'PolynomialTransform': ('numeric', False, 'Vendor elementwise square differs from public appended polynomial features; fixed-width DAG cannot silently substitute'),
}
SCALERS = tuple(n for n, (group, enabled, _) in CANDIDATES.items() if group == 'numeric' and enabled and n != 'Identity')


def fit_input_encoder(frame, categorical_columns=(), column_types=None):
    from .runtime import InputEncoder
    encoder = InputEncoder(preserve_missing=True)
    encoder.state = []
    for col in frame:
        series = frame[col]
        declared = (column_types or {}).get(col)
        if declared in ('timestamp', 'datetime') and not (pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series)):
            raise ValueError(f'Temporal column {col} must be normalized by the dataset protocol before DAGPipe')
        kind = ('category' if col in categorical_columns else
                'datetime' if pd.api.types.is_datetime64_any_dtype(series) else
                'numeric' if declared in ('numeric', 'numerical', 'continuous') or pd.api.types.is_numeric_dtype(series) else 'category')
        spec = dict(column=str(col), kind=kind, mean=0., scale=1., fill=0.)
        if kind == 'category':
            spec['mapping'] = {value: i for i, value in enumerate(sorted(series.dropna().astype(str).unique()))}
        encoder.state.append(spec)
    validate_raw_features(frame, encoder.state)
    return encoder


def row_folds(values, folds, seed):
    # Only feature bytes, never labels, query batch membership or row positions.
    array = values.detach().cpu().numpy() if torch.is_tensor(values) else values
    array = np.asarray(array, dtype='<f8').copy()
    array[np.isnan(array)] = 0.
    array[array == 0] = 0.  # canonicalize signed zero
    salt = str(int(seed)).encode() + b':'
    return np.array([int.from_bytes(hashlib.sha256(salt + row.tobytes()).digest()[:8], 'little') % folds
                     for row in array], dtype=np.int64)


def cover_reference(values, specs, permutation, count):
    """Find training rows covering numeric fit inputs, with seeded tie-breaking."""
    numeric = [i for i, spec in enumerate(specs) if spec['kind'] != 'category']
    observed = ~np.isnan(values[:, numeric])
    absent = [specs[i]['column'] for i, present in zip(numeric, observed.any(axis=0)) if not present]
    if absent:
        raise ValueError(f'DAGPipe numeric features have no observed training values: {absent}')

    selected = []
    uncovered = np.ones(len(numeric), dtype=bool)
    while uncovered.any() and len(selected) < count:
        scores = observed[:, uncovered].sum(axis=1)
        row = int(permutation[np.argmax(scores[permutation])])
        selected.append(row)
        uncovered &= ~observed[row]
    if uncovered.any():
        missing = [specs[i]['column'] for i, pending in zip(numeric, uncovered) if pending]
        raise ValueError(f'DAGPipe greedy reference selection could not cover these features in {count} rows: {missing}')

    used = set(selected)
    # Fill spare slots from the original permutation without duplicate rows.
    remaining = count - len(selected)
    selected.extend(islice((int(row) for row in permutation if row not in used), remaining))
    return selected


def allocate_layers(specs, values, budget, allow_target=True, allow_frequency=True, *,
                    task=None, ablate_elliptic_second_robust_scaler=False):
    numeric = [i for i, s in enumerate(specs) if s['kind'] != 'category']
    if np.isinf(values).any() or (numeric and np.isnan(values[:, numeric]).all(axis=0).any()):
        raise ValueError('DAGPipe requires finite observed training values in every numeric column; all-missing columns cannot fit public mean/median')
    categorical = any(s['kind'] == 'category' for s in specs)
    groups = []
    # Keep a real imputer stage even when train is complete: deployment can
    # contain missing values, and implicit zero filling must not be hidden.
    if numeric:
        groups.append([n for n, (g, enabled, _) in CANDIDATES.items() if g == 'impute' and enabled])
    if categorical:
        encoders = [name for name, allowed in (
            ('TargetEncoder', allow_target), ('FrequencyEncoder', allow_frequency)
        ) if allowed and CANDIDATES[name][1]]
        if not numeric:
            encoders.append('Identity')
        if encoders:
            groups.append(encoders)
    if numeric:
        group = [n for n, (g, enabled, _) in CANDIDATES.items() if g == 'numeric' and enabled]
        groups += [list(group), list(group)]
    if not groups or budget < len(groups):
        raise ValueError('DAGPipe budget cannot cover applicable typed stages')
    selected = [[] for _ in groups]
    # Give each stage a node before expanding it; do not hide discarded nodes.
    for position in range(max(map(len, groups))):
        for i, group in enumerate(groups):
            if position < len(group) and sum(map(len, selected)) < budget:
                selected[i].append(group[position])
    if ablate_elliptic_second_robust_scaler:
        if task != 'elliptic_bitcoin' or 'RobustScaler' not in selected[-1]:
            raise ValueError('This ablation requires Elliptic and a second-stage RobustScaler')
        # Remove after allocation so the vacant slot stays empty.
        selected[-1].remove('RobustScaler')
    return selected


def validate_raw_features(frame, specs):
    for spec in specs:
        if spec['kind'] == 'numeric':
            values = pd.to_numeric(frame[spec['column']], errors='raise').to_numpy(dtype=float, na_value=np.nan)
            if np.isinf(values).any():
                raise ValueError(f'Infinite numeric input in {spec["column"]}; DAGPipe does not clip or silently impute infinity')


def fit_tables(frame, labels, specs, values, names, seed, folds=5):
    from baselines.common.search_cost import record_work
    if folds < 2:
        raise ValueError('Target encoding requires >=2 folds')
    if not names:
        return {}
    result = fit_category_state(frame, labels, specs, [n for n in names if n != 'TargetEncoder'])
    if 'FrequencyEncoder' in names:
        record_work('operator_work.frequency_fit', fitted_rows=len(frame))
    if 'TargetEncoder' in names:
        assignment = row_folds(values, folds, seed)
        tables = []
        for fold in range(folds):
            keep = assignment != fold
            if not keep.any():
                raise ValueError('TargetEncoder has an empty out-of-fold fit set; insufficient distinct feature rows')
            tables.append(fit_category_state(frame.loc[keep], labels.loc[keep], specs, ['TargetEncoder'])['TargetEncoder'])
            record_work('operator_work.target_fold_fit', fitted_rows=int(keep.sum()), folds=1)
        result['TargetEncoder'] = dict(folds=folds, seed=int(seed), tables=tables)
    return result


class CrossFitCategory(torch.nn.Module):
    def __init__(self, specs, state):
        super().__init__()
        self.folds, self.seed = int(state['folds']), int(state['seed'])
        if self.folds < 2 or len(state['tables']) != self.folds:
            raise ValueError('Invalid cross-fit TargetEncoder state')
        self.branches = torch.nn.ModuleList([FrozenCategoryBranch(specs, t) for t in state['tables']])

    def forward(self, x, source):
        ids = torch.as_tensor(row_folds(source, self.folds, self.seed), device=x.device)
        result = x.clone()
        for fold, branch in enumerate(self.branches):
            mask = ids == fold
            if mask.any():
                result[mask] = branch(x[mask], source[mask])
        return result


class TypedOperation(torch.nn.Module):
    def __init__(self, name, specs, state):
        super().__init__()
        from .vendor import operations
        self.name = name
        self.frozen_stats = None
        self.freeze_requested = False
        self.observed_abs_max = None
        self.numeric = [i for i, spec in enumerate(specs) if spec['kind'] != 'category']
        if name == 'TargetEncoder':
            self.op = CrossFitCategory(specs, state[name])
        elif name == 'FrequencyEncoder':
            self.op = FrozenCategoryBranch(specs, state[name])
        elif name == 'Identity':
            self.op = torch.nn.Identity()
        elif name in (*SCALERS, 'MeanImputer', 'MedianImputer'):
            if not self.numeric:
                raise ValueError('Numeric operation has no applicable columns')
            self.op = getattr(operations, name)()
        else:
            raise ValueError(f'Unknown typed operation {name}')

    def execute(self, weighted, count, source, current, routing):
        out = self._execute(weighted, count, source, current, routing)
        if not torch.isfinite(out).all():
            raise ValueError(f'Nonfinite DAGPipe node output: {self.name}; public formula retained, no clipping applied')
        if out.numel():
            peak = out.detach().abs().max()
            self.observed_abs_max = peak if self.observed_abs_max is None else torch.maximum(self.observed_abs_max, peak)
        return out

    def _execute(self, weighted, count, source, current, routing):
        from .runtime import reference_operation, reference_impute
        if self.name in ('TargetEncoder', 'FrequencyEncoder'):
            return self.op(current, source) * routing
        if self.name == 'Identity':
            return weighted
        out = weighted.clone()
        if self.name.endswith('Imputer'):
            values = source[:, self.numeric]
            if self.frozen_stats is not None:
                fill = values.new_tensor(self.frozen_stats['fill'])
                filled = torch.where(values.isnan(), fill, values)
            else:
                ref = values[:count]
                fill = (torch.nanmean(ref, dim=0) if self.name == 'MeanImputer' else torch.nanquantile(ref, .5, dim=0))
                filled = torch.where(values.isnan(), fill, values)
                if self.freeze_requested:
                    self.frozen_stats = {'fill': fill.detach().cpu().tolist()}
            out[:, self.numeric] = filled * routing[:, self.numeric]
        else:
            values = weighted[:, self.numeric]
            if self.frozen_stats is not None:
                center = values.new_tensor(self.frozen_stats['center'])
                denominator = values.new_tensor(self.frozen_stats['denominator'])
                result = (values - center) / denominator
                if self.name == 'MinMaxScaler':
                    result = torch.where(torch.tensor(self.frozen_stats['constant_output'], device=values.device), 0., result)
                out[:, self.numeric] = result
            else:
                def capture(stage, value, **details):
                    if stage == 'scaler_statistics' and self.freeze_requested:
                        self.frozen_stats = {k: details[k].detach().flatten().cpu().tolist()
                                             for k in ('center', 'denominator', 'constant_output')}
                out[:, self.numeric] = reference_operation(self.op, values, count, 'public_v1', observe=capture)
        return out


def freeze_statistics(model, reference):
    for layer in model.op_layers:
        for op in layer:
            op.freeze_requested = True
    with torch.no_grad():
        model(reference, reference_rows=len(reference))
    model.frozen_statistics = True
    return [[op.frozen_stats for op in layer] for layer in model.op_layers]


def install_statistics(model, nodes):
    if len(nodes) != len(model.op_layers) or any(len(a) != len(b) for a, b in zip(nodes, model.op_layers)):
        raise ValueError('Frozen statistic topology mismatch')
    for layer, states in zip(model.op_layers, nodes):
        for op, state in zip(layer, states):
            expected = ({'fill'} if op.name.endswith('Imputer') else
                        {'center', 'denominator', 'constant_output'} if op.name in SCALERS else set())
            if set(state or {}) != expected:
                raise ValueError('Missing or unexpected frozen node statistics')
            for key, values in (state or {}).items():
                if len(values) != len(op.numeric) or not np.isfinite(values).all():
                    raise ValueError('Invalid frozen statistic dimensions/values')
                if key == 'denominator' and (np.asarray(values) <= 0).any():
                    raise ValueError('Invalid frozen scaler denominator')
                if key == 'constant_output' and any(type(value) is not bool for value in values):
                    raise ValueError('Invalid constant-column mask')
            op.frozen_stats = state
    model.frozen_statistics = True


def numerical_report(model):
    return [dict(layer=i, node=j, operator=op.name,
                 observed_abs_max=float(op.observed_abs_max) if op.observed_abs_max is not None else None)
            for i, layer in enumerate(model.op_layers) for j, op in enumerate(layer)]


def make_typed_model(layers, specs, state, device, dtype):
    from .vendor.dagpipe_hie import HierarchicalDAGPipe
    if dtype != torch.float64:
        raise ValueError('DAGPipe requires float64 for stable cross-fit row hashing')
    allowed_positions = {'MeanImputer': 0, 'MedianImputer': 0}
    for i, layer in enumerate(layers):
        if not layer:
            raise ValueError('Empty typed DAG layer')
        if len(set(layer)) != len(layer):
            raise ValueError('Duplicate typed nodes within a layer')
        for name in layer:
            if name in allowed_positions and i != 0:
                raise ValueError('Imputation must precede numeric stages')
    model = HierarchicalDAGPipe(len(specs), [[TypedOperation(n, specs, state) for n in layer] for layer in layers])
    model.operator_space = SPACE
    model.scale_protocol = 'public_v1'
    return model.to(device=device, dtype=dtype)
