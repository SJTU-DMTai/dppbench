"""Recommendation protocol v1: train-fitted categories and strict training history."""
import hashlib
import bisect
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache

VERSION = "rec_train_only_v1"


@lru_cache(maxsize=1)
def implementation_manifest():
    """Record code/config identity even in checkouts without Git metadata."""
    import json
    root = Path(__file__).resolve().parents[1]
    paths = set()
    for folder in ('dppbench', 'baselines/common', 'baselines/DiffPrep', 'baselines/DAGPipe'):
        paths.update((root/folder).rglob('*.py'))
    paths.update((root/'dppbench/tasks').rglob('model.yaml'))
    paths.update((root/'baselines').glob('*/config.yaml'))
    paths.update(root/'scripts'/name for name in ('final_model_hpo.py', 'evaluate_baseline.py'))
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    return {'protocol': VERSION, 'files': files,
            'implementation_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def frozen_file_manifest(directory):
    """Stream artifact fingerprints at final reporting, outside model selection."""
    files = {}
    for name in ('interaction_frozen.parquet', 'std_test.parquet', 'meta.json'):
        path = Path(directory)/name
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
        files[name] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
    return files


def training_rows(frame):
    return frame.loc[frame['__split__'].eq('train')] if '__split__' in frame else frame


def token(value):
    if value is None or pd.isna(value):
        return None
    return str(value)


def remap_training_categories(data):
    """Fit exclusively on model-training rows. OOV/padding=0, known IDs start at 1."""
    frame = data.interaction_df
    train = training_rows(frame)
    types = data.col_types
    for key in (data._user_id_col, data._item_id_col):
        if key in frame:
            types.setdefault(key, 'categorical')
    sequences = (data.model_cfg.get('feature', {}).get('seq_cols', {}) or {})
    related = {data._item_id_col: list(data._item_id_related_cols),
               data._user_id_col: list(data._user_id_related_cols)}
    for seq, ref in sequences.items():
        related.setdefault(ref, []).append(seq)
    maps = {}
    reference = getattr(data, '_rec_history_reference', None)
    if reference is not None and '__split__' in reference and not reference['__split__'].eq('train').all():
        raise ValueError('Category history reference must contain training events only')
    for col, kind in types.items():
        if kind != 'categorical' or col == data.model_cfg.get('feature', {}).get('target_col'):
            continue
        values = [token(v) for v in train[col]] if col in train else []
        if reference is not None and col in (data._user_id_col, data._item_id_col) and col in reference:
            # Context IDs are training observations; held-out queries cannot fit the vocabulary.
            values.extend(token(v) for v in reference[col])
        for seq in related.get(col, []):
            if seq in train:
                values.extend(token(v) for row in train[seq] if isinstance(row, (list, np.ndarray)) for v in row)
        maps[col] = {v: i + 1 for i, v in enumerate(sorted({v for v in values if v is not None}))}
    for col, kind in types.items():
        if kind == 'categorical_list' and col not in sequences and not any(col in v for v in related.values()):
            values = [token(v) for row in train.get(col, []) if isinstance(row, (list, np.ndarray)) for v in row]
            maps[col] = {v: i + 1 for i, v in enumerate(sorted({v for v in values if v is not None}))}
    for attr in ('interaction_df', 'user_df', 'item_df', 'std_test_negatives_df'):
        df = getattr(data, attr, None)
        if df is None:
            continue
        for col, mapping in maps.items():
            if col in df and types.get(col) != 'categorical_list':
                df[col] = df[col].map(lambda v: mapping.get(token(v), 0)).astype('int64')
        for seq, kind in types.items():
            if kind != 'categorical_list' or seq not in df:
                continue
            ref = sequences.get(seq)
            if ref is None:
                ref = next((c for c, names in related.items() if seq in names), seq)
            mapping = maps.get(ref, {})
            df[seq] = df[seq].map(lambda row: [mapping.get(token(v), 0) for v in row]
                                if isinstance(row, (list, np.ndarray)) else [])
    data.category_maps_ = maps
    data.model_cfg.setdefault('feature', {})['vocabulary_sizes'] = {c: len(m)+1 for c, m in maps.items()}
    data.rec_protocol_version = VERSION


def assert_disjoint(train, heldout, user='user_id', item='item_id', time='timestamp'):
    """Audit identities only; held-out labels are never needed for this check."""
    cols = [c for c in (user, item, time) if c in train and c in heldout]
    if len(cols) < 3:
        raise ValueError('Recommendation isolation requires user/item/time identity columns')
    def keys(df):
        return pd.util.hash_pandas_object(df[cols].astype('string'), index=False).to_numpy()
    a, b = keys(train), keys(heldout)
    if np.intersect1d(a, b).size:
        raise ValueError('Frozen recommendation training/test identities overlap')
    return {'protocol': VERSION, 'identity_columns': cols,
            'train_identity_sha256': hashlib.sha256(a.tobytes()).hexdigest(),
            'test_identity_sha256': hashlib.sha256(b.tobytes()).hexdigest()}


def history_transform(frame, op, time_col):
    """Strictly earlier timestamps, training observations only, including for train queries.

    Equal-time observations cannot see one another. Held-out rows never update state.
    Aggregate and rolling outputs are numeric; lag preserves the source semantic type
    (the caller registers it). No held-out labels or features fit these statistics.
    """
    if not time_col or time_col not in frame:
        raise ValueError(f'{op.name} requires an explicit timestamp for recommendation history')
    out = frame.copy()
    groups = op.group_cols or []
    groups = [groups] if isinstance(groups, str) else groups
    if op.name == 'AggregateGroupFeature':
        cols = [c for c in (op.agg_cols or []) if c in out and c not in groups]
        specs = [(c, agg, None, f'{c}_{agg}') for c in cols for agg in op.agg_funcs]
    elif op.name == 'CreateLagFeature':
        if any(int(lag) < 1 for lag in op.lags):
            raise ValueError('Recommendation lags must be positive')
        specs = [(op.target_col, 'lag', int(lag), f'{op.target_col}_lag_{lag}') for lag in op.lags]
    else:
        if any(int(w) < 1 for w in op.windows):
            raise ValueError('Recommendation rolling windows must be positive')
        specs = [(op.target_col, agg, int(w), f'{op.target_col}_roll_{agg}_{w}')
                 for w in op.windows for agg in op.aggs]
    train = training_rows(out)
    # Group-key tuples also work for an empty grouping and missing keys.
    def group_keys(df):
        return list(zip(*(df[c].map(lambda v: token(v) or '__MISSING__') for c in groups))) if groups else [()] * len(df)
    train_keys, query_keys = group_keys(train), group_keys(out)
    queries = {}
    for pos, key in enumerate(query_keys):
        queries.setdefault(key, []).append(pos)
    references = {}
    for key, idx in zip(train_keys, train.index):
        references.setdefault(key, []).append(idx)
    result = {name: [np.nan] * len(out) for _, _, _, name in specs}
    for key, indices in references.items():
        ref = train.loc[indices].sort_values(time_col, kind='stable')
        if ref[time_col].isna().any():
            raise ValueError('Training history timestamps must be non-null')
        times = ref[time_col].tolist()
        cached = {}
        for col, agg, window, name in specs:
            if col not in ref:
                continue
            s = ref[col].reset_index(drop=True)
            if agg == 'lag':
                cached[name] = s.shift(window - 1).tolist()
            else:
                numeric = s if agg == 'count' else pd.to_numeric(s, errors='raise')
                reducer = numeric.rolling(window, min_periods=1) if window else numeric.expanding(min_periods=1)
                cached[name] = reducer.agg(agg).tolist()
        # O(N log N); expanding/rolling statistics above are computed once.
        for pos in queries.get(key, []):
            when = out[time_col].iloc[pos]
            if pd.isna(when):
                continue
            previous = bisect.bisect_left(times, when) - 1
            if previous >= 0:
                for name, values in cached.items():
                    result[name][pos] = values[previous]
    for name, values in result.items():
        out[name] = values
    return out
