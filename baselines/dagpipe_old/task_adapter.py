"""Task-specific schema adapters around DAGPipe's learned feature transform.

Recommendation IDs/history remain discrete; graph topology and supervision
masks remain owned by the dataset. Only training rows fit learned state.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd


def frozen_frame(data, filename):
    path = Path(data._std_test_dir(data.data_dir)) / filename
    if not path.is_file():
        raise FileNotFoundError(f'DAGPipe requires frozen training data: {path}; run build_std_test.py')
    frame = pd.read_parquet(path)
    if '__split__' in frame and not frame['__split__'].eq('train').all():
        raise ValueError(f'{filename} contains non-training split markers')
    return frame


def rec_adapter(data, config):
    feature = config.get('feature', {})
    target = feature.get('target_col')
    if not target:
        raise ValueError('DAGPipe recommendation requires feature.target_col')
    return {
        'task_type': 'rec', 'target': target,
        'user': data._user_id_col, 'item': data._item_id_col,
        'time': feature.get('time_col', 'timestamp'),
        'sequences': dict(feature.get('seq_cols', {})),
        'max_len': int(feature.get('seq_max_len', 20)),
        'split': {'val_ratio': config.get('train', {}).get('val_ratio', .1),
                  'seed': config.get('train', {}).get('seed', 42)},
        'side_tables': [name for name in ('user', 'item')
                        if getattr(data, name + '_df', None) is not None],
    }


def rec_splits(data, config, include_std_test=False, adapter=None):
    # The original checkout uses " operators" (a leading space), so these
    # files are not necessarily importable as dppbench.operators. Respect the
    # dataset's dynamic loader; also support checkouts with the normal name.
    operator_dir = Path(data.OPERATOR_DIR)
    if not operator_dir.is_dir():
        normal_dir = operator_dir.with_name('operators')
        if normal_dir.is_dir():
            data.OPERATOR_DIR = str(normal_dir)
    JoinTable = data._load_op('JoinTable')
    CreateSequence = data._load_op('CreateSequence')

    adapter = adapter or rec_adapter(data, config)
    data.interaction_df = frozen_frame(data, 'interaction_frozen.parquet')
    data._has_std_test = False
    if include_std_test and not data.load_std_test_frozen():
        raise FileNotFoundError('DAGPipe requires frozen recommendation standard test')
    target = adapter['target']
    frame = data.interaction_df
    if target not in frame:
        raise ValueError(f'DAGPipe missing recommendation target {target!r}')
    labels = pd.to_numeric(frame[target], errors='coerce')
    if not np.isfinite(labels.to_numpy(dtype=float)).all():
        raise ValueError('DAGPipe recommendation labels must be finite')
    data.model_cfg = copy.deepcopy(config)
    data.model_cfg.setdefault('train', {}).update(adapter['split'])
    data._validation_prepared = False
    data._prepare_validation()
    frame = data.interaction_df
    # Join only new side features; metadata/labels cannot be overwritten.
    for name in adapter['side_tables']:
        side = getattr(data, name + '_df', None)
        key = adapter[name]
        if side is None or key not in side:
            raise ValueError(f'DAGPipe missing {name} side table/key {key!r}')
        columns = [key] + [c for c in side if c not in frame and c not in (target, '__split__')]
        frame = JoinTable(method='key', aux_df=side[columns], key_col=key).transform(frame)
    for sequence, reference in adapter['sequences'].items():
        frame = CreateSequence(user_col=adapter['user'], item_col=reference,
                               time_col=adapter['time'], seq_col=sequence,
                               max_len=adapter['max_len']).transform(frame)
        frame[history_length_col(sequence)] = frame[sequence].map(len).astype(float)
    data.interaction_df = frame
    return {name: frame.loc[frame['__split__'].eq(name)].reset_index(drop=True)
            for name in ('train', 'val', 'std_test')}


def history_length_col(sequence):
    return f'__dagpipe_history_length_{sequence}'


def fit_rec_schema(adapter, train, col_types):
    """Freeze embedding vocabularies on the same training rows as the encoder."""
    sparse = list(dict.fromkeys([adapter['user'], adapter['item'], *adapter['sequences'].values()]))
    adapter['id_maps'] = {
        column: {value: i + 1 for i, value in enumerate(sorted(train[column].dropna().astype(str).unique()))}
        for column in sparse
    }
    excluded = {adapter['target'], '__split__', 'sample_weight', adapter['time'], *adapter['sequences']}
    return [c for c in train if c not in excluded and (
        c in sparse or c.startswith('__dagpipe_history_length_')
        or col_types.get(c) in ('numeric', 'categorical')
        or (c not in col_types and pd.api.types.is_numeric_dtype(train[c]))
    )]


def rec_output(frame, transformed, adapter):
    out = pd.DataFrame(transformed, columns=[f'dagpipe_feature_{i}' for i in range(transformed.shape[1])])
    for col, mapping in adapter['id_maps'].items():
        out[col] = frame[col].astype('string').map(mapping).fillna(0).to_numpy(dtype=np.int64)
    for seq, reference in adapter['sequences'].items():
        mapping = adapter['id_maps'][reference]
        out[seq] = frame[seq].map(lambda values: [mapping.get(str(v), 0) for v in values]).to_numpy()
    for col in (adapter['target'], '__split__'):
        out[col] = frame[col].to_numpy()
    return out


def graph_splits(data, include_std_test=False):
    # Keep unknown and future nodes for message passing, but never train on
    # their labels. The graph dataset defines the chronological masks.
    data.train_df = data._normalize_frozen_frame(frozen_frame(data, 'train_frozen.parquet'))
    data._has_std_test = False
    if include_std_test and not data.load_std_test_frozen():
        raise FileNotFoundError('DAGPipe requires frozen graph standard test')
    data._prepare_validation()
    frame = data.train_df
    return {name: frame.loc[frame['__split__'].eq(name)].reset_index(drop=True)
            for name in ('train', 'val', 'test', 'std_test')}
