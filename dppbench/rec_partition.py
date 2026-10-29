"""Fixed recommendation partitions and budgeted query views, before feature fitting.

No rows are deduplicated/reweighted. Bookkeeping lives on the data object, never
in prediction columns. Amazon groups are conservative original user/item/time.
"""
import contextlib
import contextvars
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_VERSION = 'rec_event_group_split_v1'
BOUNDARY_VERSION = 'rec_fixed_validation_v1'
_artifact_root = contextvars.ContextVar('rec_partition_artifact_root', default=None)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _json_sha(value):
    return _sha(json.dumps(value, sort_keys=True, default=str, separators=(',', ':')).encode())


@contextlib.contextmanager
def partition_audit_scope(directory):
    token = _artifact_root.set(Path(directory))
    try:
        yield
    finally:
        _artifact_root.reset(token)


@lru_cache(maxsize=32)
def _file_sha(path, size, mtime_ns):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return _file_sha(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def _distribution(labels):
    return {str(k): int(v) for k, v in pd.Series(labels).value_counts(dropna=False).sort_index().items()}


def _group_assignment(codes, labels, ratio, seed, groups):
    """Seeded cumulative-deficit allocation; whole groups, approximate row strata.

    Minimize normalized squared class/row deficit against ratio * cumulative
    counts after each shuffled group. Ties choose train. No metric selection,
    retries, majority-label rewrite or row reassignment within a group.
    """
    if groups < 2:
        raise ValueError('Event-group validation needs at least two distinct groups')
    classes, y = np.unique(labels, return_inverse=True)
    counts = np.zeros((groups, len(classes) + 1), dtype=np.int64)
    np.add.at(counts, (codes, y), 1)
    counts[:, -1] = counts[:, :-1].sum(axis=1)
    scale = np.maximum(counts.sum(axis=0) * ratio, 1.)
    cumulative = np.zeros(counts.shape[1]); chosen = cumulative.copy()
    assignment = np.zeros(groups, dtype=np.int8)
    order = np.random.RandomState(seed).permutation(groups)
    for step, g in enumerate(order):
        row = counts[g]; cumulative += row
        goal = ratio * cumulative
        stay = np.square((chosen - goal) / scale).sum()
        take = np.square((chosen + row - goal) / scale).sum()
        use_val = take < stay
        # Nonempty partitions are mandatory even for a tiny dominant group.
        if step == groups - 1:
            if not assignment.sum():
                use_val = True
            elif assignment.sum() == groups - 1:
                use_val = False
        if use_val:
            assignment[g] = 1; chosen += row
    return assignment[codes], counts, classes


def prepare_validation(data):
    """Called once on the FULL frozen training pool, before joins/remapping."""
    if getattr(data, '_validation_prepared', False):
        return
    from sklearn.model_selection import train_test_split
    cfg = data.model_cfg or {}
    train_cfg = cfg.get('train', {})
    ratio, seed = float(train_cfg.get('val_ratio', .1)), int(train_cfg.get('seed', 42))
    if not 0 < ratio < 1:
        raise ValueError('Recommendation val_ratio must be strictly between 0 and 1')
    frame = data.interaction_df.copy()
    if '__split__' not in frame:
        frame['__split__'] = 'train'
    pool = frame.loc[frame.__split__.eq('train')].reset_index(drop=True)
    other = frame.loc[~frame.__split__.eq('train')]
    target = cfg.get('feature', {}).get('target_col')
    columns = [data._user_id_col, data._item_id_col, cfg.get('feature', {}).get('time_col', 'timestamp')]
    required = getattr(data, 'SPLIT_PROTOCOL', None)
    version = train_cfg.get('split_protocol', required or 'rec_row_stratified_v1')
    if required and version != required:
        raise ValueError(f'{data.name} requires split protocol {required}')
    if version not in (GROUP_VERSION, 'rec_row_stratified_v1'):
        raise ValueError(f'Unsupported recommendation split protocol: {version}')
    labels = pool[target] if target in pool else pd.Series(np.zeros(len(pool)))
    if version == GROUP_VERSION:
        if not set(columns).issubset(pool) or pool[columns].isna().any().any():
            raise ValueError('Event-group split needs nonmissing original user/item/time')
        if not pd.to_numeric(labels, errors='coerce').isin([0, 1]).all():
            raise ValueError('Event-group v1 requires finite binary labels')
        # Factorization compares actual tuples; hash collisions never merge groups.
        identities = pool[columns].astype('string')
        codes, unique = pd.factorize(pd.MultiIndex.from_frame(identities), sort=True)
        group_digests = np.asarray([_json_sha(tuple(key)) for key in unique], dtype='S64')
        membership, group_counts, _ = _group_assignment(codes, labels.to_numpy(), ratio, seed, len(unique))
        tr, va = np.flatnonzero(membership == 0), np.flatnonzero(membership == 1)
        if np.intersect1d(codes[tr], codes[va]).size:
            raise ValueError('Event-group train/validation overlap')
        mixed = int((group_counts[:, :-1].astype(bool).sum(axis=1) > 1).sum())
    else:
        positions = np.arange(len(pool))
        strata = labels if target in pool and 1 < labels.nunique() <= 50 else None
        tr, va = train_test_split(positions, test_size=ratio, random_state=seed, stratify=strata)
        membership = np.zeros(len(pool), dtype=np.int8); membership[va] = 1
        codes = np.arange(len(pool), dtype=np.int64)
        group_digests = np.empty(0, dtype='S64'); mixed = None
    identity_cols = [c for c in columns + [target] if c in pool]
    input_sha = _sha(pd.util.hash_pandas_object(pool[identity_cols].astype('string'), index=False).to_numpy().tobytes())
    canonical_sha = _json_sha({'input': input_sha, 'protocol': version, 'ratio': ratio, 'seed': seed,
                               'membership': _sha(membership.tobytes())})
    audit = dict(split_protocol=version, validation_boundary_protocol=BOUNDARY_VERSION,
                 algorithm='seeded_cumulative_class_row_deficit_v1' if version == GROUP_VERSION else 'sklearn_stratified_rows',
                 seed=seed, requested_val_ratio=ratio, actual_val_ratio=len(va) / len(pool),
                 pool_rows=len(pool), train_rows=len(tr), val_rows=len(va),
                 group_columns=columns if version == GROUP_VERSION else [],
                 key_normalization='original frozen column string representation; no mapped IDs',
                 groups=len(np.unique(codes)), mixed_label_groups=mixed,
                 label_distributions={'pool': _distribution(labels), 'train': _distribution(labels.iloc[tr]), 'val': _distribution(labels.iloc[va])},
                 canonical_sha256=canonical_sha, input_identity_label_sha256=input_sha,
                 event_groups_sha256=_sha(codes.tobytes() + group_digests.tobytes()),
                 train_membership_sha256=_sha(np.sort(tr).astype('int64').tobytes()),
                 val_membership_sha256=_sha(np.sort(va).astype('int64').tobytes()),
                 row_weighting='all original rows retained, no reweighting; duplicate history multiplicity retained')
    directory = getattr(data, 'data_dir', None)
    if directory:
        path = Path(data._std_test_dir(directory)) / 'interaction_frozen.parquet'
        if path.is_file():
            audit['frozen_sha256'] = file_fingerprint(path)
    data._rec_partition_state = dict(audit=audit, membership=membership, group_codes=codes,
                                     group_digests=group_digests, train_positions=np.asarray(tr), val_positions=np.asarray(va))
    data._rec_active_positions = np.concatenate([tr, va, np.full(len(other), -1)])
    data.interaction_df = pd.concat([pool.iloc[tr].assign(__split__='train'), pool.iloc[va].assign(__split__='val'), other], ignore_index=True)
    data._validation_prepared = True
    data.partition_audit_ = copy_audit(audit)


def copy_audit(audit):
    return json.loads(json.dumps(audit))


def select_queries(data, train_limit=None, val_limit=None):
    """Recent queries inside already-fixed partitions; never resplit/swap them."""
    frame = data.interaction_df
    state = data._rec_partition_state
    selected = []
    columns = state['audit']['group_columns']
    time = (data.model_cfg or {}).get('feature', {}).get('time_col', 'timestamp')
    for part, limit in (('train', train_limit), ('val', val_limit), ('std_test', None)):
        positions = np.flatnonzero(frame.__split__.eq(part).to_numpy())
        if limit is not None and len(positions) > limit:
            positions = frame.iloc[positions].sort_values(time, kind='stable').tail(limit).index.to_numpy()
        selected.extend(positions)
    selected = np.asarray(selected, dtype=np.int64)
    active = data._rec_active_positions[selected]
    data.interaction_df = frame.iloc[selected].reset_index(drop=True)
    data._rec_active_positions = active
    tr = active[data.interaction_df.__split__.eq('train')]
    va = active[data.interaction_df.__split__.eq('val')]
    if not np.isin(tr, state['train_positions']).all() or not np.isin(va, state['val_positions']).all():
        raise ValueError('Search queries crossed the fixed validation boundary')
    state['query_train_positions'], state['query_val_positions'] = tr, va
    data.partition_audit_ = copy_audit(state['audit'])
    data.partition_audit_.update(query_train_rows=len(tr), query_val_rows=len(va),
        query_train_sha256=_sha(tr.astype('int64').tobytes()), query_val_sha256=_sha(va.astype('int64').tobytes()),
        train_query_limit=train_limit, val_query_limit=val_limit,
        validation_query_policy='stable recent rows within fixed Val, independent of candidate pipeline')


def public_search_view(data, small_n):
    prepare_validation(data)
    n = data._rec_partition_state['audit']['pool_rows']
    if not small_n or n <= small_n:
        select_queries(data)
    else:
        # Original common small_n bounded TOTAL search rows, not train alone.
        val = int(math.ceil(small_n * data.model_cfg.get('train', {}).get('val_ratio', .1)))
        select_queries(data, int(small_n) - val, val)


def persist_partition(data, directory=None):
    """Persist complete positional/group membership separately from model inputs."""
    state = getattr(data, '_rec_partition_state', None)
    if state is None:
        return None
    audit = copy_audit(getattr(data, 'partition_audit_', state['audit']))
    directory = Path(directory) if directory is not None else _artifact_root.get()
    if directory is None:
        return audit
    directory.mkdir(parents=True, exist_ok=True)
    view = _json_sha({k: v for k, v in audit.items() if k != 'label_distributions'})
    path = directory / (view + '.npz')
    if not path.exists():
        np.savez_compressed(path, membership=state['membership'], event_group=state['group_codes'],
                            event_group_sha256=state['group_digests'], train_positions=state['train_positions'],
                            val_positions=state['val_positions'],
                            query_train_positions=state.get('query_train_positions', state['train_positions']),
                            query_val_positions=state.get('query_val_positions', state['val_positions']))
    cfg = data.model_cfg or {}
    config_sha = _json_sha(cfg)
    from .rec_protocol import implementation_manifest
    artifact = dict(audit, positional_membership_file=path.name, membership_file_sha256=file_fingerprint(path),
                    effective_config=cfg, effective_config_sha256=config_sha,
                    implementation=implementation_manifest())
    metadata = directory / (view + '_' + config_sha + '.json')
    if not metadata.exists():
        metadata.write_text(json.dumps(artifact, indent=2, default=str), encoding='utf8')
    audit.update(artifact=str(metadata), membership_file_sha256=artifact['membership_file_sha256'])
    return audit
