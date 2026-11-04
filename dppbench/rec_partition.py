"""Fixed recommendation partitions and budgeted query views, before feature fitting.

No rows are deduplicated/reweighted. Bookkeeping lives on the data object, never
in prediction columns. Amazon groups are conservative original user/item/time.
"""
import contextlib
import contextvars
from collections import OrderedDict
import hashlib
import json
import math
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_VERSION = 'rec_event_group_split_v1'
BOUNDARY_VERSION = 'rec_fixed_validation_v1'
_artifact_root = contextvars.ContextVar('rec_partition_artifact_root', default=None)
_group_cache = OrderedDict()
_group_cache_lock = threading.RLock()


def _cache_groups(key, arrays):
    """Bounded, within-process memoization of data-only split assignments."""
    frozen = tuple(a.copy() for a in arrays)
    for a in frozen:
        a.setflags(write=False)
    if sum(a.nbytes for a in frozen) > 64 * 1024**2:
        return
    with _group_cache_lock:
        _group_cache[key] = frozen
        _group_cache.move_to_end(key)
        while len(_group_cache) > 4 or sum(a.nbytes for entry in _group_cache.values() for a in entry) > 64 * 1024**2:
            _group_cache.popitem(last=False)


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
    counts after each shuffled group, followed by fixed correction sweeps.
    Tiny cases enumerate all nonempty assignments in seeded order. No metric
    selection, retries, majority-label rewrite or row splitting.
    """
    if groups < 2:
        raise ValueError('Event-group validation needs at least two distinct groups')
    classes, y = np.unique(labels, return_inverse=True)
    counts = np.zeros((groups, len(classes) + 1), dtype=np.int64)
    np.add.at(counts, (codes, y), 1)
    counts[:, -1] = counts[:, :-1].sum(axis=1)
    totals = counts.sum(axis=0)
    scale = np.maximum(totals * ratio, 1.)
    possible = (counts[:, :-1] > 0).sum(axis=0) >= 2
    def score(val):
        # Class presence, when attainable, precedes approximate proportion.
        missing = int(((val[:-1] == 0) | (val[:-1] == totals[:-1]))[possible].sum())
        return missing, float(np.square((val - totals * ratio) / scale).sum())
    order = np.random.RandomState(seed).permutation(groups)
    if groups <= 12:
        # Tiny/dominant-group cases need a whole-group optimum, not a forced
        # last giant group. At most 4094 fixed assignments; no metric used.
        best, best_mask = None, None
        for mask in range(1, (1 << groups) - 1):
            selected = np.asarray([(mask >> i) & 1 for i in range(groups)], dtype=bool)
            value = score(counts[order[selected]].sum(axis=0))
            if best is None or value < best:
                best, best_mask = value, selected
        assignment = np.zeros(groups, dtype=np.int8)
        assignment[order[best_mask]] = 1
        return assignment[codes], counts, classes
    cumulative = np.zeros(counts.shape[1]); chosen = cumulative.copy()
    assignment = np.zeros(groups, dtype=np.int8)
    for step, g in enumerate(order):
        row = counts[g]; cumulative += row
        goal = ratio * cumulative
        stay = np.square((chosen - goal) / scale).sum()
        take = np.square((chosen + row - goal) / scale).sum()
        use_val = take < stay
        if use_val:
            assignment[g] = 1; chosen += row
    if not assignment.any():
        g = min(order, key=lambda i: score(counts[i]))
        assignment[g] = 1; chosen = counts[g].copy()
    elif assignment.all():
        g = min(order, key=lambda i: score(totals - counts[i]))
        assignment[g] = 0; chosen = totals - counts[g]
    # Two deterministic local correction sweeps, preserving nonempty groups.
    # This handles sparse minority groups without retries or splitting them.
    val_groups = int(assignment.sum())
    for _ in range(2):
        improved = False
        current = score(chosen)
        for g in order:
            direction = -1 if assignment[g] else 1
            next_groups = val_groups + direction
            if next_groups == 0 or next_groups == groups:
                continue
            candidate = chosen + direction * counts[g]
            candidate_score = score(candidate)
            if candidate_score < current:
                assignment[g] = 1 - assignment[g]
                chosen, current, val_groups = candidate, candidate_score, next_groups
                improved = True
        if not improved:
            break
    if score(chosen)[0]:
        raise ValueError('Group constraints prevent this allocator from preserving attainable class coverage; inspect source groups, do not silently resplit')
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
    identity_cols = [c for c in columns + [target] if c in pool]
    input_sha = _sha(pd.util.hash_pandas_object(pool[identity_cols].astype('string'), index=False).to_numpy().tobytes())
    if version == GROUP_VERSION:
        if target not in pool:
            raise ValueError('Event-group split requires an explicitly configured observed target')
        if not set(columns).issubset(pool) or pool[columns].isna().any().any():
            raise ValueError('Event-group split needs nonmissing original user/item/time')
        if not pd.to_numeric(labels, errors='coerce').isin([0, 1]).all():
            raise ValueError('Event-group v1 requires finite binary labels')
        key = (version, seed, ratio, tuple(identity_cols), input_sha)
        with _group_cache_lock:
            cached = _group_cache.get(key)
            if cached is not None:
                _group_cache.move_to_end(key)
        if cached is None:
            # Factorization compares actual tuples; hashes never merge groups.
            identities = pool[columns].astype('string')
            codes, unique = pd.factorize(pd.MultiIndex.from_frame(identities), sort=True)
            group_digests = np.asarray([_json_sha(tuple(group)) for group in unique], dtype='S64')
            membership, group_counts, _ = _group_assignment(codes, labels.to_numpy(), ratio, seed, len(unique))
            _cache_groups(key, (codes, group_digests, membership, group_counts))
        else:
            codes, group_digests, membership, group_counts = (a.copy() for a in cached)
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
    canonical_sha = _json_sha({'input': input_sha, 'protocol': version, 'ratio': ratio, 'seed': seed,
                               'membership': _sha(membership.tobytes())})
    audit = dict(split_protocol=version, validation_boundary_protocol=BOUNDARY_VERSION,
                 algorithm='seeded_class_row_deficit_tiny_exact_two_sweeps_v1' if version == GROUP_VERSION else 'sklearn_stratified_rows',
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
                 history_reference_protocol='full_fixed_training_observations_v1',
                 row_weighting='all original rows retained, no reweighting; duplicate history multiplicity retained')
    if version == GROUP_VERSION:
        audit.update(max_group_rows=int(group_counts[:, -1].max()),
                     groups_per_label={str(c): int((group_counts[:, i] > 0).sum()) for i, c in enumerate(np.unique(labels))},
                     labels_not_splittable_across_partitions=[str(c) for i, c in enumerate(np.unique(labels)) if (group_counts[:, i] > 0).sum() < 2],
                     row_ratio_absolute_error=abs(len(va) / len(pool) - ratio))
    directory = getattr(data, 'data_dir', None)
    if directory:
        std_dir = Path(data._std_test_dir(directory))
        sources = {name: std_dir / name for name in ('interaction_frozen.parquet','std_test.parquet','meta.json')}
        if version == GROUP_VERSION:
            sources.update({name: Path(directory) / name for name in ('All_Beauty.json.gz','meta_All_Beauty.json.gz')})
        # Opaque file hashes are provenance only, never features or fit inputs.
        audit['data_files'] = {name: file_fingerprint(path) for name, path in sources.items() if path.is_file()}
        audit['unavailable_data_fingerprints'] = [name for name, path in sources.items() if not path.is_file()]
        if 'interaction_frozen.parquet' in audit['data_files']:
            audit['frozen_sha256'] = audit['data_files']['interaction_frozen.parquet']
    if version == GROUP_VERSION and 'auc' in train_cfg.get('metrics', []) and any(labels.iloc[ix].nunique() < 2 for ix in (tr, va)):
        data.partition_audit_ = dict(audit, status='infeasible_class_coverage_for_auc')
        raise ValueError('Event-group split cannot provide both classes to train/val required by AUC; inspect whole-group constraints, do not split copies or silently resample')
    data._rec_partition_state = dict(audit=audit, membership=membership, group_codes=codes,
                                     group_digests=group_digests, train_positions=np.asarray(tr), val_positions=np.asarray(va))
    data._rec_active_positions = np.concatenate([tr, va, np.full(len(other), -1)])
    # History is a separate observation pool in every phase, including final.
    # Query sampling/cleaning does not turn Val/test into training observations.
    if set(columns).issubset(pool):
        data._rec_history_reference = pool.iloc[tr][columns].assign(__split__='train').copy()
        audit['history_reference_sha256'] = _sha(pd.util.hash_pandas_object(
            data._rec_history_reference[columns].astype('string'), index=False).to_numpy().tobytes())
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
    time = (data.model_cfg or {}).get('feature', {}).get('time_col', 'timestamp')
    target = (data.model_cfg or {}).get('feature', {}).get('target_col')
    val_policy = 'complete fixed Val'
    parts = [('train', train_limit), ('val', val_limit)]
    parts.extend((name, None) for name in pd.unique(frame.__split__) if name not in ('train', 'val'))
    for part, limit in parts:
        positions = np.flatnonzero(frame.__split__.eq(part).to_numpy())
        if limit is not None and len(positions) > limit:
            if limit < 1:
                raise ValueError('Search query limits must be positive')
            if part == 'val':
                from sklearn.model_selection import train_test_split
                labels = frame.iloc[positions][target] if target in frame else None
                classes = labels.nunique() if labels is not None else 0
                can_stratify = (classes > 1 and labels.value_counts().min() >= 2
                                and limit >= classes and len(positions) - limit >= classes)
                _, positions = train_test_split(positions, test_size=int(limit),
                    random_state=state['audit']['seed'], stratify=labels if can_stratify else None)
                val_policy = 'seeded stratified rows within fixed Val' if can_stratify else 'seeded rows within fixed Val; stratification infeasible at unchanged budget'
            else:
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
        validation_query_policy=val_policy,
        query_label_distributions={part: _distribution(data.interaction_df.loc[data.interaction_df.__split__.eq(part), target])
                                   for part in ('train', 'val')} if target in frame else {})


def public_search_view(data, small_n):
    prepare_validation(data)
    n = data._rec_partition_state['audit']['pool_rows']
    if not small_n or n <= small_n:
        select_queries(data)
    else:
        # Original common small_n bounded TOTAL search rows, not train alone.
        val = int(math.ceil(small_n * data.model_cfg.get('train', {}).get('val_ratio', .1)))
        select_queries(data, int(small_n) - val, val)


def persist_partition(data, directory=None, effective_config=None):
    """Persist complete positional/group membership separately from model inputs."""
    state = getattr(data, '_rec_partition_state', None)
    if state is None:
        return None
    audit = copy_audit(getattr(data, 'partition_audit_', state['audit']))
    cfg = (data.model_cfg or {}) if effective_config is None else effective_config
    config_sha = _json_sha(cfg)
    audit['effective_config_sha256'] = config_sha
    directory = Path(directory) if directory is not None else _artifact_root.get()
    if directory is None:
        return audit
    directory.mkdir(parents=True, exist_ok=True)
    view = _json_sha({k: v for k, v in audit.items() if k not in ('label_distributions', 'effective_config_sha256')})
    path = directory / (view + '.npz')
    if not path.exists():
        np.savez_compressed(path, membership=state['membership'], event_group=state['group_codes'],
                            event_group_sha256=state['group_digests'], train_positions=state['train_positions'],
                            val_positions=state['val_positions'],
                            query_train_positions=state.get('query_train_positions', state['train_positions']),
                            query_val_positions=state.get('query_val_positions', state['val_positions']))
    from .rec_protocol import implementation_manifest
    artifact = dict(audit, positional_membership_file=path.name, membership_file_sha256=file_fingerprint(path),
                    effective_config=cfg, effective_config_sha256=config_sha,
                    implementation=implementation_manifest())
    metadata = directory / (view + '_' + config_sha + '.json')
    if not metadata.exists():
        metadata.write_text(json.dumps(artifact, indent=2, default=str), encoding='utf8')
    audit.update(artifact=str(metadata), membership_file_sha256=artifact['membership_file_sha256'])
    return audit
