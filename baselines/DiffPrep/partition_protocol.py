"""Current frozen train/validation boundaries, independent of legacy search ops."""
from pathlib import Path
import copy
import hashlib
import numpy as np
import pandas as pd
from dppbench.rec_protocol import assert_disjoint, training_rows, VERSION

def tabular_search_splits(data, config, seed, small_n=0):
    path = Path(data._std_test_dir(data.data_dir)) / 'train_frozen.parquet'
    if not path.is_file():
        raise FileNotFoundError(f'DiffPrep requires frozen training data: {path}')
    frame = data._normalize_frozen_frame(pd.read_parquet(path))
    if '__split__' in frame and not frame['__split__'].eq('train').all():
        raise ValueError('DiffPrep training file contains non-training split markers')
    labels = pd.to_numeric(frame[data.target_col], errors='coerce')
    data.train_df = frame.loc[labels.notna() & np.isfinite(labels)].reset_index(drop=True)
    splits = data.split(val_ratio=config.get('train', {}).get('val_ratio', .2), seed=seed)
    from dppbench.tabular_contracts import fixed_search_view
    fixed_search_view(data, splits, 0, seed)  # fingerprint the unchanged canonical boundary
    # Keep the canonical validation partition. small_n limits training only.
    if small_n and len(splits['train']) > small_n:
        splits['train'] = splits['train'].sample(n=small_n, random_state=seed).reset_index(drop=True)
    from dppbench.tabular_contracts import prepare_references
    prepare_references(data, splits['train'])
    return splits


def recommendation_search_partition(data, config, seed, small_n):
    """Audit projected test identities, never test labels, before optimization."""
    directory = Path(data._std_test_dir(data.data_dir))
    train_path, test_path = directory/'interaction_frozen.parquet', directory/'std_test.parquet'
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError('DiffPrep rec requires frozen training and test identity artifacts')
    frame = pd.read_parquet(train_path)
    if '__split__' in frame and not frame['__split__'].eq('train').all():
        raise ValueError('Frozen training file contains held-out partitions')
    time = config.get('feature', {}).get('time_col', 'timestamp')
    identity_cols = [data._user_id_col, data._item_id_col, time]
    identities = pd.read_parquet(test_path, columns=identity_cols)
    audit = assert_disjoint(frame, identities, *identity_cols)
    data.model_cfg = copy.deepcopy(config)
    data.model_cfg.setdefault('train', {})['seed'] = seed
    data.interaction_df = frame.assign(__split__='train')
    data._validation_prepared = False
    data._prepare_validation()
    train = training_rows(data.interaction_df)
    val = data.interaction_df.loc[data.interaction_df['__split__'].eq('val')]
    # Context is the full legal training partition, NOT the optimizer query sample.
    data._rec_history_reference = train[identity_cols].assign(__split__='train').copy()
    from dppbench.rec_partition import select_queries, persist_partition
    select_queries(data, train_limit=small_n or None)  # DiffPrep small_n caps TRAIN only.
    train = training_rows(data.interaction_df)
    partition = persist_partition(data)
    return audit
