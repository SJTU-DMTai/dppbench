"""Search partitions from frozen training data, without opening std_test."""
from pathlib import Path
import numpy as np
import pandas as pd


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
    # Keep the canonical validation partition. small_n limits training only.
    if small_n and len(splits['train']) > small_n:
        splits['train'] = splits['train'].sample(n=small_n, random_state=seed).reset_index(drop=True)
    return splits
