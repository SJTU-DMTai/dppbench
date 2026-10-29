"""Search partitions from frozen training data, without opening std_test."""
from pathlib import Path
import random
import copy
import numpy as np
import pandas as pd
import yaml

from baselines.common.pipeline import make_step, PipelineStep
from baselines.common.data_ctx import infer_data_context


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


def select_join(ctx, seed, join_config=None):
    """Explicit adapter or the unchanged legacy, independent seeded draw."""
    if join_config is None:
        return make_step('JoinTable', ctx, random.Random(seed))
    if not isinstance(join_config, dict):
        raise ValueError('DiffPrep join override must be a mapping')
    allowed = {'table', 'key_col', 'method', 'prefix', 'max_cols'}
    if set(join_config) - allowed or not {'table', 'key_col', 'method'} <= set(join_config):
        raise ValueError('Join override requires table/key_col/method; optional prefix/max_cols')
    table, key, method = (join_config[k] for k in ('table', 'key_col', 'method'))
    if not isinstance(table, str) or not isinstance(key, str):
        raise ValueError('Join table/key_col must be strings')
    info = ctx.tables.get(table)
    if table not in ctx.secondary_tables or info is None:
        raise ValueError(f'Unknown auxiliary table: {table}')
    if not any(k['primary'] == key and k['secondary'] == key for k in info.join_keys_to_primary):
        raise ValueError(f'No declared direct join key {key!r} for {table!r}')
    if method not in ('key', 'agg'):
        raise ValueError('Join override method must be key or agg')
    if method == 'key' and not info.join_key_unique.get(key, False):
        raise ValueError(f'Join key {table}.{key} is not unique; use agg')
    limit = join_config.get('max_cols')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('Join max_cols must be a positive integer or null')
    if join_config.get('prefix') is not None and not isinstance(join_config['prefix'], str):
        raise ValueError('Join prefix must be a string or null')
    params = copy.deepcopy(join_config)
    params.pop('table')
    params.update(aux_df=f'${table}', how='left')
    return PipelineStep('JoinTable', params=params)


def assert_join_replay(structural_steps, pipeline):
    expected = [s.params for s in structural_steps if s.op == 'JoinTable']
    actual = [s.params for s in pipeline.steps if s.op == 'JoinTable']
    if actual != expected:
        raise ValueError('DiffPrep search/export JoinTable parameters differ')
    # Verify actual YAML source wiring as well as the in-memory parameter list.
    dag = yaml.safe_load(pipeline.to_yaml())['dag']
    sources = {s['id']: s['table'] for s in dag['sources']}
    joins = [s for s in dag['ops'] if s['op'] == 'JoinTable']
    if len(joins) != len(expected):
        raise ValueError('DiffPrep YAML JoinTable count differs')
    for params, node in zip(expected, joins):
        table = params['aux_df'][1:]
        if table not in [sources.get(p) for p in node['prev']]:
            raise ValueError('DiffPrep YAML JoinTable source differs from search')
        if node.get('params', {}) != {k: v for k, v in params.items() if k != 'aux_df'}:
            raise ValueError('DiffPrep YAML JoinTable parameters differ from search')


def prepare_tabular_search(data, config, ctx, seed, small_n=0, join_config=None):
    """Execute the fixed seeded join before learning, then replay it verbatim.

    Structural joins are not differentiable. Selecting one only at export
    trains architecture weights on a different schema from the deployed one.
    Selection here depends on schema/seed, never held-out labels.
    """
    splits = tabular_search_splits(data, config, seed, small_n)
    step = select_join(ctx, seed, join_config)
    data._diffprep_join_selection = {
        'mode': 'explicit' if join_config is not None else 'legacy_seeded',
        'seed': seed,
        'candidate_tables': [name for name in ctx.aux_dfs
                             if ctx.tables.get(name) and ctx.tables[name].join_keys_to_primary],
        'params': copy.deepcopy(step.params) if step is not None else None,
        'partitions': {},
    }
    structural_steps = []
    if step is not None:
        cls = data._load_op("JoinTable")
        op = cls(**data._resolve_param(step.params))
        for name in ("train", "val"):
            before = splits[name]
            after = op.transform(before)
            if len(after) != len(before):
                raise ValueError("DiffPrep structural join must preserve partition row counts")
            if not after.columns.is_unique:
                raise ValueError('DiffPrep structural join produced duplicate column names')
            data._diffprep_join_selection['partitions'][name] = {
                'rows_before': len(before), 'rows_after': len(after),
                'added_columns': [c for c in after if c not in before],
            }
            splits[name] = after.reset_index(drop=True)
        if list(splits['train'].columns) != list(splits['val'].columns):
            raise ValueError('DiffPrep joined train/validation schemas differ')
        structural_steps.append(step)
    # Profile training columns after the join, so joined numeric/categorical
    # attributes participate in learning and feature-wise projection.
    data.train_df = splits["train"]
    ctx = infer_data_context(ctx.data_name, data, task_type="tabular")
    return splits, ctx, structural_steps
