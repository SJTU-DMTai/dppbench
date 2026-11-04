"""Compare Berka inputs on TRAIN/VALIDATION only; never evaluate std_test.

Uses one fixed LightGBM configuration, not HPO or test-driven selection.
"""
from pathlib import Path
import argparse
import json
import sys
import copy
import hashlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from baselines.common.executor import TrainingExecutor
from baselines.DAGPipe.evaluator import training_splits
from baselines.DAGPipe.runtime import InputEncoder
from baselines.DAGPipe.tabular_adapter import tabular_adapter, augment_splits
from baselines.DAGPipe.pipeline import DAGPipePipeline
from dppbench.models.lightgbm_model import LightGBMModel
from dppbench.operators.integration.join_table import JoinTable


def shared_transaction_splits(data, splits):
    """Replay the shared TRANS join used by Random in the seed-43 log."""
    join = JoinTable(
        aux_df=data.auxiliary_dfs['trans'], method='agg', key_col='account_id',
        how='left', prefix='TRANS', max_cols=20,
    )
    return {name: None if frame is None else join.transform(frame)
            for name, frame in splits.items()}


def representation_diagnostics(before, after):
    """Check feature-order preservation without consulting any labels."""
    report = {}
    for col in before:
        x, y = before[col].to_numpy(float), after[col].to_numpy(float)
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            report[col] = {'finite': False}
            continue
        order = np.argsort(x, kind='stable')
        distinct = np.diff(x[order]) > 0
        dy = np.diff(y[order])
        tolerance = 1e-10 * max(1., float(np.max(np.abs(y))))
        report[col] = {
            'finite': True,
            'input_unique': int(len(np.unique(x))),
            'output_unique': int(len(np.unique(y))),
            'order_reversals': int(np.sum(dy[distinct] < -tolerance)),
            'output_abs_max': float(np.max(np.abs(y))),
        }
    return report


def diagnose(data, config, seed=42, pipeline=None):
    config = copy.deepcopy(config)
    config.setdefault('train', {})['seed'] = seed
    splits = training_splits(data, config, include_std_test=False)
    target = data.target_col
    excluded = {target, data.id_col, 'sample_weight', '__split__', getattr(data, '_sort_col', None)}
    account = tabular_adapter('berka')
    account.pop('berka_history_version')
    account['ops'] = ['JoinTable']
    variants = {
        'loan_only': splits,
        'loan_account': augment_splits(data, splits, account),
        'loan_account_preloan_history': augment_splits(data, splits, tabular_adapter('berka')),
        'loan_shared_trans_join': shared_transaction_splits(data, splits),
    }
    report = {'selection_split': 'validation_only', 'seed': seed, 'results': {},
              'protocol': 'one_fixed_LightGBM_config_no_HPO',
              'model_params': config.get('model_params', {})}

    def evaluate(name, train_x, val_x, representation):
        params = dict(config.get('model_params', {}))
        params.pop('seed', None)
        model = LightGBMModel(seed=seed, **params)
        model.fit(train_x, splits['train'][target].to_numpy(),
                  X_val=val_x, y_val=splits['val'][target].to_numpy())
        metrics = model.evaluate(val_x, splits['val'][target].to_numpy(), metrics=['auc'])
        report['results'][name] = {'validation_auc': float(metrics['auc']),
                                    'representation': representation,
                                    'columns': list(train_x), 'n_features': len(train_x.columns),
                                    'best_iteration': model.model.best_iteration_,
                                    'train_profile': feature_profile(train_x),
                                    'val_profile': feature_profile(val_x)}

    for name, frames in variants.items():
        columns = [c for c in frames['train'] if c not in excluded]
        raw_train, raw_val = frames['train'][columns].copy(), frames['val'][columns].copy()
        for col in columns:
            if not (pd.api.types.is_numeric_dtype(raw_train[col]) or
                    pd.api.types.is_datetime64_any_dtype(raw_train[col])):
                categories = pd.Index(raw_train[col].dropna().unique())
                raw_train[col] = pd.Categorical(raw_train[col], categories=categories)
                raw_val[col] = pd.Categorical(raw_val[col], categories=categories)
        evaluate(name + '_native', raw_train, raw_val,
                 'native_numeric_missing_values_and_train_fitted_categories')
        encoder = InputEncoder().fit(frames['train'][columns])
        evaluate(name, pd.DataFrame(encoder.transform(frames['train'][columns]), columns=columns),
                 pd.DataFrame(encoder.transform(frames['val'][columns]), columns=columns),
                 'DAGPipe_InputEncoder_imputed_ordinal_standardized')
    if pipeline is not None:
        if pipeline.payload['task'] != 'berka' or pipeline.payload['seed'] != seed:
            raise ValueError('Diagnostic checkpoint must match Berka and the split seed')
        frames = augment_splits(data, splits, pipeline.payload.get('adapter', {}))
        columns = [s['column'] for s in pipeline.payload['encoder']]
        encoder = InputEncoder(pipeline.payload['encoder'])
        for name, transform in (('checkpoint_input_only', encoder.transform),
                                ('checkpoint_learned', pipeline.transform)):
            evaluate(name, pd.DataFrame(transform(frames['train'][columns]), columns=columns),
                     pd.DataFrame(transform(frames['val'][columns]), columns=columns), name)
        report['checkpoint_transform_diagnostics'] = {
            name: representation_diagnostics(
                pd.DataFrame(encoder.transform(frames[name][columns]), columns=columns),
                pd.DataFrame(pipeline.transform(frames[name][columns]), columns=columns),
            ) for name in ('train', 'val')
        }
    report['split_rows'] = {name: len(splits[name]) for name in ('train', 'val')}
    report['split_labels'] = {name: {str(k): int(v) for k, v in splits[name][target].value_counts().items()}
                              for name in ('train', 'val')}
    report['split_fingerprints'] = {name: hashlib.sha256(pd.util.hash_pandas_object(
        splits[name][[c for c in (data.id_col, target) if c in splits[name]]], index=False
    ).values.tobytes()).hexdigest() for name in ('train', 'val')}
    return report


def feature_profile(frame):
    numeric = frame.select_dtypes(include='number').to_numpy(dtype=float)
    return {'missing_cells': int(frame.isna().sum().sum()),
            'infinite_numeric_cells': int(np.isinf(numeric).sum()),
            'constant_columns': [str(c) for c in frame if frame[c].nunique(dropna=False) <= 1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pipeline', help='Optional v2 checkpoint for a same-input ablation')
    parser.add_argument('--output', default='berka_input_diagnostics.json')
    args = parser.parse_args()
    executor = TrainingExecutor(str(ROOT / 'dppbench/tasks/berka'), 'berka', args.data_dir,
                                device='cpu', model_name='LightGBM', seed=args.seed)
    try:
        pipeline = DAGPipePipeline.from_yaml(Path(args.pipeline).read_text(encoding='utf-8')) if args.pipeline else None
        report = diagnose(executor.load_data(), executor.get_model_config(), args.seed, pipeline)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        executor.cleanup()


if __name__ == '__main__':
    main()
