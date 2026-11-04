"""DPPBench adapter around the released README's hierarchical DAGPipe model."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import time
import uuid

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from baselines.common.budget import get_active_budget
from baselines.common.executor import TrainingExecutor
from .evaluator import training_splits
from .pipeline import DAGPipePipeline
from .tabular_adapter import tabular_adapter, augment_splits
from .runtime import make_model, check_resources, reference_forward
from .protocol_v4 import REFERENCE_ROWS, CANDIDATES, fit_input_encoder, validate_raw_features, allocate_layers, fit_tables, row_folds
from .task_adapter import rec_adapter, rec_splits, fit_rec_schema, graph_splits
from .performance import (clip_gradients, legacy_clip_gradients, TrainingData, ProfileWindow,
                          StepMeasurements, ResourceGuard)


class DAGPipe:
    def __init__(self, task_dir, data_name, data_dir=None, seed=42,
                 output_dir=None, verbose=True, device='cpu', eval_full=False,
                 max_pipeline_len=None, model_name=None, config_path=None):
        self.task_dir, self.data_name, self.data_dir = task_dir, data_name, data_dir
        self.seed, self.device, self.verbose = int(seed), device, verbose
        self.model_name = model_name
        self.output_dir = Path(output_dir or Path('outputs')/'DAGPipe'/data_name)
        self.cfg = yaml.safe_load(Path(config_path or Path(__file__).with_name('config.yaml')).read_text(encoding='utf-8'))
        if self.cfg.get('operator_space') != 'typed_public_v4':
            raise ValueError('New DAGPipe training requires typed_public_v4; v2/v3 are replay-only')
        if self.cfg.get('dtype', 'float64') != 'float64':
            raise ValueError('v4 requires float64 for numerical precision and stable feature-hash folds')
        if self.cfg.get('dtype', 'float64') not in ('float32', 'float64'):
            raise ValueError('DAGPipe dtype must be float32 or float64')
        self.dtype = getattr(torch, self.cfg.get('dtype', 'float64'))
        self.max_len = int(max_pipeline_len) if max_pipeline_len is not None else 10
        if not 2 <= self.max_len <= 10:
            raise ValueError('DAGPipe max_pipeline_len must be in [2, 10]')
        for key in ('epochs', 'batch_size', 'patience'):
            if int(self.cfg[key]) < (2 if key == 'batch_size' else 1):
                raise ValueError(f'Invalid DAGPipe config {key}={self.cfg[key]}')
        self.eval_full = eval_full  # Final evaluation is owned by the harness.
        if float(self.cfg.get('min_delta', 0)) < 0 or int(self.cfg.get('profile_steps', 0)) < 0:
            raise ValueError('min_delta and profile_steps must be nonnegative')
        self.legacy_sync = self.cfg.get('sync_mode', 'optimized') == 'legacy'
        if self.cfg.get('sync_mode', 'optimized') not in ('legacy', 'optimized'):
            raise ValueError('sync_mode must be legacy or optimized')
        self.diagnostic_steps = int(self.cfg.get('diagnostic_steps', 0))
        if self.diagnostic_steps < 0:
            raise ValueError('diagnostic_steps must be nonnegative')
        if self.diagnostic_steps:
            self.output_dir = self.output_dir/'diagnostics'/uuid.uuid4().hex

    def run(self):
        started = time.monotonic()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if str(self.device).startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('DAGPipe CUDA requested but unavailable')
            torch.cuda.set_device(self.device)
            torch.cuda.manual_seed_all(self.seed)
        budget = get_active_budget()
        check_resources()
        executor = TrainingExecutor(self.task_dir, self.data_name, self.data_dir,
                                    device=self.device, model_name=self.model_name, seed=self.seed)
        try:
            cfg = executor.get_model_config()
            data = executor.load_data()
            task_type = executor.task_type
            adapter = {'task_type': task_type}
            if task_type == 'rec':
                adapter = rec_adapter(data, cfg)
                splits = rec_splits(data, cfg, adapter=adapter)
                from dppbench.rec_partition import persist_partition
                adapter['partition'] = persist_partition(data, self.output_dir / 'partition_audits')
                target = adapter['target']
                columns = fit_rec_schema(adapter, splits['train'], data.col_types)
                exclude = {target, '__split__', adapter['time'], 'sample_weight'}
                adapter['ops'] = (['JoinTable'] if adapter['side_tables'] else []) + [
                    'CreateSequence' for _ in adapter['sequences']] + ['DAGPipeRecEncoding']
            elif task_type == 'graph':
                splits = graph_splits(data)
                target = data.target_col
                columns = [c for c in splits['train'] if c.startswith('feat_')]
                exclude = {target, data.id_col, 'time_step', '__split__', 'sample_weight'}
            else:
                splits = training_splits(data, cfg, include_std_test=False)
                adapter = tabular_adapter(self.data_name)
                adapter['task_type'] = task_type
                splits = augment_splits(data, splits, adapter)
                target = data.target_col
                exclude = {target, data.id_col, 'sample_weight', '__split__', getattr(data, '_sort_col', None)}
                columns = [c for c in splits['train'] if c not in exclude]
            train, val = splits['train'], splits['val']
            exclude.update(self.cfg.get('exclude_columns', {}).get(self.data_name, []))
            columns = [c for c in columns if c not in exclude]
            input_diagnostics = None
            if self.data_name == 'berka':
                input_diagnostics = {
                    'berka_history_version': adapter.get('berka_history_version', 0),
                    'feature_columns': columns,
                    'splits': {name: {
                        'rows': len(frame),
                        'positive_labels': int((frame[target] == 1).sum()),
                        'rows_with_preloan_history': int((frame['trans__count'] > 0).sum()),
                    } for name, frame in (('train', train), ('val', val))},
                }
                print('[DAGPipe inputs] ' + json.dumps(input_diagnostics, ensure_ascii=False))
            if len(train) < 2 or len(val) < 2:
                raise ValueError('DAGPipe needs >=2 train and >=2 validation rows')
            categorical_columns = {col for col, kind in (getattr(data, 'col_types', {}) or {}).items()
                                   if kind == 'categorical'}
            operator_space = self.cfg.get('operator_space', 'current')
            temporal = task_type in ('rec', 'graph') or bool(getattr(data, '_sort_col', None) or adapter.get('time') or
                            any(kind in ('timestamp', 'datetime') for kind in (getattr(data, 'col_types', {}) or {}).values()))
            encoder = fit_input_encoder(train[columns], categorical_columns, getattr(data, 'col_types', {}))
            validate_raw_features(train[columns], encoder.state)
            validate_raw_features(val[columns], encoder.state)
            fit_labels = train[target].copy()
            x_train = torch.from_numpy(encoder.transform(train[columns])).to(self.dtype)
            x_val = torch.from_numpy(encoder.transform(val[columns])).to(self.dtype)
            if budget:
                if budget.max_output_rows and len(train)+len(val) > budget.max_output_rows:
                    raise ValueError('DAGPipe exceeds max_output_rows')
                if budget.max_output_columns and len(columns) > budget.max_output_columns:
                    raise ValueError('DAGPipe exceeds max_output_columns')
            task = cfg.get('model_params', {}).get('task', 'binary')
            regression = task == 'regression'
            if task not in ('regression', 'binary', 'multiclass'):
                raise ValueError(f'DAGPipe unsupported prediction task {task!r}')
            if regression:
                yt = train[target].to_numpy(dtype=np.float32)
                yv = val[target].to_numpy(dtype=np.float32)
                y_mean, y_scale = float(yt.mean()), max(float(yt.std()), 1e-6)
                y_train = torch.tensor((yt-y_mean)/y_scale, dtype=self.dtype)
                y_val = torch.tensor((yv-y_mean)/y_scale, dtype=self.dtype)
                n_outputs = 1
            else:
                classes = sorted(train[target].unique().tolist())
                mapping = {v: i for i, v in enumerate(classes)}
                if len(classes) < 2 or not val[target].isin(classes).all():
                    raise ValueError('DAGPipe validation has unseen labels or training has <2 classes')
                y_train = torch.tensor(train[target].map(mapping).to_numpy(), dtype=torch.long)
                y_val = torch.tensor(val[target].map(mapping).to_numpy(), dtype=torch.long)
                n_outputs = len(classes)
                if not pd.api.types.is_numeric_dtype(fit_labels):
                    fit_labels = fit_labels.map(mapping)
            if not torch.isfinite(y_train).all() or not torch.isfinite(y_val).all():
                raise ValueError('DAGPipe encountered non-finite train/validation labels')
        finally:
            executor.cleanup()

        preparation_seconds = time.monotonic()-started

        fixed_steps = 1 + len(adapter.get('ops', []))
        limit = min(self.max_len, budget.max_pipeline_length or self.max_len) if budget else self.max_len
        if limit <= fixed_steps:
            raise ValueError(f'DAGPipe {task_type} needs max_pipeline_len >= {fixed_steps + 1} '
                             'to include task adapters and a learned operation')
        categorical = any(spec['kind'] == 'category' for spec in encoder.state)
        v4 = operator_space == 'typed_public_v4'
        allow_target = not temporal and (regression or n_outputs == 2)
        folds = int(self.cfg.get('target_encoding_folds', 5))
        if folds < 2:
            raise ValueError('Target encoding fold count must be >=2')
        if categorical and allow_target:
            allow_target = len(set(row_folds(x_train.numpy(), folds, self.seed))) >= 2
        ablate_second_robust = self.cfg.get('ablate_elliptic_second_robust_scaler', False)
        layers = allocate_layers(
            encoder.state, x_train.numpy(), limit-fixed_steps,
            allow_target=allow_target, allow_frequency=not temporal,
            task=self.data_name, ablate_elliptic_second_robust_scaler=ablate_second_robust,
        )
        names = [name for layer in layers for name in layer if CANDIDATES[name][0] == 'encode']
        from baselines.common.search_cost import measured_call
        category_state = measured_call('operator_fit.category_tables', lambda:
            fit_tables(train[columns], fit_labels, encoder.state, x_train.numpy(), names,
                       self.seed, folds))
        reference_rows = self.cfg.get('elliptic_reference_rows', REFERENCE_ROWS)
        if type(reference_rows) is not int or reference_rows < REFERENCE_ROWS:
            raise ValueError('Invalid Elliptic reference row count')
        if reference_rows != REFERENCE_ROWS and (
            self.data_name != 'elliptic_bitcoin' or reference_rows > len(x_train)
        ):
            raise ValueError('Custom reference row counts require enough Elliptic training rows')
        print(f'[DAGPipe] protocol={operator_space} nodes={layers} reference={min(reference_rows, len(x_train))} '
              f'steps={fixed_steps + sum(map(len, layers))}/{limit} adapters_and_encoding={fixed_steps}')
        if v4:
            print('[DAGPipe] unavailable=' + json.dumps({n: reason for n, (_, enabled, reason) in CANDIDATES.items() if not enabled}))
            if temporal:
                print('[DAGPipe] TargetEncoder and FrequencyEncoder disabled: event-time visibility is not established')
            elif categorical and not allow_target:
                print('[DAGPipe] TargetEncoder skipped: requires binary/regression target and at least two nonempty feature-hash folds')
        model = make_model(layers, len(columns), self.device, self.dtype, specs=encoder.state,
                           category_state=category_state, scale_protocol='public_v1',
                           operator_space=operator_space)
        predictor = nn.Sequential(nn.Linear(len(columns), int(self.cfg['hidden'])), nn.ReLU(),
                                  nn.Linear(int(self.cfg['hidden']), n_outputs)).to(device=self.device, dtype=self.dtype)
        optimizer = torch.optim.Adam(list(model.parameters())+list(predictor.parameters()),
                                     lr=float(self.cfg['learning_rate']))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        generator = torch.Generator().manual_seed(self.seed)
        # Fixed before optimization and reused by validation and deployment.
        reference_idx = torch.randperm(len(x_train), generator=generator)[:reference_rows]
        sampled = x_train[reference_idx]
        numeric = [i for i, spec in enumerate(encoder.state) if spec['kind'] != 'category']
        missing = [encoder.state[i]['column'] for i in numeric if sampled[:, i].isnan().all()]
        if missing:
            raise ValueError(f'DAGPipe sampled reference has no observed values for: {missing}')
        reference = sampled.to(self.device)
        training_data = TrainingData((x_train, y_train, x_val, y_val), self.device, self.cfg, budget,
                                     parameters=list(model.parameters())+list(predictor.parameters()))
        print('[DAGPipe memory plan] ' + json.dumps(training_data.report))
        epochs = int(self.cfg['epochs'])
        best_loss, best_state, stale, updates = float('inf'), None, 0, 0
        stopping_loss = float('inf')
        history = []

        from baselines.common.search_cost import measured_call, record_work

        def task_loss(z, y):
            return nn.functional.mse_loss(z.squeeze(-1), y) if regression else nn.functional.cross_entropy(z, y)

        batch_size = int(self.cfg['batch_size'])
        profiler = ProfileWindow(self.cfg.get('profile_steps', 0), self.device, self.output_dir)
        measurements = StepMeasurements(self.device, bool(self.diagnostic_steps or self.cfg.get('profile_steps', 0)))
        resources = ResourceGuard(check_resources, float(self.cfg.get('resource_check_interval_s', .1)))

        def diagnostic_result():
            # No validation sweep, checkpoint serialization, downstream fit or HPO.
            profiler.close()
            result = {'diagnostic_only': True, 'architecture_steps': updates,
                      'requested_steps': self.diagnostic_steps, 'memory_plan': training_data.report,
                      'preparation_wall_seconds': preparation_seconds,
                      'timing_kind': 'host wall / CUDA stream elapsed; not pure kernel time',
                      'measurements': measurements.summary(),
                      'profile_enabled': bool(self.cfg.get('profile_steps', 0))}
            result['total_wall_seconds'] = time.monotonic()-started
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir/'diagnostic.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            result['output_dir'] = str(self.output_dir)
            return result

        try:
            for epoch in range(epochs):
                if str(self.device).startswith('cuda'):
                    torch.cuda.synchronize(self.device)
                epoch_started = time.monotonic()
                resources(force=True)
                model.train(); predictor.train()
                perm = training_data.indices(torch.randperm(len(x_train), generator=generator))
                total_train = torch.zeros((), dtype=torch.float64, device=self.device)
                for idx in perm.split(batch_size):
                    measurements.enabled = bool(self.diagnostic_steps or profiler.profiler is not None)
                    with measurements.scope('resource_check'):
                        resources()
                    with measurements.scope('batch_transfer'):
                        x, y = training_data.batch(False, idx)
                    with measurements.scope('forward'):
                        loss = measured_call('proxy_train.joint', lambda:
                            task_loss(predictor(reference_forward(model, x, reference)), y))
                        record_work('proxy_work.train', supervised_rows=len(y),
                                    forwarded_rows=len(y) + len(reference))
                        sparse = torch.triu(torch.sigmoid(model.layer_adj_logits), diagonal=1).sum()
                        sparse = sparse + sum(torch.triu(torch.sigmoid(a), diagonal=1).sum() for a in model.intra_adj_logits)
                        loss = loss + float(self.cfg['lambda_sparse'])*sparse
                        if self.legacy_sync and not torch.isfinite(loss):
                            raise RuntimeError('DAGPipe non-finite loss')
                    with measurements.scope('backward'):
                        optimizer.zero_grad()
                        measured_call('gradient.backward', loss.backward)
                    with measurements.scope('gradient_norm_and_clip'):
                        if self.legacy_sync:
                            legacy_clip_gradients(model.parameters())
                            legacy_clip_gradients(predictor.parameters())
                        else:
                            model_finite = clip_gradients(model.parameters(), check=False, instrument=measurements.enabled)
                            predictor_finite = clip_gradients(predictor.parameters(), check=False, instrument=measurements.enabled)
                    with measurements.scope('finite_check'):
                        # Deliberately retain ONE host check before every update.
                        # Removing it would delay detection until after parameter mutation.
                        if not self.legacy_sync and not bool(torch.isfinite(loss.detach()) & model_finite & predictor_finite):
                            raise RuntimeError(f'DAGPipe non-finite training loss/gradient at epoch {epoch+1}')
                    with measurements.scope('optimizer'):
                        measured_call('optimizer.joint', optimizer.step)
                    updates += 1
                    total_train += (loss.item() if self.legacy_sync else loss.detach().double())*len(y)
                    profiler.step()
                    if self.diagnostic_steps and updates >= self.diagnostic_steps:
                        resources(force=True)
                        return diagnostic_result()
                train_loss = total_train.item()/len(x_train)
                if str(self.device).startswith('cuda'):
                    torch.cuda.synchronize(self.device)
                training_seconds = time.monotonic()-epoch_started
                # Diagnostic runs only measure bounded train steps, no full validation.
                if self.diagnostic_steps:
                    continue
                validation_started = time.monotonic()
                model.eval(); predictor.eval()
                total_val = torch.zeros((), dtype=torch.float64, device=self.device)
                val_finite = torch.ones((), dtype=torch.bool, device=self.device)
                with torch.no_grad():
                    for start in range(0, len(x_val), batch_size):
                        resources()
                        x, y = training_data.batch(True, slice(start, start+batch_size))
                        loss = measured_call('proxy_val.selection', lambda:
                            task_loss(predictor(reference_forward(model, x, reference)), y))
                        record_work('proxy_work.val', supervised_rows=len(y),
                                    forwarded_rows=len(y) + len(reference))
                        if self.legacy_sync and not torch.isfinite(loss):
                            raise RuntimeError('DAGPipe non-finite validation loss')
                        val_finite &= torch.isfinite(loss)
                        total_val += (loss.item() if self.legacy_sync else loss.double())*len(y)
                if not bool(val_finite):
                    raise RuntimeError(f'DAGPipe non-finite validation loss at epoch {epoch+1}')
                value = total_val.item()/len(x_val)
                if str(self.device).startswith('cuda'):
                    torch.cuda.synchronize(self.device)
                validation_seconds = time.monotonic()-validation_started
                resources(force=True)
                scheduler.step(value)
                history.append({'epoch': epoch+1, 'train_loss': train_loss, 'validation_task_loss': value,
                                'training_wall_seconds': training_seconds, 'validation_wall_seconds': validation_seconds,
                                'profile_enabled': bool(self.cfg.get('profile_steps', 0))})
                if value < best_loss:
                    best_loss, best_state = value, copy.deepcopy(model.state_dict())
                    best_epoch = epoch+1
                if value < stopping_loss-float(self.cfg.get('min_delta', 0)):
                    stopping_loss, stale = value, 0
                else:
                    stale += 1
                if self.verbose:
                    print(f'[DAGPipe] epoch={epoch+1}/{epochs} val_loss={value:.12g} '
                          f'train_wall_s={training_seconds:.3f} val_wall_s={validation_seconds:.3f}')
                self.output_dir.mkdir(parents=True, exist_ok=True)
                (self.output_dir/'training_progress.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
                if stale >= int(self.cfg['patience']):
                    break
            if self.diagnostic_steps:
                return diagnostic_result()
        finally:
            profiler.close()
        if best_state is None:
            raise RuntimeError('DAGPipe produced no valid checkpoint')
        frozen_nodes = None
        freeze_verification = None
        if v4:
            from .protocol_v4 import freeze_statistics
            model.load_state_dict(best_state, strict=True)
            probe = x_val[:int(self.cfg['batch_size'])].to(self.device)
            with torch.no_grad():
                before_freeze = measured_call('operator_check.before_freeze', lambda: reference_forward(model, probe, reference))
            frozen_nodes = measured_call('operator_fit.final_statistics', lambda: freeze_statistics(model, reference))
            record_work('operator_work.final_statistics', forwarded_rows=len(reference))
            with torch.no_grad():
                after_freeze = measured_call('operator_check.after_freeze', lambda: reference_forward(model, probe, reference))
            torch.testing.assert_close(before_freeze, after_freeze, rtol=1e-10, atol=1e-12)
            freeze_verification = {'rows': len(probe), 'max_abs_diff': float((before_freeze-after_freeze).abs().max()),
                                   'rtol': 1e-10, 'atol': 1e-12}
            record_work('operator_work.freeze_verification', forwarded_rows=len(reference)+2*len(probe))
        payload = {
            'format': 'dppbench.DAGPipe.reference.v4', 'task': self.data_name,
            'seed': self.seed, 'variant': 'typed_crossfit_full_train_v4',
            'adapter': adapter,
            'dtype': self.cfg.get('dtype', 'float64'),
            'excluded_columns': sorted(c for c in exclude if c),
            'encoder': encoder.state, 'layers': layers,
            'category_state': category_state, 'scale_protocol': 'public_v1',
            'operator_space': operator_space,
            'fit_protocol': 'seeded_training_sample_16',
            'training_rows': len(x_train),
            'max_pipeline_len': limit,
            'allow_target_encoding': allow_target,
            'allow_frequency_encoding': not temporal,
            'frozen_nodes': frozen_nodes,
            'state_dict': {k: v.detach().cpu().tolist() for k,v in best_state.items()},
            'reference': sampled.tolist(),
            'cache_entries': int(self.cfg['transform_cache_entries']),
            'cache_max_mb': float(self.cfg.get('transform_cache_max_mb', 512)),
            'transform_batch_size': int(self.cfg.get('transform_batch_size', 1024)),
            'selection': {'epoch': best_epoch, 'validation_loss': best_loss},
        }
        if ablate_second_robust:
            payload['ablate_elliptic_second_robust_scaler'] = True
        if reference_rows != REFERENCE_ROWS:
            payload.update(fit_protocol='seeded_elliptic_training_sample', reference_rows=reference_rows)
        pipeline = DAGPipePipeline(payload)
        self.best_pipeline = pipeline
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pipeline.save(self.output_dir/'best_pipeline.yaml')
        log = {'variant': payload['variant'], 'config': self.cfg, 'seed': self.seed,
               'task_type': task_type,
               'surrogate': 'node_attribute_mlp' if task_type == 'graph' else 'feature_mlp',
               'n_train': len(x_train), 'n_val': len(x_val), 'n_features': len(columns),
               'epochs_completed': len(history), 'architecture_steps': updates,
               'n_unique_evaluations': 0, 'best_epoch': best_epoch,
               'n_steps': len(pipeline), 'step_semantics': 'weighted_dag_nodes_including_adapters',
               'evaluation_semantics': 'downstream_pipeline_evaluations_excluding_final_hpo',
               'ops': pipeline.op_names(), 'history': history,
               'construct_time_s': time.monotonic()-started,
               'preparation_seconds': preparation_seconds, 'memory_plan': training_data.report,
               'profile_steps': int(self.cfg.get('profile_steps', 0)),
               'step_measurements': measurements.summary(),
               'timing_kind': 'synchronized phase wall time; includes Python/resource/statistics/profiler overhead',
               'search_metric': 'negative_normalized_mse' if regression else 'negative_cross_entropy',
               'budget': budget.snapshot() if budget else None}
        if v4:
            from .protocol_v4 import numerical_report
            log['numerical_nodes'] = numerical_report(model)
            log['candidate_registry'] = CANDIDATES
            log['reference_protocol'] = 'fixed seeded sample of up to 16 legal training rows; node statistics frozen at selected weights for query-only inference'
            log['freeze_verification'] = freeze_verification
            log['environment'] = {'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
                                  'device': str(self.device), 'cuda': torch.version.cuda}
        if input_diagnostics is not None:
            log['input_diagnostics'] = input_diagnostics
        (self.output_dir/'DAGPipe_search.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
        check_resources()
        return {'best_pipeline_yaml': pipeline.to_yaml(), 'best_fitness': -best_loss,
                'n_unique_evaluations': 0, 'is_legal': True, 'eval_error': None}

    def get_pipeline(self):
        if not hasattr(self, 'best_pipeline'):
            raise RuntimeError('Call DAGPipe.run() before get_pipeline()')
        return self.best_pipeline
