"""DPPBench adapter around the released README's hierarchical DAGPipe model."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import yaml

from baselines.common.budget import get_active_budget
from baselines.common.executor import TrainingExecutor
from .evaluator import training_splits
from .pipeline import DAGPipePipeline
from .tabular_adapter import tabular_adapter, augment_splits
from .runtime import InputEncoder, limited_layers, make_model, check_resources, reference_forward
from .task_adapter import rec_adapter, rec_splits, fit_rec_schema, graph_splits


def clip_gradients(parameters):
    """Same L2 clip at 1, with float64 norm to avoid float32 square overflow."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if any(not torch.isfinite(g).all() for g in grads):
        raise RuntimeError('DAGPipe encountered non-finite gradients')
    norm = torch.stack([torch.linalg.vector_norm(g.double()) for g in grads]).norm()
    if not torch.isfinite(norm):
        raise RuntimeError('DAGPipe encountered non-finite gradient norm')
    factor = min(1.0, 1.0/(norm.item()+1e-6))
    for grad in grads:
        grad.mul_(factor)


class DAGPipe:
    def __init__(self, task_dir, data_name, data_dir=None, seed=42,
                 output_dir=None, verbose=True, device='cpu', eval_full=False,
                 max_pipeline_len=None, model_name=None, config_path=None):
        self.task_dir, self.data_name, self.data_dir = task_dir, data_name, data_dir
        self.seed, self.device, self.verbose = int(seed), device, verbose
        self.model_name = model_name
        self.output_dir = Path(output_dir or Path('outputs')/'DAGPipe'/data_name)
        self.cfg = yaml.safe_load(Path(config_path or Path(__file__).with_name('config.yaml')).read_text(encoding='utf-8'))
        if self.cfg.get('dtype', 'float64') not in ('float32', 'float64'):
            raise ValueError('DAGPipe dtype must be float32 or float64')
        self.dtype = getattr(torch, self.cfg.get('dtype', 'float64'))
        self.max_len = int(max_pipeline_len) if max_pipeline_len is not None else 18
        if self.max_len < 2:
            raise ValueError('DAGPipe max_pipeline_len must be >=2: input encoding plus operation nodes')
        for key in ('epochs', 'batch_size', 'reference_rows', 'patience'):
            if int(self.cfg[key]) < (2 if key in ('batch_size', 'reference_rows') else 1):
                raise ValueError(f'Invalid DAGPipe config {key}={self.cfg[key]}')
        self.eval_full = eval_full  # Final evaluation is owned by the harness.

    def run(self):
        started = time.monotonic()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if str(self.device).startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError('DAGPipe CUDA requested but unavailable')
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
            encoder = InputEncoder().fit(train[columns])
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
            if not torch.isfinite(y_train).all() or not torch.isfinite(y_val).all():
                raise ValueError('DAGPipe encountered non-finite train/validation labels')
        finally:
            executor.cleanup()

        fixed_steps = 1 + len(adapter.get('ops', []))
        if self.max_len <= fixed_steps:
            raise ValueError(f'DAGPipe {task_type} needs max_pipeline_len >= {fixed_steps + 1} '
                             'to include task adapters and a learned operation')
        layers = limited_layers(self.max_len-fixed_steps)
        model = make_model(layers, len(columns), self.device, self.dtype)
        predictor = nn.Sequential(nn.Linear(len(columns), int(self.cfg['hidden'])), nn.ReLU(),
                                  nn.Linear(int(self.cfg['hidden']), n_outputs)).to(device=self.device, dtype=self.dtype)
        optimizer = torch.optim.Adam(list(model.parameters())+list(predictor.parameters()),
                                     lr=float(self.cfg['learning_rate']))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        generator = torch.Generator().manual_seed(self.seed)
        # Fixed before optimization and reused by validation and deployment.
        reference_idx = torch.randperm(len(x_train), generator=generator)[:int(self.cfg['reference_rows'])]
        reference = x_train[reference_idx].to(self.device)
        epochs = int(self.cfg['epochs'])
        best_loss, best_state, stale, updates = float('inf'), None, 0, 0
        history = []

        def task_loss(z, y):
            return nn.functional.mse_loss(z.squeeze(-1), y) if regression else nn.functional.cross_entropy(z, y)

        batch_size = int(self.cfg['batch_size'])
        for epoch in range(epochs):
            check_resources()
            model.train(); predictor.train()
            perm = torch.randperm(len(x_train), generator=generator)
            total_train = 0.0
            for idx in perm.split(batch_size):
                check_resources()
                x, y = x_train[idx].to(self.device), y_train[idx].to(self.device)
                predictions = predictor(reference_forward(model, x, reference))
                loss = task_loss(predictions, y)
                penalty = model.dag_penalty()/(model.num_layers+1)
                sparse = torch.triu(torch.sigmoid(model.layer_adj_logits), diagonal=1).sum()
                sparse = sparse + sum(torch.triu(torch.sigmoid(a), diagonal=1).sum() for a in model.intra_adj_logits)
                loss = loss + float(self.cfg['lambda_dag'])*penalty + float(self.cfg['lambda_sparse'])*sparse
                if not torch.isfinite(loss):
                    raise RuntimeError(f'DAGPipe non-finite training loss at epoch {epoch+1}')
                optimizer.zero_grad()
                loss.backward()
                clip_gradients(model.parameters())
                clip_gradients(predictor.parameters())
                optimizer.step()
                updates += 1
                total_train += loss.item()*len(y)
            # Same train-only reference protocol as final downstream evaluation.
            model.eval(); predictor.eval()
            total_val = 0.0
            with torch.no_grad():
                for start in range(0, len(x_val), batch_size):
                    check_resources()
                    x = x_val[start:start+batch_size].to(self.device)
                    y = y_val[start:start+batch_size].to(self.device)
                    loss = task_loss(predictor(reference_forward(model, x, reference)), y)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'DAGPipe non-finite validation loss at epoch {epoch+1}')
                    total_val += loss.item()*len(y)
            value = total_val/len(x_val)
            scheduler.step(value)
            history.append({'epoch': epoch+1, 'train_loss': total_train/len(x_train), 'validation_task_loss': value})
            if value < best_loss:
                best_loss, best_state, stale = value, copy.deepcopy(model.state_dict()), 0
                best_epoch = epoch+1
            else:
                stale += 1
            if self.verbose:
                print(f'[DAGPipe] epoch={epoch+1}/{epochs} val_loss={value:.6g}')
            if stale >= int(self.cfg['patience']):
                break
        if best_state is None:
            raise RuntimeError('DAGPipe produced no valid checkpoint')
        payload = {
            'format': 'dppbench.DAGPipe.reference.v2', 'task': self.data_name,
            'seed': self.seed, 'variant': 'daghie_inductive_reference_v2',
            'adapter': adapter,
            'dtype': self.cfg.get('dtype', 'float64'),
            'excluded_columns': sorted(c for c in exclude if c),
            'encoder': encoder.state, 'layers': layers,
            'state_dict': {k: v.detach().cpu().tolist() for k,v in best_state.items()},
            'reference': x_train[reference_idx].tolist(),
            'cache_entries': int(self.cfg['transform_cache_entries']),
            'selection': {'epoch': best_epoch, 'validation_loss': best_loss},
        }
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
               'search_metric': 'negative_normalized_mse' if regression else 'negative_cross_entropy',
               'budget': budget.snapshot() if budget else None}
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
