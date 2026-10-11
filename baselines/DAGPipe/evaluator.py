"""Use shared TrainingExecutor and final HPO with an instance-local data hook."""
from __future__ import annotations

import os
import math
import time
import pandas as pd

from baselines.common.executor import TrainingExecutor
from baselines.common.evaluator import PipelineEvaluator, EvaluationResult
from baselines.common.budget import get_active_budget
from baselines.common.pipeline import PipelineCheckReport
from .pipeline import DAGPipePipeline
from .task_adapter import rec_splits, rec_output, graph_splits


def training_splits(data, config, include_std_test=False):
    """Canonical task split applied to the canonical frozen training subset.

    In search/HPO, never read std_test.parquet (not even its labels/schema).
    Requiring train_frozen avoids accidentally training on the holdout.
    """
    std_dir = data._std_test_dir(data.data_dir)
    frozen = os.path.join(std_dir, 'train_frozen.parquet')
    if not os.path.isfile(frozen):
        raise FileNotFoundError(f'DAGPipe requires frozen training data: {frozen}; run build_std_test.py')
    data.train_df = data._normalize_frozen_frame(pd.read_parquet(frozen))
    target = data.target_col
    labels = pd.to_numeric(data.train_df[target], errors='coerce')
    import numpy as np
    data.train_df = data.train_df.loc[labels.notna() & np.isfinite(labels)].reset_index(drop=True)
    if include_std_test:
        if not data.load_std_test_frozen():
            raise FileNotFoundError(f'DAGPipe frozen standard test missing in {std_dir}')
    train = config.get('train', {})
    return data.split(val_ratio=train.get('val_ratio', .2), seed=train.get('seed', 42))


class DAGPipeExecutor(TrainingExecutor):
    pipeline = None

    def _make_configured_data_instance(self):
        data = super()._make_configured_data_instance()
        if self.pipeline is None:
            return data

        def preprocess(_yaml_path):
            cfg = self.get_model_config()
            feature_cols = [s['column'] for s in self.pipeline.payload['encoder']]
            if self.task_type == 'rec':
                adapter = self.pipeline.payload['adapter']
                splits = rec_splits(data, cfg, include_std_test=self._evaluate_std_test, adapter=adapter)
                encoded = []
                for name in ('train', 'val', 'std_test'):
                    frame = splits[name]
                    if len(frame):
                        encoded.append(rec_output(frame, self.pipeline.transform(frame[feature_cols], self.device), adapter))
                data.interaction_df = pd.concat(encoded, ignore_index=True)
                data.col_types = {f'dagpipe_feature_{i}': 'numeric' for i in range(len(feature_cols))}
                data.col_types.update({c: 'categorical' for c in adapter['id_maps']})
                data.col_types.update({c: 'categorical_list' for c in adapter['sequences']})
                data.col_types[adapter['target']] = 'numeric'
                return data.interaction_df
            if self.task_type == 'graph':
                graph_splits(data, include_std_test=self._evaluate_std_test)
                # Transform node attributes in place: identifiers, edges,
                # unknown labels and fixed supervision masks remain intact.
                frame = data.train_df.copy()
                frame[feature_cols] = self.pipeline.transform(frame[feature_cols], self.device)
                data.train_df = frame
                return frame
            splits = training_splits(data, cfg, include_std_test=self._evaluate_std_test)
            encoded = {}
            for split_name in ('train', 'val', 'std_test'):
                frame = splits.get(split_name)
                if frame is None:
                    continue
                transformed = self.pipeline.transform(frame[feature_cols], self.device)
                result = pd.DataFrame(transformed, columns=feature_cols)
                for col in (data.target_col, data.id_col, 'sample_weight'):
                    if col and col in frame and col not in result:
                        result[col] = frame[col].to_numpy()
                encoded[split_name] = result
            encoded['test'] = None
            data.train_df = pd.concat([encoded['train'], encoded['val']], ignore_index=True)
            # Downstream common executor asks for split again. Return exactly
            # the already-fixed task split rather than re-splitting transformed rows.
            data.split = lambda **kwargs: encoded
            return data.train_df

        data.run_pre_process = preprocess
        return data


class DAGPipeEvaluator(PipelineEvaluator):
    executor_class = DAGPipeExecutor

    def __init__(self, task_dir, data_name, data_dir=None, metric_key=None,
                 verbose=False, device='cpu', model_name=None,
                 scoring_method='full_train', budget=None, seed=None, model_config=None):
        # Do not depend on the common constructor's executor_class hook:
        # older deployed versions hard-code TrainingExecutor there.
        self.task_dir, self.data_name, self.data_dir = task_dir, data_name, data_dir
        self.verbose, self.device, self.model_name = verbose, device, model_name
        self.scoring_method = scoring_method
        self.budget = budget if budget is not None else get_active_budget()
        self._executor = DAGPipeExecutor(
            task_dir, data_name, data_dir=data_dir, device=device,
            model_name=model_name, seed=seed, model_config=model_config,
        )
        self.metric_key = metric_key or self._executor.get_model_config().get('train', {}).get('metrics', ['auc'])[0]
        self._cache = {}
        self._data_context = None

    def _get_data_context(self):
        # No public-operator registry conversion: legality is checked below.
        return None

    def _check_pipeline(self, pipeline, ctx, budget=None, execution_data=None, **kwargs):
        errors = []
        if not isinstance(pipeline, DAGPipePipeline):
            return PipelineCheckReport(False, ['DAGPipeEvaluator requires DAGPipePipeline'])
        adapter_type = pipeline.payload.get('adapter', {}).get('task_type', 'tabular')
        if adapter_type != self.task_type:
            errors.append('DAGPipe checkpoint task type does not match evaluator task type')
        if pipeline.payload['task'] != self.data_name:
            errors.append('DAGPipe checkpoint task does not match evaluator task')
        if budget is not None:
            budget.check_resources()
            if budget.max_pipeline_length and len(pipeline) > budget.max_pipeline_length:
                errors.append('DAGPipe exceeds max_pipeline_length')
            if execution_data:
                shape = execution_data.get('shape', [0, 0])
                if budget.max_output_rows and shape[0] > budget.max_output_rows:
                    errors.append('DAGPipe exceeds max_output_rows')
                if budget.max_output_columns and shape[1] > budget.max_output_columns:
                    errors.append('DAGPipe exceeds max_output_columns')
        return PipelineCheckReport(not errors, errors)

    def evaluate(self, pipeline, scoring_method=None):
        """Evaluate a learned checkpoint, never the public operator-DAG schema.

        Older common evaluate() versions call check_pipeline() directly instead
        of the subclass hook. Keep dispatch explicit here so updating this
        adapter also works with those installations, without global patches.
        """
        method = scoring_method or self.scoring_method
        key = f'{method}:{pipeline.hash()}'
        if key in self._cache:
            return self._cache[key]
        started = time.monotonic()
        metrics, error, fitness = {}, None, -math.inf
        try:
            if method not in ('full_train', 'fast_train'):
                raise ValueError(f'DAGPipe checkpoint does not support scoring_method={method!r}')
            if self.budget is not None:
                self.budget.consume_evaluation()
            report = self._check_pipeline(pipeline, None, budget=self.budget)
            if not report.valid:
                raise ValueError('; '.join(report.errors))
            self._executor.pipeline = pipeline
            self._executor.write_yaml(pipeline.to_yaml())
            result = self._run_scoring_method(method)
            metrics = dict(result.get('metrics') or {})
            report = self._check_pipeline(
                pipeline, None, budget=self.budget,
                execution_data=result.get('execution_data'),
            )
            if not result.get('success') or result.get('error') or not report.valid:
                reasons = [result.get('error'), *report.errors]
                raise RuntimeError('; '.join(str(reason) for reason in reasons if reason)
                                   or 'DAGPipe downstream training failed')
            value = metrics.get(self.metric_key)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f'Missing or non-finite evaluation metric {self.metric_key!r}: {value!r}')
            fitness = float(value)
            if self.metric_key in {'rmse', 'mse', 'mae', 'logloss'}:
                fitness = -fitness
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        label = self._display_method(method)
        ev = EvaluationResult(
            fitness=fitness, success=error is None, error=error,
            metrics={**metrics, 'scoring_method': label},
            duration_seconds=time.monotonic() - started,
        )
        self._cache[key] = ev
        if self.verbose:
            status = f'fitness={fitness:.4f}' if ev.success else f'FAILED {error}'
            print(f'  [DAGPipe eval:{label}] {status} ({ev.duration_seconds:.1f}s)')
        return ev

    def run_preprocessing(self, pipeline):
        report = self._check_pipeline(pipeline, None, budget=self.budget)
        if not report.valid:
            raise ValueError('; '.join(report.errors))
        self._executor.pipeline = pipeline
        self._executor.write_yaml(pipeline.to_yaml())
        return self._executor.run_preprocessing()
