"""Use shared TrainingExecutor and final HPO with an instance-local data hook."""
from __future__ import annotations

import os
import pandas as pd

from baselines.common.executor import TrainingExecutor
from baselines.common.evaluator import PipelineEvaluator
from baselines.common.pipeline import PipelineCheckReport


def training_splits(data, config, include_std_test=False):
    """Canonical task split applied to the canonical frozen training subset.

    In search/HPO, never read std_test.parquet (not even its labels/schema).
    Requiring train_frozen avoids accidentally training on the holdout.
    """
    std_dir = data._std_test_dir(data.data_dir)
    frozen = os.path.join(std_dir, 'train_frozen.parquet')
    if not os.path.isfile(frozen):
        raise FileNotFoundError(f'DagPipe requires frozen training data: {frozen}; run build_std_test.py')
    data.train_df = data._normalize_frozen_frame(pd.read_parquet(frozen))
    target = data.target_col
    labels = pd.to_numeric(data.train_df[target], errors='coerce')
    import numpy as np
    data.train_df = data.train_df.loc[labels.notna() & np.isfinite(labels)].reset_index(drop=True)
    if include_std_test:
        if not data.load_std_test_frozen():
            raise FileNotFoundError(f'DagPipe frozen standard test missing in {std_dir}')
    train = config.get('train', {})
    return data.split(val_ratio=train.get('val_ratio', .2), seed=train.get('seed', 42))


class DagPipeExecutor(TrainingExecutor):
    pipeline = None

    def _make_configured_data_instance(self):
        data = super()._make_configured_data_instance()
        if self.pipeline is None:
            return data
        if self.task_type != 'tabular':
            raise ValueError(f'DagPipe unsupported_task_type={self.task_type}')

        def preprocess(_yaml_path):
            cfg = self.get_model_config()
            splits = training_splits(data, cfg, include_std_test=self._evaluate_std_test)
            encoded = {}
            feature_cols = [s['column'] for s in self.pipeline.payload['encoder']]
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


class DagPipeEvaluator(PipelineEvaluator):
    executor_class = DagPipeExecutor

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_key = self._executor.get_model_config().get('train', {}).get('metrics', ['auc'])[0]

    def _get_data_context(self):
        # No public-operator registry conversion: legality is checked below.
        return None

    def _check_pipeline(self, pipeline, ctx, budget=None, execution_data=None, **kwargs):
        errors = []
        if self.task_type != 'tabular':
            errors.append(f'DagPipe unsupported_task_type={self.task_type}')
        if pipeline.payload['task'] != self.data_name:
            errors.append('DagPipe checkpoint task does not match evaluator task')
        if budget is not None:
            budget.check_resources()
            if budget.max_pipeline_length and len(pipeline) > budget.max_pipeline_length:
                errors.append('DagPipe exceeds max_pipeline_length')
            if execution_data:
                shape = execution_data.get('shape', [0, 0])
                if budget.max_output_rows and shape[0] > budget.max_output_rows:
                    errors.append('DagPipe exceeds max_output_rows')
                if budget.max_output_columns and shape[1] > budget.max_output_columns:
                    errors.append('DagPipe exceeds max_output_columns')
        return PipelineCheckReport(not errors, errors)

    def evaluate(self, pipeline, scoring_method=None):
        self._executor.pipeline = pipeline
        # Preserve shared cache, error reporting, evaluation counters and metrics.
        return super().evaluate(pipeline, scoring_method)
