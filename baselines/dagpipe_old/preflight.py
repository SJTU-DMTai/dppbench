"""Data-free functional check of the installed checkpoint evaluation route."""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path


def source_info(obj):
    path = Path(inspect.getfile(obj)).resolve()
    return {'symbol': f'{obj.__module__}.{obj.__qualname__}', 'source': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def probe_pipeline(task, task_type):
    from .pipeline import DAGPipePipeline
    from .runtime import make_model
    layers = [['Identity']]
    model = make_model(layers, 1)
    return DAGPipePipeline({
        'format': 'dppbench.DAGPipe.reference.v2', 'task': task, 'seed': 42,
        'adapter': {'task_type': task_type}, 'layers': layers,
        'encoder': [{'column': 'probe', 'kind': 'numeric', 'fill': 0., 'mean': 0., 'scale': 1.}],
        'state_dict': {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()},
        'reference': [[0.], [1.]],
    })


def check_runtime(*, task_dir, data_name, model_name=None, evaluator_class=None):
    """Exercise actual evaluate/validate/dispatch methods; never load task data.

    Only downstream training is replaced on this disposable executor instance.
    Passing this check proves routing compatibility, not dataset/model fitness.
    """
    from .evaluator import DAGPipeEvaluator, DAGPipeExecutor
    from .tabular_adapter import augment_splits
    from .runtime import reference_forward
    from scripts.final_model_hpo import resolve_evaluator_class
    evaluator_class = evaluator_class or DAGPipeEvaluator
    evaluator = evaluator_class(task_dir=task_dir, data_name=data_name,
                                model_name=model_name, device='cpu', budget=None)
    try:
        if not isinstance(evaluator._executor, DAGPipeExecutor):
            raise RuntimeError('DAGPipe runtime uses the wrong executor')
        evaluator.set_evaluate_std_test(False)
        pipeline = probe_pipeline(data_name, evaluator.task_type)
        resolve_evaluator_class(pipeline, evaluator_class)
        calls = []

        def fake_training():
            calls.append(True)
            if evaluator._executor.pipeline is not pipeline:
                raise RuntimeError('Checkpoint was not attached to DAGPipeExecutor')
            return {'success': True, 'metrics': {evaluator.metric_key: .5},
                    'execution_data': {'shape': [2, 1]}}

        evaluator._executor.run_training = fake_training
        result = evaluator.evaluate(pipeline, scoring_method='full_train')
        if not result.success or len(calls) != 1:
            raise RuntimeError(f'DAGPipe runtime preflight failed before search: {result.error}; '
                               'synchronize DAGPipe and common evaluation files, then rerun diagnostics')
        wrong = probe_pipeline('__wrong_task__', evaluator.task_type)
        rejected = evaluator.evaluate(wrong, scoring_method='full_train')
        if rejected.success or len(calls) != 1:
            raise RuntimeError('DAGPipe runtime failed to reject a mismatched checkpoint')
        return {'task': data_name, 'status': 'passed',
                'tabular_adapter': source_info(augment_splits),
                'reference_transform': source_info(reference_forward),
                'evaluate': source_info(evaluator_class.evaluate),
                'validator': source_info(evaluator_class._check_pipeline),
                'executor': source_info(type(evaluator._executor)),
                'pipeline': source_info(type(pipeline))}
    finally:
        evaluator._executor.cleanup()
