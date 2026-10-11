from pathlib import Path
import inspect
import json
import random
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from baselines.DiffPrep.diffprep import DiffPrep
from baselines.Random.random_baseline import RandomBaseline
from baselines.common.evaluator import PipelineEvaluator
from baselines.common.pipeline import Pipeline
from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.pipeline import build_default_params, validate_step_params

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def graph_task(tmp_path):
    task = tmp_path / 'task'
    task.mkdir()
    source = ROOT / 'dppbench/tasks/elliptic_bitcoin'
    (task / 'prepare.yaml').write_text((source / 'prepare.yaml').read_text(), encoding='utf-8')
    cfg = yaml.safe_load((source / 'model.yaml').read_text())
    cfg.setdefault('model_params', {}).update(epochs=1, hidden_dim=8, dropout=0.)
    (task / 'model.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    std = tmp_path / 'std_test'
    std.mkdir()
    n = 48
    frame = pd.DataFrame({'txId': np.arange(n), 'time_step': [10]*20+[36]*16+[42]*12,
                          'feat_0': np.sin(np.arange(n)), 'feat_1': np.cos(np.arange(n))})
    frame.to_csv(data_dir/'elliptic_txs_features.csv', header=False, index=False)
    pd.DataFrame({'txId': np.arange(n), 'class': [str(1+i%2) if i<44 else 'unknown' for i in range(n)]}).to_csv(data_dir/'elliptic_txs_classes.csv', index=False)
    pd.DataFrame({'txId1': np.arange(n-1), 'txId2': np.arange(1,n)}).to_csv(data_dir/'elliptic_txs_edgelist.csv', index=False)
    held = frame.index.isin([0,1,20,21,40,41])
    frame[~held].to_parquet(std/'train_frozen.parquet', index=False)
    frame[held].to_parquet(std/'std_test.parquet', index=False)
    return dict(task_dir=str(task), data_name='elliptic_bitcoin', data_dir=str(data_dir),
                device='cpu', verbose=False, seed=42, model_name='GraphSAGE')


@pytest.mark.parametrize('baseline', ['Random', 'DiffPrep'])
def test_graph_search_and_real_downstream_training(graph_task, tmp_path, monkeypatch, baseline):
    original = pd.read_parquet
    def guarded(path, *a, **kw):
        if Path(path).name == 'std_test.parquet':
            raise AssertionError('Search must not read standard test')
        return original(path, *a, **kw)
    with monkeypatch.context() as m:
        m.setattr(pd, 'read_parquet', guarded)
        if baseline == 'Random':
            runner = RandomBaseline(**graph_task, n_trials=2, random_pipeline_p_optional=0.,
                                    small_n=0, eval_full=True, max_pipeline_len=8, output_dir=str(tmp_path/'run'))
        else:
            runner = DiffPrep(**graph_task, n_epochs=2, small_n=0, eval_full=True,
                              second_order=True, hard_sample=False, max_features=2,
                              surrogate_hidden_dim=8, max_pipeline_len=12, output_dir=str(tmp_path/'run'))
        result = runner.run()
    diagnostic = json.dumps({
        'runner_source': inspect.getfile(type(runner)),
        'context_source': inspect.getfile(DataContext),
        'pipeline_source': inspect.getfile(build_default_params),
        'result': result,
    }, ensure_ascii=False, indent=2, default=str)
    assert result['is_legal'], diagnostic
    assert result['eval_error'] is None, diagnostic
    assert np.isfinite(result['best_fitness'])
    assert Path(result['best_pipeline_path']).is_file()
    pipe = Pipeline.from_yaml(result['best_pipeline_yaml'])
    for model in ('GraphSAGE', 'GCN', 'GAT'):
        args = dict(graph_task, model_name=model)
        evaluator = PipelineEvaluator(**args)
        try:
            evaluator.set_evaluate_std_test(True)
            ev = evaluator.evaluate(pipe)
            assert ev.success, ev.error
            assert np.isfinite(ev.metrics['std_test_auc'])
        finally:
            evaluator._executor.cleanup()


def test_graph_transform_statistics_use_only_training_rows():
    from baselines.DiffPrep.soft_ops import apply_soft_op
    x = torch.tensor([[1.], [3.], [100.]])
    tr = torch.tensor([True, True, False])
    a = apply_soft_op('ScaleFeature', x, torch.zeros_like(x), tr)
    x[-1] = -999.
    b = apply_soft_op('ScaleFeature', x, torch.zeros_like(x), tr)
    torch.testing.assert_close(a[:2], b[:2])
    torch.testing.assert_close(a[:2], torch.tensor([[-1.], [1.]]))


def test_graph_pipeline_accepts_context_without_prediction_task():
    ctx = DataContext(task_type='graph', data_name='elliptic_bitcoin',
                      primary_table='main', tables={
                          'main': TableContext(name='main', role='primary',
                                               dtypes={'txId': 'int64', 'feat_0': 'float64'})})

    class LegacyContext:
        def __getattr__(self, name):
            if name == 'prediction_task':
                raise AttributeError(name)
            return getattr(ctx, name)

    legacy = LegacyContext()
    params = build_default_params('ScaleFeature', legacy, random.Random(42))
    assert params is not None
    assert validate_step_params('ScaleFeature', params, legacy) == []
    # Missing optional metadata must not disable graph topology protection.
    assert build_default_params('Oversample', legacy, random.Random(42)) is None
