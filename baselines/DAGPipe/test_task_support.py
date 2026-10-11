"""Synthetic task-contract tests; no external datasets or downloads required."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from baselines.common.budget import Budget, budget_scope
from baselines.common.data_loader import DATASET_REGISTRY, import_class
from baselines.DAGPipe.dagpipe import DAGPipe
from baselines.DAGPipe.evaluator import DAGPipeEvaluator
from baselines.DAGPipe.pipeline import DAGPipePipeline
from baselines.DAGPipe.task_adapter import rec_splits

ROOT = Path(__file__).resolve().parents[2]
REC_TASKS = ['amazon_beauty', 'kuairec', 'movielens', 'yelp']


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def forbid_std_reads():
    original = pd.read_parquet

    def read(path, *args, **kwargs):
        assert Path(path).name != 'std_test.parquet', 'search/HPO accessed held-out data'
        return original(path, *args, **kwargs)
    return patch.object(pd, 'read_parquet', side_effect=read)


@pytest.fixture(params=REC_TASKS)
def rec_run(request, tmp_path, monkeypatch):
    name = request.param
    task_dir = ROOT / 'dppbench/tasks' / name
    config = yaml.safe_load((task_dir / 'model.yaml').read_text(encoding='utf-8'))
    target = config['feature']['target_col']
    data_dir = tmp_path / name / 'data'
    data_dir.mkdir(parents=True)
    std_dir = data_dir.parent / 'std_test'
    std_dir.mkdir()
    frame = pd.DataFrame({
        'user_id': [f'u{i % 4}' for i in range(60)],
        'item_id': [f'i{i % 5}' for i in range(60)],
        'timestamp': np.arange(60), target: np.tile([0., 1.], 30),
        'duration': np.arange(60, dtype=float),
    })
    frame.iloc[:48].to_parquet(std_dir / 'interaction_frozen.parquet', index=False)
    frame.iloc[48:].to_parquet(std_dir / 'std_test.parquet', index=False)
    cls = import_class(DATASET_REGISTRY[name][1])

    def load(data):
        data.interaction_df = frame.copy()
        data.user_df = pd.DataFrame({'user_id': [f'u{i}' for i in range(4)], 'age': [20., 30., 40., np.nan]})
        data.item_df = pd.DataFrame({'item_id': [f'i{i}' for i in range(5)], 'item_brand': ['a', 'b', 'c', 'd', 'e']})
        data.col_types = {'user_id': 'categorical', 'item_id': 'categorical',
                          'timestamp': 'timestamp', target: 'numeric', 'duration': 'numeric',
                          'age': 'numeric', 'item_brand': 'categorical'}
    monkeypatch.setattr(cls, 'load_data', load)
    kwargs = dict(task_dir=str(task_dir), data_name=name, data_dir=str(data_dir),
                  seed=42, device='cpu', verbose=False)
    with forbid_std_reads(), budget_scope(Budget(max_evaluations=1, max_pipeline_length=8)):
        runner = DAGPipe(**kwargs, max_pipeline_len=8, output_dir=tmp_path / 'run')
        runner.run()
    return kwargs, runner.get_pipeline(), cls, frame


@pytest.mark.parametrize('model', ['FNN', 'DeepFM', 'DIN', 'DIEN', 'SIM'])
def test_rec_downstream_models_and_frozen_holdout(rec_run, model):
    kwargs, pipeline, _, _ = rec_run
    pipeline = DAGPipePipeline.from_yaml(pipeline.to_yaml())
    assert len(pipeline) == 8
    evaluator = DAGPipeEvaluator(**kwargs, model_name=model, model_config={
        'model_params': {'embedding_dim': 4, 'dnn_hidden_units': [8, 4],
                         'att_hidden_size': [8, 4], 'dnn_dropout': 0.0},
        'train': {'epochs': 1, 'batch_size': 16},
    })
    try:
        evaluator.set_evaluate_std_test(False)
        with forbid_std_reads():
            result = evaluator.evaluate(pipeline)
        assert result.success, result.error
        assert np.isfinite(result.metrics['auc'])
        assert not any(k.startswith('std_test_') for k in result.metrics)
        evaluator.set_evaluate_std_test(True)
        result = evaluator.evaluate(pipeline)
        assert result.success, result.error
        assert result.metrics['std_test_n_rows'] == 12
        assert np.isfinite(result.metrics['std_test_auc'])
    finally:
        evaluator._executor.cleanup()


def test_rec_metadata_and_validation_history(rec_run):
    kwargs, pipeline, cls, _ = rec_run
    evaluator = DAGPipeEvaluator(**kwargs)
    try:
        adapter = pipeline.payload['adapter']
        data = cls(data_dir=kwargs['data_dir'])
        data.set_model_config(evaluator._executor.get_model_config())
        data.load_data()
        with forbid_std_reads():
            splits = rec_splits(data, data.model_cfg, adapter=adapter)
        train = splits['train']
        for row in splits['val'].itertuples(index=False):
            history = train.loc[(train.user_id == row.user_id) & (train.timestamp <= row.timestamp)]
            expected = history.sort_values('timestamp').item_id.tolist()[-adapter['max_len']:]
            assert row.item_id_seq == expected
        assert adapter['target'] not in [s['column'] for s in pipeline.payload['encoder']]
        assert min(adapter['id_maps']['item_id'].values()) == 1
        # Unknown IDs must use padding/OOV=0, not fit a new embedding vocabulary.
        from baselines.DAGPipe.task_adapter import rec_output
        unseen = splits['val'].iloc[:1].copy()
        unseen['item_id'] = 'unseen'
        out = rec_output(unseen, pipeline.transform(unseen), adapter)
        assert out.item_id.iloc[0] == 0
    finally:
        evaluator._executor.cleanup()


@pytest.fixture
def graph_run(tmp_path):
    data_dir = tmp_path / 'elliptic_bitcoin/data'
    data_dir.mkdir(parents=True)
    std_dir = data_dir.parent / 'std_test'
    std_dir.mkdir()
    n = 64
    frame = pd.DataFrame({'txId': np.arange(n),
                          'time_step': [10] * 24 + [36] * 16 + [42] * 24,
                          'feat_0': np.sin(np.arange(n)), 'feat_1': np.cos(np.arange(n))})
    frame.to_csv(data_dir / 'elliptic_txs_features.csv', header=False, index=False)
    labels = pd.DataFrame({'txId': np.arange(n), 'class': [str(1 + i % 2) if i < 56 else 'unknown' for i in range(n)]})
    labels.to_csv(data_dir / 'elliptic_txs_classes.csv', index=False)
    pd.DataFrame({'txId1': np.arange(n - 1), 'txId2': np.arange(1, n)}).to_csv(
        data_dir / 'elliptic_txs_edgelist.csv', index=False)
    held_out = frame.index.isin([0, 1, 24, 25, 40, 41, 42, 43])
    frame.loc[~held_out].to_parquet(std_dir / 'train_frozen.parquet', index=False)
    frame.loc[held_out].to_parquet(std_dir / 'std_test.parquet', index=False)
    kwargs = dict(task_dir=str(ROOT / 'dppbench/tasks/elliptic_bitcoin'),
                  data_name='elliptic_bitcoin', data_dir=str(data_dir),
                  seed=42, device='cpu', verbose=False)
    with forbid_std_reads(), budget_scope(Budget(max_evaluations=1)):
        runner = DAGPipe(**kwargs, max_pipeline_len=5, output_dir=tmp_path / 'run')
        runner.run()
    return kwargs, runner.get_pipeline()


@pytest.mark.parametrize('model', ['GCN', 'GraphSAGE', 'GAT'])
def test_graph_downstream_models_and_masks(graph_run, model):
    kwargs, pipeline = graph_run
    pipeline = DAGPipePipeline.from_yaml(pipeline.to_yaml())
    evaluator = DAGPipeEvaluator(**kwargs, model_name=model, model_config={
        'model_params': {'epochs': 1, 'hidden_dim': 8, 'dropout': 0.0},
    })
    try:
        evaluator.set_evaluate_std_test(False)
        with forbid_std_reads():
            result = evaluator.evaluate(pipeline)
        assert result.success, result.error
        assert not any(k.startswith('std_test_') for k in result.metrics)
        evaluator.set_evaluate_std_test(True)
        result = evaluator.evaluate(pipeline)
        assert result.success, result.error
        assert result.metrics['std_test_n_rows'] == 8
        assert np.isfinite(result.metrics['std_test_auc'])
        data = evaluator._executor._make_configured_data_instance()
        data.load_data()
        data.run_pre_process('unused.yaml')
        graph = data.build_graph()
        assert graph['num_nodes'] == 64
        assert graph['edge_index'].shape[1] == 63
        assert (graph['y'] == -1).sum() == 8
        assert not (graph['train_mask'] & graph['std_test_mask']).any()
        assert not (graph['val_mask'] & graph['std_test_mask']).any()
        assert not ((graph['y'] == -1) & graph['train_mask']).any()
        assert all(s['column'].startswith('feat_') for s in pipeline.payload['encoder'])
    finally:
        evaluator._executor.cleanup()


def test_every_shipped_dataset_has_a_supported_task_type():
    tasks = [p for p in (ROOT / 'dppbench/tasks').iterdir() if (p / 'model.yaml').is_file()]
    assert len(tasks) == 14
    for task in tasks:
        assert DATASET_REGISTRY[task.name][0] in {'tabular', 'rec', 'graph'}


TABULAR_TASKS = [name for name, (kind, _) in DATASET_REGISTRY.items() if kind == 'tabular']


@pytest.mark.parametrize('name', TABULAR_TASKS)
def test_all_tabular_dataset_split_contracts(name, tmp_path, monkeypatch):
    """Real per-task split/normalization/config; synthetic loaded observations."""
    cls = import_class(DATASET_REGISTRY[name][1])
    data_dir = tmp_path / name / 'data'
    data_dir.mkdir(parents=True)
    std_dir = data_dir.parent / 'std_test'
    std_dir.mkdir()
    template = cls(data_dir=str(data_dir))
    config = yaml.safe_load((ROOT / 'dppbench/tasks' / name / 'model.yaml').read_text(encoding='utf-8'))
    regression = config['model_params']['task'] == 'regression'
    n = 80
    frame = pd.DataFrame({'x': np.sin(np.arange(n)), 'category': np.tile(['a', 'b'], n // 2)})
    frame[template.target_col] = np.arange(n) * .1 if regression else np.tile([0, 1], n // 2)
    if template.id_col:
        frame[template.id_col] = np.arange(n)
    if getattr(template, '_sort_col', None):
        frame[template._sort_col] = np.arange(n)
    frame.iloc[:64].to_parquet(std_dir / 'train_frozen.parquet', index=False)
    frame.iloc[64:].to_parquet(std_dir / 'std_test.parquet', index=False)

    def load(data):
        data.train_df, data.test_df = frame.copy(), None
    monkeypatch.setattr(cls, 'load_data', load)
    kwargs = dict(task_dir=str(ROOT / 'dppbench/tasks' / name), data_name=name,
                  data_dir=str(data_dir), seed=42, verbose=False, model_name='LightGBM')
    with forbid_std_reads(), budget_scope(Budget(max_evaluations=1)):
        runner = DAGPipe(**kwargs, max_pipeline_len=4, output_dir=tmp_path / 'run')
        runner.run()
    evaluator = DAGPipeEvaluator(**kwargs, model_config={
        'model_params': {'n_estimators': 2, 'min_child_samples': 2},
    })
    try:
        result = evaluator.evaluate(runner.get_pipeline())
        assert result.success, result.error
        assert result.metrics['std_test_n_rows'] == 16
    finally:
        evaluator._executor.cleanup()


def _hpo_contract(kwargs, pipeline, model, overrides, tmp_path, monkeypatch, automatic=False):
    from scripts.final_model_hpo import tune_final_model
    monkeypatch.setattr('scripts.final_model_hpo.candidate_configs', lambda _: [overrides])
    before = pipeline.hash()
    selection = {} if automatic else {'evaluator_class': DAGPipeEvaluator}
    with forbid_std_reads(), patch('baselines.common.evaluator.check_pipeline',
                                  side_effect=AssertionError('DAGPipe reached public operator validator')):
        evaluator, summary = tune_final_model(
            pipeline, **kwargs, model_name=model, **selection,
            output_dir=str(tmp_path),
        )
    try:
        assert isinstance(evaluator, DAGPipeEvaluator)
        assert len(summary['trials']) == 1
        assert summary['trials'][0]['success']
        assert pipeline.hash() == before
        result = evaluator.evaluate(pipeline)
        assert result.success, result.error
        assert result.metrics['std_test_n_rows'] > 0
    finally:
        evaluator._executor.cleanup()


@pytest.mark.parametrize('automatic', [False, True])
def test_rec_final_hpo_is_validation_only(rec_run, tmp_path, monkeypatch, automatic):
    kwargs, pipeline, _, _ = rec_run
    _hpo_contract(kwargs, pipeline, 'DIN', {'train': {'epochs': 1, 'batch_size': 16}}, tmp_path, monkeypatch, automatic)


@pytest.mark.parametrize('automatic', [False, True])
def test_graph_final_hpo_is_validation_only(graph_run, tmp_path, monkeypatch, automatic):
    kwargs, pipeline = graph_run
    _hpo_contract(kwargs, pipeline, 'GCN', {'model_params': {'epochs': 1, 'hidden_dim': 8}}, tmp_path, monkeypatch, automatic)


def test_evaluator_selection_rejects_incompatible_explicit_class():
    from scripts.final_model_hpo import resolve_evaluator_class, tune_final_model
    from baselines.common.evaluator import PipelineEvaluator
    pipeline = object.__new__(DAGPipePipeline)
    assert resolve_evaluator_class(pipeline) is DAGPipeEvaluator
    with pytest.raises(TypeError, match='requires .*DAGPipeEvaluator'):
        tune_final_model(pipeline, output_dir='unused', evaluator_class=PipelineEvaluator)
    common = object.__new__(PipelineEvaluator)
    report = common._check_pipeline(pipeline, None)
    assert not report.valid
    assert 'DAGPipeEvaluator' in report.errors[0]


def test_standard_pipeline_keeps_standard_evaluator():
    from scripts.final_model_hpo import resolve_evaluator_class
    from baselines.common.pipeline import Pipeline
    from baselines.common.evaluator import PipelineEvaluator
    assert resolve_evaluator_class(Pipeline([])) is PipelineEvaluator


def forbid_legacy_common_dispatch(monkeypatch):
    from baselines.common.evaluator import PipelineEvaluator

    def incompatible(*args, **kwargs):
        raise AssertionError('Old common evaluator cannot execute a DAGPipe checkpoint')
    monkeypatch.setattr(PipelineEvaluator, '__init__', incompatible)
    monkeypatch.setattr(PipelineEvaluator, 'evaluate', incompatible)


def test_rec_hpo_with_incompatible_common_evaluator(rec_run, tmp_path, monkeypatch):
    kwargs, pipeline, _, _ = rec_run
    forbid_legacy_common_dispatch(monkeypatch)
    _hpo_contract(kwargs, pipeline, 'DIN', {'train': {'epochs': 1, 'batch_size': 16}},
                  tmp_path, monkeypatch, automatic=True)


def test_graph_hpo_with_incompatible_common_evaluator(graph_run, tmp_path, monkeypatch):
    kwargs, pipeline = graph_run
    forbid_legacy_common_dispatch(monkeypatch)
    _hpo_contract(kwargs, pipeline, 'GCN', {'model_params': {'epochs': 1, 'hidden_dim': 8}},
                  tmp_path, monkeypatch, automatic=True)


def test_custom_evaluation_cache_budget_and_rejection(rec_run, monkeypatch):
    kwargs, pipeline, _, _ = rec_run
    budget = Budget(max_evaluations=1)
    evaluator = DAGPipeEvaluator(**kwargs, budget=budget)
    calls = []
    def score(method):
        calls.append(method)
        return {'success': True, 'metrics': {'auc': .7}}
    monkeypatch.setattr(evaluator, '_run_scoring_method', score)
    try:
        result = evaluator.evaluate(pipeline)
        assert result.success
        assert evaluator.evaluate(pipeline) is result
        assert budget.evaluations_used == 1
        assert calls == ['full_train']
        evaluator.set_evaluate_std_test(False)
        exhausted = evaluator.evaluate(pipeline)
        assert not exhausted.success
        assert 'BudgetExceededError' in exhausted.error
        assert calls == ['full_train']
    finally:
        evaluator._executor.cleanup()
