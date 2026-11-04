"""Regression coverage for early routing checks and search-free recovery."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from baselines.DAGPipe.preflight import check_runtime, probe_pipeline
from baselines.DAGPipe.evaluator import DAGPipeEvaluator
from baselines.common.evaluator import EvaluationResult
from scripts import evaluate_baseline as harness

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('task', ['amazon_beauty', 'kuairec', 'elliptic_bitcoin', 'bike_sharing'])
def test_preflight_is_data_free(task):
    with patch('pandas.read_parquet', side_effect=AssertionError('dataset read')):
        assert check_runtime(task_dir=str(ROOT / 'dppbench/tasks' / task), data_name=task)['status'] == 'passed'


def test_preflight_rejects_legacy_validator(monkeypatch):
    from baselines.common.pipeline import check_pipeline

    def legacy(self, pipeline, scoring_method=None):
        report = check_pipeline(pipeline)
        return EvaluationResult(fitness=float('-inf'), success=False,
                                error='; '.join(report.errors), metrics={}, duration_seconds=0)

    monkeypatch.setattr(DAGPipeEvaluator, 'evaluate', legacy)
    with pytest.raises(RuntimeError, match='Unsupported pipeline type'):
        check_runtime(task_dir=str(ROOT / 'dppbench/tasks/amazon_beauty'), data_name='amazon_beauty')


def test_preflight_rejects_bypassed_validation(monkeypatch):
    from baselines.common.pipeline import PipelineCheckReport
    monkeypatch.setattr(DAGPipeEvaluator, '_check_pipeline',
                        lambda *a, **kw: PipelineCheckReport(True, []))
    with pytest.raises(RuntimeError, match='failed to reject'):
        check_runtime(task_dir=str(ROOT / 'dppbench/tasks/amazon_beauty'), data_name='amazon_beauty')


def test_bad_runtime_stops_before_search(tmp_path, monkeypatch):
    from baselines.DAGPipe.dagpipe import DAGPipe
    args = SimpleNamespace(data_dir=None, model='DIN', max_pipeline_len=10, seed=42,
                           quiet=True, device='cpu', final_hpo=True, resume_pipeline=None)
    monkeypatch.setattr(DAGPipe, 'run', lambda self: pytest.fail('Search started before check'))
    with patch('baselines.DAGPipe.preflight.check_runtime', side_effect=RuntimeError('bad runtime')):
        with pytest.raises(RuntimeError, match='bad runtime'):
            harness._run_baseline_and_eval('amazon_beauty', 'DAGPipe', args, str(tmp_path))


def test_resume_skips_search_and_preserves_costs(tmp_path, monkeypatch):
    from baselines.DAGPipe.dagpipe import DAGPipe
    pipeline = probe_pipeline('amazon_beauty', 'rec')
    source = tmp_path / 'original' / 'best_pipeline.yaml'
    source.parent.mkdir()
    pipeline.save(source)
    original = source.read_bytes()
    args = SimpleNamespace(data_dir=None, model='DIN', max_pipeline_len=10, seed=42,
                           quiet=True, device='cpu', final_hpo=True, resume_pipeline=str(source))
    monkeypatch.setattr(DAGPipe, 'run', lambda self: pytest.fail('Search must not run'))
    calls = []

    def evaluate(self, checkpoint, scoring_method=None):
        # Data-free preflight has its own synthetic pipeline and stub training.
        if hasattr(self._executor.run_training, '__name__') and self._executor.run_training.__name__ == 'fake_training':
            return real_evaluate(self, checkpoint, scoring_method)
        calls.append(self._executor._evaluate_std_test)
        return EvaluationResult(fitness=.7, success=True, error=None,
                                metrics={'auc': .7, 'std_test_auc': .6} if calls[-1] else {'auc': .7},
                                duration_seconds=0)

    real_evaluate = DAGPipeEvaluator.evaluate
    monkeypatch.setattr(DAGPipeEvaluator, 'evaluate', evaluate)
    monkeypatch.setattr('scripts.final_model_hpo.candidate_configs', lambda model: [{'lr': .001}])
    row = harness._run_baseline_and_eval('amazon_beauty', 'DAGPipe', args, str(tmp_path / 'replay'))
    assert calls == [False, True]
    assert row['pipeline_reused'] and row['pipeline_sha256'] == pipeline.hash()
    assert row['construct_time_s'] is None and row['baseline_training_seconds'] is None
    assert row['n_unique_evals'] is None and row['error'] is None
    assert source.read_bytes() == original
    args.seed = 43
    with pytest.raises(ValueError, match='task/seed'):
        harness._run_baseline_and_eval('amazon_beauty', 'DAGPipe', args, str(tmp_path / 'replay'))
    args.seed = 42
    args.resume_pipeline = str(tmp_path / 'missing.yaml')
    with pytest.raises(FileNotFoundError):
        harness._run_baseline_and_eval('amazon_beauty', 'DAGPipe', args, str(tmp_path / 'replay'))
