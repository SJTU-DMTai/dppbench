"""Search-free replay and model-transfer experiment regressions."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scripts import evaluate_model_transfer as transfer


def arguments(tmp_path, **overrides):
    values = dict(source_root=str(tmp_path / 'original'),
                  pipeline_template='{source_root}/{baseline}/{task}_{source_model}/best_pipeline.yaml',
                  data_names='default_credit', baselines='SAGA,Random',
                  models='LightGBM,MLP', source_model=None, seed=42,
                  output_dir=str(tmp_path / 'transfer'), data_dir=None, gpu_id=-1)
    values.update(overrides)
    return SimpleNamespace(**values)


def sources(args):
    for baseline in args.baselines.split(','):
        source = Path(args.source_root) / baseline / 'default_credit_LightGBM/best_pipeline.yaml'
        source.parent.mkdir(parents=True)
        source.write_text('pipeline: []\n', encoding='utf-8')


def test_matrix_uses_same_source_for_each_target(tmp_path):
    args = arguments(tmp_path)
    sources(args)
    plan = transfer.build_plan(args)
    assert len(plan) == 4
    assert plan[0]['pipeline_source'] == plan[1]['pipeline_source']
    assert plan[0]['source_sha256'] == plan[1]['source_sha256']
    for job in plan:
        command = transfer.command_for(job, args)
        assert '--resume_pipeline' in command and '--skip_build' in command
        assert '--final_hpo' not in command
        assert command[command.index('--model') + 1] == job['model']


def test_missing_source_and_unsupported_model_fail_before_training(tmp_path):
    args = arguments(tmp_path)
    with pytest.raises(FileNotFoundError):
        transfer.build_plan(args)
    sources(args)
    args.models = 'DIN'
    with pytest.raises(ValueError, match='configured models'):
        transfer.build_plan(args)


def test_rank_agreement_excludes_failures_and_ties():
    rows = []
    for model, values in [('LightGBM', [0.9, 0.8, 0.7]), ('MLP', [0.6, 0.8, 0.7])]:
        for baseline, value in zip(['SAGA', 'Random', 'DiffPrep'], values):
            rows.append(dict(task='default_credit', seed=42, source_model='LightGBM',
                             model=model, baseline=baseline, std_test_metric=value,
                             std_test_metric_name='auc', error=''))
    result = transfer.comparison_rows(rows)[0]
    assert result['compared_pairs'] == 3
    assert result['ordering_agreement'] == pytest.approx(1 / 3)
    rows[-1]['error'] = 'failed'
    assert transfer.comparison_rows(rows)[0]['matched_baselines'] == 2
    rows[-2]['std_test_metric'] = 0.6
    assert transfer.comparison_rows(rows)[0]['ordering_agreement'] is None


def test_all_common_baselines_replay_without_importing_search(tmp_path):
    from scripts import evaluate_baseline as harness
    from baselines.common.evaluator import EvaluationResult, PipelineEvaluator
    from baselines.common.pipeline import Pipeline

    args = SimpleNamespace(data_dir=None, model='MLP', max_pipeline_len=None,
                           seed=42, quiet=True, device='cpu', final_hpo=False)
    source = tmp_path / 'original/best_pipeline.yaml'
    source.parent.mkdir()
    source.write_text('pipeline: []\n', encoding='utf-8')
    args.resume_pipeline = str(source)
    original = source.read_bytes()
    seen = []

    def fake_evaluate(self, pipeline, scoring_method=None):
        seen.append((self.model_name, scoring_method))
        return EvaluationResult(fitness=0.7, success=True,
                                metrics={'auc': 0.7, 'std_test_auc': 0.6})

    with patch.object(harness, '_load_class', side_effect=AssertionError('search class imported')), \
         patch.object(harness, '_validate_final_pipeline'), \
         patch.object(PipelineEvaluator, 'evaluate', fake_evaluate):
        for baseline in harness.BASELINE_CONFIGS:
            if baseline == 'DAGPipe':
                continue
            row = harness._run_baseline_and_eval('default_credit', baseline, args, str(tmp_path / 'replay'))
            assert row['error'] is None
            assert row['pipeline_reused']
            assert row['pipeline_sha256'] == hashlib.sha256(Pipeline.from_yaml(original.decode()).to_yaml().encode()).hexdigest()
            assert row['construct_time_s'] is None
            assert row['n_unique_evals'] is None
        args.resume_pipeline = str(tmp_path / 'missing.yaml')
        with pytest.raises(FileNotFoundError):
            harness._run_baseline_and_eval('default_credit', 'SAGA', args, str(tmp_path / 'replay'))
    assert seen and all(item == ('MLP', 'full_train') for item in seen)
    assert source.read_bytes() == original
