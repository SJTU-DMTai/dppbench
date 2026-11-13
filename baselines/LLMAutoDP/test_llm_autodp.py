import json
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from baselines.LLMAutoDP.llm_autodp import LLMAutoDP, decode_pipeline, parse_group
from baselines.common.budget import Budget, budget_scope
from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.pipeline import Pipeline, check_pipeline, repair
from scripts.run_task_baselines import TASKS

ROOT = Path(__file__).resolve().parents[2]


def context(kind="tabular"):
    primary = "interaction" if kind == "rec" else "main"
    columns = {"id": "int64", "x": "float64", "label": "int64"}
    if kind == "rec":
        columns.update(user_id="int64", item_id="int64", timestamp="int64")
    return DataContext(kind, "fixture", primary, tables={primary: TableContext(
        primary, "primary", dtypes=columns, configured_target_col="label",
        configured_id_col="id", configured_time_col="timestamp" if kind == "rec" else None)})


def runner(tmp_path):
    return LLMAutoDP(str(ROOT / "dppbench/tasks/default_credit"), "default_credit",
                     output_dir=tmp_path)


@pytest.mark.parametrize("kind", ["tabular", "rec", "graph"])
def test_anchor_roundtrip_legal(kind):
    ctx = context(kind)
    anchor = repair(Pipeline(), kind, ctx, rng=random.Random(42))
    parsed = decode_pipeline(yaml.safe_load(anchor.to_yaml()))
    report = check_pipeline(parsed, ctx)
    assert report.valid, report.errors


@pytest.mark.parametrize("text", ['{}', '[]', '{"stop":"false"}',
    '{"stop":true,"candidates":[{}]}', '{"candidates":[{}, {}, {}]}'])
def test_bad_protocol(text):
    with pytest.raises(ValueError):
        parse_group(text, 2)


def test_group_feedback_cache_and_regression_direction(tmp_path):
    obj, ctx = runner(tmp_path), context()
    anchor = repair(Pipeline(), "tabular", ctx, rng=random.Random(42))
    # Use legal alternative order; it is distinct without relying on optional params.
    candidate = anchor.copy()
    candidate.steps.reverse()
    payload = yaml.safe_load(candidate.to_yaml())
    evaluator = Mock()
    evaluator.evaluate.side_effect = [
        SimpleNamespace(success=True, fitness=-5., metrics={"rmse": 5.}, error=None),
        SimpleNamespace(success=True, fitness=-3., metrics={"rmse": 3.}, error=None)]
    llm = Mock()
    llm.chat.side_effect = [json.dumps({"candidates": [payload, payload]}),
                            '{"stop":true,"candidates":[]}']
    best, fit, _, history, messages, reason = obj._search(evaluator, llm, ctx)
    assert fit == -3. and best.hash() == candidate.hash()
    assert evaluator.evaluate.call_count == 2  # anchor + one unique candidate
    assert history[1]["group"][0]["relative_improvement"] == 2.
    assert history[1]["group"][1]["cache_hit"]
    assert reason == "llm_stop"
    assert "relative_improvement" in messages[-2]["content"]


def test_invalid_candidate_and_parse_retry_keep_anchor(tmp_path):
    obj, ctx = runner(tmp_path), context()
    evaluator = Mock()
    evaluator.evaluate.return_value = SimpleNamespace(success=True, fitness=.6, metrics={}, error=None)
    llm = Mock()
    llm.chat.side_effect = ['not json', '{"candidates":[{}]}', '{"stop":true}']
    _, fit, _, history, _, _ = obj._search(evaluator, llm, ctx)
    assert fit == .6
    assert evaluator.evaluate.call_count == 1
    assert "error" in history[1] and history[2]["group"][0]["error"]


def test_no_success_fails_explicitly(tmp_path):
    obj = runner(tmp_path)
    evaluator = Mock()
    evaluator.evaluate.return_value = SimpleNamespace(success=False, fitness=float('-inf'), metrics={}, error="training failed")
    llm = Mock()
    llm.chat.return_value = '{"stop":true}'
    with pytest.raises(RuntimeError, match="no successful"):
        obj._search(evaluator, llm, context())


def test_budget_stops_calls_preserves_best(tmp_path):
    obj = runner(tmp_path)
    evaluator = Mock()
    evaluator.evaluate.return_value = SimpleNamespace(success=True, fitness=.7, metrics={}, error=None)
    llm = Mock()
    budget = Budget(max_llm_calls=1)
    budget.consume_llm_call()
    llm.chat.side_effect = lambda *a, **k: budget.consume_llm_call()
    with budget_scope(budget):
        _, fit, _, _, _, reason = obj._search(evaluator, llm, context())
    assert fit == .7 and "budget exhausted" in reason


def test_run_artifacts_and_validation_only(tmp_path):
    obj = runner(tmp_path)
    evaluator = Mock()
    evaluator._get_data_context.return_value = context()
    evaluator.evaluate.return_value = SimpleNamespace(success=True, fitness=.7, metrics={"auc": .7}, error=None)
    llm = Mock()
    llm.chat.return_value = '{"stop":true}'
    obj._make_evaluator = lambda: evaluator
    obj._make_llm = lambda: llm
    result = obj.run()
    assert result["method"] == "LLMAutoDP"
    assert result["original_paper_reproduction"] is False
    assert Path(result["best_pipeline_path"]).exists()
    assert json.loads((tmp_path / "run_summary.json").read_text())["best_fitness_in_loop"] == .7
    evaluator._executor.cleanup.assert_called_once()


@pytest.mark.parametrize("task,model,device", TASKS)
def test_all_fourteen_task_evaluator_contracts(task, model, device, tmp_path):
    obj = LLMAutoDP(str(ROOT / "dppbench/tasks" / task), task, model_name=model, output_dir=tmp_path)
    evaluator = obj._make_evaluator()
    try:
        cfg = yaml.safe_load((ROOT / "dppbench/tasks" / task / "model.yaml").read_text())
        assert evaluator.metric_key == cfg["train"]["metrics"][0]
        assert evaluator.small_n == obj.cfg["downstream_eval_n"] == 10000
        assert evaluator._executor._evaluate_std_test is False
        assert evaluator.task_type in {"rec", "graph", "tabular"}
    finally:
        evaluator._executor.cleanup()


@pytest.mark.parametrize("name", ["LLMAutoDP", "llmautodp", " LLM-AutoDP-Adapted "])
def test_public_name_and_legacy_alias(name):
    from scripts.evaluate_baseline import BASELINE_CONFIGS, _canonical_baseline_name
    canonical = _canonical_baseline_name(name)
    assert canonical == "LLMAutoDP"
    assert BASELINE_CONFIGS[canonical]["cls_path"] == "baselines.LLMAutoDP.llm_autodp.LLMAutoDP"


def test_default_output_uses_public_name():
    obj = LLMAutoDP(str(ROOT / "dppbench/tasks/default_credit"), "default_credit")
    assert obj.output_dir == Path("outputs") / "LLMAutoDP" / "default_credit"


def test_local_llm_runtime_configuration(tmp_path):
    obj = runner(tmp_path)
    client = obj._make_llm()
    assert client.cfg.backend == "local"
    assert client.cfg.model == obj.cfg["llm_model"]
    assert client.cfg.max_tokens == 2048
    assert client.cfg.timeout == 600
    assert client.cfg.enable_thinking is False
    assert client.cfg.torch_dtype == "bfloat16"
    assert client.cfg.device_map == "auto"
    assert client.cfg.temperature == .6
    assert client.cfg.top_p == 1.0
    assert client.cfg.repetition_penalty == 1.0
    assert obj.cfg["fast_train"] is True


def test_qwen_path_is_forwarded():
    from scripts.evaluate_baseline import _apply_baseline_model_paths
    kwargs = {}
    _apply_baseline_model_paths("LLMAutoDP", kwargs, SimpleNamespace(qwen3_model_path="/custom/Qwen3-8B"))
    assert kwargs == {"llm_backend": "local", "llm_model": "/custom/Qwen3-8B"}


def test_optional_full_evaluation_removes_search_cap(tmp_path):
    obj = runner(tmp_path)
    obj.cfg["eval_full"] = True
    evaluator = Mock()
    evaluator._get_data_context.return_value = context()
    evaluator.evaluate.return_value = SimpleNamespace(success=True, fitness=.7, metrics={}, error=None)
    llm = Mock()
    llm.chat.return_value = '{"stop":true}'
    obj._make_evaluator = lambda: evaluator
    obj._make_llm = lambda: llm
    obj.run()
    evaluator.set_small_n.assert_called_once_with(0)
    evaluator.set_fast_mode.assert_called_with(False)
