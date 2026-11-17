"""Regressions derived from exp1_10_30_llm failures; no checkpoint required."""
import os
import sys
import random
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import torch

from baselines.DeepPrep.gpu_config import resolve_device, llm_placement
from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.pipeline import Pipeline, PipelineStep
from baselines.DeepPrep.candidate_contracts import assign_dag_structure, parameter_errors as validate_step_params, frame_errors
from baselines.BAT.node import MCTSNode, MCTSNodeType as T
from baselines.BAT.mcts import MCTSSolver
from baselines.BAT.reward import BATRewardConfig
from baselines.DeepPrep.tree_agent import chain_to_steps, ChainParseError, TreeAgent
from baselines.DeepPrep.llm_client import LLMClient
from dppbench.operators.integration.rename_column import RenameColumn
from dppbench.operators.integration.join_table import JoinTable


def context():
    return DataContext('tabular', 'fixture', 'main', tables={
        'main': TableContext('main', 'primary', shape=(2, 3),
                             dtypes={'id': 'int64', 'x': 'float64', 'target': 'int64'})})


def legal_steps():
    return [PipelineStep('HandleMV', 'both', {}), PipelineStep('LabelEncode', 'both', {})]


@pytest.mark.parametrize('visible,gpu,expected', [('0,1', 0, '0,1'), ('1,0', 0, '1,0'), (None, 1, '1')])
def test_visibility_preserves_external_mapping(visible, gpu, expected):
    env = {} if visible is None else {'CUDA_VISIBLE_DEVICES': visible}
    props = SimpleNamespace(name='fake', total_memory=24 * 1024**3)
    with patch.dict(os.environ, env, clear=True), patch.object(torch.cuda, 'is_initialized', return_value=False), \
            patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'device_count', return_value=2), \
            patch.object(torch.cuda, 'get_device_properties', return_value=props):
        assert resolve_device(gpu) == 'cuda:0'
        assert os.environ['CUDA_VISIBLE_DEVICES'] == expected


def test_visibility_rejects_late_changes_and_unavailable_gpu():
    with patch.object(torch.cuda, 'is_initialized', return_value=True):
        with pytest.raises(RuntimeError, match='before CUDA'):
            resolve_device(0, '0,1')
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(torch.cuda, 'is_available', return_value=False):
        with pytest.raises(RuntimeError, match='unavailable'):
            resolve_device(0)


def test_llm_per_device_limits_use_logical_indices():
    with patch.dict(os.environ, {'DPPBENCH_LLM_DEVICE_MAP': 'balanced_low_0',
                                'DPPBENCH_LLM_MAX_MEMORY': '{"0":"12GiB","1":"18GiB"}'}):
        assert llm_placement() == {'device_map': 'balanced_low_0', 'max_memory': {0: '12GiB', 1: '18GiB'}}


def test_failed_rollout_expanded_unvisited_node_is_not_selected_for_expansion():
    solver = MCTSSolver(object(), context(), object(), object(), verbose=False)
    root = MCTSNode(T.ROOT, expanded=True)
    branch = MCTSNode(T.SCHEMA_MATCH, parent=root, expanded=True, N=0)
    dead = MCTSNode(T.TRANSFORMATION, parent=branch, expanded=True, exhausted=True)
    leaf = MCTSNode(T.IDENTIFY_COLUMN_FUNCTIONS, parent=branch)
    branch.children = [dead, leaf]
    root.children = [branch]
    assert solver.select(root) is leaf
    leaf.expanded = leaf.exhausted = True
    assert solver.select(root).exhausted


def test_simulation_reuses_children_after_partial_rollout():
    solver = MCTSSolver(object(), context(), object(), object(), verbose=False)
    root = MCTSNode(T.SCHEMA_MATCH, expanded=True)
    end = MCTSNode(T.END, parent=root)
    root.children = [end]
    with patch.object(solver, 'expand', side_effect=AssertionError('duplicate expansion')):
        assert solver.simulate(root) is end


def test_invalid_terminal_cannot_be_selected_even_with_high_reward():
    reward = SimpleNamespace(config=BATRewardConfig(use_downstream=True))
    solver = MCTSSolver(object(), context(), object(), reward, verbose=False)
    end = MCTSNode(T.END, final_pipeline_steps=legal_steps(), reward_value=1,
                   exec_error='JoinTable failed', downstream_fitness=.9)
    assert not solver._candidate_valid(end)
    end.exec_error = None
    end.downstream_fitness = None
    assert not solver._candidate_valid(end)
    end.downstream_fitness = -4.3
    end.reward_value = 0
    assert solver._candidate_valid(end)


def test_multiple_aggregate_sources_keep_distinct_names_and_row_alignment():
    ctx = context()
    ctx.secondary_tables = ['history_a', 'history_b']
    ctx.tables.update({name: TableContext(name, 'secondary', dtypes={'id': 'int64', 'v': 'float64'})
                       for name in ctx.secondary_tables})
    pipe = Pipeline(steps=[PipelineStep('JoinTable', 'main', {'method': 'agg', 'key_col': 'id', 'aux_df': '$' + name + '_df'})
                           for name in ctx.secondary_tables])
    assign_dag_structure(pipe, ctx)
    frame = pd.DataFrame({'id': [2, 1], 'target': [0, 1]})
    aux = pd.DataFrame({'id': [1, 1, 2], 'v': [2., 4., 9.]})
    for step in pipe.steps:
        params = dict(step.params, aux_df=aux)
        frame = JoinTable(**params).transform(frame)
    assert frame.id.tolist() == [2, 1]
    assert frame.target.tolist() == [0, 1]
    assert frame.columns.is_unique
    assert frame.history_a___v_mean.tolist() == [9., 3.]
    assert 'history_b___v_mean' in frame


def test_llm_contracts_reject_bad_proposals_without_changing_operators():
    ctx = context()
    frame = pd.DataFrame({'id': [1], 'x': [2.], 'target': [0]})
    assert frame_errors([PipelineStep('RenameColumn', 'both', {'rename_map': {'x': 'id'}})], ctx, frame)
    assert frame_errors([PipelineStep('RenameColumn', 'both', {'rename_map': {'absent': 'y'}})], ctx, frame)
    assert frame_errors([PipelineStep('JoinTable', 'both', {'method': 'agg', 'key_col': 'missing'})], ctx, frame)
    assert validate_step_params('JoinTable', {'agg_funcs': {'v': {'mean': 'mean'}}}, ctx)
    assert validate_step_params('DropColumns', {'cols': ['target']}, ctx)
    assert validate_step_params('RenameColumn', {'rename_map': {'id': 'other'}}, ctx)
    # Shared operators retain their pre-chat semantics.
    assert not RenameColumn({'x': 'id'}).transform(frame).columns.is_unique
    pd.testing.assert_frame_equal(RenameColumn({'absent': 'y'}).transform(frame), frame)
    pd.testing.assert_frame_equal(JoinTable(method='agg', key_col='missing', aux_df=frame).transform(frame), frame)


def test_deepprep_never_accepts_trailing_arrow_or_nested_aggregation():
    with pytest.raises(ChainParseError, match='Incomplete'):
        chain_to_steps('HandleMV(method="median") -->', context(), random.Random(0), strict_instances=True)
    with pytest.raises(ChainParseError, match='agg_funcs'):
        chain_to_steps('JoinTable(method="agg", key_col="id", agg_funcs={"v":{"mean":"mean"}})',
                       context(), random.Random(0), strict_instances=True)


def test_deepprep_oom_is_not_retried_as_protocol_failure():
    sandbox = Mock()
    sandbox.reset.return_value = SimpleNamespace(success=True, snapshot=b'root', obs=SimpleNamespace(text='data'))
    llm = Mock()
    llm.chat.side_effect = torch.cuda.OutOfMemoryError('CUDA out of memory')
    agent = TreeAgent(llm, sandbox, context(), verbose=False)
    with patch.object(agent, '_initial_messages', return_value=[{'role': 'system', 'content': 'test'}]):
        with pytest.raises(torch.cuda.OutOfMemoryError):
            agent.run()
    assert llm.chat.call_count == 1
    assert agent.search_stats['llm_call_failures'] == 1
    assert agent.search_stats['response_parse_failures'] == 0
    assert agent.transcript[-1]['stage'] == 'llm_generation'


def test_loading_oom_preserves_exception_type():
    client = LLMClient(backend='local', model='fake')
    with patch.object(client, '_load_local_model', side_effect=torch.cuda.OutOfMemoryError('CUDA out of memory')):
        with pytest.raises(torch.cuda.OutOfMemoryError):
            client._ensure_local_model()


def test_llm_loader_forwards_placement_and_records_actual_map():
    model = Mock()
    model.hf_device_map = {'embed': 0, 'layers.1': 1}
    factory = Mock()
    factory.from_pretrained.return_value = model
    tokenizer = Mock()
    tokenizer.from_pretrained.return_value.chat_template = 'template'
    module = SimpleNamespace(AutoModelForCausalLM=factory, AutoTokenizer=tokenizer)
    client = LLMClient(backend='local', model='fake')
    with patch.dict(sys.modules, {'transformers': module}), \
            patch.dict(os.environ, {'DPPBENCH_LLM_DEVICE_MAP': 'balanced_low_0',
                                    'DPPBENCH_LLM_MAX_MEMORY': '{"0":"12GiB","1":"18GiB"}'}), \
            patch('baselines.DeepPrep.llm_client.require_accelerate'), \
            patch('baselines.DeepPrep.llm_client.fp8_quantization_config', return_value=None), \
            patch('baselines.DeepPrep.llm_client.local_torch_dtype', return_value=torch.float32), \
            patch('baselines.DeepPrep.llm_client.model_dtype_kwargs', return_value={'torch_dtype': torch.float32}):
        client._load_local_model()
    kwargs = factory.from_pretrained.call_args.kwargs
    assert kwargs['device_map'] == 'balanced_low_0'
    assert kwargs['max_memory'] == {0: '12GiB', 1: '18GiB'}
    assert client._actual_device_map == model.hf_device_map


def test_right_key_cardinality_rejected_before_execution():
    ctx = context()
    ctx.secondary_tables = ['history']
    ctx.tables['history'] = TableContext('history', 'secondary', dtypes={'id': 'int64'},
                                         join_key_unique={'id': False})
    params = {'method': 'key', 'aux_df': '$history_df', 'key_col': 'id'}
    assert any('repeated' in e for e in validate_step_params('JoinTable', params, ctx))
    params['method'] = 'agg'
    assert not validate_step_params('JoinTable', params, ctx)


def test_graph_operator_catalog_and_contract_are_available():
    from baselines.BAT.operator_catalog import format_op_descriptions
    ctx = context()
    ctx.task_type = 'graph'
    assert 'HandleMV' in format_op_descriptions('graph')
    pipe = Pipeline(steps=legal_steps())
    assign_dag_structure(pipe, ctx)
    from baselines.common.pipeline import check_pipeline
    assert check_pipeline(pipe, ctx).valid


def test_repeated_failed_deepprep_expansion_executes_once():
    from scripts.test_deepprep_stopping import Sandbox
    from baselines.DeepPrep.sandbox import ExecutionResult
    sandbox = Sandbox()
    sandbox.execute_chain = Mock(return_value=ExecutionResult(False, error='invalid candidate feature type'))
    llm = Mock()
    llm.chat.return_value = 'HandleMV(method="median", action="impute") </execute>'
    agent = TreeAgent(llm, sandbox, context(), max_explore_turn=3, verbose=False)
    result = agent.run()
    assert not result.success
    assert sandbox.execute_chain.call_count == 1
    assert agent.search_stats['execution_failure_cache_hits'] == 2


def test_llm_gpu_budget_does_not_change_common_budget():
    from baselines.common.budget import Budget
    from baselines.DeepPrep.resource_budget import LLMResourceBudget
    unit = 1024**2
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'device_count', return_value=2), \
            patch.object(torch.cuda, 'max_memory_allocated', side_effect=lambda i=0: [3, 7][i] * unit), \
            patch.object(torch.cuda, 'max_memory_reserved', side_effect=lambda i=0: [5, 9][i] * unit):
        shared, llm = Budget(), LLMResourceBudget()
        shared.violations()
        llm.violations()
        assert shared.peak_gpu_memory_mb == 5
        assert llm.peak_gpu_memory_mb == 14


def test_nonllm_entry_keeps_legacy_device_selection():
    from scripts.evaluate_baseline import _resolve_device
    props = SimpleNamespace(name='fake', total_memory=24 * 1024**3)
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'}), \
            patch.object(torch.cuda, 'is_available', return_value=True):
        assert _resolve_device(1) == 'cuda:0'
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'}), \
            patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'device_count', return_value=2), \
            patch.object(torch.cuda, 'get_device_properties', return_value=props):
        assert _resolve_device(0, llm=True) == 'cuda:0'
        assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'


