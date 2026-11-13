"""Regression for the seed42 AmazonBeauty invalid interaction-side join."""
import random
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from baselines.common.pipeline import Pipeline, PipelineStep, assign_dag_structure, check_pipeline
from baselines.DeepPrep.rec_validation import check_deepprep_pipeline, lookup_reference_errors
from baselines.DeepPrep.tree_agent import chain_to_steps, ChainParseError, TreeAgent
from scripts.test_deepprep_join_protocol import context


@pytest.mark.parametrize("ref", ["$interaction_df", "interaction", "interaction_df"])
def test_primary_side_reference_is_rejected_before_execution(ref):
    with pytest.raises(ChainParseError, match="primary interaction"):
        chain_to_steps(f"JoinTable(method='rec', user_df='{ref}', item_df='$item_df')",
                       context(), random.Random(1), strict_instances=True, validate_rec_lookup=True)


def test_framework_does_not_fill_user_reference():
    steps = chain_to_steps("JoinTable(method='rec', item_df='item')", context(), random.Random(1), strict_instances=True, validate_rec_lookup=True)
    assert steps[0].params == {"method": "rec", "item_df": "$item_df"}


def test_wrong_or_absent_side_roles_are_rejected():
    with pytest.raises(ChainParseError, match="must reference"):
        chain_to_steps("JoinTable(method='rec', user_df='$item_df')", context(), random.Random(1), strict_instances=True, validate_rec_lookup=True)
    with pytest.raises(ChainParseError, match="unavailable"):
        chain_to_steps("JoinTable(method='rec', user_df='$user_df')", context(), random.Random(1), strict_instances=True, validate_rec_lookup=True)


def test_raw_and_serialized_bad_join_both_fail():
    ctx = context()
    pipe = Pipeline([PipelineStep('HandleMV', 'interaction', {'method': 'median', 'action': 'impute'}),
                     PipelineStep('JoinTable', 'interaction', {'method': 'rec', 'user_df': '$interaction_df', 'item_df': '$item_df'})])
    assign_dag_structure(pipe, ctx)
    assert not check_deepprep_pipeline(pipe, ctx, require_mandatory=False).valid
    report = check_deepprep_pipeline(pipe.to_yaml(), ctx, require_mandatory=False)
    assert any('primary branches' in error for error in report.errors)


def test_direct_primary_secondary_edge_fails_even_without_prefix():
    dag = {'dag': {'sources': [{'id':'s0','table':'interaction'}, {'id':'s1','table':'item'}],
        'ops':[{'id':'o1','op':'JoinTable','prev':['s0','s0','s1'],'params':{'method':'rec'}}],
        'train':{'prev':['o1']}}}
    assert not check_deepprep_pipeline(dag, context(), require_mandatory=False).valid


def test_legal_side_join_stays_legal_after_serialization():
    ctx = context()
    steps = chain_to_steps("HandleMV(method='median', action='impute') --> JoinTable(method='rec', item_df='$item_df')",
                           ctx, random.Random(1), strict_instances=True, validate_rec_lookup=True)
    pipe = assign_dag_structure(Pipeline(steps), ctx)
    assert check_deepprep_pipeline(pipe, ctx, require_mandatory=False).valid
    assert check_deepprep_pipeline(pipe.to_yaml(), ctx, require_mandatory=False).valid


def test_sandbox_rejects_primary_ref_before_deduplication(tmp_path):
    from baselines.DeepPrep.sandbox import Sandbox
    sandbox = Sandbox.__new__(Sandbox)
    sandbox.data = SimpleNamespace()
    sandbox.seed = 42
    sandbox.validate_rec_lookup = True
    sandbox._tmpdir = str(tmp_path)
    sandbox._context = lambda: context()
    sandbox.snapshot = lambda: __import__('pickle').dumps(SimpleNamespace())
    result = sandbox.execute_chain([PipelineStep('JoinTable', 'interaction', {'method':'rec', 'user_df':'$interaction_df', 'item_df':'$item_df'})])
    assert not result.success and 'primary interaction' in result.error


def test_best_scored_candidate_survives_later_failed_candidate():
    agent = TreeAgent(Mock(), Mock(), context(), verbose=False)
    agent.tree.add_root(b'root', 'root')
    steps = chain_to_steps("JoinTable(method='rec', item_df='$item_df')", context(), random.Random(1), strict_instances=True, validate_rec_lookup=True)
    node = agent.tree.add_child(steps, b'valid', 'valid')
    from baselines.DeepPrep.tree_node import canonical_path
    path = canonical_path(steps)
    agent.candidate_scores[path] = {'path':path, 'fitness':.8}
    agent.candidate_scores['bad'] = {'path':'bad', 'fitness':None}
    assert agent._fallback_pipeline().steps[0].params['item_df'] == '$item_df'


def test_no_scored_candidate_stops_before_hpo(monkeypatch, tmp_path):
    from scripts import evaluate_baseline as harness
    runner = Mock()
    runner.run.return_value = {'no_valid_scored_candidate':True, 'candidate_errors':['invalid side'], 'agent_log_path':'trace.json'}
    class Runner:
        def __init__(self, **kwargs): pass
        def run(self): return runner.run()
    monkeypatch.setitem(harness.BASELINE_CONFIGS, 'Unit', {'cls_path':'unit', 'kwargs':{}, 'supports_eval_full':False, 'max_len_arg':'max_pipeline_len'})
    monkeypatch.setattr(harness, '_load_class', lambda _: Runner)
    args = SimpleNamespace(max_pipeline_len=None, data_dir=str(tmp_path), model='DIN', seed=42,
                           quiet=True, device='cpu', resume_pipeline=None, final_hpo=True)
    with pytest.raises(RuntimeError, match='skipping final HPO/test'):
        harness._run_baseline_and_eval('amazon_beauty','Unit',args,str(tmp_path))


@pytest.mark.parametrize("generated,reason", [([7,8], "length"), ([7,0], "stop_token")])
def test_generation_diagnostics_distinguish_length_and_stop(generated, reason):
    import torch
    from baselines.DeepPrep.llm_client import LLMClient
    class Inputs(dict):
        def to(self, device): return self
    class Tokenizer:
        eos_token_id = 0
        def apply_chat_template(self, *args, **kwargs): return "prompt"
        def __call__(self, *args, **kwargs): return Inputs(input_ids=torch.tensor([[1,2]]))
        def encode(self, marker, **kwargs): return [10,11]
        def decode(self, ids, **kwargs): return "<plan>unfinished</plan>"
    model = SimpleNamespace(device='cpu', generate=lambda **kwargs: torch.tensor([[1,2]+generated]))
    client = LLMClient(backend='local', model='fixture', max_tokens=2)
    client.attach_local_model(model, Tokenizer())
    client.chat([{'role':'user','content':'go'}])
    assert client.last_generation['finish_reason'] == reason
    assert client.last_generation['generated_tokens'] == 2


def test_format_failure_preserves_raw_response_and_generation():
    from baselines.DeepPrep.sandbox import ObsState, ExecutionResult
    obs = ObsState((1,4), ['user_id','item_id','timestamp','rating'],
                   {'user_id':'object','item_id':'object','timestamp':'int64','rating':'int8'}, [], 0., 'root')
    sandbox = Mock()
    sandbox.reset.return_value = ExecutionResult(True, obs=obs, snapshot=b'root')
    sandbox.restore.return_value = obs
    llm = Mock()
    llm.chat.return_value = '<plan>unfinished</plan>'
    llm.last_generation = {'finish_reason':'length','hit_token_limit':True}
    result = TreeAgent(llm, sandbox, context(), max_explore_turn=1, verbose=False).run()
    response = next(t for t in result.transcript if t['role']=='assistant')
    assert response['content'] == '<plan>unfinished</plan>'
    assert response['generation']['finish_reason'] == 'length'
    assert any('parsed_kind' in t['content'] for t in result.transcript if t['role']=='environment')


@pytest.mark.parametrize("user_ref", ["$user_df", "$user", "user_df", "user"])
def test_legitimate_user_and_item_references_are_preserved(user_ref):
    from baselines.common.data_ctx import TableContext
    ctx = context()
    ctx.tables['user'] = TableContext('user', 'secondary', dtypes={'user_id':'object','age':'int64'})
    ctx.secondary_tables.append('user')
    steps = chain_to_steps(f"JoinTable(method='rec', user_df='{user_ref}', item_df='$item_df')",
                           ctx, random.Random(1), strict_instances=True, validate_rec_lookup=True)
    assert steps[0].params['user_df'] == '$user_df' if user_ref in ('user','user_df') else steps[0].params['user_df'] == user_ref
    assert steps[0].params['item_df'] == '$item_df'
    pipe = assign_dag_structure(Pipeline(steps), ctx)
    assert check_deepprep_pipeline(pipe, ctx, require_mandatory=False).valid
    assert check_deepprep_pipeline(pipe.to_yaml(), ctx, require_mandatory=False).valid


@pytest.mark.parametrize("params", [{"method":"rec", "item_df":"$item_df"},
                                     {"method":"rec", "item_df":"$item_df", "user_df":None}])
def test_item_only_lookup_never_requires_user_table(params):
    from baselines.common.pipeline import validate_step_params
    ctx = context()
    assert 'user' not in ctx.tables
    assert lookup_reference_errors('JoinTable',params,ctx) == []
    pipe = assign_dag_structure(Pipeline([PipelineStep('JoinTable','interaction',params)]),ctx)
    assert check_deepprep_pipeline(pipe,ctx,require_mandatory=False).valid


def test_lineage_gate_is_lookup_specific_not_generic_multi_input(monkeypatch):
    from dataclasses import replace
    from baselines.common.op_spec import CATALOG
    # A placeholder verifies checker scope only, not runtime support for a new op.
    monkeypatch.setitem(CATALOG,'FutureMerge',replace(CATALOG['JoinTable'],op_name='FutureMerge',default_params={}))
    dag = {'dag':{'sources':[{'id':'s0','table':'interaction'}],
        'ops':[{'id':'o1','op':'FutureMerge','prev':['s0','s0']}], 'train':{'prev':['o1']}}}
    report = check_deepprep_pipeline(dag,context(),require_mandatory=False)
    assert not any('primary branches' in error for error in report.errors)


def test_shared_parser_keeps_original_behavior_without_opt_in():
    steps = chain_to_steps("JoinTable(method='rec', user_df='$interaction_df', item_df='$item_df')",
                           context(), random.Random(1), strict_instances=True)
    assert steps[0].params['user_df'] == '$interaction_df'
