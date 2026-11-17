"""DeepPrep final-answer and repeat-stop regressions, without LLM weights."""
from unittest.mock import Mock
import pytest
from baselines.DeepPrep.tree_agent import TreeAgent,parse_response
from baselines.DeepPrep.sandbox import ObsState,ExecutionResult
from baselines.DeepPrep.test_deepprep_protocol import _ctx

PATH = 'HandleMV(method="median", action="impute") --> LabelEncode(cols=None)'
OTHER = 'HandleMV(method="mean", action="impute") --> LabelEncode(cols=None)'

class Sandbox:
    def __init__(self):
        self.obs=ObsState((3,2),['x','target'],{'x':'float64','target':'int64'},[],0.,'state',tables={'main':{'columns':['x','target']}})
        self.steps=0
    def reset(self):return ExecutionResult(True,obs=self.obs,snapshot=b'root')
    def restore(self,snapshot):return self.obs
    def execute_chain(self,steps):
        self.steps+=1
        return ExecutionResult(True,obs=self.obs,snapshot=str(self.steps).encode())

def run(responses,scores=None):
    llm=Mock();llm.chat.side_effect=responses
    evaluator=Mock(metric_key='auc')
    evaluator.evaluate_for_agent.side_effect=scores or [(.7,{'auc':.7},None)]
    agent=TreeAgent(llm,Sandbox(),_ctx(),downstream_evaluator=evaluator,
                    max_explore_turn=len(responses),verbose=False)
    return agent,agent.run(),evaluator

@pytest.mark.parametrize('tag',['answer','solution'])
def test_explicit_successful_submission_stops_immediately(tag):
    agent,result,ev=run([PATH+' </execute>',f'<{tag}>{PATH}</{tag}>','unused'])
    assert result.success and result.n_turns==2 and result.stop_reason=='model_answer'
    assert ev.evaluate_for_agent.call_count==1
    assert agent.search_stats['candidate_coverage']=='single_candidate'
    assert agent.search_stats['submitted_path']==agent.search_stats['selected_path']

def test_repeat_warns_once_then_stops_without_retraining():
    agent,result,ev=run([PATH+' </execute>']*6)
    assert result.success and result.n_turns==3 and result.stop_reason=='repeat_stagnation'
    assert ev.evaluate_for_agent.call_count==1
    assert agent.search_stats['repeated_paths']==2 and agent.search_stats['cache_hits']==2
    assert any('Submit <answer>' in t['content'] for t in result.transcript)

def test_failed_scoring_cannot_be_submitted_as_success():
    agent,result,ev=run([PATH+' </execute>',f'<answer>{PATH}</answer>'],[(None,{},'training failed')])
    assert not result.success and result.stop_reason!='model_answer'
    assert ev.evaluate_for_agent.call_count==1

def test_best_scored_candidate_survives_answer_for_weaker_path():
    agent,result,ev=run([PATH+' </execute>',OTHER+' </execute>',f'<answer>{PATH}</answer>'],
                        [(.6,{'auc':.6},None),(.8,{'auc':.8},None)])
    assert result.stop_reason=='model_answer' and result.pipeline.steps[0].params['method']=='mean'
    assert agent.search_stats['candidate_coverage']=='multiple_candidates'
    assert agent.search_stats['submitted_path']!=agent.search_stats['selected_path']
    assert ev.evaluate_for_agent.call_count==2

def test_new_path_resets_consecutive_repeat_count():
    agent,result,ev=run([PATH+' </execute>',PATH+' </execute>',OTHER+' </execute>',
                        OTHER+' </execute>',f'<answer>{OTHER}</answer>'],
                        [(.6,{'auc':.6},None),(.8,{'auc':.8},None)])
    assert result.stop_reason=='model_answer' and result.n_turns==5
    assert ev.evaluate_for_agent.call_count==2

def test_aliases_are_deepprep_opt_in_only():
    assert parse_response('<solution>root</solution>').kind=='solution'
    assert parse_response('<solution>root</solution>',strict_native=True).kind=='answer'
    assert parse_response('<operator>HandleMV()</operator>',strict_native=True).kind=='expand'

def test_repeat_policy_is_validated():
    with pytest.raises(ValueError,match='>= 2'):
        TreeAgent(Mock(),Mock(),_ctx(),max_repeated_paths=1)


def test_illegal_final_path_does_not_end_successfully():
    agent,result,ev=run(['<answer>root</answer>'])
    assert not result.success and result.stop_reason!='model_answer'
    ev.evaluate_for_agent.assert_not_called()


def test_downloaded_repeats_stop_after_one_warning():
    import json
    from pathlib import Path
    from scripts.test_deepprep_join_protocol import context
    from baselines.DeepPrep.rec_validation import check_deepprep_pipeline
    p=Path(__file__).resolve().parents[1]/'logs/deepprep_amazon/amazon_beauty_DIN/agent_log.json'
    if not p.exists():pytest.skip('Optional downloaded user trace')
    data=json.loads(p.read_text(encoding='utf-8'))
    responses=[m['content'] for m in data['transcript'] if m['role']=='assistant']
    llm=Mock();llm.chat.side_effect=responses
    ev=Mock(metric_key='auc');ev.evaluate_for_agent.return_value=(.62,{'auc':.62},None)
    ctx=context()
    agent=TreeAgent(llm,Sandbox(),ctx,downstream_evaluator=ev,max_explore_turn=6,verbose=False)
    # Real pipeline contracts; downstream/schema data execution is simulated.
    agent._answer_errors=lambda node: check_deepprep_pipeline(agent._pipeline_for_node(node),ctx).errors
    result=agent.run()
    assert result.success and result.stop_reason=='repeat_stagnation' and result.n_turns==4
    assert ev.evaluate_for_agent.call_count==1
