import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.pipeline import Pipeline, PipelineStep
from baselines.DeepPrep.tree_agent import TreeAgent


class SearchNoveltyTest(unittest.TestCase):
    def test_repaired_sequence_and_explicit_sequence_share_execution_key(self):
        repaired = Pipeline([
            PipelineStep('JoinTable', target='interaction', id='repair_op1', prev=['s0', 's1'],
                         params={'method': 'rec', 'how': 'left', 'user_col': 'user_id', 'item_col': 'item_id'}),
            PipelineStep('CreateSequence', target='interaction', id='o1', prev=['repair_op1'],
                         params={'user_col': 'user_id', 'item_col': 'item_id', 'time_col': 'timestamp',
                                 'seq_col': 'item_id_seq', 'max_len': 20}),
        ], sources=[{'id': 's0', 'table': 'interaction'}, {'id': 's1', 'table': 'item'}], train_prev=['o1'])
        explicit = Pipeline.from_yaml(repaired.to_yaml())
        self.assertNotEqual(repaired.steps[1].target, explicit.steps[1].target)
        self.assertEqual(TreeAgent._effective_key(repaired), TreeAgent._effective_key(explicit))
        agent, evaluator = self.agent()
        agent._pipeline_for_node = Mock(side_effect=[repaired, explicit])
        agent._downstream_feedback(SimpleNamespace(node_id='a', accumulated_steps=[PipelineStep('CreateSequence')]))
        agent._downstream_feedback(SimpleNamespace(node_id='b', accumulated_steps=explicit.steps))
        self.assertEqual(evaluator.evaluate_for_agent.call_count, 1)

    def agent(self):
        ctx = DataContext('tabular', 'unit', 'main')
        ctx.tables['main'] = TableContext('main', 'primary', dtypes={'x': 'float64', 'y': 'int64'})
        evaluator = Mock(metric_key='auc')
        evaluator.evaluate_for_agent.return_value = (0.7, {'audit_blob': 'x' * 10000}, None)
        return TreeAgent(Mock(), Mock(), ctx, downstream_evaluator=evaluator), evaluator

    def pipeline(self, op_id='o1', source='s0', method='median'):
        return Pipeline([PipelineStep('HandleMV', params={'method': method}, id=op_id, prev=[source])],
                        sources=[{'id': source, 'table': 'main'}], train_prev=[op_id])

    def test_equivalent_repaired_nodes_score_once_and_feedback_is_compact(self):
        agent, evaluator = self.agent()
        agent._pipeline_for_node = Mock(side_effect=[self.pipeline(), self.pipeline('repair_op1', 's2')])
        first = SimpleNamespace(node_id='a', accumulated_steps=[PipelineStep('HandleMV')])
        second = SimpleNamespace(node_id='b', accumulated_steps=[PipelineStep('HandleMV'), PipelineStep('LabelEncode')])
        agent._downstream_feedback(first)
        feedback = agent._downstream_feedback(second)
        self.assertEqual(evaluator.evaluate_for_agent.call_count, 1)
        self.assertIn('DUPLICATE', feedback)
        self.assertNotIn('audit_blob', feedback)
        self.assertLess(len(feedback), 4000)
        self.assertFalse(list(agent.candidate_scores.values())[1]['evaluated'])

    def test_new_parameters_are_evaluated_after_duplicate_correction(self):
        agent, evaluator = self.agent()
        node = SimpleNamespace(node_id='a', accumulated_steps=[PipelineStep('HandleMV')])
        agent._pipeline_for_node = Mock(return_value=self.pipeline())
        agent._downstream_feedback(node)
        agent._downstream_feedback(node)
        feedback = agent._downstream_feedback(node)
        self.assertIn('last 2 proposals all reused earlier evaluations', feedback)
        different = SimpleNamespace(node_id='b', accumulated_steps=[PipelineStep('HandleMV', params={'method': 'mean'})])
        agent._pipeline_for_node.return_value = self.pipeline(method='mean')
        agent._downstream_feedback(different)
        self.assertEqual(evaluator.evaluate_for_agent.call_count, 2)
        self.assertEqual(agent._duplicate_streak, 0)


if __name__ == '__main__':
    unittest.main()
