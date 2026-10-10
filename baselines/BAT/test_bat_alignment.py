from __future__ import annotations

import math
import random
import unittest
from unittest.mock import Mock, patch
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

from baselines.BAT.action import (
    ActionContext,
    EndAction,
    _column_similarity,
    _pipeline_from_response,
    _observation_text,
    _pipeline_key,
    TransformationAction,
    TransformationRevisionAction,
    evaluate_downstream_feedback,
    get_valid_action_space_for_node,
)
from baselines.BAT.operator_catalog import format_op_descriptions, format_mandatory_examples
from baselines.BAT.prompts import render_schema_match
from baselines.BAT.llm_client import BATLLMClient, BATSearchTimeLimit
from baselines.DeepPrep.llm_client import LLMClient
from baselines.DeepPrep.sandbox import Sandbox
from baselines.common.budget import BudgetExceededError
from baselines.BAT.mcts import MCTSSolver
from baselines.BAT.node import MCTSNode, MCTSNodeType
from baselines.BAT.reward import BATReward, BATRewardConfig
from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.pipeline import PipelineStep


def _ctx() -> DataContext:
    return DataContext(
        task_type="tabular",
        data_name="unit",
        primary_table="main",
        tables={
            "main": TableContext(
                name="main",
                role="primary",
                shape=(2, 2),
                dtypes={"x": "float64", "target": "int64"},
            )
        },
    )


class BATAlignmentTest(unittest.TestCase):
    def test_tabular_examples_cover_all_object_columns(self):
        examples = format_mandatory_examples(_ctx())
        self.assertIn('LabelEncode(cols=None)', examples)
        self.assertIn('cols=None', examples)
        import pandas as pd
        from dppbench.dataset import TabularData
        operator = TabularData('unit')._load_op('LabelEncode')
        frame = pd.DataFrame({'ProductCD': ['W', 'C'], 'M1': ['T', 'F'],
                              'M9': ['F', None], 'value': [1.0, 2.0]})
        encoded = operator(cols=None).transform(frame)
        self.assertFalse(any(dtype == object for dtype in encoded.dtypes))

    def test_budget_reserve_stops_before_calling_model(self):
        with tempfile.TemporaryDirectory() as directory:
            client = BATLLMClient(trace_path=str(Path(directory) / 'trace.jsonl'),
                                  backend='local', model='fake')
            with patch('baselines.BAT.llm_client.get_active_budget',
                       return_value=SimpleNamespace(max_seconds=1200, elapsed_seconds=1100)), patch.object(
                LLMClient, 'chat', return_value='answer',
            ) as chat:
                with self.assertRaises(BATSearchTimeLimit):
                    client.chat([])
                chat.assert_not_called()

    def test_time_reserve_keeps_evaluated_terminal_candidate(self):
        solver = MCTSSolver(object(), _ctx(), object(), object(), verbose=False)

        def expand(root):
            child = MCTSNode(MCTSNodeType.TRANSFORMATION, parent=root)
            child.children.append(MCTSNode(
                MCTSNodeType.END, parent=child, reward_value=0.9,
                final_pipeline_steps=[PipelineStep('LabelEncode', 'both', {'cols': None})],
            ))
            root.children.append(child)
            return [child]

        with patch.object(solver, 'expand', side_effect=expand), patch.object(
            solver, 'simulate', side_effect=BATSearchTimeLimit('reserve'),
        ):
            result = solver.solve()
        self.assertEqual(result.best_node.reward_value, 0.9)
        self.assertEqual(len(result.selected_nodes), 1)

    def test_object_pipeline_accepts_logged_name_kwargs_format(self):
        response = json.dumps({'pipeline': [
            {'name': 'LabelEncode', 'kwargs': {'cols': None}},
            {'name': 'Terminate', 'params': {}},
        ]})
        self.assertEqual(_pipeline_from_response(response),
                         'LabelEncode(cols=None) --> Terminate()')

    def test_training_failure_reaches_revision_even_when_schema_matches(self):
        llm = Mock()
        llm.chat.return_value = '{"pipeline": "HandleMV --> LabelEncode --> Terminate"}'
        sandbox = Mock()
        sandbox._observe.return_value.text = 'source'
        evaluator = Mock()
        evaluator.evaluate_for_agent.return_value = (None, {}, 'M1: object')
        context = ActionContext(llm, _ctx(), sandbox, random.Random(1),
                                downstream_evaluator=evaluator)
        parent = MCTSNode(MCTSNodeType.TRANSFORMATION, columns_match=True,
                          pipeline_steps=[PipelineStep('LabelEncode', 'both', {})])
        with patch('baselines.BAT.action.check_pipeline', return_value=SimpleNamespace(valid=True)):
            evaluate_downstream_feedback(parent, context)
        self.assertIn(TransformationRevisionAction, get_valid_action_space_for_node(parent))
        with patch('baselines.BAT.action._score_steps_structurally', return_value=([], False, 0.0, 'test')):
            TransformationRevisionAction().create_children_nodes(parent, context)
        self.assertIn('M1: object', llm.chat.call_args.args[0][-1]['content'])

    def test_pipeline_dedup_distinguishes_parameters(self):
        mean = [PipelineStep('HandleMV', 'both', {'method': 'mean'})]
        median = [PipelineStep('HandleMV', 'both', {'method': 'median'})]
        self.assertNotEqual(_pipeline_key(mean), _pipeline_key(median))
        self.assertEqual(_pipeline_key(mean), _pipeline_key(list(mean)))

    def test_later_transformation_sees_previous_task_pipeline(self):
        llm = Mock()
        llm.chat.return_value = '{"pipeline": "HandleMV --> LabelEncode --> Terminate"}'
        sandbox = Mock()
        sandbox._observe.return_value.text = 'source'
        context = ActionContext(
            llm, _ctx(), sandbox, random.Random(1),
            previous_pipeline_texts=["HandleMV(method='mean') --> LabelEncode()"],
        )
        with patch('baselines.BAT.action._score_steps_structurally',
                   return_value=([], False, 0.0, 'test')):
            TransformationAction().create_children_nodes(MCTSNode(MCTSNodeType.ROOT), context)
        prompt = llm.chat.call_args.args[0][-1]['content']
        self.assertIn("HandleMV(method='mean') --> LabelEncode()", prompt)
        self.assertIn('do not add irrelevant operators solely for novelty', prompt)

    def test_sandbox_loads_frozen_data_before_sampling(self):
        data = Mock()
        events = []
        data.load_data.side_effect = lambda: events.append('load')
        data.load_std_test_frozen.side_effect = lambda: events.append('frozen')
        with patch('baselines.DeepPrep.sandbox.TrainingExecutor') as executor, patch(
            'baselines.DeepPrep.sandbox.subsample_data_inplace',
            side_effect=lambda *args, **kwargs: events.append('sample'),
        ), patch('baselines.DeepPrep.sandbox.get_data_ctx', return_value=_ctx()), patch.object(
            Sandbox, 'snapshot', return_value=b'snapshot',
        ), patch.object(Sandbox, '_observe', return_value=Mock()):
            executor.return_value.task_type = 'tabular'
            executor.return_value._make_configured_data_instance.return_value = data
            sandbox = Sandbox('task', 'unit', small_n=128)
            try:
                sandbox.reset()
            finally:
                sandbox.cleanup()
        self.assertEqual(events, ['load', 'frozen', 'sample'])

    def test_wide_observation_retains_schema_and_bounds_sample_values(self):
        columns = [f"col_{i}" for i in range(434)]
        profile = {
            "shape": (128, 434), "null_ratio": 0.1,
            "columns": columns, "dtypes": {col: "float64" for col in columns},
            "head": [{col: i for i, col in enumerate(columns)} for _ in range(5)],
        }
        obs = SimpleNamespace(text="x" * 50000, tables={"main": profile})
        compact = _observation_text(obs)
        self.assertIn("col_433", compact)
        self.assertIn("col_0", compact)
        self.assertIn("sample_rows_first_20_columns", compact)
        self.assertLess(len(compact), 12000)

    def test_wide_schema_prompt_only_requests_non_identity_matches(self):
        prompt = render_schema_match(
            task_type="tabular", data_name="wide", summary_text="431 columns",
            obs_text="source columns", expected_target_columns=[f"c{i}" for i in range(431)],
        )
        self.assertIn("Direct same-name columns are implicit identity matches", prompt)
        self.assertNotIn("c430", prompt)

    def test_logged_object_list_pipeline_uses_strict_operator_calls(self):
        response = json.dumps({"pipeline": [
            {"operator": "JoinTable", "params": {"method": "rec", "item_df": "$item_df"}},
            {"operator": "CreateSequence", "params": {"user_col": "user_id", "item_col": "item_id"}},
            {"operator": "Terminate", "params": {}},
        ]})
        chain = _pipeline_from_response(response)
        self.assertEqual(
            chain,
            "JoinTable(method='rec', item_df='$item_df') --> "
            "CreateSequence(user_col='user_id', item_col='item_id') --> Terminate()",
        )
        self.assertIsNone(_pipeline_from_response(json.dumps({"pipeline": [{"operator": "JoinTable", "params": []}]})))

    def test_catalog_exposes_real_constructor_keywords(self):
        catalog = format_op_descriptions("rec")
        self.assertIn("accepted_keywords=[item_col, max_len, seq_col, time_col, user_col]", catalog)
        self.assertIn("accepted_keywords=[rename_map]", catalog)
        rec = DataContext(
            task_type="rec", data_name="unit", primary_table="interaction",
            secondary_tables=["item"],
            tables={
                "interaction": TableContext("interaction", "primary", dtypes={
                    "user_id": "int64", "item_id": "int64", "timestamp": "int64", "rating": "float64",
                }),
                "item": TableContext("item", "secondary", dtypes={"item_id": "int64"}),
            },
        )
        examples = format_mandatory_examples(rec)
        self.assertIn("CreateSequence(", examples)
        self.assertIn("JoinTable(", examples)
        self.assertIn("seq_col='item_id_seq'", examples)

    def test_revision_sees_unparsed_candidate_and_error(self):
        llm = Mock()
        llm.chat.return_value = '{"pipeline": "HandleMV --> LabelEncode --> Terminate"}'
        sandbox = Mock()
        sandbox._observe.return_value.text = 'source'
        context = ActionContext(llm, _ctx(), sandbox, random.Random(1))
        parent = MCTSNode(MCTSNodeType.TRANSFORMATION, pipeline_text="RenameColumn(column_mapping={})",
                          exec_error="RenameColumn has unknown params: ['column_mapping']")
        with patch('baselines.BAT.action._score_steps_structurally', return_value=([], False, 0.0, 'test')):
            TransformationRevisionAction().create_children_nodes(parent, context)
        prompt = llm.chat.call_args.args[0][-1]['content']
        self.assertIn("RenameColumn(column_mapping={})", prompt)
        self.assertIn("rename_map", prompt)
        self.assertIn("column_mapping", prompt)

    def test_qwen_wrappers_and_truncated_reasoning(self):
        payload = '{"pipeline": "HandleMV --> LabelEncode --> Terminate"}'
        self.assertEqual(
            _pipeline_from_response('<think>draft</think>' + payload + '<|im_end|>'),
            'HandleMV --> LabelEncode --> Terminate',
        )
        self.assertIsNone(_pipeline_from_response('<think>' + payload))
        self.assertIsNone(_pipeline_from_response('{"pipeline": "HandleMV'))

    def test_call_failure_does_not_create_empty_candidate_or_repair(self):
        for error in (RuntimeError('model failed'), BudgetExceededError('limit')):
            llm = Mock()
            llm.chat.side_effect = error
            sandbox = Mock()
            sandbox._observe.return_value.text = 'source'
            context = ActionContext(llm, _ctx(), sandbox, random.Random(1))
            with self.assertRaises(type(error)):
                TransformationAction().create_children_nodes(
                    MCTSNode(MCTSNodeType.ROOT), context,
                )
            self.assertEqual(llm.chat.call_count, 1)

    def test_mcts_propagates_budget_failure(self):
        solver = MCTSSolver(object(), _ctx(), object(), object(), verbose=False)
        with patch.object(solver, '_restore_node_state', return_value=True), patch(
            'baselines.BAT.action.SchemaMatchAction.create_children_nodes',
            side_effect=BudgetExceededError('limit'),
        ):
            with self.assertRaises(BudgetExceededError):
                solver.solve()

    def test_mcts_keeps_completed_candidate_when_next_rollout_would_overrun(self):
        reward = Mock()
        reward.score.return_value = (1.0, {})
        solver = MCTSSolver(object(), _ctx(), object(), reward,
                            max_rollout_steps=2, verbose=False)

        def expand(root):
            child = MCTSNode(MCTSNodeType.END, parent=root,
                             final_pipeline_steps=[PipelineStep("HandleMV", "both", {})])
            root.children.append(child)
            return [child]

        with patch.object(solver, 'expand', side_effect=expand) as expansion, patch(
            'baselines.BAT.mcts.get_active_budget',
            return_value=SimpleNamespace(max_seconds=600, elapsed_seconds=500),
        ), patch('baselines.BAT.mcts.time.monotonic', side_effect=[0, 200]):
            result = solver.solve()
        self.assertEqual(result.n_rollouts, 1)
        self.assertEqual(expansion.call_count, 1)
        self.assertEqual([s.op for s in result.best_node.final_pipeline_steps], ["HandleMV"])

    def test_repeated_terminal_pipeline_is_not_a_second_best_path(self):
        reward = Mock()
        reward.score.return_value = (1.0, {})
        solver = MCTSSolver(object(), _ctx(), object(), reward,
                            max_rollout_steps=2, early_stop_n_paths=2, verbose=False)

        def expand(root):
            return [MCTSNode(MCTSNodeType.END, parent=root,
                             final_pipeline_steps=[PipelineStep('HandleMV', 'both', {'method': 'mean'})])]

        with patch.object(solver, 'select', side_effect=lambda root: root), patch.object(
            solver, 'expand', side_effect=expand,
        ):
            result = solver.solve()
        self.assertEqual(result.n_rollouts, 2)
        self.assertEqual(len(result.selected_nodes), 1)
        self.assertEqual(len(solver.previous_pipeline_texts), 1)

    def test_trace_records_success_and_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.jsonl'
            client = BATLLMClient(trace_path=str(path), backend='local', model='fake',
                                  enable_thinking=False)
            self.assertFalse(client.cfg.enable_thinking)
            self.assertIsNone(LLMClient(backend='local', model='fake').cfg.enable_thinking)
            with patch.object(LLMClient, 'chat', return_value='answer'):
                self.assertEqual(client.chat([]), 'answer')
            with patch.object(LLMClient, 'chat', side_effect=BudgetExceededError('limit')):
                with self.assertRaises(BudgetExceededError):
                    client.chat([])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0]['response'], 'answer')
            self.assertIn('BudgetExceededError', rows[1]['error'])
            self.assertIsNone(rows[1]['response'])

    def test_column_similarity_is_official_target_recall(self):
        self.assertEqual(_column_similarity(["a", "b", "extra"], ["a", "b"]), 1.0)
        self.assertEqual(_column_similarity(["a"], ["a", "b"]), 0.5)

    def test_default_reward_includes_downstream_auc(self):
        cfg = BATRewardConfig()
        self.assertEqual(cfg.alpha, 1.0)
        self.assertEqual(cfg.beta, 1.0)
        self.assertTrue(cfg.use_downstream)

    def test_sampled_auc_changes_terminal_reward_and_failure_is_zero(self):
        sandbox = Mock()
        sandbox._initial_snapshot = b'initial'
        sandbox.execute_chain.return_value.success = True
        evaluator = Mock()
        reward = BATReward(_ctx(), sandbox, downstream_evaluator=evaluator)
        with patch('baselines.BAT.reward.expected_target_columns', return_value=['x']), patch(
            'baselines.BAT.reward._sandbox_columns', return_value=['x'],
        ):
            for auc, error, expected in ((0.8, None, 0.9), (0.6, None, 0.8),
                                         (None, 'training failed', 0.0),
                                         (float('nan'), None, 0.0)):
                evaluator.evaluate_for_agent.return_value = (auc, {}, error)
                node = MCTSNode(MCTSNodeType.END,
                                final_pipeline_steps=[PipelineStep('HandleMV', 'both', {})])
                value, breakdown = reward.score(node)
                self.assertAlmostEqual(value, expected)
                if expected == 0.0:
                    self.assertIsNotNone(breakdown['downstream_error'])

    def test_end_action_does_not_repair_candidate(self):
        parent = MCTSNode(
            node_type=MCTSNodeType.TRANSFORMATION,
            pipeline_steps=[PipelineStep("LabelEncode", "both", {})],
        )
        context = ActionContext(
            llm=object(),
            ctx=_ctx(),
            sandbox=object(),
            rng=random.Random(1),
            max_chain_len=6,
        )
        end = EndAction().create_children_nodes(parent, context)[0]
        self.assertEqual([step.op for step in end.final_pipeline_steps], ["LabelEncode"])
        self.assertIn("HandleMV", end.exec_error or "")

    def test_official_json_pipeline_protocol(self):
        response = '''```json
        {"chain_of_thought_reasoning": "clean then encode",
         "pipeline": "HandleMV --> LabelEncode --> Terminate"}
        ```'''
        self.assertEqual(
            _pipeline_from_response(response),
            "HandleMV --> LabelEncode --> Terminate",
        )

    def test_ucb_and_unvisited_selection_match_official(self):
        solver = MCTSSolver(
            llm=object(), ctx=_ctx(), sandbox=object(), reward_model=object(),
            exploration_constant=1.0, verbose=False,
        )
        root = MCTSNode(MCTSNodeType.ROOT, N=4)
        first = MCTSNode(MCTSNodeType.SCHEMA_MATCH, parent=root, N=1, Q=0.0)
        unvisited = MCTSNode(MCTSNodeType.TRANSFORMATION, parent=root, N=0, Q=0.0)
        root.children = [first, unvisited]
        self.assertIs(solver.select(root), unvisited)
        self.assertAlmostEqual(
            solver._ucb_score(first, root.N),
            math.sqrt(math.log(4)),
        )


if __name__ == "__main__":
    unittest.main()
