from __future__ import annotations

import random
import os
import unittest

from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.pipeline import PipelineStep
from baselines.DeepPrep.llm_client import LLMClient
from baselines.DeepPrep.operator_catalog import mandatory_ops_for_context
from baselines.DeepPrep.sandbox import ExecutionResult, ObsState
from baselines.DeepPrep.target_schema import TargetSchema
from baselines.DeepPrep.tree_agent import TreeAgent, chain_to_steps, parse_response
from baselines.DeepPrep.tree_node import SearchTree, canonical_path


def _ctx() -> DataContext:
    return DataContext(
        task_type="tabular",
        data_name="unit",
        primary_table="main",
        tables={
            "main": TableContext(
                name="main",
                role="primary",
                shape=(3, 2),
                dtypes={"x": "float64", "target": "int64"},
            )
        },
    )


class DeepPrepProtocolTest(unittest.TestCase):
    def test_deepprep_generation_settings_do_not_change_shared_defaults(self):
        from baselines.DeepPrep.deepprep import DeepPrep
        from types import SimpleNamespace
        settings = SimpleNamespace(
            llm_backend="local", llm_model="unit-checkpoint", api_key=None,
            base_url=None, temperature=0.01, top_p=1.0, max_tokens=2048,
            timeout=600, torch_dtype="bfloat16", device_map="auto",
        )
        client = DeepPrep._build_llm(settings)
        self.assertFalse(client.cfg.enable_thinking)
        self.assertEqual(client.cfg.repetition_penalty, 1.1)
        shared = LLMClient(backend="local", model="unit-checkpoint")
        self.assertEqual(shared.cfg.repetition_penalty, 1.0)

    def test_invalid_table_arguments_receive_signature_feedback(self):
        for chain, expected in (
            ("HandleMV(df=main)", "accepted keywords"),
            ("SortRows(main, timestamp)", "Accepted keywords"),
        ):
            with self.assertRaisesRegex(Exception, expected) as caught:
                chain_to_steps(chain, _ctx(), random.Random(1), strict_instances=True)
            self.assertIn("environment", str(caught.exception))

    def test_prompt_exposes_legal_calls_and_actual_keywords(self):
        agent = TreeAgent(object(), object(), _ctx(), verbose=False)
        messages = agent._initial_messages("root observation")
        self.assertIn("LabelEncode(cols=None)", messages[0]["content"])
        self.assertIn("accepted_keywords=", messages[1]["content"])
        self.assertIn("at most 80 words", messages[0]["content"])

    def test_native_action_without_reasoning_and_plan_only(self):
        action = parse_response("HandleMV(method='median', action='impute') </execute>")
        self.assertEqual(action.kind, "expand")
        self.assertEqual(parse_response("<plan>clean</plan>").kind, "noop")

    def test_quoted_answer_tag_does_not_hide_native_action(self):
        action = parse_response(
            'The environment said "<answer>... </answer>".\n'
            "HandleMV(method='median', action='impute') </execute>"
        )
        self.assertEqual(action.kind, "expand")
        action = parse_response(
            '<think>Example: <answer>...</answer></think>\n'
            '<answer>root</answer>'
        )
        self.assertEqual(action.chain_str, "root")

    def test_rec_single_edge_executes_without_requiring_future_join(self):
        from unittest.mock import Mock, mock_open, patch
        from baselines.DeepPrep.sandbox import Sandbox
        from baselines.common.pipeline import Pipeline, assign_dag_structure, check_pipeline
        ctx = DataContext(
            task_type="rec", data_name="unit", primary_table="interaction",
            secondary_tables=["item"],
            tables={
                "interaction": TableContext(name="interaction", role="primary", shape=(3, 3),
                    dtypes={"user_id": "int64", "item_id": "int64", "timestamp": "int64"}),
                "item": TableContext(name="item", role="secondary", shape=(2, 1),
                    dtypes={"item_id": "int64"}),
            },
        )
        step = PipelineStep("CreateSequence", "interaction", {})
        sandbox = Sandbox.__new__(Sandbox)
        sandbox.data = Mock()
        sandbox.seed = 42
        sandbox.small_n = 0
        sandbox._tmpdir = "/tmp/deepprep-unit"
        sandbox._context = Mock(return_value=ctx)
        sandbox.snapshot = Mock(return_value=b"snapshot")
        sandbox._observe = Mock()
        with patch("baselines.DeepPrep.sandbox.open", mock_open(), create=True):
            result = sandbox.execute_chain([step])
        self.assertTrue(result.success, result.error)
        sandbox.data._run_relation_dag.assert_called_once()
        sandbox.data.run_pre_process.assert_not_called()
        final = Pipeline(steps=[step])
        assign_dag_structure(final, ctx)
        self.assertFalse(check_pipeline(final, ctx).valid)

        # Use real shared operators on synthetic frames: IDs and auxiliary
        # join keys must not be remapped between materialized tree edges.
        import pandas as pd
        from dppbench.dataset import RecData
        data = RecData.__new__(RecData)
        data.interaction_df = pd.DataFrame({
            'user_id': [71, 71], 'item_id': [23746, 22572],
            'timestamp': [1, 2], 'rating': [1, 1], '__split__': ['train', 'train'],
        })
        data.item_df = pd.DataFrame({'item_id': [23746, 22572], 'brand': [7, 8]})
        data.user_df = None
        data.col_types = {}
        data.model_cfg = {}
        sandbox.data = data
        with patch('baselines.DeepPrep.sandbox.open', mock_open(), create=True):
            first = sandbox.execute_chain([step])
            joined = sandbox.execute_chain([PipelineStep('JoinTable', 'interaction',
                {'method': 'rec', 'item_df': 'item', 'item_col': 'item_id'})])
        self.assertTrue(first.success, first.error)
        self.assertTrue(joined.success, joined.error)
        self.assertEqual(data.interaction_df['item_id'].tolist(), [23746, 22572])
        self.assertEqual(data.interaction_df['brand'].tolist(), [7, 8])

    def test_revision_hint_provides_exact_sequence_continuation(self):
        agent = TreeAgent(object(), object(), _ctx(), verbose=False)
        agent.ctx.task_type = 'rec'
        agent.tree.add_root(b'root', 'root')
        hint = agent._revision_hint()
        self.assertIn('CreateSequence(', hint)
        self.assertIn('time_col="timestamp"', hint)
        self.assertIn('not optional', hint)

    def test_wide_observation_limits_sample_columns(self):
        from baselines.DeepPrep.sandbox import Sandbox
        cols = [f"v{i}" for i in range(435)]
        profile = {"columns": cols, "shape": (192, 435), "null_ratio": 0.0,
                   "dtypes": dict.fromkeys(cols, "float64"),
                   "head": [dict.fromkeys(cols, 0)]}
        text = Sandbox._render_text(Sandbox.__new__(Sandbox), {"main": profile})
        self.assertNotIn("v434", text)
        self.assertLess(len(text), 2500)

    def test_timeout_fallback_uses_cached_candidate_without_revalidation(self):
        from unittest.mock import Mock
        from baselines.common.pipeline import Pipeline
        from baselines.common.budget import BudgetExceededError
        agent = TreeAgent(object(), object(), _ctx(), verbose=False)
        agent._validated_pipelines["node"] = Pipeline(steps=[PipelineStep("LabelEncode", "both", {})])
        agent._answer_errors = Mock(side_effect=BudgetExceededError("expired"))
        self.assertEqual(agent._fallback_pipeline().op_names(), ["LabelEncode"])
        agent._answer_errors.assert_not_called()

    def test_official_action_protocol_parser(self):
        action = parse_response(
            '<plan>clean missing values</plan>'
            '<expand>HandleMV(method="median", action="impute")</expand>'
        )
        self.assertEqual(action.kind, "expand")
        self.assertEqual(action.plan, "clean missing values")
        self.assertIsNone(action.parent_chain)

        answer = parse_response(
            '<plan>the target schema is satisfied</plan><answer>root</answer>'
        )
        self.assertEqual(answer.kind, "answer")

        self.assertEqual(
            parse_response("<plan>x</plan><expand parent='root'>A</expand><answer></answer>").kind,
            "noop",
        )

    def test_qwen_thinking_block_is_accepted_as_plan(self):
        action = parse_response(
            "<think>include all required operators</think>"
            "<expand>HandleMV(method='median', action='impute')</expand>"
        )
        self.assertEqual(action.kind, "expand")
        self.assertEqual(action.plan, "include all required operators")

    def test_qwen_prompt_owned_opening_think_tag_is_accepted(self):
        action = parse_response(
            "include all required operators</think>"
            "<expand>HandleMV(method='median', action='impute')</expand>"
        )
        self.assertEqual(action.kind, "expand")
        self.assertEqual(action.plan, "include all required operators")

    def test_initial_prompt_exposes_context_contract(self):
        ctx = DataContext(
            task_type="rec",
            data_name="unit_rec",
            primary_table="interaction",
            secondary_tables=["item"],
            tables={
                "interaction": TableContext(
                    name="interaction",
                    role="primary",
                    shape=(3, 3),
                    dtypes={"user_id": "int64", "item_id": "int64", "label": "int64"},
                ),
                "item": TableContext(
                    name="item",
                    role="secondary",
                    shape=(2, 2),
                    dtypes={"item_id": "int64", "brand": "object"},
                    join_keys_to_primary=[{"primary": "item_id", "secondary": "item_id"}],
                ),
            },
        )
        self.assertEqual(
            mandatory_ops_for_context(ctx),
            ["CreateSequence", "JoinTable"],
        )
        agent = TreeAgent(object(), object(), ctx, verbose=False)
        prompt = agent._initial_messages("root observation")[1]["content"]
        self.assertIn("# Mandatory DPPBench contract", prompt)
        self.assertIn("- CreateSequence", prompt)
        self.assertIn("- JoinTable", prompt)
        self.assertNotIn("- SampleNegative", prompt)
        self.assertIn("join_keys_to_primary=[{'primary': 'item_id'", prompt)

    def test_strict_chain_uses_only_declared_defaults(self):
        steps = chain_to_steps(
            'HandleMV(method="mean", action="impute")',
            _ctx(),
            random.Random(1),
            strict_instances=True,
        )
        self.assertEqual(steps[0].params, {"method": "mean", "action": "impute"})

    def test_released_operator_syntax_maps_explicitly(self):
        steps = chain_to_steps(
            'MissingValueImputation(table="main", column="x", mode="mean")',
            _ctx(),
            random.Random(1),
            strict_instances=True,
        )
        self.assertEqual(steps[0].op, "HandleMV")
        self.assertEqual(
            steps[0].params,
            {"cols": ["x"], "method": "mean", "action": "impute"},
        )

    def test_tree_parent_path_includes_parameters(self):
        tree = SearchTree()
        tree.add_root(b"root", "root")
        mean = PipelineStep("HandleMV", "both", {"method": "mean", "action": "impute"})
        median = PipelineStep("HandleMV", "both", {"method": "median", "action": "impute"})
        mean_node = tree.add_child([mean], b"mean", "mean")
        tree.set_cursor(tree.root_id)
        median_node = tree.add_child([median], b"median", "median")

        self.assertNotEqual(canonical_path(mean_node.accumulated_steps), canonical_path(median_node.accumulated_steps))
        self.assertIs(tree.find_by_steps([mean]), mean_node)
        self.assertIs(tree.find_by_steps([median]), median_node)

    def test_target_schema_does_not_read_prepare_yaml(self):
        schema = TargetSchema.from_context(_ctx())
        self.assertEqual(schema.required_tables, ("main",))
        self.assertEqual(schema.required_columns, ("target",))

    def test_local_backend_requires_released_checkpoint(self):
        old = os.environ.pop("DEEPPREP_MODEL", None)
        try:
            with self.assertRaisesRegex(ValueError, "official DeepPrep checkpoint"):
                LLMClient(backend="local", model=None)
        finally:
            if old is not None:
                os.environ["DEEPPREP_MODEL"] = old

    def test_agent_runs_expand_execute_answer_without_downstream_feedback(self):
        obs = ObsState(
            shape=(3, 2),
            columns=["x", "target"],
            dtypes={"x": "float64", "target": "int64"},
            head=[],
            null_ratio=0.0,
            text="materialized main table",
            tables={"main": {"columns": ["x", "target"]}},
        )

        class FakeSandbox:
            def __init__(self):
                self.counter = 0

            def reset(self):
                return ExecutionResult(True, obs=obs, snapshot=b"root")

            def restore(self, snapshot):
                return obs

            def execute_chain(self, steps):
                self.counter += 1
                return ExecutionResult(
                    True,
                    obs=obs,
                    snapshot=f"s{self.counter}".encode(),
                    applied_ops=[steps[0].op],
                )

        first = 'HandleMV(action="impute", method="median")'
        second = 'LabelEncode'

        class FakeLLM:
            def __init__(self):
                self.responses = iter([
                    f"<plan>clean</plan><expand>{first}</expand>",
                    f"<plan>encode</plan><expand>{first} --> {second}</expand>",
                    f"<plan>done</plan><answer>{first} --> {second}</answer>",
                ])

            def chat(self, messages):
                return next(self.responses)

        result = TreeAgent(
            FakeLLM(), FakeSandbox(), _ctx(), max_explore_turn=3, verbose=False
        ).run()
        self.assertTrue(result.success)
        self.assertEqual([step.op for step in result.pipeline.steps], ["HandleMV", "LabelEncode"])
        environment_turns = [t for t in result.transcript if t["role"] == "environment"]
        self.assertTrue(all("<execute" in turn["content"] for turn in environment_turns))

        from baselines.common.budget import BudgetExceededError

        class TimeoutLLM:
            def __init__(self):
                self.calls = 0

            def chat(self, messages):
                self.calls += 1
                if self.calls == 1:
                    return f"{first} --> LabelEncode() </execute>"
                raise BudgetExceededError("wall-time budget exceeded")

        fallback = TreeAgent(
            TimeoutLLM(), FakeSandbox(), _ctx(), max_explore_turn=3, verbose=False
        ).run()
        self.assertFalse(fallback.success)  # No explicit model answer was received.
        self.assertEqual(fallback.pipeline.op_names(), ["HandleMV", "LabelEncode"])
        self.assertIn("wall-time", fallback.transcript[-1]["content"])

        class RepeatLLM:
            def chat(self, messages):
                return f"{first} --> LabelEncode() </execute>"

        repeated = TreeAgent(
            RepeatLLM(), FakeSandbox(), _ctx(), max_explore_turn=3, verbose=False
        ).run()
        self.assertTrue(repeated.success)
        self.assertEqual(repeated.n_turns, 2)
        self.assertEqual(repeated.n_errors, 0)

        class AUCEvaluator:
            def __init__(self):
                self.calls = 0

            def evaluate_for_agent(self, pipeline):
                self.calls += 1
                auc = 0.8 if pipeline.steps[0].params['method'] == 'mean' else 0.6
                return auc, {'auc': auc}, None

        class AUCSearchLLM:
            def __init__(self):
                self.responses = iter([
                    f"{first} --> LabelEncode() </execute>",
                    "HandleMV(method='mean', action='impute') --> LabelEncode() </execute>",
                    f"{first} --> LabelEncode() </execute>",
                ])

            def chat(self, messages):
                return next(self.responses)

        evaluator = AUCEvaluator()
        searched = TreeAgent(
            AUCSearchLLM(), FakeSandbox(), _ctx(), downstream_evaluator=evaluator,
            max_explore_turn=3, verbose=False,
        ).run()
        self.assertTrue(searched.success)
        self.assertEqual(evaluator.calls, 2)  # Repeated candidate uses cached AUC.
        self.assertEqual(searched.pipeline.steps[0].params['method'], 'mean')
        self.assertEqual(len(searched.solution_attempts), 2)
        self.assertTrue(any('Sampled downstream feedback' in t['content'] for t in searched.transcript))

    def test_attention_contract_rejects_alias_and_secondary_only_columns(self):
        config = {'model': 'DIN', 'feature': {'seq_cols': {'item_id_seq': 'item_id', 'cate_id_seq': 'item_brand'},
                                             'seq_max_len': 20}}
        schema = TargetSchema.from_context(_ctx(), config)
        obs = ObsState((3, 3), ['target', 'item_id', 'item_history'], {}, [], 0.0, '',
                       tables={'main': {'columns': ['target', 'item_id', 'item_history']},
                               'item': {'columns': ['item_id_seq']}},
                       col_types={'target': 'numeric', 'item_id': 'categorical', 'item_history': 'categorical_list'})
        self.assertTrue(any('No valid attention' in e for e in schema.validate_observation(obs)))
        obs.columns[-1] = 'item_id_seq'
        obs.col_types['item_id_seq'] = 'categorical_list'
        self.assertEqual(schema.validate_observation(obs), [])
        obs.col_types['item_id'] = 'numeric'
        self.assertTrue(any('categorical query' in e for e in schema.validate_observation(obs)))
        obs.col_types['item_id'] = 'categorical'
        obs.col_types['item_id_seq'] = 'categorical'
        self.assertTrue(schema.validate_observation(obs))
        self.assertEqual(TargetSchema.from_context(_ctx(), {'model': 'LightGBM'}).attention_sequences, {})

    def test_attention_candidate_and_answer_rejected_before_scoring(self):
        from unittest.mock import Mock, patch
        from types import SimpleNamespace
        config = {'model': 'DIN', 'feature': {'seq_cols': {'custom_history': 'item_id'}, 'seq_max_len': 12}}
        ctx = _ctx()
        ctx.task_type = 'rec'
        ctx.tables['main'].dtypes = {'target': 'int64', 'user_id': 'int64', 'item_id': 'int64'}
        obs = ObsState((3, 4), ['target', 'user_id', 'item_id', 'item_history'], {}, [], 0.0, '',
                       tables={'main': {'columns': ['target', 'user_id', 'item_id', 'item_history']}},
                       col_types={'user_id': 'categorical', 'item_id': 'categorical', 'item_history': 'categorical_list'})
        sandbox = Mock()
        sandbox.reset.return_value = ExecutionResult(True, obs=obs, snapshot=b'root')
        sandbox.restore.return_value = obs
        sandbox.execute_chain.return_value = ExecutionResult(True, obs=obs, snapshot=b'child')
        chain = 'CreateSequence(user_col="user_id", item_col="item_id", time_col="timestamp", seq_col="item_history")'
        llm = Mock()
        llm.chat.side_effect = [chain + ' </execute>', '<answer>' + chain + '</answer>']
        evaluator = Mock()
        agent = TreeAgent(llm, sandbox, ctx, model_config=config, downstream_evaluator=evaluator,
                          max_explore_turn=2, verbose=False)
        with patch('baselines.DeepPrep.tree_agent.check_pipeline', return_value=SimpleNamespace(errors=[])):
            result = agent.run()
        evaluator.evaluate_for_agent.assert_not_called()
        self.assertFalse(result.success)
        self.assertFalse(agent._validated_pipelines)
        feedback = '\n'.join(t['content'] for t in result.transcript if t['role'] == 'environment')
        self.assertIn('Candidate contract errors', feedback)
        self.assertIn('Target schema or pipeline contract not satisfied', feedback)
        self.assertIn('custom_history', feedback)
        self.assertIn('max_len=12', agent._revision_hint())
        self.assertIn('attention_sequences', agent._initial_messages('root')[1]['content'])
        child = next(node for node in agent.tree.nodes.values() if node.depth == 1)
        self.assertEqual(child.accumulated_steps[0].params['seq_col'], 'item_history')

    def test_sandbox_passes_model_selection_to_executor(self):
        from unittest.mock import patch
        from baselines.DeepPrep.sandbox import Sandbox
        config = {'model': 'DIN', 'feature': {'seq_cols': {'item_id_seq': 'item_id'}}}
        with patch('baselines.DeepPrep.sandbox.TrainingExecutor') as executor:
            executor.return_value.get_model_config.return_value = config
            executor.return_value.task_type = 'rec'
            sandbox = Sandbox('task', 'unit', model_name='DIN')
            try:
                self.assertEqual(executor.call_args.kwargs['model_name'], 'DIN')
                self.assertEqual(sandbox.model_config, config)
            finally:
                sandbox.cleanup()

    def test_tabular_join_hints_use_actual_keys_and_cardinality(self):
        ctx = _ctx()
        ctx.tables['main'].dtypes['customer_id'] = 'int64'
        ctx.secondary_tables = ['history', 'lookup']
        for name, unique in [('history', False), ('lookup', True)]:
            ctx.tables[name] = TableContext(name=name, role='secondary',
                dtypes={'customer_id': 'int64', 'amount': 'float64'},
                join_keys_to_primary=[{'primary': 'customer_id', 'secondary': 'customer_id'}],
                join_key_unique={'customer_id': unique})
        agent = TreeAgent(object(), object(), ctx, verbose=False)
        prompt = agent._initial_messages('root')[1]['content']
        self.assertIn('**JoinTable** [mandatory]', prompt)
        hint = agent._revision_hint()
        self.assertIn('aux_df="$history_df"', hint)
        self.assertIn('method="agg"', hint)
        self.assertIn('aux_df="$lookup_df"', hint)
        self.assertIn('method="key"', hint)
        self.assertIn('key_col="customer_id"', hint)
        self.assertFalse(agent.tree.nodes)


if __name__ == "__main__":
    unittest.main()
