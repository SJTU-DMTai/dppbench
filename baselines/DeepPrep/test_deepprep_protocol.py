from __future__ import annotations

import random
import os
import unittest

from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.pipeline import PipelineStep
from baselines.DeepPrep.llm_client import LLMClient
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


if __name__ == "__main__":
    unittest.main()
