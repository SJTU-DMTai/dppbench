from __future__ import annotations

import math
import random
import unittest

from baselines.BAT.action import (
    ActionContext,
    EndAction,
    _column_similarity,
    _pipeline_from_response,
)
from baselines.BAT.mcts import MCTSSolver
from baselines.BAT.node import MCTSNode, MCTSNodeType
from baselines.BAT.reward import BATRewardConfig
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
    def test_column_similarity_is_official_target_recall(self):
        self.assertEqual(_column_similarity(["a", "b", "extra"], ["a", "b"]), 1.0)
        self.assertEqual(_column_similarity(["a"], ["a", "b"]), 0.5)

    def test_default_reward_is_schema_only(self):
        cfg = BATRewardConfig()
        self.assertEqual(cfg.alpha, 1.0)
        self.assertEqual(cfg.beta, 0.0)
        self.assertFalse(cfg.use_downstream)

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
