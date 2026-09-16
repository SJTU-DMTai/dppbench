from __future__ import annotations

import random
import unittest

from baselines.common.data_ctx import DataContext, TableContext
from baselines.common.op_spec import operators_for_task
from baselines.common.pipeline import check_pipeline, random_pipeline


def _graph_ctx() -> DataContext:
    return DataContext(
        task_type="graph",
        data_name="elliptic_bitcoin",
        primary_table="main",
        secondary_tables=["classes", "edges"],
        tables={
            "main": TableContext(
                name="main",
                role="primary",
                dtypes={
                    "txId": "int64",
                    "time_step": "int64",
                    "feat_0": "float64",
                    "feat_1": "float64",
                    "class": "int64",
                },
            ),
            "classes": TableContext(
                name="classes",
                role="secondary",
                dtypes={"txId": "int64", "class": "object"},
                join_keys_to_primary=[{"primary": "txId", "secondary": "txId"}],
            ),
            "edges": TableContext(
                name="edges",
                role="secondary",
                dtypes={"txId1": "int64", "txId2": "int64"},
            ),
        },
    )


class SagaGraphAlignmentTest(unittest.TestCase):
    def test_graph_operator_space_uses_shared_generic_ops(self):
        ops = set(operators_for_task("graph"))
        self.assertIn("JoinTable", ops)
        self.assertIn("ScaleFeature", ops)
        self.assertNotIn("SampleNegative", ops)
        self.assertNotIn("ConcatTable", ops)

    def test_graph_random_pipeline_is_legal_and_joins_joinable_source(self):
        ctx = _graph_ctx()
        self.assertEqual(ctx.target_col, "class")
        pipeline = random_pipeline(
            "graph", ctx, random.Random(42), p_optional=0.0, max_len=10,
        )
        report = check_pipeline(pipeline, ctx)
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(pipeline.steps[0].op, "JoinTable")
        self.assertEqual(set(pipeline.op_names()), {"JoinTable", "LabelEncode", "HandleMV"})
        join = pipeline.steps[0]
        self.assertEqual(join.params["aux_df"], "$classes")
        self.assertNotIn("$edges", str(join.params))
        self.assertEqual(join.target, "both")


if __name__ == "__main__":
    unittest.main()
