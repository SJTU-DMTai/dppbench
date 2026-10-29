"""Regressions for KuaiRec targets, HAIPipe config, and baseline aliases."""
from pathlib import Path
from types import SimpleNamespace
import unittest

import pandas as pd

from baselines.common.data_ctx import infer_data_context
from scripts.evaluate_baseline import _canonical_baseline_name

ROOT = Path(__file__).resolve().parents[1]


def rec_data(target="watch_ratio", configured=True):
    return SimpleNamespace(
        interaction_df=pd.DataFrame({
            "user_id": [1, 1, 2, 2], "item_id": [1, 2, 1, 2],
            target: [0., 1., 0., 1.], "duration": [1., 2., 3., 4.],
        }),
        user_df=None,
        item_df=pd.DataFrame({"item_id": [1, 2], "side_numeric": [10., 20.]}),
        _user_id_col="user_id", _item_id_col="item_id", col_types={},
        model_cfg={"feature": {"target_col": target}} if configured else {},
    )


class ExperimentRegressions(unittest.TestCase):
    def test_configured_rec_target_is_excluded_from_features(self):
        for target in ("watch_ratio", "custom_feedback"):
            with self.subTest(target=target):
                ctx = infer_data_context("kuairec", rec_data(target), "rec")
                self.assertEqual(ctx.target_col, target)
                self.assertNotIn(target, ctx.numeric_cols)
                self.assertNotIn(target, ctx.categorical_cols)

    def test_legacy_rating_target_still_inferred(self):
        ctx = infer_data_context("amazon_beauty", rec_data("rating", False), "rec")
        self.assertEqual(ctx.target_col, "rating")

    def test_baseline_alias_and_unknown_name(self):
        for name in ("DAGPipe", "DagPipe", "dagpipe", " DAGPIPE "):
            self.assertEqual(_canonical_baseline_name(name), "DAGPipe")
        self.assertEqual(_canonical_baseline_name("HAIPipe"), "HAIPipe")
        with self.assertRaisesRegex(ValueError, "Unknown baseline"):
            _canonical_baseline_name("missing_baseline")

    def test_haipipe_default_config_constructs(self):
        import tempfile
        from baselines.HAIPipe.haipipe import HAIPipe
        with tempfile.TemporaryDirectory() as output:
            runner = HAIPipe(
                task_dir=str(ROOT / "dppbench/tasks/kuairec"),
                data_name="kuairec", output_dir=output,
            )
            self.assertEqual(runner.al_k, 20)
            self.assertEqual(runner.al_iterations, 7)

    # Recommendation tensor regressions now live in
    # baselines/DiffPrep/test_rec_correctness.py and include the real joined
    # features, frozen partitions, architecture gradients, and exported inputs.

    def test_tabular_tensors_ignore_unjoined_auxiliary_columns(self):
        import torch
        from baselines.DiffPrep.diffprep import _tabular_feature_tensors
        data = SimpleNamespace(
            train_df=pd.DataFrame({"id": [1, 2], "label": [0, 1], "x": [1., 2.]}),
            id_col="id", target_col="label",
            auxiliary_dfs={"aux": pd.DataFrame({"id": [1, 2], "side": [3., 4.]})},
        )
        ctx = infer_data_context("test", data, "tabular")
        x, _, _, names, _ = _tabular_feature_tensors(data, ctx, 64, torch.device("cpu"))
        self.assertEqual(names, ["x"])
        self.assertEqual(tuple(x.shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
