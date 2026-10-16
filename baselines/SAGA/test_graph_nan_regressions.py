"""Offline reproductions from exp1_fix_saga42.log."""
import unittest
import random
import warnings
import numpy as np
import pandas as pd

from dppbench.dataset import TabularData
from dppbench.graph_input import prepare_graph_features, validate_graph_training
from dppbench.tasks.elliptic_bitcoin.elliptic_bitcoin_data import EllipticBitcoinData


class GraphNaNRegressions(unittest.TestCase):
    def test_generated_graph_selector_protects_metadata(self):
        from baselines.SAGA.test_saga_graph import _graph_ctx
        from baselines.common.pipeline import make_step, get_param_space
        ctx = _graph_ctx()
        for seed in range(10):
            step = make_step("SelectFeature", ctx, random.Random(seed))
            self.assertEqual(set(step.params["exclude_cols"]), {"txId", "time_step", "class"})
        self.assertEqual(set(get_param_space("SelectFeature", ctx)["exclude_cols"]),
                         {"txId", "time_step", "class"})

    def test_legacy_selector_keeps_graph_identity_and_temporal_masks(self):
        data = EllipticBitcoinData()
        frame = pd.DataFrame({"txId": [1, 2, 3, 4], "time_step": [1, 2, 36, 40],
                              "class": [0, 1, 0, 1], "feat_0": [0., 100., 200., 300.],
                              "feat_constant": [1., 1., 1., 1.],
                              "__split__": ["train", "train", "train", "std_test"]})
        # Reproduce an older YAML protecting only the target. A high variance
        # threshold would otherwise remove txId and time_step.
        op = data._load_op("SelectFeature")(
            target_col="class", method="variance", threshold=1000, exclude_cols=["class"])
        out = data._transform_tabular_dag_df(frame, op, "SelectFeature")
        self.assertEqual(out.txId.tolist(), [1, 2, 3, 4])
        self.assertEqual(out.time_step.tolist(), [1, 2, 36, 40])
        self.assertNotIn("feat_constant", out.columns)
        self.assertEqual(op._candidate_cols(frame.drop(columns="__split__")), ["feat_0", "feat_constant"])
        data.train_df = out
        data.auxiliary_dfs["edges"] = pd.DataFrame({"txId1": [1, 2], "txId2": [2, 3]})
        graph = data.build_graph()
        self.assertEqual(graph["x"].shape, (4, 1))
        self.assertEqual(graph["train_mask"].tolist(), [True, True, False, False])
        self.assertEqual(graph["val_mask"].tolist(), [False, False, True, False])
        self.assertEqual(graph["std_test_mask"].tolist(), [False, False, False, True])
        self.assertEqual(graph["edge_index"].tolist(), [[0, 1], [1, 2]])

    def test_regular_tabular_selector_is_not_given_graph_metadata(self):
        data = TabularData("fixture")
        op = data._load_op("SelectFeature")(target_col="y", threshold=1.)
        frame = pd.DataFrame({"x": [0., 0., 0.], "y": [0, 1, 0],
                              "__split__": ["train", "train", "std_test"]})
        out = data._transform_tabular_dag_df(frame, op, "SelectFeature")
        self.assertNotIn("x", out.columns)
        self.assertEqual(op.exclude_cols, [])

    def test_cleaning_search_spaces_match_operator_contracts(self):
        from baselines.common.op_spec import CATALOG
        self.assertEqual(set(CATALOG["HandleError"].param_space["repair_method"]), {"clip", "set_missing", "fill_constant", "median", "mode"})
        self.assertEqual(set(CATALOG["HandleOutlier"].param_space["repair_method"]), {"clip", "set_missing", "median"})
        self.assertEqual(set(CATALOG["HandleOutlier"].param_space["method"]), {"iqr", "zscore"})
        self.assertIn("iterative", CATALOG["HandleMV"].param_space["method"])

    def test_synthetic_oversample_assembly_is_not_fragmented(self):
        data = TabularData("fixture")
        op = data._load_op("Oversample")(target_col="target", method="smote")
        cols = [f"f{i}" for i in range(435)]
        xr = np.arange(3 * len(cols), dtype=float).reshape(3, len(cols))
        yr = np.array([0, 1, 1])
        original = [*cols[:8], "target", *cols[8:]]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", pd.errors.PerformanceWarning)
            out = op._assemble_resampled(xr, yr, cols, original)
        self.assertEqual(out.columns.tolist(), original)
        self.assertEqual(out.target.tolist(), [0, 1, 1])
        self.assertFalse(any(issubclass(w.category, pd.errors.PerformanceWarning) for w in caught))
        self.assertLessEqual(out._mgr.nblocks, 2)

    def test_singleton_aggregate_is_imputed_at_graph_boundary(self):
        data = EllipticBitcoinData()
        frame = pd.DataFrame({"txId": [1, 2, 3, 4], "time_step": [1, 2, 36, 40],
                              "class": [1, 0, 1, 0], "feat_0": [1., 3., 100., 200.],
                              "__split__": ["train", "train", "train", "std_test"]})
        op = data._load_op("AggregateGroupFeature")(
            group_cols=["txId"], agg_cols=["feat_0"], agg_funcs=["mean", "std"])
        data.train_df = data._transform_tabular_dag_df(frame, op, "AggregateGroupFeature")
        self.assertTrue(data.train_df.feat_0_std.isna().all())
        data.auxiliary_dfs["edges"] = pd.DataFrame({"txId1": [1], "txId2": [2]})
        graph = data.build_graph()
        self.assertTrue(np.isfinite(graph["x"]).all())
        self.assertEqual(graph["x"][:, -1].tolist(), [0., 0., 0., 0.])
        self.assertEqual(graph["y"].tolist(), [1, 0, 1, 0])
        self.assertEqual(graph["std_test_mask"].tolist(), [False, False, False, True])

    def test_imputation_uses_training_nodes_only(self):
        raw = np.array([[1., np.nan], [3., np.nan], [1000., 999.], [np.nan, np.nan]])
        train = np.array([True, True, False, False])
        x = prepare_graph_features(raw, train, ["observed", "unobserved"])
        self.assertEqual(x[-1].tolist(), [2., 0.])
        self.assertEqual(x[2].tolist(), [1000., 999.])
        self.assertTrue(np.isnan(raw[-1]).all())

    def test_infinity_is_not_silently_replaced(self):
        with self.assertRaisesRegex(ValueError, "infinity.*feat_bad"):
            prepare_graph_features([[np.inf]], [True], ["feat_bad"])

    def test_sort_keeps_frozen_membership_even_when_length_unchanged(self):
        data = TabularData("fixture")
        frame = pd.DataFrame({"id": [1, 2, 3, 4], "time": [40, 35, 1, 2],
                              "__split__": ["train", "train", "std_test", "std_test"]})
        op = data._load_op("SortRows")(by=["time"])
        out = data._transform_tabular_dag_df(frame, op, "SortRows")
        self.assertEqual(set(out.loc[out["__split__"] == "train", "id"]), {1, 2})
        self.assertEqual(set(out.loc[out["__split__"] == "std_test", "id"]), {3, 4})
        self.assertEqual(out.loc[out["__split__"] == "train", "time"].tolist(), [35, 40])

    def test_splitwise_transform_executes_once_per_partition(self):
        data = TabularData("fixture")
        class Reverse:
            calls = 0
            def transform(self, frame):
                self.calls += 1
                return frame.iloc[::-1].reset_index(drop=True)
        op = Reverse()
        frame = pd.DataFrame({"id": [1, 2, 3, 4], "__split__": ["train", "train", "std_test", "std_test"]})
        result = data._transform_tabular_dag_df(frame, op, "Reverse")
        self.assertEqual(op.calls, 2)
        self.assertEqual(result.id.tolist(), [2, 1, 4, 3])

    def test_empty_or_single_class_validation_fails_before_training(self):
        graph = {"x": np.ones((4, 1)), "y": np.array([0, 1, 0, 1]),
                 "train_mask": np.array([True, True, False, False]),
                 "val_mask": np.array([False, False, False, False])}
        with self.assertRaisesRegex(ValueError, "val_mask contains no labeled"):
            validate_graph_training(graph, ["auc"])
        graph["val_mask"][2] = True
        with self.assertRaisesRegex(ValueError, "requires both label classes"):
            validate_graph_training(graph, ["auc"])
        graph["val_mask"][3] = True
        validate_graph_training(graph, ["auc"])


if __name__ == "__main__":
    unittest.main()
