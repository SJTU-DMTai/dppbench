"""Small, offline reproductions of full42 failures; no benchmark files changed."""
import random
import json
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from baselines.common.data_ctx import infer_data_context
from baselines.common.evaluator import PipelineEvaluator
from baselines.common.op_spec import CATALOG
from baselines.common.pipeline import Pipeline, check_pipeline, random_pipeline, get_param_space
from dppbench.dataset import BaseData, RecData, TabularData
from dppbench.tasks.elliptic_bitcoin.elliptic_bitcoin_data import EllipticBitcoinData
from dppbench.tasks.bike_sharing.bike_sharing_data import BikeSharingData
from dppbench.tasks.beijing_air_quality.beijing_air_quality_data import BeijingAirQualityData


class Full42Regressions(unittest.TestCase):
    def rec(self, col="item_genres"):
        data = RecData("fixture")
        data.interaction_df = pd.DataFrame({"user_id": [1, 2], "item_id": [1, 2], "rating": [1, 0]})
        data.item_df = pd.DataFrame({"item_id": [1, 2], col: [["a", "b"], ["b"]]})
        data.register_col_types({col: data.CATEGORICAL_LIST})
        return data

    def test_multivalue_schema_excluded_from_scalar_search(self):
        for col in ("item_genres", "item_categories"):
            data = self.rec(col)
            ctx = infer_data_context("fixture", data, "rec")
            self.assertIn(col, ctx.list_cols)
            self.assertNotIn(col, ctx.categorical_cols)
            self.assertNotIn(col, get_param_space("TargetEncode", ctx).get("cols", []))

    def test_label_encode_preserves_lists_and_unseen_scalar_policy(self):
        data = self.rec()
        op = data._load_op("LabelEncode")()
        train = pd.DataFrame({"city": ["a", "b"], "item_genres": [[1, 2], [2]]})
        result = op.transform(train)
        self.assertEqual(result.item_genres.tolist(), [[1, 2], [2]])
        held = op.transform(pd.DataFrame({"city": ["new"], "item_genres": [[3]]}))
        self.assertEqual(held.city.tolist(), [-1])
        self.assertEqual(held.item_genres.tolist(), [[3]])

    def test_deduplicate_ordered_lists_without_modifying_cells(self):
        data = self.rec()
        frame = pd.DataFrame({"id": [1, 1, 1], "tags": [[1, 2], [1, 2], [2, 1]]})
        result = data._load_op("Deduplicate")().transform(frame)
        self.assertEqual(result.tags.tolist(), [[1, 2], [2, 1]])
        self.assertIsInstance(result.tags.iloc[0], list)
        self.assertEqual(len(frame), 3)

    def test_join_without_id_column(self):
        for key, aux in (("station", "station_meta"), ("instant", "day")):
            data = TabularData("fixture")
            data.target_col = "y"
            data.train_df = pd.DataFrame({key: [1, 2], "x": [3., 4.], "y": [0., 1.]})
            data.auxiliary_dfs[aux] = pd.DataFrame({key: [1, 2], "z": [5., 6.]})
            ctx = infer_data_context("fixture", data, "tabular")
            self.assertIsNone(ctx.id_col)
            pipe = random_pipeline("tabular", ctx, random.Random(42), p_optional=0., max_len=10)
            self.assertTrue(check_pipeline(pipe, ctx).valid, check_pipeline(pipe, ctx).errors)
            join = next(s for s in pipe.steps if s.op == "JoinTable")
            self.assertEqual(join.params["key_col"], key)

    def test_every_advertised_oversample_method_constructs(self):
        cls = TabularData("fixture")._load_op("Oversample")
        for method in CATALOG["Oversample"].param_space["method"]:
            self.assertEqual(cls(target_col="y", method=method).method, method)

    def test_real_task_join_metadata_overrides_incidental_shared_columns(self):
        for cls, aux, key in ((BikeSharingData, "day", "dteday"),
                              (BeijingAirQualityData, "station_meta", "station")):
            data = cls()
            data.train_df = pd.DataFrame({"instant": [100, 200], "SO2": [1., 2.], key: ["a", "b"], data.target_col: [3., 4.]})
            data.auxiliary_dfs[aux] = pd.DataFrame({"instant": [1, 2], "SO2": [3., 4.], key: ["a", "b"], "extra": [5., 6.]})
            ctx = infer_data_context("fixture", data, "tabular")
            self.assertEqual(ctx.tables[aux].join_keys_to_primary, [{"primary": key, "secondary": key}])
            pipe = random_pipeline("tabular", ctx, random.Random(42), p_optional=0.)
            step = next(s for s in pipe.steps if s.op == "JoinTable")
            params = dict(step.params, aux_df=data.auxiliary_dfs[aux])
            joined = data._load_op("JoinTable")(**params).transform(data.train_df)
            self.assertEqual(joined.extra.tolist(), [5., 6.])

    def test_empty_rec_training_rejected_at_filter(self):
        data = self.rec()
        frame = pd.DataFrame({"user_id": [1, 2, 3], "item_id": [1, 2, 3],
                              "__split__": ["train", "train", "std_test"]})
        op = data._load_op("FilterKCore")(k=5)
        with self.assertRaisesRegex(ValueError, "FilterKCore removed all training rows"):
            data._transform_rec_dag_df(frame, op, "FilterKCore")
        self.assertEqual(frame["__split__"].tolist(), ["train", "train", "std_test"])

    def test_empty_tabular_training_rejected_at_delete(self):
        data = TabularData("fixture")
        frame = pd.DataFrame({"x": [np.nan, np.nan, 1.], "__split__": ["train", "train", "std_test"]})
        op = data._load_op("HandleMV")(action="delete")
        with self.assertRaisesRegex(ValueError, "HandleMV removed all training rows"):
            data._transform_tabular_dag_df(frame, op, "HandleMV")

    def test_legacy_graph_frozen_labels_and_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "std_test").mkdir()
            train = pd.DataFrame({"txId": [1, 2, 3], "time_step": [1, 2, 36], "feat_0": [1., 2., 3.]})
            held = pd.DataFrame({"txId": [4], "time_step": [3], "feat_0": [4.], "class": ["2"]})
            (root / "std_test/train_frozen.parquet").touch()
            (root / "std_test/std_test.parquet").touch()
            data = EllipticBitcoinData(str(root / "data"))
            data.auxiliary_dfs["classes"] = pd.DataFrame({"txId": [1, 2, 3, 4], "class": ["1", "unknown", "2", "2"]})
            data.auxiliary_dfs["edges"] = pd.DataFrame({"txId1": [1, 2], "txId2": [2, 4]})
            # Test the loader's schema boundary independently of parquet engines.
            with patch("pandas.read_parquet", side_effect=[train, held]):
                self.assertTrue(data.load_std_test_frozen())
            graph = data.build_graph()
            self.assertEqual(graph["y"].tolist(), [1, -1, 0, 0])
            self.assertEqual(graph["train_mask"].tolist(), [True, False, False, False])
            self.assertEqual(graph["std_test_mask"].tolist(), [False, False, False, True])
            self.assertEqual(graph["val_mask"].tolist(), [False, False, True, False])

    def test_dag_releases_intermediate_after_last_branch_consumer(self):
        data = BaseData("fixture")
        refs = {}
        class Op:
            def __init__(self, tag):
                self.tag = tag
        data._load_op = lambda name: Op
        data._resolve_param = lambda value: value
        def transform(frame, op, name):
            if op.tag == "third":
                self.assertIsNone(refs["first"]())
            result = frame.copy()
            refs[op.tag] = weakref.ref(result)
            return result
        config = {"dag": {"sources": [{"id": "s0", "table": "main"}],
            "ops": [{"id": f"o{i}", "op": "Op", "prev": ["s0" if i == 1 else f"o{i-1}"],
                     "params": {"tag": tag}} for i, tag in enumerate(["first", "second", "third"], 1)],
            "train": {"prev": ["o3"]}}}
        result = data._run_relation_dag(config, {"main": pd.DataFrame({"x": [1]})}, transform)
        self.assertEqual(result.x.tolist(), [1])

    def test_regression_fitness_and_nonfinite_metrics(self):
        evaluator = object.__new__(PipelineEvaluator)
        evaluator.metric_key = "rmse"
        evaluator.scoring_method = "full_train"
        evaluator._cache = {}
        evaluator.budget = None
        evaluator.verbose = False
        class Executor:
            _fast_mode = False
            def write_yaml(self, text): pass
        evaluator._executor = Executor()
        evaluator._get_data_context = lambda: None
        evaluator._check_pipeline = lambda *a, **kw: type("Report", (), {"valid": True})()
        for value, expected in ((3., -3.), (1., -1.), (float("nan"), -float("inf"))):
            evaluator._cache.clear()
            evaluator._run_scoring_method = lambda method: {"success": True, "metrics": {"rmse": value}}
            result = evaluator.evaluate(Pipeline([]))
            self.assertEqual(result.fitness, expected)
            self.assertEqual(result.success, np.isfinite(value))

    def test_shared_dag_branch_stays_live_for_all_consumers(self):
        data = BaseData("fixture")
        refs = {}
        class Op:
            def __init__(self, tag, other_dfs=None):
                self.tag, self.other_dfs = tag, other_dfs
        data._load_op = lambda name: Op
        data._resolve_param = lambda value: value
        def transform(frame, op, name):
            if op.tag in ("left", "right"):
                self.assertIsNotNone(refs["shared"]())
            if op.tag == "join":
                self.assertIsNone(refs["shared"]())
                return frame + op.other_dfs[0]
            result = frame.copy()
            refs[op.tag] = weakref.ref(result)
            return result
        config = {"dag": {"sources": [{"id": "s0", "table": "main"}], "ops": [
            {"id": "o1", "op": "Op", "prev": ["s0"], "params": {"tag": "shared"}},
            {"id": "o2", "op": "Op", "prev": ["o1"], "params": {"tag": "left"}},
            {"id": "o3", "op": "Op", "prev": ["o1"], "params": {"tag": "right"}},
            {"id": "o4", "op": "Op", "prev": ["o2", "o3"], "params": {"tag": "join"}},
        ], "train": {"prev": ["o4"]}}}
        result = data._run_relation_dag(config, {"main": pd.DataFrame({"x": [1]})}, transform)
        self.assertEqual(result.x.tolist(), [2])

    def test_default_metric_is_read_from_model_config(self):
        from baselines.common.search_evaluator import SearchPipelineEvaluator
        root = Path(__file__).resolve().parents[2]
        evaluator = SearchPipelineEvaluator(str(root / "dppbench/tasks/bike_sharing"), "bike_sharing")
        try:
            self.assertEqual(evaluator.metric_key, "rmse")
        finally:
            evaluator._executor.cleanup()

    def test_hpo_selects_validation_metric_without_test_evaluation(self):
        from scripts.final_model_hpo import tune_final_model
        from baselines.common.evaluator import EvaluationResult
        instances = []
        class FakeEvaluator:
            def __init__(self, model_config, **kwargs):
                self.config = model_config
                self.std_test = True
                self._executor = self
                self.cleaned = False
                instances.append(self)
            def set_evaluate_std_test(self, enabled): self.std_test = enabled
            def get_model_config(self): return {"train": {"metrics": ["rmse"]}}
            def cleanup(self): self.cleaned = True
            def evaluate(self, pipeline, scoring_method):
                self_test.assertFalse(self.std_test)
                value = self.config["model_params"]["value"]
                return EvaluationResult(-value, True, {"rmse": value, "std_test_rmse": 100 - value})
        self_test = self
        candidates = [{"model_params": {"value": value}} for value in (3., 1., 2.)]
        with tempfile.TemporaryDirectory() as directory, patch("scripts.final_model_hpo.candidate_configs", return_value=candidates):
            final, summary = tune_final_model(Pipeline([]), output_dir=directory,
                                              evaluator_class=FakeEvaluator, model_name="LightGBM")
            self.assertEqual(summary["best_params"], candidates[1])
            self.assertTrue(final.std_test)
            self.assertTrue(all(item.cleaned for item in instances[:-1]))
            self.assertEqual(len(json.loads((Path(directory) / "final_hpo.json").read_text())["trials"]), 3)


if __name__ == "__main__":
    unittest.main()
