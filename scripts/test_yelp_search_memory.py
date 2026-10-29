"""Small synthetic fixtures; never download or train on the real Yelp data."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from baselines.common.search_evaluator import SearchPipelineEvaluator, _subsample_rec_split_frame
from dppbench.tasks.yelp.yelp_data import YelpData
from dppbench.tasks.yelp.search_data import load_frozen_search_interactions, read_selected_rows
from scripts.run_memory_limited import build_command


class YelpSearchMemoryTests(unittest.TestCase):
    def setUp(self):
        scope = patch.dict(os.environ, {"DPPBENCH_YELP_SIDE_TABLE_SCOPE": "sampled"})
        scope.start()
        self.addCleanup(scope.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "data" / "yelp"
        self.raw.mkdir(parents=True)
        frozen = self.root / "std_test"
        frozen.mkdir()
        rng = np.random.RandomState(7)
        self.train = pd.DataFrame({
            "user_id": [f"u{i % 25}" for i in range(400)],
            "item_id": [f"b{i % 30}" for i in range(400)],
            "timestamp": rng.randint(0, 100, 400),
            "stars": rng.randint(0, 2, 400),
            "useful": np.arange(400),
        })
        self.std = self.train.iloc[:250].copy().reset_index(drop=True)
        # Held-out events are distinct; duplicating training identities is now
        # correctly rejected before any search/model optimization.
        self.std['timestamp'] += 1000
        self.train.to_parquet(frozen / "interaction_frozen.parquet")
        self.std.to_parquet(frozen / "std_test.parquet")
        for name, rows in (
            ("user", [{"user_id": f"u{i}", "yelping_since": "2010-01-01", "friends": "long text"} for i in range(50)]),
            ("business", [{"business_id": f"b{i}", "categories": "Food, Cafe"} for i in range(60)]),
        ):
            (self.raw / f"yelp_academic_dataset_{name}.json").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        self.cfg = {"feature": {"target_col": "stars", "label_rule": {"mode": "ge", "threshold": 3, "positive_label": 1, "negative_label": 0}}}

    def data(self):
        data = YelpData(str(self.root / "data"))
        data.set_model_config(self.cfg)
        return data

    def test_projected_sampling_matches_existing_sampler_including_order(self):
        combined = pd.concat([self.train.assign(__split__="train"),
                              self.std.assign(__split__="std_test")], ignore_index=True)
        for n in (5, 101, 300, 1000):
            for seed in (43, 44):
                expected = _subsample_rec_split_frame(combined, self.data(), n, seed)
                actual = load_frozen_search_interactions(self.data(), n, seed)
                pd.testing.assert_frame_equal(actual, expected)

    def test_batch_selection_preserves_requested_order(self):
        actual = read_selected_rows(self.root / "std_test" / "interaction_frozen.parquet", [21, 2, 18], batch_size=7)
        pd.testing.assert_frame_equal(actual, self.train.iloc[[21, 2, 18]].reset_index(drop=True))

    def test_no_review_file_required_and_side_tables_are_trimmed(self):
        data = self.data()
        data._search_small_n, data._search_seed = 5, 43
        data.load_data()
        self.assertTrue(data._has_std_test)
        self.assertEqual(set(data.user_df.user_id), set(data.interaction_df.user_id))
        self.assertEqual(set(data.item_df.item_id), set(data.interaction_df.item_id))
        self.assertLess(len(data.user_df), 50)
        data._remap_ids()
        self.assertLessEqual(data.interaction_df.user_id.max(), len(data.user_df))

    def test_snapshot_reuses_loading_and_isolates_nested_objects(self):
        evaluator = SearchPipelineEvaluator("dppbench/tasks/yelp", "yelp", data_dir=str(self.root / "data"), small_n=5, seed=43)
        self.addCleanup(evaluator._executor.cleanup)
        first = evaluator.load_data()
        original_categories = list(first.item_df.iloc[0].item_categories)
        first.item_df.iloc[0].item_categories.append("MUTATED")
        evaluator._executor.clear_data_cache()
        with patch.object(YelpData, "_load_json_lines", side_effect=AssertionError("unexpected reload")):
            second = evaluator.load_data()
        with patch.object(pd, "read_parquet", side_effect=AssertionError("unexpected full frozen load")):
            self.assertTrue(second.load_std_test_frozen())
        self.assertEqual(second.item_df.iloc[0].item_categories, original_categories)
        self.assertIsNot(first, second)
        evaluator.set_small_n(7)
        self.assertIsNone(evaluator._search_snapshot)

    def test_candidate_order_isolated_by_real_search_snapshot(self):
        import pickle
        from dppbench.operators.preprocessing.one_hot_encode import OneHotEncode
        from dppbench.operators.preprocessing.augment_noise import AugmentNoise
        evaluator = SearchPipelineEvaluator('dppbench/tasks/yelp', 'yelp',
            data_dir=str(self.root / 'data'), small_n=101, seed=43)
        self.addCleanup(evaluator._executor.cleanup)
        first = evaluator.load_data()
        pristine = pickle.loads(pickle.dumps(first.interaction_df))
        original_items = pickle.loads(pickle.dumps(first.item_df))
        def candidate_b(frame):
            return OneHotEncode(['item_id']).transform(frame)
        expected = candidate_b(pristine)
        # Candidate A uses the changed operators, then deliberately mutates its
        # owned data, including an object-cell list, to stress snapshot isolation.
        candidate_a = AugmentNoise(['useful']).transform(candidate_b(first.interaction_df))
        self.assertGreater(len(candidate_a), len(pristine))
        pd.testing.assert_frame_equal(first.interaction_df, pristine)
        first.interaction_df.loc[:, 'useful'] = -999
        first.item_df.iloc[0].item_categories.append('candidate-A-only')
        evaluator._executor.clear_data_cache()
        second = evaluator.load_data()
        pd.testing.assert_frame_equal(second.interaction_df, pristine)
        pd.testing.assert_frame_equal(second.item_df, original_items)
        pd.testing.assert_frame_equal(candidate_b(second.interaction_df), expected)

    def test_missing_frozen_data_fails_instead_of_full_load(self):
        (self.root / "std_test" / "std_test.parquet").unlink()
        data = self.data()
        data._search_small_n, data._search_seed = 5, 43
        with self.assertRaises(FileNotFoundError):
            data.load_data()

    def test_full_data_path_still_reads_all_side_rows(self):
        (self.raw / "yelp_academic_dataset_review.json").write_text(json.dumps({
            "user_id": "u0", "business_id": "b0", "date": "2020-01-01", "stars": 5,
        }), encoding="utf-8")
        data = self.data()
        data.load_data()
        self.assertEqual(len(data.user_df), 50)
        self.assertEqual(len(data.item_df), 60)
        self.assertFalse(data._has_std_test)

    def test_default_search_preserves_full_side_tables_and_friend_text(self):
        with patch.dict(os.environ):
            os.environ.pop("DPPBENCH_YELP_SIDE_TABLE_SCOPE", None)
            data = self.data()
            data._search_small_n, data._search_seed = 5, 43
            data.load_data()
        self.assertEqual(len(data.user_df), 50)
        self.assertEqual(len(data.item_df), 60)
        self.assertTrue(data.user_df.user_friends.eq("long text").all())

    def test_full_scope_join_matches_original_side_table_inputs(self):
        from dppbench.operators.integration.join_table import JoinTable
        (self.raw / "yelp_academic_dataset_review.json").write_text(json.dumps({
            "user_id": "u0", "business_id": "b0", "date": "2020-01-01", "stars": 5,
        }), encoding="utf-8")
        old = self.data()
        old.load_data()
        with patch.dict(os.environ, {"DPPBENCH_YELP_SIDE_TABLE_SCOPE": "full"}):
            new = self.data()
            new._search_small_n, new._search_seed = 101, 43
            new.load_data()
        for how in ("left", "outer"):
            expected = JoinTable(method="rec", how=how, user_df=old.user_df,
                                 item_df=old.item_df).transform(new.interaction_df)
            actual = JoinTable(method="rec", how=how, user_df=new.user_df,
                               item_df=new.item_df).transform(new.interaction_df)
            pd.testing.assert_frame_equal(actual, expected)

    def test_validation_membership_matches_legacy_frozen_load(self):
        from baselines.common.search_evaluator import subsample_data_inplace
        for seed in (43, 44):
            old = self.data()
            old.load_std_test_frozen()
            subsample_data_inplace(old, 101, seed)
            old._prepare_validation()
            new = self.data()
            new.interaction_df = load_frozen_search_interactions(new, 101, seed)
            new._prepare_validation()
            pd.testing.assert_frame_equal(new.interaction_df, old.interaction_df)

    def test_sequence_reference_uses_same_mapping_as_item_id(self):
        data = self.data()
        data.model_cfg["feature"]["seq_cols"] = {"item_id_seq": "item_id"}
        data.interaction_df = pd.DataFrame({"user_id": ["u", "u"],
            "item_id": ["a", "z"], "item_id_seq": [["z"], ["history_only"]]})
        data.item_df = pd.DataFrame({"item_id": ["a", "z", "history_only"]})
        data.col_types = {"item_id_seq": "categorical_list"}
        data._remap_ids()
        self.assertEqual(data.interaction_df.item_id_seq.iloc[0][0], data.interaction_df.item_id.iloc[1])
        self.assertEqual(data.interaction_df.item_id_seq.iloc[1][0], data.item_df.item_id.iloc[2])

    def test_sequence_history_excludes_future_and_heldout_updates(self):
        from dppbench.operators.feature_engineering.create_sequence import CreateSequence
        frame = pd.DataFrame({"user_id": ["u"] * 5,
            "item_id": ["future", "past", "validation", "test", "tie"],
            "timestamp": [4, 1, 2, 3, 3],
            "__split__": ["train", "train", "val", "std_test", "train"]})
        out = CreateSequence().transform(frame)
        self.assertEqual(out.item_id_seq.iloc[2], ["past"])
        # Strict causal policy excludes all observations at the query timestamp.
        self.assertEqual(out.item_id_seq.iloc[3], ["past"])
        self.assertEqual(out.item_id_seq.iloc[0], ["past", "tie"])

    def test_single_row_final_batch_is_trained_not_dropped(self):
        import torch
        from dppbench.models.base_model import BaseModel
        model = BaseModel()
        model.layer = torch.nn.Linear(1, 1)
        model.forward = lambda x: torch.sigmoid(model.layer(x))
        model.compile(optimizer="sgd")
        sizes = []
        original_loss = model.loss_func
        def loss(prediction, target, **kwargs):
            sizes.append(len(target))
            self.assertEqual(prediction.shape, target.shape)
            return original_loss(prediction, target, **kwargs)
        model.loss_func = loss
        model.fit([np.array([[1.], [2.], [3.]], dtype=np.float32)],
                  np.array([0., 1., 0.], dtype=np.float32), batch_size=2,
                  epochs=1, shuffle=False, verbose=0)
        self.assertEqual(sizes, [2, 1])

    def test_deepprep_reset_uses_small_snapshot_without_reloading(self):
        from baselines.DeepPrep.sandbox import Sandbox
        sandbox = Sandbox("dppbench/tasks/yelp", "yelp", data_dir=str(self.root / "data"), small_n=101, seed=43)
        self.addCleanup(sandbox.cleanup)
        self.addCleanup(sandbox._executor.cleanup)
        sandbox.reset()
        original = sandbox.data.interaction_df.copy()
        sandbox.data.interaction_df = sandbox.data.interaction_df.iloc[:1]
        with patch.object(YelpData, "load_data", side_effect=AssertionError("unexpected reload")):
            sandbox.reset()
        pd.testing.assert_frame_equal(sandbox.data.interaction_df, original)

    def test_hard_limit_command(self):
        command = build_command(70, ["--", "python", "experiment.py"])
        self.assertIn("MemoryMax=75161927680", command)
        self.assertIn("MemorySwapMax=0", command)
        self.assertEqual(command[-2:], ["python", "experiment.py"])
        with self.assertRaises(ValueError):
            build_command(0, ["python"])

    def test_small_search_training_smoke(self):
        import torch
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        evaluator = SearchPipelineEvaluator(
            "dppbench/tasks/yelp", "yelp", data_dir=str(self.root / "data"),
            small_n=101, seed=43, model_name="DIN",
        )
        self.addCleanup(evaluator._executor.cleanup)
        evaluator.set_evaluate_std_test(False)
        evaluator.set_fast_mode(True, {"rec": {"epochs": 1, "batch_size": 32}})
        result = evaluator._executor.run_training()
        self.assertTrue(result["success"], result["error"])
        self.assertIn("auc", result["metrics"])


if __name__ == "__main__":
    unittest.main()
