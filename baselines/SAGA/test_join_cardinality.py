"""Synthetic-only regression checks; no task loader or dataset access."""
import random
import unittest
from types import SimpleNamespace

import pandas as pd

from baselines.common.data_ctx import _infer_tabular_context
from baselines.common.pipeline import build_default_params
from dppbench.dataset import TabularData


class JoinCardinalityTests(unittest.TestCase):
    def make_context(self, right_keys):
        main = pd.DataFrame({"ID": [1, 2], "target": [0, 1]})
        aux = pd.DataFrame({"ID": right_keys, "amount": range(len(right_keys))})
        data = SimpleNamespace(train_df=main, id_col="ID", target_col="target",
                               auxiliary_dfs={"history": aux})
        return main, aux, _infer_tabular_context("synthetic", data)

    def test_unique_right_key_uses_lookup(self):
        _, _, ctx = self.make_context([1, 2])
        params = build_default_params("JoinTable", ctx, random.Random(42))
        self.assertEqual(params["method"], "key")

    def test_repeated_history_aggregates_without_expanding_rows(self):
        main, aux, ctx = self.make_context([1, 1, 2, 2])
        params = build_default_params("JoinTable", ctx, random.Random(42))
        self.assertEqual(params["method"], "agg")
        params["aux_df"] = aux
        join_class = TabularData("synthetic")._load_op("JoinTable")
        result = join_class(**params).transform(main)
        self.assertEqual(len(result), len(main))
        self.assertEqual(result["target"].tolist(), [0, 1])
        mean_col = next(c for c in result if c.endswith("amount_mean"))
        self.assertEqual(result[mean_col].tolist(), [0.5, 2.5])
        with self.assertRaises(pd.errors.MergeError):
            join_class(method="key", aux_df=aux, key_col="ID").transform(main)

    def test_declared_cohort_key_wins_over_shared_features(self):
        main = pd.DataFrame({"Attr1": [0.1, 0.2], "_join_key": [1, 1], "TARGET": [0, 1]})
        aux = pd.DataFrame({"Attr1": [0.3, 0.4], "_join_key": [1, 1]})
        data = SimpleNamespace(train_df=main, id_col=None, target_col="TARGET",
                               auxiliary_dfs={"year1": aux},
                               auxiliary_join_keys={"year1": ["_join_key"]})
        ctx = _infer_tabular_context("synthetic", data)
        params = build_default_params("JoinTable", ctx, random.Random(42))
        self.assertEqual((params["key_col"], params["method"]), ("_join_key", "agg"))


if __name__ == "__main__":
    unittest.main()
