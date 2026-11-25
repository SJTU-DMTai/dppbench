"""Synthetic-data regression checks; no real dataset or GPU required."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("missingness", Path(__file__).with_name("default_credit_missingness.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MissingnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        rng = np.random.default_rng(19)
        n = 300
        cls.main = pd.DataFrame({"ID": np.arange(1, n + 1), "TARGET": np.arange(n) % 2,
                                "AGE": rng.integers(20, 70, n), "SEX": rng.integers(1, 3, n),
                                "LIMIT_BAL": rng.integers(1, 100, n) * 1000.,
                                "EDUCATION": rng.integers(1, 5, n), "MARRIAGE": rng.integers(1, 4, n)})
        cls.wide = cls.main.copy()
        for field, columns in m.MONTHS.items():
            for col in columns:
                cls.wide[col] = rng.integers(0, 10, n).astype(float)
        cls.aux = pd.concat([pd.DataFrame({"ID": cls.wide.ID, "MONTH_OFFSET": i,
                                           **{f: cls.wide[c[i]] for f, c in m.MONTHS.items()}})
                             for i in range(6)], ignore_index=True)
        cls.ids = cls.main.ID.iloc[:180].tolist()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_determinism_nesting_and_source_immutability(self):
        original = self.main.copy(deep=True)
        original_aux = self.aux.copy(deep=True)
        for layer in m.FIELDS:
            for kind in ("MCAR", "MAR", "MNAR", "Mixed"):
                low = m.inject(self.main, self.aux, self.ids, layer, kind, .1, 1001)
                high = m.inject(self.main, self.aux, self.ids, layer, kind, .5, 1001)
                again = m.inject(self.main, self.aux, self.ids, layer, kind, .1, 1001)
                for index in (0, 1, 2, 3):
                    pd.testing.assert_frame_equal(low[index], again[index])
                for index in (2, 3):
                    self.assertFalse((low[index] & ~high[index]).any().any())
                for info in low[4].values():
                    self.assertAlmostEqual(info["training_expected_rate"], .1)
                for col in ("ID", "TARGET", "AGE", "SEX"):
                    pd.testing.assert_series_equal(low[0][col], self.main[col])
        pd.testing.assert_frame_equal(self.main, original)
        pd.testing.assert_frame_equal(self.aux, original_aux)

    def test_history_blocks_survive_aggregation(self):
        dirty, aux, _, masks, _ = m.inject(self.main, self.aux, self.ids, "B", "MNAR", .3, 1001)
        pd.testing.assert_frame_equal(dirty, self.main)
        for col in m.FIELDS["B"]:
            counts = masks.groupby(self.aux.ID)[col].sum()
            self.assertTrue(counts.isin([0, 6]).all())
            aggregated = aux.groupby("ID")[col].agg(["mean", "max", "min"])
            self.assertEqual(int(aggregated.isna().all(axis=1).sum()), int((counts == 6).sum()))

    def test_calibration_never_uses_heldout_values(self):
        changed = self.main.copy()
        changed.loc[~changed.ID.isin(self.ids), ["AGE", "LIMIT_BAL"]] = 999999
        a = m.inject(self.main, self.aux, self.ids, "A", "MNAR", .3, 1001)
        b = m.inject(changed, self.aux, self.ids, "A", "MNAR", .3, 1001)
        for col in m.FIELDS["A"]:
            self.assertEqual(a[4][col]["alpha"], b[4][col]["alpha"])
            self.assertEqual(a[4][col]["condition_threshold"], b[4][col]["condition_threshold"])

    def test_isolated_preparation_and_frozen_loader(self):
        source = self.root / "source"
        std = source / "default_credit/std_test"
        std.mkdir(parents=True)
        self.main.iloc[:240].to_parquet(std / "train_frozen.parquet", index=False)
        self.main.iloc[240:].to_parquet(std / "std_test.parquet", index=False)
        m.dump(std / "meta.json", {"test": "synthetic"})
        raw = source / "wide.parquet"
        self.wide.to_parquet(raw, index=False)
        before = {str(p): m.sha(p) for p in source.rglob("*") if p.is_file()}
        output = self.root / "experiment"
        args = SimpleNamespace(source_root=str(source), output_root=str(output), wide_parquet=str(raw),
                               mask_seeds=[1001], layers=["A", "B"], mechanisms=["MCAR", "Mixed"], rates=[.1, .5])
        m.prepare(args)
        self.assertEqual(before, {str(p): m.sha(p) for p in source.rglob("*") if p.is_file()})
        with self.assertRaises(FileExistsError):
            m.prepare(args)
        self.assertEqual(len(json.loads((output / "suite.json").read_text())["scenarios"]), 9)
        data = output / "scenarios/B_Mixed_q50_mask1001/datasets/default_credit/data"
        check = """from dppbench.tasks.default_credit.default_credit_data import DefaultCreditData
import sys
d=DefaultCreditData(sys.argv[1]);d.load_data();assert d.load_std_test_frozen()
a=d.split(seed=42);b=d.split(seed=44)
assert list(a['train'].ID)==list(b['train'].ID)
assert len(a['train'])==192 and len(a['val'])==48 and len(a['std_test'])==60
assert not set(a['train'].ID)&set(a['val'].ID)
print('isolated loader OK')
from pathlib import Path
import yaml
config=Path('dppbench/tasks/default_credit/prepare.yaml')
for method in ('mean','median','constant','knn','iterative'):
    d=DefaultCreditData(sys.argv[1]);d.load_data()
    cfg=yaml.safe_load(config.read_text())
    for op in cfg['dag']['ops']:
        if op['op']=='HandleMV': op['params']={'method':method}
    target=Path(sys.argv[1]) / ('test_'+method+'.yaml')
    target.write_text(yaml.safe_dump(cfg))
    d.run_pre_process(str(target))
    parts=d.split(seed=99)
    assert len(parts['train'])==192 and len(parts['val'])==48
    assert not parts['train'].isna().any().any()
    target.unlink()
print('five imputation pipelines OK')
"""
        subprocess.run([sys.executable, "-c", check, str(data)], cwd=output / "code", check=True)
        # Validate command generation without launching training.
        subprocess.run([sys.executable, str(Path(m.__file__)), "screen", "--output-root", str(output),
                        "--scenarios", "clean", "--search-seeds", "42"], check=True)
        for bad in (source, source / "child", source.parent):
            with self.assertRaises(ValueError):
                m.disjoint(bad, source)


if __name__ == "__main__":
    unittest.main()
