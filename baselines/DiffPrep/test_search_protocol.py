import random
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from baselines.DiffPrep.data_protocol import tabular_search_splits
from baselines.DiffPrep.search_space import ContinuousPipeline
from baselines.DiffPrep.slot_planner import Slot, make_slots
from baselines.common.op_spec import OpCategory
from baselines.common.pipeline import Pipeline, repair


def test_hard_candidate_does_not_supply_fake_gradients():
    x = torch.randn(5, 2, requires_grad=True)
    fixed = Slot(0, OpCategory.PREPROCESSING, ['JoinTable'], 'hard', True, 'JoinTable')
    model = ContinuousPipeline([fixed], n_features=2)
    torch.testing.assert_close(model(x, torch.zeros_like(x)), x)
    optional = Slot(0, OpCategory.PREPROCESSING, ['JoinTable', 'Identity'], 'hard')
    with pytest.raises(ValueError, match='unsupported hard'):
        ContinuousPipeline([optional], n_features=2)(x, torch.zeros_like(x))


def test_mixed_category_keeps_supported_soft_choices():
    with patch('baselines.DiffPrep.slot_planner.get_category_order', return_value=[OpCategory.PREPROCESSING]), \
         patch('baselines.DiffPrep.slot_planner._candidates_in_category', return_value=['Deduplicate', 'ScaleFeature']), \
         patch('baselines.DiffPrep.slot_planner.is_mandatory', return_value=False):
        slots = make_slots('tabular', SimpleNamespace(task_type='tabular'))
    assert slots[0].candidates == ['ScaleFeature', 'Identity']
    assert slots[0].kind == 'soft'


def test_repair_uses_callers_seed_for_join_and_required_tail():
    ctx = SimpleNamespace(secondary_tables=['account'])
    def draws(seed):
        seen = []
        def fake_step(name, context, rng):
            seen.append((name, rng.random()))
            return None
        with patch('baselines.common.pipeline.make_step', side_effect=fake_step):
            repair(Pipeline([]), 'tabular', ctx, rng=random.Random(seed))
        return seen
    assert draws(42) == draws(42)
    assert draws(42) != draws(43)
    assert [name for name, _ in draws(42)] == ['JoinTable', 'LabelEncode', 'HandleMV']


def test_search_does_not_open_standard_test_and_keeps_validation(tmp_path):
    frame = pd.DataFrame({'id': range(20), 'target': [i % 2 for i in range(20)]})
    frame.to_parquet(tmp_path / 'train_frozen.parquet', index=False)
    data = SimpleNamespace(data_dir=str(tmp_path), target_col='target',
                           _std_test_dir=lambda _: str(tmp_path), _normalize_frozen_frame=lambda x: x)
    data.load_std_test_frozen = lambda: pytest.fail('opened standard test')
    def split(val_ratio, seed):
        assert seed == 43 and val_ratio == .2
        return {'train': data.train_df.iloc[:16], 'val': data.train_df.iloc[16:]}
    data.split = split
    read = pd.read_parquet
    def read_train_only(path, *args, **kwargs):
        assert str(path).endswith('train_frozen.parquet')
        return read(path, *args, **kwargs)
    with patch.object(pd, 'read_parquet', side_effect=read_train_only):
        result = tabular_search_splits(data, {'train': {'val_ratio': .2}}, 43, small_n=5)
    assert len(result['train']) == 5
    assert result['val'].id.tolist() == [16, 17, 18, 19]
    assert set(result['train'].id).isdisjoint(result['val'].id)
    frame['__split__'] = 'std_test'
    frame.to_parquet(tmp_path / 'train_frozen.parquet', index=False)
    with pytest.raises(ValueError, match='non-training'):
        tabular_search_splits(data, {}, 43)


def test_berka_search_reproducible_without_reading_std_test(tmp_path, monkeypatch):
    from baselines.DiffPrep.diffprep import DiffPrep
    from dppbench.tasks.berka.berka_data import BerkaData
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    std_dir = tmp_path / 'std_test'
    std_dir.mkdir()
    n = 40
    frame = pd.DataFrame({'loan_id': range(n), 'account_id': range(n), 'date': 930102,
                          'amount': [100. + i for i in range(n)], 'target': [i % 2 for i in range(n)]})
    frame.to_parquet(std_dir / 'train_frozen.parquet', index=False)
    def load(data):
        data.train_df = frame.copy()
        data.test_df = None
        data.auxiliary_dfs = {'account': pd.DataFrame({'account_id': range(n), 'district_id': [i % 4 for i in range(n)]})}
        return data.train_df, None
    monkeypatch.setattr(BerkaData, 'load_data', load)
    original = pd.read_parquet
    def guarded(path, *args, **kwargs):
        assert Path(path).name != 'std_test.parquet'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_parquet', guarded)
    kwargs = dict(task_dir=str(Path(__file__).resolve().parents[2] / 'dppbench/tasks/berka'),
                  data_name='berka', data_dir=str(data_dir), seed=42, device='cpu',
                  n_epochs=2, small_n=0, eval_full=False, max_pipeline_len=10, verbose=False)
    first = DiffPrep(**kwargs, output_dir=str(tmp_path / 'first')).run()
    second = DiffPrep(**kwargs, output_dir=str(tmp_path / 'second')).run()
    assert first['best_pipeline_yaml'] == second['best_pipeline_yaml']
    contract = json.loads((tmp_path / 'first/search_contract.json').read_text(encoding='utf-8'))
    assert contract == first['search_contract']
    assert contract['version'] == 2
    assert contract['exported_joins']
