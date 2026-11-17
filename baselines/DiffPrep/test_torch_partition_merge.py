"""Exercise the tabular orchestrator without external data/model dependencies."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import pytest

from baselines.DiffPrep import diffprep
from baselines.DiffPrep.slot_planner import Slot
from baselines.common.op_spec import OpCategory
from baselines.common.pipeline import Pipeline


@pytest.mark.parametrize('task_type', ['tabular', 'rec'])
def test_minibatch_search_keeps_prepared_partitions(tmp_path, monkeypatch, task_type):
    from baselines.DiffPrep import data_protocol

    train = pd.DataFrame({'x': np.arange(12, dtype=float), 'target': [0, 1] * 6})
    val = pd.DataFrame({'x': np.arange(100, 106, dtype=float), 'target': [1, 0] * 3})
    ctx = SimpleNamespace(task_type=task_type, data_name='fixture', target_col='target',
                          id_col=None, time_col=None, numeric_cols=['x'], categorical_cols=[],
                          user_col='user_id', item_col='item_id')
    data = SimpleNamespace(_tabular_split_contract={'sentinel': 'canonical'},
                           _diffprep_join_selection={'mode': 'seeded'})
    data.split = lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not re-split'))
    prepared = []

    def prepare(actual, config, context, seed, small_n, **kwargs):
        assert actual is data and seed == 43 and small_n == 12
        prepared.append(True)
        return {'train': train.copy(), 'val': val.copy()}, ctx, []

    monkeypatch.setattr(data_protocol, 'prepare_tabular_search', prepare)
    if task_type == 'rec':
        from baselines.DiffPrep import rec_search
        from dppbench import rec_partition
        train['user_id'], train['item_id'] = 1, 1
        val['user_id'], val['item_id'] = 1, 1
        def prepare_rec(actual, config, seed, small_n):
            prepared.append(True)
            actual.interaction_df = pd.concat([train.assign(__split__='train'),
                                               val.assign(__split__='val')], ignore_index=True)
            actual.category_maps_ = {'user_id': {'u': 1}, 'item_id': {'i': 1}}
            actual._remap_ids = lambda: None
            actual._diffprep_sequence_step = None
            return ctx, [], {}
        monkeypatch.setattr(rec_search, 'prepare_rec_search', prepare_rec)
        monkeypatch.setattr(rec_partition, 'persist_partition', lambda *args: {'sentinel': 'canonical'})
        monkeypatch.setattr(rec_search, 'RecTransformBank', lambda *a, **k: pytest.fail('real bank called'))
    monkeypatch.setattr(data_protocol, 'assert_join_replay', lambda *a: None)
    evaluator = SimpleNamespace(n_unique_evaluations=0, _executor=SimpleNamespace(get_model_config=lambda: {}),
                                set_evaluate_std_test=lambda enabled: None)
    monkeypatch.setattr(diffprep, 'DiffPrepEvaluator', lambda **kwargs: evaluator)
    monkeypatch.setattr(diffprep.DiffPrep, '_build_context', lambda self, ev: (ctx, data))
    slots = [Slot(0, OpCategory.PREPROCESSING, ['ScaleFeature', 'Identity'], 'soft')]
    monkeypatch.setattr(diffprep, 'make_slots', lambda *a, **k: slots)
    def export(*args, **kwargs):
        pipe = Pipeline([])
        pipe.metadata = {}
        return pipe
    monkeypatch.setattr(diffprep, 'discretize', export)
    monkeypatch.setattr(diffprep, 'check_pipeline', lambda *a: SimpleNamespace(valid=True, errors=[], warnings=[]))
    calls = []
    forward = diffprep.ContinuousPipeline.forward

    def observed(self, x, mask, **kwargs):
        assert self.legacy_torch and not hasattr(self, 'tabular_bank')
        calls.append(x.detach().clone())
        return forward(self, x, mask, **kwargs)

    monkeypatch.setattr(diffprep.ContinuousPipeline, 'forward', observed)
    runner = diffprep.DiffPrep(task_dir='unused', data_name='fixture', seed=43,
                              n_epochs=2, small_n=12, batch_size=4, eval_full=False,
                              output_dir=str(tmp_path), verbose=False, join_overrides={})
    result = runner.run()
    assert prepared == [True]
    actual = data.train_df if task_type == 'tabular' else data.interaction_df.drop(columns='__split__')
    pd.testing.assert_frame_equal(actual, pd.concat([train, val], ignore_index=True))
    assert len(calls) == 10  # five proxy forwards per second-order epoch
    assert all(len(x) == 4 for x in calls)
    # Train features remain below 2 after train-fitted scaling; held-out values clip to 5.
    for epoch in range(2):
        group = calls[epoch * 5:(epoch + 1) * 5]
        assert torch.all(group[1] == 5)  # architecture validation
        assert all(torch.all(x < 2) for i, x in enumerate(group) if i != 1)
    contract = result['search_contract']
    assert contract['partition'] == data._tabular_split_contract
    assert contract['version'] == task_type + '_legacy_torch_fixed_partition_v1'


def test_real_partition_helper_preserves_validation_and_reads_train_only(tmp_path, monkeypatch):
    from baselines.DiffPrep.data_protocol import tabular_search_splits

    frame = pd.DataFrame({'id': range(20), 'target': [0, 1] * 10})
    (tmp_path / 'train_frozen.parquet').touch()
    reads = []
    def read(path, *args, **kwargs):
        assert path.name == 'train_frozen.parquet'
        reads.append(path.name)
        return frame.copy()
    monkeypatch.setattr(pd, 'read_parquet', read)
    data = SimpleNamespace(data_dir=str(tmp_path), target_col='target',
                           _std_test_dir=lambda _: str(tmp_path),
                           _normalize_frozen_frame=lambda x: x)
    data.split = lambda **kwargs: {'train': data.train_df.iloc[:16].copy(),
                                   'val': data.train_df.iloc[16:].copy()}
    splits = tabular_search_splits(data, {'train': {'val_ratio': .2}}, 43, small_n=5)
    assert reads == ['train_frozen.parquet']
    assert len(splits['train']) == 5
    assert splits['val'].id.tolist() == [16, 17, 18, 19]
    assert set(splits['train'].id).isdisjoint(splits['val'].id)
