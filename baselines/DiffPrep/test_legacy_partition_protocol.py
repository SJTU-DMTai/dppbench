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


def test_legacy_four_hard_slots_and_candidate_index_gate(monkeypatch):
    from baselines.DiffPrep import slot_planner
    categories = [OpCategory.INTEGRATION, OpCategory.CLEANING,
                  OpCategory.PREPROCESSING, OpCategory.FEATURE_ENGINEERING]
    candidates = {
        categories[0]: ['JoinTable'] + ['integration_' + str(i) for i in range(5)],
        categories[1]: ['Deduplicate', 'HandleError', 'FilterSample'],
        categories[2]: ['DropColumns'] + ['preprocessing_' + str(i) for i in range(6)],
        categories[3]: ['CreateSequence'] + ['features_' + str(i) for i in range(6)],
    }
    monkeypatch.setattr(slot_planner, 'get_category_order', lambda task: categories)
    monkeypatch.setattr(slot_planner, '_candidates_in_category', lambda cat, *a: candidates[cat])
    monkeypatch.setattr(slot_planner, 'is_mandatory', lambda name, task: name == 'CreateSequence')
    slots = slot_planner.make_slots('rec', SimpleNamespace(task_type='rec'))
    assert [(s.kind, s.n_candidates) for s in slots] == [('hard', 7), ('hard', 4), ('hard', 8), ('hard', 7)]
    model = diffprep.ContinuousPipeline(slots, n_features=2)
    x = torch.tensor([[1., 2.], [3., 4.]])
    expected = x.clone()
    for i, slot in enumerate(slots):
        indices = torch.arange(slot.n_candidates, dtype=x.dtype)
        indices -= indices.mean()
        weights = model.beta()[i, :, :slot.n_candidates]
        expected = expected * (1. + 1e-3 * (weights * indices).sum(-1))
    torch.testing.assert_close(model(x, torch.zeros_like(x), hard_sample=False), expected)


@pytest.mark.parametrize('task_type', ['tabular', 'rec'])
def test_minibatch_search_keeps_prepared_partitions(tmp_path, monkeypatch, task_type):
    from baselines.DiffPrep import partition_protocol as data_protocol

    train = pd.DataFrame({'x': np.arange(12, dtype=float), 'target': [0, 1] * 6})
    val = pd.DataFrame({'x': np.arange(100, 106, dtype=float), 'target': [1, 0] * 3})
    ctx = SimpleNamespace(task_type=task_type, data_name='fixture', target_col='target',
                          id_col=None, time_col=None, numeric_cols=['x'], categorical_cols=[],
                          user_col='user_id', item_col='item_id')
    data = SimpleNamespace(_tabular_split_contract={'sentinel': 'canonical'},
                           _diffprep_join_selection={'mode': 'seeded'})
    data.split = lambda *a, **k: (_ for _ in ()).throw(AssertionError('must not re-split'))
    prepared = []

    def prepare(actual, config, seed, small_n, **kwargs):
        assert actual is data and seed == 43 and small_n == 12
        prepared.append(True)
        return {'train': train.copy(), 'val': val.copy()}

    monkeypatch.setattr(data_protocol, 'tabular_search_splits', prepare)
    if task_type == 'rec':
        from baselines.DiffPrep import partition_protocol as rec_search
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
            return {}
        monkeypatch.setattr(rec_search, 'recommendation_search_partition', prepare_rec)
        monkeypatch.setattr(rec_partition, 'persist_partition', lambda *args: {'sentinel': 'canonical'})
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
        assert not hasattr(self, 'tabular_bank') and not hasattr(self, 'rec_bank')
        calls.append(x.detach().clone())
        return forward(self, x, mask, **kwargs)

    monkeypatch.setattr(diffprep.ContinuousPipeline, 'forward', observed)
    runner = diffprep.DiffPrep(task_dir='unused', data_name='fixture', seed=43,
                              n_epochs=2, small_n=12, batch_size=4, eval_full=False,
                              output_dir=str(tmp_path), verbose=False)
    from baselines.common.search_cost import SearchCost, search_cost_scope
    cost = SearchCost('DiffPrep')
    with search_cost_scope(cost):
        result = runner.run()
    ops = cost.snapshot()['operations']
    assert ops['search/proxy_work.train'] == {'supervised_rows': 32, 'forwarded_rows': 32}
    assert ops['search/proxy_work.val'] == {'supervised_rows': 8, 'forwarded_rows': 8}
    assert cost.fields()['search_proxy_train_loss_calls'] == 8
    assert cost.fields()['search_proxy_val_loss_calls'] == 2
    assert prepared == [True]
    actual = data.train_df if task_type == 'tabular' else data.interaction_df.drop(columns='__split__')
    pd.testing.assert_frame_equal(actual, pd.concat([train, val], ignore_index=True))
    assert len(calls) == 10  # five proxy forwards per second-order epoch
    assert all(len(x) == 4 for x in calls)
    # Check that each forward received rows from its own prepared partition.
    from baselines.DiffPrep.diffprep import _rec_feature_tensors, _tabular_feature_tensors
    if task_type == 'rec':
        full = _rec_feature_tensors(data, ctx, None, torch.device('cpu'))[0]
    else:
        full = _tabular_feature_tensors(data, ctx, None, torch.device('cpu'), fit_indices=np.arange(12))[0]
    for epoch in range(2):
        for i, batch in enumerate(calls[epoch * 5:(epoch + 1) * 5]):
            allowed = full[12:] if i == 1 else full[:12]
            assert all(any(torch.equal(row, candidate) for candidate in allowed) for row in batch)
    contract = result['search_contract']
    if task_type == 'tabular':
        assert contract['partition'] == data._tabular_split_contract
    else:
        assert contract['partition']['partition'] == data._tabular_split_contract
    assert contract['version'] == 'legacy_logic_current_partitions_v1'


def test_real_partition_helper_preserves_validation_and_reads_train_only(tmp_path, monkeypatch):
    from baselines.DiffPrep.partition_protocol import tabular_search_splits

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
